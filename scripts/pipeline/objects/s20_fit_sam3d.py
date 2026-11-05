"""Stage 20 (objects): place each SAM 3D mesh (s19) in the building by fitting it to the object's LiDAR points.
SAM 3D meshes are upright (glb Y-up -> Z-up), so the fit is yaw + uniform scale + translation (7 -> 5 dof):
36 yaw starts, each refined by trimmed point-to-surface ICP (LiDAR points -> nearest mesh sample; 2-D Procrustes about Z + scale),
best = lowest trimmed RMS. Scale starts from the footprint-area ratio (x 0.85/1/1.15) and is kept within 0.6-1.6 of it; the mesh bottom stays on the floor.
Quality: cover = fraction of LiDAR points within 3 cm of the fitted mesh; fits with cover < BH_FIT_MIN (0.6) keep the LiDAR mesh.
Out: objects/sam3d_fit/obj_<id>_<label>.ply (building coordinates, vertex colours) + fit.json"""
import os, sys, json, glob
import numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log

MINC = float(os.environ.get('BH_FIT_MIN', 0.6))
I = np.load(W('objects', 'instances.npz')); P = np.load(W('color', 'hq_colored.npz'))['xyz'][I['point']]; inst = I['inst']
index = {o['id']: o for o in json.load(open(W('objects', 'meshes', 'index.json')))['objects']}
OUT = W('objects', 'sam3d_fit'); os.makedirs(OUT, exist_ok=True)
fp = os.path.join(OUT, 'fit.json'); fits = json.load(open(fp)) if os.path.exists(fp) else {}
YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)                   # glb Y-up -> Z-up
rng = np.random.default_rng(0)
FLOORS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', '2026-11-04', 'meta.json')))['floors'][:4]


def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


for d in sorted(os.listdir(W('objects', 'sam3d_out'))):
    i = int(d); o = index.get(i); mp = W('objects', 'sam3d_out', d, 'mesh.glb')
    if o is None or not os.path.exists(mp): continue
    m = trimesh.load(mp, force='mesh'); m.vertices = np.asarray(m.vertices) @ YUP.T
    X = P[inst == i]; X = X[rng.choice(len(X), min(len(X), 4000), replace=False)]
    S, fi = trimesh.sample.sample_surface_even(m, 20000) if len(m.faces) else (None, None)
    S = np.asarray(S); mlo, mhi = S.min(0), S.max(0); lo, hi = np.percentile(X, 1, 0), np.percentile(X, 99, 0)
    # scale from the footprint (labelled LiDAR often misses legs / the lower part, so height under-estimates);
    # the object stands on the floor of its level: mesh bottom pinned to the floor height
    fz = max([f['z'] for f in FLOORS if f['z'] <= lo[2] + 0.1] or [lo[2]])
    s0 = np.sqrt(np.prod(hi[:2] - lo[:2]) / np.prod(mhi[:2] - mlo[:2])); best = None
    for a0, sm in [(a, b) for a in np.radians(np.arange(0, 360, 10)) for b in (0.85, 1.0, 1.15)]:
        R, s = Rz(a0), s0 * sm; t = np.r_[(lo[:2] + hi[:2]) / 2, 0] - s * R @ np.r_[(mlo[:2] + mhi[:2]) / 2, 0]; t[2] = fz - s * mlo[2]
        for it in range(30):
            Y = s * S @ R.T + t; tree = cKDTree(Y); dd, nn = tree.query(X)
            k = dd <= np.quantile(dd, 0.9); A, B = Y[nn[k]], X[k]                 # trimmed correspondences mesh->LiDAR
            # solve B ~ s' Rz(da) A + t' (yaw-only increment, uniform scale)
            ca, cb = A.mean(0), B.mean(0); A0, B0 = A - ca, B - cb
            h = (A0[:, 0] * B0[:, 1] - A0[:, 1] * B0[:, 0]).sum(); c = (A0[:, 0] * B0[:, 0] + A0[:, 1] * B0[:, 1]).sum(); da = np.arctan2(h, c)
            Rd = Rz(da); Ar = A0 @ Rd.T; ds = (Ar * B0).sum() / (Ar * Ar).sum(); ds = np.clip(ds * s, 0.6 * s0, 1.6 * s0) / s
            R = Rd @ R; s = s * ds; t = ds * Rd @ (t - ca) + cb; t[2] = fz - s * mlo[2]
            if abs(da) < 1e-4 and abs(ds - 1) < 1e-4: break
        Y = s * S @ R.T + t; dd, _ = cKDTree(Y).query(X); rms = np.sqrt(np.mean(np.sort(dd)[:int(0.9 * len(dd))] ** 2))
        if best is None or rms < best[0]: best = (rms, R, s, t, (dd < 0.03).mean(), fz)
    rms, R, s, t, cover, fz = best
    v = s * np.asarray(m.vertices) @ R.T + t; out = m.copy(); out.vertices = v
    fn = f"obj_{i:03d}_{o['label'].replace(' ', '_')}.ply"; out.export(os.path.join(OUT, fn))
    fits[d] = dict(id=i, label=o['label'], file=fn, rms_cm=round(rms * 100, 2), cover=round(float(cover), 3), scale=round(float(s), 3),
                   yaw_deg=round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 1), floor_z=round(float(fz), 3), use=bool(cover >= MINC))
    log('fit', fits[d])
json.dump(fits, open(fp, 'w'), indent=1)
