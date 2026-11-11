"""Comparison stage 8: put the LingBot-Map video mesh (c07b) into the LiDAR building frame and measure its error.
The video model has no scale, no gravity and no position, so a similarity transform (scale, rotation, shift) is found:
  1. level: 'up' = mean camera up (-y of each pose), refined with the floor/ceiling normals of the mesh
  2. scale guess: the floor-to-ceiling distance in the mesh vs the LiDAR storey height
  3. plan search: wall points (vertical faces) rasterised at 20 cm and cross-correlated (FFT, GPU) with the LiDAR wall
     plan over scale (guess +-20 %) x rotation (1 deg) x shift; best peak refined at 10 cm
  4. 3D refinement: trimmed similarity ICP (Umeyama) of the wall points against the LiDAR wall points
Then: distance of every video wall point to the nearest LiDAR wall point, and that error along the walk (drift).
  BH_WORK=... LB_DIR=work/<date>-video/lingbot LB_TAG=fine python c08_lingbot_align.py
Out (LB_DIR): <tag>_aligned_mesh.ply (building frame, metres), <tag>_align.json, <tag>_align.png"""
import os, sys, json, time
import numpy as np, open3d as o3d, torch
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

WORK = os.environ['BH_WORK']; FI = int(os.environ.get('BH_FLOOR', 2)); D = os.path.join(WORK, 'compare', f'f{FI}')
LB = os.environ['LB_DIR']; TAG = os.environ.get('LB_TAG', 'fine'); t0 = time.time()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- LiDAR reference -------------------------------------------------------------------------------------------
ref = np.load(os.path.join(D, 'walls_ref.npz')); fz = float(ref['floor_z'])
R = ref['xyz']; R = R[np.unique(np.floor(R / 0.05).astype(np.int64), axis=0, return_index=True)[1]]
P = np.load(os.path.join(D, 'points.npz')); Pz, Pn = P['xyz'], P['nrm']
hz = Pz[np.abs(Pn[:, 2]) > 0.9, 2]; hz = hz[(hz > fz + 1.8) & (hz < fz + 5)]
hist, edges = np.histogram(hz, bins=np.arange(fz + 1.8, fz + 5, 0.02)); ceil_l = float(edges[hist.argmax()] + 0.01)
H_l = ceil_l - fz
Rw = R[(R[:, 2] > fz + 0.3) & (R[:, 2] < fz + 2.0)]; treeR = cKDTree(Rw)

# ---- video mesh ------------------------------------------------------------------------------------------------
m = o3d.io.read_triangle_mesh(os.path.join(LB, f'{TAG}_mesh.ply')); m.compute_vertex_normals()
V = np.asarray(m.vertices); N = np.asarray(m.vertex_normals)
pred = np.load(os.path.join(LB, 'pred_long_w.npz'), mmap_mode=None) if not os.path.exists(os.path.join(LB, 'poses.npz')) else None
if pred is not None:
    np.savez(os.path.join(LB, 'poses.npz'), extrinsic=pred['extrinsic']); del pred
E = np.load(os.path.join(LB, 'poses.npz'))['extrinsic']
cam = np.array([-e[:, :3].T @ e[:, 3] for e in E])
up = -np.mean(E[:, 1, :3], 0); up /= np.linalg.norm(up)
for _ in range(3):                                                   # refine with near-horizontal surfaces
    c = N @ up; h = N[np.abs(c) > 0.95] * np.sign(c[np.abs(c) > 0.95])[:, None]
    up = h.mean(0); up /= np.linalg.norm(up)
def rot_to_z(u):
    v = np.cross(u, [0, 0, 1]); s = np.linalg.norm(v); c = u[2]
    if s < 1e-9: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    a = np.arctan2(s, c); return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
Rl = rot_to_z(up); Vl = V @ Rl.T; Nl = N @ Rl.T; cl = cam @ Rl.T
camz = np.median(cl[:, 2])
fl = Vl[(Nl[:, 2] > 0.95) & (Vl[:, 2] < camz)][:, 2]; ce = Vl[(Nl[:, 2] < -0.95) & (Vl[:, 2] > camz)][:, 2]
def mode(x, w):
    hh, ee = np.histogram(x, bins=np.arange(x.min(), x.max() + w, w)); return float(ee[hh.argmax()] + w / 2)
span = np.ptp(Vl[:, 2]); floor_v = mode(fl, span / 400); ceil_v = mode(ce, span / 400) if len(ce) > 100 else np.nan
s0 = H_l / (ceil_v - floor_v) if np.isfinite(ceil_v) else (1.5 / (camz - floor_v))
print(f'LiDAR storey height {H_l:.2f} m; video floor {floor_v:.4f} ceiling {ceil_v:.4f} -> scale guess {s0:.2f} m/unit; camera height {(camz - floor_v) * s0:.2f} m')

# wall points of the video, 0.3-2.0 m above its floor
Vm = (Vl - [0, 0, floor_v]) * s0
wall = (np.abs(Nl[:, 2]) < 0.3) & (Vm[:, 2] > 0.3) & (Vm[:, 2] < 2.0)
Wv = Vl[wall] - [0, 0, floor_v]                                       # units, floor at 0
Wv = Wv[np.unique(np.floor(Wv * s0 / 0.05).astype(np.int64), axis=0, return_index=True)[1]]

# ---- plan search -----------------------------------------------------------------------------------------------
def raster(xy, lo, shape, res):
    q = np.floor((xy - lo) / res).astype(int); ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < shape[1]) & (q[:, 1] < shape[0])
    img = np.zeros(shape, np.float32); np.add.at(img, (q[ok, 1], q[ok, 0]), 1); return np.minimum(img, 3)
def search(res, scales, thetas, centre=None, win=None):
    ext = np.ptp(Wv[:, :2], 0).max() * max(scales) * 0.75 + 2
    lo = Rw[:, :2].min(0) - ext; hi = Rw[:, :2].max(0) + ext; shape = tuple((np.ceil((hi - lo) / res)).astype(int)[::-1])
    A = gaussian_filter(raster(Rw[:, :2], lo, shape, res), 1.0)
    FA = torch.fft.rfft2(torch.tensor(A, device=dev)); best = (-1,)
    for s in scales:
        for th in thetas:
            c, sn = np.cos(th), np.sin(th); xy = Wv[:, :2] @ np.array([[c, sn], [-sn, c]]) * s
            mc = xy.mean(0); xy0 = xy - mc
            B = gaussian_filter(raster(xy0, -np.array(shape[::-1]) * res / 2, shape, res), 1.0)
            FB = torch.fft.rfft2(torch.tensor(B, device=dev))
            cc = torch.fft.irfft2(FA * FB.conj(), s=shape); cc = torch.fft.fftshift(cc)
            v, i = torch.max(cc.flatten(), 0); v = float(v) / (np.sqrt((B * B).sum()) + 1e-9)
            if v > best[0]:
                iy, ix = np.unravel_index(int(i), shape)
                shift = np.array([ix - shape[1] // 2, iy - shape[0] // 2]) * res        # B centre -> A position
                t = lo + np.array(shape[::-1]) * res / 2 + shift - mc
                best = (v, s, th, t)
    return best
coarse = search(0.2, s0 * np.arange(0.8, 1.21, 0.02), np.radians(np.arange(0, 360, 1.0)))
print('coarse', coarse[0], coarse[1], np.degrees(coarse[2]))
fine = search(0.1, coarse[1] * np.arange(0.97, 1.031, 0.005), coarse[2] + np.radians(np.arange(-1.5, 1.6, 0.25)))
_, s1, th1, t1 = fine; print('fine', fine[0], s1, np.degrees(th1), t1)

# ---- similarity ICP in 3D on wall points --------------------------------------------------------------------------
def sim_of(s, th, t):
    c, sn = np.cos(th), np.sin(th); Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    return s, Rz, np.array([t[0], t[1], fz])
def apply(X, T): s, Rr, t = T; return s * X @ Rr.T + t
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; S = Yc.T @ Xc / len(X)
    U, d, Vt = np.linalg.svd(S); Dm = np.eye(3); Dm[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rr = U @ Dm @ Vt; s = np.trace(np.diag(d) @ Dm) / Xc.var(0).sum(); return s, Rr, my - s * Rr @ mx
# the ICP acts on wall points expressed in units with the floor at 0, levelled
T = sim_of(s1, th1, t1)
# note: the plan search rotates row-vectors by [[c, s], [-s, c]] (i.e. angle th) -> Rz(th) for column vectors
for tol in (0.6, 0.4, 0.25, 0.15, 0.1, 0.1):
    Y = apply(Wv, T); d, j = treeR.query(Y); ok = d < tol
    T = umeyama(Wv[ok], Rw[j[ok]])
Y = apply(Wv, T); d, _ = treeR.query(Y)
s_f = T[0]
print(f'ICP: scale {s_f:.3f} m/unit; inliers(<10cm) {np.mean(d < 0.1):.3f}; median error {np.median(d) * 100:.1f} cm')

# ---- error along the walk (drift) ---------------------------------------------------------------------------------
camB = apply(cl - [0, 0, floor_v], T)
step = np.r_[0, np.cumsum(np.linalg.norm(np.diff(camB[:, :2], axis=0), axis=1))]
tw = cKDTree(Y[:, :2]); seg = []
for k in range(0, len(camB), 25):
    idx = tw.query_ball_point(camB[k, :2], 3.0)
    if len(idx) > 50: seg.append([float(step[k]), float(np.median(d[idx])), float(np.mean(d[idx] < 0.1))])
seg = np.array(seg)

# ---- outputs ---------------------------------------------------------------------------------------------------
m2 = o3d.geometry.TriangleMesh(m); Vb = apply(Vl - [0, 0, floor_v], T); m2.vertices = o3d.utility.Vector3dVector(Vb)
m2.compute_vertex_normals(); o3d.io.write_triangle_mesh(os.path.join(LB, f'{TAG}_aligned_mesh.ply'), m2)
np.save(os.path.join(LB, f'{TAG}_cams_building.npy'), camB)
out = dict(lidar_storey_height=H_l, scale_guess=s0, plan_search=dict(score=fine[0], scale=s1, rot_deg=float(np.degrees(th1))),
           scale_m_per_unit=float(s_f), camera_height_m=float((camz - floor_v) * s_f), walk_length_m=float(step[-1]),
           wall_points=int(len(Wv)), median_error_cm=float(np.median(d) * 100), within_10cm=float(np.mean(d < 0.1)),
           within_25cm=float(np.mean(d < 0.25)), along_walk=seg.tolist(), seconds=round(time.time() - t0, 1),
           transform=dict(level=Rl.tolist(), floor_units=floor_v, s=float(s_f), R=T[1].tolist(), t=T[2].tolist()))
json.dump(out, open(os.path.join(LB, f'{TAG}_align.json'), 'w'), indent=1, default=float)
fig, ax = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw=dict(width_ratios=[1.3, 1]))
lo_, hi_ = Y[:, :2].min(0) - 3, Y[:, :2].max(0) + 3; sel = np.all((Rw[:, :2] > lo_) & (Rw[:, :2] < hi_), 1)
ax[0].scatter(Rw[sel, 0], Rw[sel, 1], s=0.05, c='0.6', label='LiDAR walls')
ax[0].scatter(Y[:, 0], Y[:, 1], s=0.05, c=np.clip(d / 0.3, 0, 1), cmap='RdYlGn_r', vmin=0, vmax=1)
ax[0].plot(camB[:, 0], camB[:, 1], 'b', lw=0.6); ax[0].set_aspect('equal'); ax[0].set_title('video walls (green <= 0 cm ... red >= 30 cm from LiDAR) over LiDAR (grey)')
ax[1].plot(seg[:, 0], seg[:, 1] * 100); ax[1].set_xlabel('distance walked (m)'); ax[1].set_ylabel('median wall error within 3 m (cm)'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(LB, f'{TAG}_align.png'), dpi=70)
print(json.dumps({k: v for k, v in out.items() if k not in ('along_walk', 'transform')}, indent=1, default=float))
