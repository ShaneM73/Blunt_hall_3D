"""Comparison stage 6: 3D data for the comparison page (compare.html, "3D models" section).
Per method: the wall mesh split into its walls (connected pieces); each wall is checked against the measured LiDAR wall
points the same way as the plan score (surface samples 0.3-2.0 m above the floor within 10 cm + half the wall's
thickness of a measured point) and flagged "on a measured wall" when at least half its samples are.
Out: data/compare/m3d_<method>.bin   float32 xyz of non-indexed triangles (building frame, metres, z up) followed by one
                                     uint8 flag per triangle (1 = on a measured wall, 0 = not)
     data/compare/lidar_3d.bin       int16 xyz in cm (relative to 'origin') of the measured wall points, 6 cm grid
     data/compare/compare3d.json     triangle counts, per-method wall counts / share on measured walls, doors, origin"""
import os, sys, json
import numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log

FI = int(os.environ.get('BH_FLOOR', 2)); D = W('compare', os.environ.get('BH_CMP_SET', f'f{FI}')); TOL = 0.10
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'compare')
SHOW = ['ours', 'cloud2bim', 'scenescript', 'scenescript-t8-20k']
ref = np.load(os.path.join(D, 'walls_ref.npz')); fz = float(ref['floor_z'])
R = ref['xyz']; R = R[np.unique(np.floor(R / 0.05).astype(np.int64), axis=0, return_index=True)[1]]; tree = cKDTree(R)
rng = np.random.default_rng(0); meta = dict(floor_z=fz, methods={})
for m in SHOW:
    z = np.load(os.path.join(D, f'{m}_walls.npz')); mesh = trimesh.Trimesh(z['V'], z['F'], process=True)
    parts = mesh.split(only_watertight=False) if len(mesh.faces) else []
    tris, flags, n_ok = [], [], 0
    for p in parts:
        ext = p.bounding_box_oriented.primitive.extents if len(p.vertices) > 3 else np.zeros(3)
        half = float(np.sort(ext)[0]) / 2 if p.is_volume or len(p.faces) > 2 else 0.0             # planes: no thickness
        S, _ = trimesh.sample.sample_surface(p, max(60, int(p.area / 0.01)), seed=0)
        S = S[(S[:, 2] > fz + 0.3) & (S[:, 2] < fz + 2.0)]
        ok = len(S) > 0 and (tree.query(S)[0] <= TOL + half).mean() >= 0.5
        n_ok += ok; tris.append(p.vertices[p.faces].reshape(-1, 3)); flags.append(np.full(len(p.faces), ok, np.uint8))
    T = np.vstack(tris).astype(np.float32) if tris else np.zeros((0, 3), np.float32); Fl = np.concatenate(flags) if flags else np.zeros(0, np.uint8)
    with open(os.path.join(OUT, f'm3d_{m}.bin'), 'wb') as f: f.write(T.tobytes()); f.write(Fl.tobytes())
    d = z['doors'] if 'doors' in z.files else np.zeros((0, 6))
    meta['methods'][m] = dict(triangles=int(len(Fl)), walls=len(parts), walls_on_measured=int(n_ok),
                              doors=[[round(float(v), 2) for v in r] for r in d])
    log(m, len(parts), 'walls,', n_ok, 'on measured walls,', len(Fl), 'triangles')
P = ref['xyz']                                                       # measured wall surfaces (0.3-2.0 m), the scoring reference
P = P[np.unique(np.floor(P / 0.06).astype(np.int64), axis=0, return_index=True)[1]]
if len(P) > 300000: P = P[rng.choice(len(P), 300000, replace=False)]
org = np.floor(P.min(0)); q = np.round((P - org) * 100).astype(np.int16)
q.tofile(os.path.join(OUT, 'lidar_3d.bin')); meta.update(origin=org.tolist(), lidar_points=int(len(q)))
json.dump(meta, open(os.path.join(OUT, 'compare3d.json'), 'w'), separators=(',', ':'))
log('3D data ->', OUT, {f: os.path.getsize(os.path.join(OUT, f)) // 1024 for f in os.listdir(OUT) if f.endswith('.bin')})
