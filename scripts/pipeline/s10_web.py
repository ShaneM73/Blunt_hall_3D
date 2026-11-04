"""Stage 10: web data for one dataset -> data/<id>/
- floors: detected from horizontal surfaces (|nz| > 0.9) that the walking path stood on
- points: people-free colored cloud on a VOXEL grid, one file per floor (split into parts <= 4 M points)
- mesh: one PLY per mesh tile (s08), with per-tile z range for floor filtering
- SLAM map per floor, trajectory, meta.json
Usage: s10_web.py <id> "<title>"     env BH_VOXEL (cm, default 2)"""
import os, sys, json
import numpy as np, open3d as o3d, cv2
from common import W, REPO, RAW, traj, log

DID, TITLE = sys.argv[1], sys.argv[2]
OUT = os.path.join(REPO, 'data', DID); os.makedirs(OUT, exist_ok=True)
VOX = float(os.environ.get('BH_VOXEL', 2.0)) / 100
d = np.load(W('color', 'hq_colored.npz')); rm = np.load(W('clouds', 'person_rm.npy')); k = d['has'] & ~rm
P, C, I, Nr = d['xyz'][k], d['rgb'][k], d['inten'][k], d['nrm'][k]
T = traj(); tz = T[:, 3]

# ---- floors: histogram of up-facing horizontal points; keep peaks the operator walked on (path ~0.9-1.6 m above)
hz = P[Nr[:, 2] > 0.9, 2]           # up-facing only (normals face the path): ceilings are excluded
h, e = np.histogram(hz, bins=np.arange(hz.min(), hz.max() + 0.05, 0.05))
cand = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.01 * h.max()]
floors = []
for i in sorted(cand, key=lambda i: -h[i]):
    z = (e[i] + e[i + 1]) / 2; above = tz - z; on = ((above > 0.8) & (above < 1.7)).sum()
    if on > 150 and all(abs(z - f['z']) > 2.0 for f in floors):     # >= 15 s of walking on it
        floors.append(dict(z=round(float(z), 2), dwell_s=round(on / 10, 1)))
floors.sort(key=lambda f: f['z'])
for j, f in enumerate(floors):
    f['zmin'] = round(f['z'] - 0.3, 2); f['zmax'] = round(floors[j + 1]['z'] - 0.3, 2) if j + 1 < len(floors) else round(float(P[:, 2].max()) + 0.1, 2)
    dz = P[(Nr[:, 2] < -0.9) & (P[:, 2] > f['z'] + 2.0) & (P[:, 2] < min(f['zmax'] + 0.3, f['z'] + 6)), 2]   # down-facing = ceilings
    if len(dz) > 1000:
        hc, ec = np.histogram(dz, bins=np.arange(dz.min(), dz.max() + 0.1, 0.05)); cz_ = float(ec[np.argmax(hc)] + 0.025)
    else: cz_ = min(f['zmax'] + 0.3, f['z'] + 3.0)
    f['ceiling_z'] = round(cz_, 2); f['ceiling'] = round(cz_ - 0.15, 2)
main = int(np.argmax([f['dwell_s'] for f in floors]))
for j, f in enumerate(floors): f['name'] = 'Main level' if j == main else f'Level {j - main:+d}'
log('floors', floors)

# ---- points per floor
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64))); pc.colors = o3d.utility.Vector3dVector(C / 255.)
v, _, tr = pc.voxel_down_sample_and_trace(VOX, pc.get_min_bound(), pc.get_max_bound())
Pv = np.asarray(v.points); Cv = (np.asarray(v.colors) * 255).round().astype(np.uint8); Iv = I[np.array([t[0] for t in tr])]
log('web points', len(Pv), 'at', VOX * 100, 'cm')
def write_pts(fn, X, Cc, Ii):
    o = X.min(0); sc = float(max((X.max(0) - o).max(), 1e-3) / 65535); q = np.round((X - o) / sc).astype(np.uint16)
    with open(os.path.join(OUT, fn), 'wb') as f:
        f.write(b'BHPC'); f.write(np.array([len(X), 1], np.uint32).tobytes()); f.write(np.array([*o, sc], np.float32).tobytes())
        f.write(q.tobytes()); f.write(Ii.astype(np.uint8).tobytes()); f.write(Cc.tobytes())
for j, f in enumerate(floors):
    m = (Pv[:, 2] >= f['zmin']) & (Pv[:, 2] < f['zmax']); ii = np.flatnonzero(m)
    ii = ii[np.argsort(Pv[ii, 0] + Pv[ii, 1])]          # spatially coherent parts
    parts = np.array_split(ii, max(1, int(np.ceil(len(ii) / 4_000_000))))
    f['points'] = []
    for p_, part in enumerate(parts):
        fn = f'points_f{j}_{p_}.bin'; write_pts(fn, Pv[part], Cv[part], Iv[part]); f['points'].append(fn)
    f['n_points'] = int(m.sum())
    log(f['name'], 'points', f['n_points'], 'files', len(parts))

# ---- mesh: each tile split by floor band (triangle centroid z), decimated to WEB_TRI_M2 per m^2 of piece footprint
WEB_TRI_M2 = float(os.environ.get('BH_WEB_TRI_M2', 2000))
tj = json.load(open(W('mesh', 'tiles.json'))); tiles = []; ntri = 0
for t in tj['tiles']:
    M = np.load(W('mesh', 'tiles', t['file'])); V0, F0, C0 = M['v'], M['f'], M['c']; cz = V0[F0][:, :, 2].mean(1)
    for j, fl in enumerate(floors):
        sel = (cz >= fl['zmin']) & (cz < fl['zmax'])
        if sel.sum() < 2000: continue
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V0.astype(np.float64)), o3d.utility.Vector3iVector(F0[sel]))
        m.vertex_colors = o3d.utility.Vector3dVector(C0 / 255.); m.remove_unreferenced_vertices()
        Vp = np.asarray(m.vertices); area = len(np.unique(np.floor(Vp[:, :2] / 0.25), axis=0)) * 0.0625
        target = int(max(5000, WEB_TRI_M2 * area))
        if len(m.triangles) > target: m = m.simplify_quadric_decimation(target)
        m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
        V = np.asarray(m.vertices).astype(np.float32); F = np.asarray(m.triangles).astype(np.int32); Cm = (np.asarray(m.vertex_colors) * 255).round().astype(np.uint8)
        hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n"
        vb = np.zeros(len(V), dtype=[('p', '<f4', 3), ('c', 'u1', 3)]); vb['p'] = V; vb['c'] = Cm
        fb = np.zeros(len(F), dtype=[('n', 'u1'), ('i', '<i4', 3)]); fb['n'] = 3; fb['i'] = F
        fn = f"mesh_f{j}_{t['i']}_{t['j']}.ply"
        with open(os.path.join(OUT, fn), 'wb') as f: f.write(hdr.encode()); f.write(vb.tobytes()); f.write(fb.tobytes())
        tiles.append(dict(file=fn, tris=len(F), floors=[j])); ntri += len(F)
log('web mesh pieces', len(tiles), 'triangles', ntri, '(full-resolution mesh: %d)' % sum(t['tris'] for t in tj['tiles']))

# ---- SLAM map per floor: 5 cm cells; wall = occupied over >= 1.2 m of a 0.15-2.2 m slice, furniture = partly, floor = seen
res = 0.05; xmin, ymin = np.floor(P[:, :2].min(0)) - 1; xmax, ymax = np.ceil(P[:, :2].max(0)) + 1
nx, ny = int((xmax - xmin) / res), int((ymax - ymin) / res)
for j, f in enumerate(floors):
    fl = f['z']; Q = P[(P[:, 2] > fl + 0.15) & (P[:, 2] < fl + 2.2)]
    ix = ((Q[:, 0] - xmin) / res).astype(int); iy = ((ymax - Q[:, 1]) / res).astype(int); iz = ((Q[:, 2] - fl - 0.15) / 0.1).astype(int)
    occ = np.zeros((ny, nx, 21), bool); occ[iy, ix, iz] = True; vb = occ.sum(2)
    Fp = P[np.abs(P[:, 2] - fl) < 0.08]; fx = ((Fp[:, 0] - xmin) / res).astype(int); fy = ((ymax - Fp[:, 1]) / res).astype(int)
    Fl = np.zeros((ny, nx), bool); Fl[fy, fx] = True
    img = np.full((ny, nx, 3), (225, 229, 233), np.uint8); img[Fl] = (252, 252, 250); img[(vb >= 3) & (vb < 12)] = (150, 150, 155); img[vb >= 12] = (30, 32, 36)
    fn = f'slam_map_f{j}.png'; cv2.imwrite(os.path.join(OUT, fn), img); f['slam_map'] = fn
slam = dict(res=res, xmin=float(xmin), ymax=float(ymax), grid_cm=5, slice='0.15–2.2 m above each floor',
            note='Black = walls (occupied over ≥1.2 m of height), gray = furniture and low objects, white = floor seen by the scanner, blue-gray = never observed. Grid 5 cm.')

# ---- trajectory (every 5th pose), camera stations, meta
cams = np.load(W('decoded', 'cam_positions.npy'))
json.dump({'t': (T[::5, 0] - T[0, 0]).round(1).tolist(), 'p': T[::5, 1:4].round(3).tolist(), 'cams': cams[::3, 2:5].round(2).tolist()},
          open(os.path.join(OUT, 'trajectory.json'), 'w'))
fz = floors[main]['z']; near = T[np.abs(T[:, 3] - fz - 1.2) < 0.6, 1:4]
info = json.load(open(W('logs', 'dataset_info.json'))) if os.path.exists(W('logs', 'dataset_info.json')) else {}
meta = dict(id=DID, title=TITLE, voxel_cm=VOX * 100, floors=floors, main_floor=main, slam_map=slam,
            n_raw=int(len(np.load(W('decoded', 'world_int.npy'), mmap_mode='r'))), n_frames=len(T), n_images=len(json.load(open(W('decoded', 'images.json')))),
            duration_s=round(float(T[-1, 0] - T[0, 0]), 1), path_m=round(float(np.linalg.norm(np.diff(T[:, 1:4], axis=0), axis=1).sum()), 1),
            rgb_frac=1.0, crop_m=5, focus=np.median(near, 0).round(2).tolist() if len(near) else np.median(T[:, 1:4], 0).round(2).tolist(),
            rgb_note='Cropped to within 5 m of the walking path. Photo exposure and white balance equalized per image before blending; people removed with photo masks and LiDAR free-space carving.',
            model=dict(tiles=tiles, tris=ntri, short='Screened Poisson, tiled',
                       method=f'Screened Poisson reconstruction (~9 mm octree cells) of the denoised, people-free 1 cm LiDAR cloud in {len(tj["tiles"])} tiles of {tj["tile"]:.0f} m, trimmed to within 3 cm of real points and decimated to {ntri:,} triangles for the web (~{WEB_TRI_M2:.0f} per m²); vertex colors from exposure-equalized photos.'),
            **info)
lf = os.path.join(RAW, 'SLAM_DATA', 'TMP_LoopSearch', 'loop_ends.txt')
meta['loop_closures'] = sum(1 for l in open(lf) if l.strip()) if os.path.exists(lf) else 0
old = os.path.join(OUT, 'meta.json')
if os.path.exists(old):
    o = json.load(open(old))
    for k_ in ('splat',):
        if k_ in o: meta[k_] = o[k_]
json.dump(meta, open(old, 'w'), indent=1)
log('wrote', OUT, 'MB', round(sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e6, 1))
