"""Stage 1: decode trajectory, register all LiDAR frames to world, index photos, overview plot.
Out: decoded/slam_trajectory.npy (n,14)  decoded/world_{xyz,frame,int}.npy  decoded/images.json  renders/overview.png
Optional: --laz writes decoded/registered.laz"""
import re, sys, json, os
import numpy as np
from common import W, RAW, read_slam_all, frame_points, load_tra, quat_R, log

T = load_tra(); np.save(W('decoded', 'slam_trajectory.npy'), T)
idx, pts = read_slam_all()
log('frames: trajectory', len(T), 'points index', len(idx), 'points', sum(len(p) for p in pts))
assert len(T) == len(idx), 'trajectory / point-frame count mismatch'
log('max |index time - traj time| s', np.abs(idx[:, 1] - np.floor(T[:, 0])).max())

N = int(idx[:, 0].sum())
xyz = np.lib.format.open_memmap(W('decoded', 'world_xyz.npy'), 'w+', np.float32, (N, 3))
fr = np.empty(N, np.int32); it = np.empty(N, np.uint8); rg = np.empty(N, np.uint8)
n = 0
for k, p in frame_points(idx, pts):
    B = np.stack([p['x'], p['y'], p['z']], 1).astype(np.float64)
    m = np.linalg.norm(B, axis=1) > 0.3          # drop operator / near-sensor hits
    c = int(m.sum())
    xyz[n:n + c] = B[m] @ quat_R(T[k, 4:8]).T + T[k, 1:4]
    fr[n:n + c] = k; it[n:n + c] = p['a'][m]; rg[n:n + c] = p['b'][m]; n += c
    if k % 2000 == 0: log('frame', k)
xyz.flush(); del xyz
xyz = np.lib.format.open_memmap(W('decoded', 'world_xyz.npy'), 'r+')
# trim to n (open_memmap cannot shrink: rewrite header shape)
if n < N:
    a = np.array(xyz[:n]); del xyz; np.save(W('decoded', 'world_xyz.npy'), a); xyz = a
np.save(W('decoded', 'world_frame.npy'), fr[:n]); np.save(W('decoded', 'world_int.npy'), it[:n]); np.save(W('decoded', 'world_ring.npy'), rg[:n])
log('registered points', n, 'extent', np.asarray(xyz).min(0).round(1), np.asarray(xyz).max(0).round(1))

# photo index: JPEGs inside IMG/CameraN/CameraN.cam, each preceded by 0xAAAAAAAA + '<unix>.jpg'
recs = []
for cam in (1, 2, 3):
    d = open(os.path.join(RAW, 'IMG', f'Camera{cam}', f'Camera{cam}.cam'), 'rb').read()
    for m in re.finditer(rb'\xaa\xaa\xaa\xaa([0-9.]+)\.jpg', d):
        s = d.find(b'\xff\xd8\xff', m.end()); e = d.find(b'\xff\xd9', s) + 2
        recs.append(dict(cam=cam, t=float(m.group(1)), off=s, len=e - s))
    log('camera', cam, 'images', sum(r['cam'] == cam for r in recs))
inside = [r for r in recs if T[0, 0] <= r['t'] <= T[-1, 0]]
json.dump(inside, open(W('decoded', 'images.json'), 'w'))
log('images within trajectory time', len(inside), 'of', len(recs))

if '--laz' in sys.argv:
    import laspy
    h = laspy.LasHeader(point_format=1, version='1.4'); X = np.asarray(xyz); h.offsets = X.min(0); h.scales = [0.001] * 3
    las = laspy.LasData(h); las.x, las.y, las.z = X.T; las.intensity = it[:n].astype(np.uint16) * 256
    las.gps_time = T[fr[:n], 0]; las.point_source_id = rg[:n]; las.write(W('decoded', 'registered.laz'))

# overview: plan slice, side view, height histogram (finds floors)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X = np.asarray(xyz[::8]); x, y, z = X.T
fig, ax = plt.subplots(1, 3, figsize=(30, 10), gridspec_kw=dict(width_ratios=[3, 3, 1]))
ax[0].hist2d(x, y, bins=1500, cmap='gray_r', vmax=40); ax[0].plot(T[:, 1], T[:, 2], 'r-', lw=0.6); ax[0].set_aspect('equal'); ax[0].set_title('plan (all heights) + path')
ax[1].hist2d(x, z, bins=1500, cmap='gray_r', vmax=60); ax[1].plot(T[:, 1], T[:, 3], 'r-', lw=0.6); ax[1].set_aspect('equal'); ax[1].set_title('side X-Z')
ax[2].hist(z, bins=400, orientation='horizontal', log=True); ax[2].hist(T[:, 3], bins=200, orientation='horizontal', alpha=.6); ax[2].set_title('z histogram (points, path)')
plt.tight_layout(); plt.savefig(W('renders', 'overview.png'), dpi=60)
log('path length m', np.linalg.norm(np.diff(T[:, 1:4], axis=0), axis=1).sum().round(1), 'duration s', round(T[-1, 0] - T[0, 0], 1),
    'path z range', T[:, 3].min().round(2), T[:, 3].max().round(2))
