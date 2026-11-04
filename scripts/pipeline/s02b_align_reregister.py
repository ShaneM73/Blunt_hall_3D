"""Stage 2b: remove the rigid gauge drift of the pose-graph result (best rigid fit of refined -> on-device positions),
keeping only the non-rigid correction, then re-register all LiDAR frames with the refined trajectory so every later
stage uses one consistent frame.   Out: decoded/traj_refined.npy (aligned), decoded/world_xyz.npy (re-registered)"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from common import W, read_slam_all, frame_points, traj, quat_R, log
T0 = traj(refined=False); T = traj(); A, B = T0[:, 1:4], T[:, 1:4]
ma, mb = A.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ D @ U.T                                   # maps refined -> on-device frame
log('removed rigid drift: %.3f deg, %.1f cm' % (np.rad2deg(Rot.from_matrix(R).magnitude()), np.linalg.norm(ma - R @ mb) * 100))
T2 = T.copy(); T2[:, 1:4] = (B - mb) @ R.T + ma
q = (Rot.from_matrix(R) * Rot.from_quat(T[:, [5, 6, 7, 4]])).as_quat(); T2[:, 4:8] = q[:, [3, 0, 1, 2]]
d = np.linalg.norm(T2[:, 1:4] - A, axis=1) * 100; log('remaining correction: mean %.1f cm  p95 %.1f  max %.1f' % (d.mean(), np.percentile(d, 95), d.max()))
np.save(W('decoded', 'traj_refined.npy'), T2)
idx, pts = read_slam_all(); xyz = np.load(W('decoded', 'world_xyz.npy'), mmap_mode='r+'); n = 0
for k, p in frame_points(idx, pts):
    Bp = np.stack([p['x'], p['y'], p['z']], 1).astype(np.float64); m = np.linalg.norm(Bp, axis=1) > 0.3; c = int(m.sum())
    xyz[n:n + c] = Bp[m] @ quat_R(T2[k, 4:8]).T + T2[k, 1:4]; n += c
xyz.flush(); log('re-registered', n)
