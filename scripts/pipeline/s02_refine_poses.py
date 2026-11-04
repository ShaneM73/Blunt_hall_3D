"""Stage 2: refine the on-device SLAM trajectory. 3 s submaps (30 frames) are aligned pairwise with point-to-plane ICP
(Tukey loss) and optimized in a pose graph; the per-submap correction is interpolated to every frame.
Out: decoded/traj_refined.npy, logs printed to stdout.   Arg: submap length in frames (default 30)."""
import sys
import numpy as np, open3d as o3d
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.spatial import cKDTree
from common import W, read_slam_all, frame_points, traj, quat_R, log

T = traj(refined=False); nF = len(T)
idx, pts = read_slam_all()
def pose(i):
    M = np.eye(4); M[:3, :3] = quat_R(T[i, 4:8]); M[:3, 3] = T[i, 1:4]; return M
Pw = [pose(i) for i in range(nF)]
L = int(sys.argv[1]) if len(sys.argv) > 1 else 30
frames = {}
for k, p in frame_points(idx, pts):
    B = np.stack([p['x'], p['y'], p['z']], 1).astype(np.float64); r = np.linalg.norm(B, axis=1)
    frames[k] = B[(r > 0.5) & (r < 15)][::2]
subs, anchors = [], []
for s0 in range(0, nF, L):
    A = Pw[s0]; Ai = np.linalg.inv(A); q = []
    for i in range(s0, min(s0 + L, nF)):
        M = Ai @ Pw[i]; q.append(frames[i] @ M[:3, :3].T + M[:3, 3])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(q))).voxel_down_sample(0.03)
    pc, _ = pc.remove_statistical_outlier(15, 2.0)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30))
    subs.append(pc); anchors.append(A)
del frames
n = len(subs); log('submaps', n)
reg_ = o3d.pipelines.registration
pg = reg_.PoseGraph()
for A in anchors: pg.nodes.append(reg_.PoseGraphNode(A))
crit = reg_.ICPConvergenceCriteria(max_iteration=60)
est = reg_.TransformationEstimationPointToPlane(reg_.TukeyLoss(k=0.05))
ctr = np.array([a[:3, 3] for a in anchors]); stats = []
pairs = sorted(cKDTree(ctr).query_pairs(5.0) | {(i, i + 1) for i in range(n - 1)})
log('candidate pairs', len(pairs))
for c, (i, j) in enumerate(pairs):
    init = np.linalg.inv(anchors[j]) @ anchors[i]
    r = reg_.registration_icp(subs[i], subs[j], 0.15, init, est, crit)
    r = reg_.registration_icp(subs[i], subs[j], 0.05, r.transformation, est, crit)
    if r.fitness < 0.3 and j != i + 1: continue
    info = reg_.get_information_matrix_from_point_clouds(subs[i], subs[j], 0.05, r.transformation)
    dT = np.linalg.inv(init) @ r.transformation
    stats.append((i, j, np.linalg.norm(dT[:3, 3]) * 100, np.rad2deg(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1))), r.fitness, r.inlier_rmse * 100))
    pg.edges.append(reg_.PoseGraphEdge(i, j, r.transformation, info, uncertain=(j != i + 1)))
    if c % 2000 == 0: log('pair', c)
S = np.array(stats)
loop = S[S[:, 1] - S[:, 0] > 3]
log('edges', len(S), 'ICP correction vs SLAM: median %.1f cm %.2f deg; p90 %.1f cm; rmse median %.1f cm' % (np.median(S[:, 2]), np.median(S[:, 3]), np.percentile(S[:, 2], 90), np.median(S[:, 5])))
log('non-sequential (revisit) edges', len(loop), 'median %.1f cm  p90 %.1f cm  max %.1f cm' % (np.median(loop[:, 2]), np.percentile(loop[:, 2], 90), loop[:, 2].max()) if len(loop) else '')
opt = reg_.GlobalOptimizationOption(max_correspondence_distance=0.05, edge_prune_threshold=0.25, reference_node=0)
reg_.global_optimization(pg, reg_.GlobalOptimizationLevenbergMarquardt(), reg_.GlobalOptimizationConvergenceCriteria(), opt)
newA = [nd.pose for nd in pg.nodes]
Cs = [newA[k] @ np.linalg.inv(anchors[k]) for k in range(n)]
mid = np.array([T[min(k * L + L // 2, nF - 1), 0] for k in range(n)])
rots = Rot.from_matrix(np.array([c[:3, :3] for c in Cs])); tr = np.array([c[:3, 3] for c in Cs])
tq = np.clip(T[:, 0], mid[0], mid[-1]); Rc = Slerp(mid, rots)(tq).as_matrix()
tc = np.stack([np.interp(tq, mid, tr[:, k]) for k in range(3)], 1)
T2 = T.copy()
for i in range(nF):
    M = Pw[i]; R = Rc[i] @ M[:3, :3]; q = Rot.from_matrix(R).as_quat()
    T2[i, 1:4] = Rc[i] @ M[:3, 3] + tc[i]; T2[i, 4:8] = [q[3], q[0], q[1], q[2]]
np.save(W('decoded', 'traj_refined.npy'), T2)
d = np.linalg.norm(T2[:, 1:4] - T[:, 1:4], axis=1) * 100
log('trajectory change: mean %.1f cm, p95 %.1f cm, max %.1f cm' % (d.mean(), np.percentile(d, 95), d.max()))
