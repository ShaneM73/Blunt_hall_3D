"""Stage 7: remove person blobs that carving missed (e.g. someone standing still): points seen inside a photo person-mask
in > 60 % of >= 5 views, clustered (DBSCAN 5 cm, >= 2000 pts), grown by 3 cm.   Out: clouds/person_rm.npy"""
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
from common import W, log
d = np.load(W('color', 'hq_colored.npz')); P = d['xyz']; f = np.load(W('color', 'person_frac.npy')); vc = np.load(W('color', 'vcount.npy'))
pers = (f > 0.6) & (vc >= 5)
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[pers].astype(np.float64)))
lab = np.array(pc.cluster_dbscan(0.05, 20)); cnt = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.zeros(0, int)
core = P[pers][(lab >= 0) & (cnt[np.maximum(lab, 0)] >= 2000)] if len(cnt) else P[:0]
rm = np.zeros(len(P), bool)
if len(core): dd, _ = cKDTree(core).query(P, distance_upper_bound=0.03, workers=64); rm = np.isfinite(dd)
log('person-flagged', pers.sum(), 'blobs', int((cnt >= 2000).sum()), 'removed', rm.sum())
np.save(W('clouds', 'person_rm.npy'), rm)
