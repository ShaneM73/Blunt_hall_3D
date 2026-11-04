"""Stage 4: free-space carving. A cloud point that LiDAR beams later passed through (nearest hit per 0.5 deg bin lies
beyond it by > 0.15 m + 3 % of range) belonged to something that moved, i.e. a person.
Out: clouds/hq_transient.npy (bool per hq_cloud point)"""
import os
import numpy as np, torch
from common import W, traj, log

dev = 'cuda:' + os.environ.get('BH_GPU', '1')
d = np.load(W('clouds', 'hq_cloud.npz')); P = d['xyz']; N = len(P)
Wp = np.load(W('decoded', 'world_xyz.npy'), mmap_mode='r'); WF = np.load(W('decoded', 'world_frame.npy'))
T = traj()          # world_xyz is re-registered with the refined trajectory (s02b)
# spatial blocks so each frame only tests points within 15 m
BS = 4.0; key = np.floor(P / BS).astype(np.int64); kmin = key.min(0); key -= kmin
dims = key.max(0) + 1; lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
order = np.argsort(lin, kind='stable'); lin_s = lin[order]
ublk, start = np.unique(lin_s, return_index=True); end = np.r_[start[1:], len(lin_s)]
bc = np.stack(np.unravel_index(ublk, dims), 1) * BS + kmin * BS + BS / 2
Pg = torch.tensor(P[order], device=dev)
free = torch.zeros(N, device=dev, dtype=torch.int32); seen = torch.zeros(N, device=dev, dtype=torch.int32)
AZ, EL = 720, 360
bounds = np.searchsorted(WF, np.arange(len(T) + 1))
def sph(X):
    r = X.norm(dim=1); az = torch.atan2(X[:, 1], X[:, 0]); el = torch.asin((X[:, 2] / r.clamp(min=1e-6)).clamp(-1, 1))
    return r, ((az + np.pi) / (2 * np.pi) * AZ).long().clamp(0, AZ - 1) * EL + ((el + np.pi / 2) / np.pi * EL).long().clamp(0, EL - 1)
for f in range(len(T)):
    a, b = bounds[f], bounds[f + 1]
    if b - a < 100: continue
    o = T[f, 1:4]; near_b = np.flatnonzero(np.linalg.norm(bc - o, axis=1) < 15 + BS)
    if len(near_b) == 0: continue
    sel = torch.tensor(np.concatenate([np.arange(start[i], end[i]) for i in near_b]), device=dev)
    og = torch.tensor(o, device=dev, dtype=torch.float32)
    rq, pq = sph(torch.tensor(np.asarray(Wp[a:b]), device=dev) - og)
    rng = torch.full((AZ * EL,), 1e9, device=dev).scatter_reduce(0, pq, rq, 'amin', include_self=False)
    rp, pp = sph(Pg[sel] - og); rr = rng[pp]
    m = (rp < 15) & (rp > 0.3) & (rr < 1e8)
    free[sel] += (m & (rp < rr - (0.15 + 0.03 * rr))).int()
    seen[sel] += (m & ((rp - rr).abs() < 0.06)).int()
    if f % 1000 == 0: log('frame', f, 'of', len(T))
tr_s = ((free >= 4) & (free.float() > seen.float() * 0.5)).cpu().numpy()
tr = np.empty(N, bool); tr[order] = tr_s
log('transient', tr.sum(), 'of', N, '(%.2f%%)' % (tr.mean() * 100))
np.save(W('clouds', 'hq_transient.npy'), tr)
