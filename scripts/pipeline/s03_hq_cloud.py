"""Stage 3: high-quality cloud. Re-register returns 0.5-6 m from the scanner with the refined trajectory, keep those within
CROP m of the path, 1 cm voxels; outliers removed / planar points pulled onto local planes fitted on a 2.5 cm base grid.
Out: clouds/hq_cloud.npz (xyz, nrm, inten)"""
import os, sys
import numpy as np, torch, open3d as o3d
from scipy.spatial import cKDTree
from common import W, read_slam_all, frame_points, traj, quat_R, log

CROP = float(os.environ.get('BH_CROP', 5.0)); dev = 'cuda:' + os.environ.get('BH_GPU', '1')
T = traj(); idx, pts = read_slam_all()
# range: indoors 0.5-6 m; frames taken inside a 'range_box' of regions.json (e.g. outside the entrance) keep returns up to
# RMAX so tall facades are modeled; points are kept within max(CROP, r_max - 1) of the path
RMAX = float(os.environ.get('BH_RMAX', 20.0))
import json
RBOX = [r['range_box'] for r in (json.load(open(W('regions.json'))) if os.path.exists(W('regions.json')) else []) if 'range_box' in r]
out, ints, thr = [], [], []; rmax = np.zeros(len(T))
for k, p in frame_points(idx, pts):
    B = np.stack([p['x'], p['y'], p['z']], 1).astype(np.float64); r = np.linalg.norm(B, axis=1)
    x, y, z = T[k, 1:4]
    rmax[k] = RMAX if any(b[0] <= x < b[1] and b[2] <= y < b[3] and b[4] <= z < b[5] for b in RBOX) else 6.0
    a = p['a']; m = (r > 0.5) & (r < rmax[k])
    out.append((B[m] @ quat_R(T[k, 4:8]).T + T[k, 1:4]).astype(np.float32)); ints.append(a[m]); thr.append(np.full(m.sum(), max(CROP, rmax[k] - 1), np.float32))
Wp = np.vstack(out); WI = np.concatenate(ints); TH = np.concatenate(thr); del out, ints, thr
np.save(W('decoded', 'frame_rmax.npy'), rmax)
log('frames with extended range (> 6 m): %d of %d, max %.1f m' % ((rmax > 6).sum(), len(T), rmax.max()))
dist, _ = cKDTree(T[::2, 1:4]).query(Wp, workers=64, distance_upper_bound=RMAX); k = dist < TH
Wp, WI = Wp[k], WI[k]; del TH; log('raw returns kept (adaptive range / crop)', len(Wp))
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Wp.astype(np.float64))); del Wp
pc.colors = o3d.utility.Vector3dVector(np.repeat(WI[:, None] / 255., 3, 1)); del WI
v = pc.voxel_down_sample(0.01); P = np.asarray(v.points).astype(np.float32); Iv = (np.asarray(v.colors)[:, 0] * 255).astype(np.uint8); del v
base = np.asarray(pc.voxel_down_sample(0.025).points).astype(np.float32); del pc
log('1 cm voxels', len(P), ' 2.5 cm base', len(base))
bt = cKDTree(base)
bn = np.empty((len(base), 3), np.float32); blam = np.empty(len(base), np.float32); bsig = np.empty(len(base), np.float32); bmu = np.empty((len(base), 3), np.float32)
CH = 2_000_000
for s in range(0, len(base), CH):
    _, nb = bt.query(base[s:s + CH], k=40, workers=64)
    X = torch.tensor(base[nb], device=dev)
    mu = X.mean(1); Xc = X - mu[:, None]
    ev, evec = torch.linalg.eigh(torch.einsum('nki,nkj->nij', Xc, Xc) / 40)
    bn[s:s + CH] = evec[:, :, 0].cpu().numpy(); blam[s:s + CH] = (ev[:, 0] / ev.sum(1).clamp(min=1e-12)).cpu().numpy()
    bsig[s:s + CH] = ev[:, 0].clamp(min=0).sqrt().cpu().numpy(); bmu[s:s + CH] = mu.cpu().numpy()
    del X, Xc, ev, evec
log('base planar(lam<0.02) %.0f%%  median sigma %.2f cm' % ((blam < 0.02).mean() * 100, np.median(bsig) * 100))
keep = np.zeros(len(P), bool); newP = np.empty_like(P); nrm = np.empty_like(P)
for s in range(0, len(P), 4 * CH):
    _, j = bt.query(P[s:s + 4 * CH], k=1, workers=64)
    Pt = torch.tensor(P[s:s + 4 * CH], device=dev); n = torch.tensor(bn[j], device=dev); mu = torch.tensor(bmu[j], device=dev)
    sig = torch.tensor(bsig[j], device=dev); planar = torch.tensor(blam[j] < 0.02, device=dev)
    res = ((Pt - mu) * n).sum(1)
    keep[s:s + 4 * CH] = ((~planar) | (res.abs() < torch.clamp(3 * sig, min=0.012))).cpu().numpy()
    newP[s:s + 4 * CH] = torch.where(planar[:, None], Pt - res[:, None] * n * 0.85, Pt).cpu().numpy(); nrm[s:s + 4 * CH] = n.cpu().numpy()
log('kept after denoise', keep.sum(), '(%.1f%%)' % (keep.mean() * 100))
P2, N2, I2 = newP[keep], nrm[keep], Iv[keep]
_, j = cKDTree(T[:, 1:4]).query(P2, workers=64); N2[(N2 * (T[j, 1:4] - P2)).sum(1) < 0] *= -1   # normals face the path
dp, _ = cKDTree(T[::2, 1:4]).query(P2, workers=64)
log('points farther than 5.5 m from the path (open spaces): %d' % (dp > 5.5).sum())
np.savez(W('clouds', 'hq_cloud.npz'), xyz=P2.astype(np.float32), nrm=N2.astype(np.float32), inten=I2, dpath=dp.astype(np.float16))
