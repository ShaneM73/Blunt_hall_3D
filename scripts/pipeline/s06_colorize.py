"""Stage 6: colorize the people-free cloud from the fisheye photos.
Visibility per photo: points within 6 m, front-facing, z-buffer from both the cloud and that moment's raw LiDAR returns
(occluders that are not in the cloud, e.g. people), Mask R-CNN people skipped. Per-photo, per-channel gains equalize
exposure/white balance against the blend (3 iterations); final blend uses sharper best-view weights + outlier rejection.
Visibility lists are kept in RAM (no disk cache).
Out: color/hq_colored.npz, calib/gains.npy, color/person_frac.npy, color/vcount.npy, decoded/cam_positions.npy"""
import os
import numpy as np, torch, cv2
from concurrent.futures import ThreadPoolExecutor
from common import W, CAL, Rcb, Poser, traj, image_index, read_image, log
import json

dev = 'cuda:' + os.environ.get('BH_GPU', '1')
d = np.load(W('clouds', 'hq_cloud.npz')); tr = np.load(W('clouds', 'hq_transient.npy'))
P = d['xyz'][~tr]; inten = d['inten'][~tr]; nrm = d['nrm'][~tr]; N = len(P); log('points', N)
T = traj(); pose = Poser(T)
Wp = np.load(W('decoded', 'world_xyz.npy'), mmap_mode='r'); WF = np.load(W('decoded', 'world_frame.npy'))
T0 = T; fb = np.searchsorted(WF, np.arange(len(T0) + 1))
GM = json.load(open(W('images', 'cameras.json'))); KN = np.array(GM['K']); MW, MH = GM['W'], GM['H']
recs = image_index()
# spatial blocks for per-photo preselection
BS = 3.0; key = np.floor(P / BS).astype(np.int64); kmin = key.min(0); key -= kmin; dims = key.max(0) + 1
lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]; order = np.argsort(lin, kind='stable')
ublk, start = np.unique(lin[order], return_index=True); end = np.r_[start[1:], N]
bc = np.stack(np.unravel_index(ublk, dims), 1) * BS + kmin * BS + BS / 2
order_g = torch.tensor(order, device=dev)
Pg = torch.tensor(P, device=dev, dtype=torch.float64); Ng = torch.tensor(nrm, device=dev, dtype=torch.float64)
S = 4; H, Wd = 2592 // S, 1944 // S
vis_cache = []; vcount = torch.zeros(N, device=dev); mcount = torch.zeros(N, device=dev); cam_pos = []
pool = ThreadPoolExecutor(8)
def load(r):
    im = read_image(r); mk = cv2.imread(W('images', 'masks', f"c{r['cam']}_{r['t']:.6f}.png"), 0); return im, mk
futs = [pool.submit(load, r) for r in recs[:16]]
for k, r in enumerate(recs):
    im, mk = futs[k].result()
    if k + 16 < len(recs): futs.append(pool.submit(load, recs[k + 16]))
    futs[k] = None
    cam, t = r['cam'], r['t']; R_cb = torch.tensor(Rcb(cam), device=dev); tb = torch.tensor(CAL[cam]['t'], device=dev, dtype=torch.float64)
    fx, fy, cx, cy = CAL[cam]['K']; kk = CAL[cam]['D']
    Rwb, twb = pose(t); cpos = twb + Rwb @ np.array(CAL[cam]['t']); cam_pos.append((cam, t, *cpos))
    nb = np.flatnonzero(np.linalg.norm(bc - cpos, axis=1) < 6 + BS)
    if len(nb) == 0: vis_cache.append(None); continue
    sel = order_g[torch.tensor(np.concatenate([np.arange(start[i], end[i]) for i in nb]), device=dev)]
    Rwb_g = torch.tensor(Rwb, device=dev); twb_g = torch.tensor(twb, device=dev)
    def proj(X):
        Pc = ((X - twb_g) @ Rwb_g - tb) @ R_cb.T; x, y, z = Pc.unbind(1); rr = torch.hypot(x, y) + 1e-9; th = torch.atan2(rr, z); t2 = th * th
        thd = th * (1 + kk[0] * t2 + kk[1] * t2 ** 2 + kk[2] * t2 ** 3 + kk[3] * t2 ** 4); return Pc, fx * thd * x / rr + cx, fy * thd * y / rr + cy, th
    Pc, u, v, th = proj(Pg[sel]); dd = Pc.norm(dim=1)
    ok = (th < 1.2) & (u >= 0) & (u < 1944) & (v >= 0) & (v < 2592) & (dd < 6) & (dd > 0.3)
    vd = torch.tensor(cpos, device=dev) - Pg[sel]; vd = vd / vd.norm(dim=1, keepdim=True); cosi = (vd * Ng[sel]).sum(1).abs()
    ok &= cosi > 0.15
    li = ok.nonzero().squeeze(1)
    if len(li) == 0: vis_cache.append(None); continue
    pix = (v[li] / S).long() * Wd + (u[li] / S).long()
    zb = torch.full((H * Wd,), 1e9, device=dev, dtype=torch.float64).scatter_reduce(0, pix, dd[li], 'amin')
    f0 = np.searchsorted(T0[:, 0], t - 0.3); f1 = np.searchsorted(T0[:, 0], t + 0.3)
    Rp = torch.tensor(np.asarray(Wp[fb[f0]:fb[f1]]), device=dev, dtype=torch.float64)
    if len(Rp):
        Qc, qu, qv, qt = proj(Rp); qn = Qc.norm(dim=1); qo = (qt < 1.2) & (qu >= 0) & (qu < 1944) & (qv >= 0) & (qv < 2592) & (qn > 0.3)
        qpix = (qv[qo] / S).long() * Wd + (qu[qo] / S).long()
        for dy in (-Wd, 0, Wd):
            for dx in (-1, 0, 1): zb.scatter_reduce_(0, (qpix + dy + dx).clamp(0, H * Wd - 1), qn[qo], 'amin')
    vis = dd[li] <= zb[pix] * 1.02 + 0.03; li = li[vis]
    if mk is not None:
        x, y, z = Pc[li].unbind(1); zz = z.clamp(min=1e-3); pu = KN[0, 0] * x / zz + KN[0, 2]; pv = KN[1, 1] * y / zz + KN[1, 2]
        inside = (pu >= 0) & (pu < MW) & (pv >= 0) & (pv < MH) & (z > 0.05)
        M = torch.tensor(mk > 127, device=dev); good = torch.zeros_like(inside); ii = inside.nonzero().squeeze(1)
        good[ii] = M[pv[ii].long(), pu[ii].long()]
        if not (mk[MH // 4:3 * MH // 4, MW // 4:3 * MW // 4] == 0).any(): good |= ~inside   # no person: keep outside-pinhole fisheye rim
        gi = sel[li]
        vcount.index_add_(0, gi[inside], torch.ones(int(inside.sum()), device=dev))
        mcount.index_add_(0, gi[inside & ~good], torch.ones(int((inside & ~good).sum()), device=dev))
        li = li[good]
    gi = sel[li]
    w = (cosi[li] ** 1.5) / (dd[li] ** 2 + 0.05) * torch.cos(th[li]) ** 2
    img = torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), device=dev)
    col = img[v[li].long(), u[li].long()]
    vis_cache.append((gi.int().cpu(), col.cpu(), w.half().cpu()))
    if k % 500 == 0: log('vis', k, 'of', len(recs), 'cached pts %.0f M' % (sum(len(c[0]) for c in vis_cache if c) / 1e6))
pool.shutdown()
np.save(W('decoded', 'cam_positions.npy'), np.array(cam_pos))

def blend(gains, ref=None, p=1.0, sigma=30.0):
    acc = torch.zeros(N, 3, device=dev); ws = torch.zeros(N, device=dev)
    for k, c in enumerate(vis_cache):
        if c is None: continue
        idx = c[0].to(dev).long(); col = c[1].to(dev).float() * gains[k]; w = c[2].to(dev).float() ** p
        if ref is not None: w = w * torch.exp(-((col - ref[idx]) ** 2).sum(1) / (2 * sigma ** 2))
        acc.index_add_(0, idx, col * w[:, None]); ws.index_add_(0, idx, w)
    return acc / ws.clamp(min=1e-12)[:, None], ws
gains = torch.ones(len(recs), 3, device=dev)
ref, ws = blend(gains)
for it in range(3):
    newg = torch.ones_like(gains)
    for k, c in enumerate(vis_cache):
        if c is None or len(c[0]) < 500: continue
        col = c[1].to(dev).float(); rf = ref[c[0].to(dev).long()]
        m = (col.min(1).values > 15) & (col.max(1).values < 240)
        if m.sum() < 300: continue
        newg[k] = torch.median(rf[m] / col[m].clamp(min=1), dim=0).values.clamp(0.5, 2.0)
    gains = newg / newg.mean(0, keepdim=True)
    ref, ws = blend(gains, ref, p=1.0)
    g = gains.cpu().numpy(); log(f'iter {it}: gain spread (std) R {g[:,0].std():.3f} G {g[:,1].std():.3f} B {g[:,2].std():.3f}')
final, ws = blend(gains, ref, p=3.0, sigma=25.0)
has = (ws > 0).cpu().numpy(); rgb = final.clamp(0, 255).byte().cpu().numpy()
log('colored fraction %.3f' % has.mean())
np.savez(W('color', 'hq_colored.npz'), xyz=P.astype(np.float32), nrm=nrm, rgb=rgb, inten=inten, has=has)
np.save(W('calib', 'gains.npy'), gains.cpu().numpy())
np.save(W('color', 'person_frac.npy'), (mcount / vcount.clamp(min=1)).cpu().numpy()); np.save(W('color', 'vcount.npy'), vcount.cpu().numpy())
