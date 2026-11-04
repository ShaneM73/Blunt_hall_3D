"""Stage 9: 3D Gaussian Splatting (gsplat, DefaultStrategy with absgrad) with LiDAR depth supervision and per-image affine
exposure. Init = people-free colored 1 cm cloud near the path + 2 cm raw points farther out (intensity gray).
Images stay in pinned CPU memory; depth targets come from LiDAR points within 12 m of each camera.
Args: [iterations] (default: 27 per training view).   Out: splat/splat_params.pt, renders/splat_test_views.jpg, logs."""
import os, sys, json, math, time
import numpy as np, torch, cv2, torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
from common import W, traj, log, removed_mask

torch.manual_seed(0); dev = 'cuda:' + os.environ.get('BH_GPU', '0'); torch.cuda.set_device(dev)
meta = json.load(open(W('images', 'cameras.json'))); Wd, Hd = meta['W'], meta['H']; K = torch.tensor(meta['K'], dtype=torch.float32, device=dev)
frames = meta['frames']; n = len(frames)
test = set(range(0, n, 25)); train = [i for i in range(n) if i not in test]
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 27 * len(train); SHD = 1; MAXINIT = int(os.environ.get('BH_GS_INIT', 6_000_000))
log('views', n, 'iterations', ITERS)
def ld(f):
    im = cv2.cvtColor(cv2.imread(W('images', 'pinhole', f['file'])), cv2.COLOR_BGR2RGB)
    mk = cv2.imread(W('images', 'masks', f['file'].replace('.jpg', '.png')), 0) > 127
    return im, mk
imgs = torch.empty((n, Hd, Wd, 3), dtype=torch.uint8).pin_memory(); masks = torch.empty((n, Hd, Wd), dtype=torch.bool).pin_memory()
with ThreadPoolExecutor(32) as ex:
    for i, (im, mk) in enumerate(ex.map(ld, frames)):
        imgs[i] = torch.from_numpy(im); masks[i] = torch.from_numpy(mk)
log('images loaded')
c2w = torch.tensor([f['c2w'] for f in frames], dtype=torch.float32, device=dev); w2c = torch.linalg.inv(c2w)
scene_scale = float(os.environ.get('BH_GS_SCALE', 7.0))   # local scale (as scan 1), not building extent: keeps densify thresholds
# init cloud
d = np.load(W('color', 'hq_colored.npz')); k = d['has'] & ~removed_mask()
Pn, Cn = d['xyz'][k], d['rgb'][k] / 255.
T = traj(); Wx = np.load(W('decoded', 'world_xyz.npy'), mmap_mode='r'); sub = np.asarray(Wx[::4])
dist, _ = cKDTree(T[::2, 1:4]).query(sub, workers=64, distance_upper_bound=5.0); far = sub[~np.isfinite(dist)]
far = np.unique(np.floor(far / 0.02).astype(np.int64), axis=0) * 0.02 + 0.01 if len(far) else far
Pf = far.astype(np.float32); Cf = np.full((len(Pf), 3), 0.6)
log('init near', len(Pn), 'far', len(Pf))
P = np.vstack([Pn, Pf]); C = np.vstack([Cn, Cf])
if len(P) > MAXINIT: s = np.random.default_rng(0).choice(len(P), MAXINIT, replace=False); P, C = P[s], C[s]
dd, _ = cKDTree(P).query(P, k=4, workers=64); s0 = np.log(np.clip(dd[:, 1:].mean(1), 0.003, 0.1)); N = len(P)
# LiDAR depth source, blocked for per-camera selection
LPn = Pn.astype(np.float32); BS = 4.0; key = np.floor(LPn / BS).astype(np.int64); kmin = key.min(0); key -= kmin; dims = key.max(0) + 1
lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]; order = np.argsort(lin, kind='stable')
ub, st = np.unique(lin[order], return_index=True); en = np.r_[st[1:], len(order)]
bc = np.stack(np.unravel_index(ub, dims), 1) * BS + kmin * BS + BS / 2
LP = torch.tensor(LPn[order], device=dev); st_t, en_t = st, en
cposn = c2w[:, :3, 3].cpu().numpy()
near_blocks = [np.flatnonzero(np.linalg.norm(bc - c, axis=1) < 12 + BS) for c in cposn]
st_g = torch.tensor(st, device=dev); len_t = torch.tensor(en - st, device=dev)
near_blocks_t = [torch.tensor(b, device=dev) for b in near_blocks]; cnt_t = [int((en - st)[b].sum()) for b in near_blocks]
DS = 4; DW, DH = Wd // DS, Hd // DS
def lidar_depth(i):
    nb = near_blocks[i]
    if len(nb) == 0: return None
    nbt = near_blocks_t[i]; lens = len_t[nbt]; tot = int(cnt_t[i])
    sel = torch.repeat_interleave(st_g[nbt] - (torch.cumsum(lens, 0) - lens), lens, output_size=tot) + torch.arange(tot, device=dev)
    Pc = LP[sel] @ w2c[i, :3, :3].T + w2c[i, :3, 3]; z = Pc[:, 2]; ok = z > 0.2
    u = K[0, 0] * Pc[:, 0] / z.clamp(min=1e-3) + K[0, 2]; v = K[1, 1] * Pc[:, 1] / z.clamp(min=1e-3) + K[1, 2]
    ok &= (u >= 0) & (u < Wd) & (v >= 0) & (v < Hd) & (z < 12); idx = ok.nonzero().squeeze(1)
    pix = (v[idx] / DS).long().clamp(max=DH - 1) * DW + (u[idx] / DS).long().clamp(max=DW - 1)
    zb = torch.full((DH * DW,), 1e9, device=dev).scatter_reduce(0, pix, z[idx], 'amin')
    idx = idx[z[idx] <= zb[pix] * 1.02 + 0.02]
    return u[idx].long(), v[idx].long(), z[idx]
params = torch.nn.ParameterDict({
    'means': torch.nn.Parameter(torch.tensor(P, dtype=torch.float32, device=dev)),
    'scales': torch.nn.Parameter(torch.tensor(s0, dtype=torch.float32, device=dev)[:, None].repeat(1, 3)),
    'quats': torch.nn.Parameter(torch.rand(N, 4, device=dev)),
    'opacities': torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
    'sh0': torch.nn.Parameter(((torch.tensor(C, dtype=torch.float32, device=dev) - 0.5) / 0.28209479)[:, None, :]),
    'shN': torch.nn.Parameter(torch.zeros(N, (SHD + 1) ** 2 - 1, 3, device=dev))})
lrs = {'means': 1.6e-4 * scene_scale, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2, 'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}
opts = {k: torch.optim.Adam([{'params': params[k], 'lr': lrs[k], 'name': k}], eps=1e-15) for k in params}
sched = torch.optim.lr_scheduler.ExponentialLR(opts['means'], gamma=0.01 ** (1 / ITERS))
expo = torch.nn.Parameter(torch.zeros(n, 3, 4, device=dev)); expo_opt = torch.optim.Adam([expo], lr=1e-3)
# large scenes: every area is seen only now and then, so periodic opacity resets prune unrevisited regions -> no resets
G2D = float(os.environ.get('BH_GS_GROW', 0.0005)); RESET = int(os.environ.get('BH_GS_RESET', 10 ** 9))
strategy = DefaultStrategy(refine_stop_iter=int(ITERS * 0.6), reset_every=RESET, refine_every=100, absgrad=True, grow_grad2d=G2D, verbose=False)
log('strategy: grow_grad2d', G2D, 'reset_every', RESET)
strategy.check_sanity(params, opts); state = strategy.initialize_state(scene_scale=scene_scale)
gk = torch.tensor(cv2.getGaussianKernel(11, 1.5), dtype=torch.float32, device=dev); gk = (gk @ gk.T)[None, None].repeat(3, 1, 1, 1)
def ssim(a, b):
    f = lambda x: F.conv2d(x, gk, padding=5, groups=3)
    mu1, mu2 = f(a), f(b); s11 = f(a * a) - mu1 ** 2; s22 = f(b * b) - mu2 ** 2; s12 = f(a * b) - mu1 * mu2
    return ((2 * mu1 * mu2 + 1e-4) * (2 * s12 + 9e-4)) / ((mu1 ** 2 + mu2 ** 2 + 1e-4) * (s11 + s22 + 9e-4))
def render(i, sh):
    colors = torch.cat([params['sh0'], params['shN']], 1)
    out, _, info = rasterization(params['means'], params['quats'], torch.exp(params['scales']), torch.sigmoid(params['opacities']), colors,
                                 w2c[i:i + 1], K[None], Wd, Hd, sh_degree=sh, packed=False, absgrad=True, near_plane=0.05, far_plane=60, render_mode='RGB+ED')
    return out[0], info
t0 = time.time()
for step in range(ITERS):
    i = train[np.random.randint(len(train))]
    gt = imgs[i].to(dev, non_blocking=True).float() / 255; mm = masks[i].to(dev, non_blocking=True)
    out4, info = render(i, min(SHD, step // 1000)); out = out4[..., :3]; dep = out4[..., 3]
    A = torch.eye(3, 4, device=dev) + expo[i]; out = out @ A[:, :3].T + A[:, 3]
    m = mm[..., None].float()
    strategy.step_pre_backward(params, opts, state, step, info)
    l1 = ((out - gt).abs() * m).sum() / (m.sum() * 3)
    ss = (ssim(out.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])[0].permute(1, 2, 0) * m).sum() / (m.sum() * 3)
    dl = torch.zeros((), device=dev); r = lidar_depth(i)
    if r is not None:
        lu, lv, lz = r; mk = mm[lv, lu]; lu, lv, lz = lu[mk], lv[mk], lz[mk]
        if len(lz) > 100: dl = ((dep[lv, lu] - lz).abs() / lz).mean()
    lam = 0.5 if step < ITERS // 4 else 0.2
    vis = (info['radii'][0] > 0) if info['radii'].dim() == 2 else (info['radii'][0] > 0).any(-1)   # opacity penalty on visible Gaussians only
    loss = 0.8 * l1 + 0.2 * (1 - ss) + lam * dl + 1e-3 * torch.sigmoid(params['opacities'][vis]).mean()
    loss.backward()
    strategy.step_post_backward(params, opts, state, step, info, packed=False)
    for o in opts.values(): o.step(); o.zero_grad(set_to_none=True)
    expo_opt.step(); expo_opt.zero_grad(); sched.step()
    if step == 200: torch.cuda.synchronize(); log('ms/step (first 200)', round((time.time() - t0) / 200 * 1000, 1))
    if step % 2000 == 0 or step == ITERS - 1:
        with torch.no_grad():
            ps = []
            for j in sorted(test)[::6]:
                o, _ = render(j, SHD); A = torch.eye(3, 4, device=dev) + expo[max(j - 1, 0)]; o = (o[..., :3] @ A[:, :3].T + A[:, 3]).clamp(0, 1)
                g = imgs[j].to(dev).float() / 255; mj = masks[j].to(dev)[..., None]; mse = (((o - g) ** 2) * mj).sum() / (mj.sum() * 3); ps.append(-10 * math.log10(mse.item()))
        log(f'step {step} loss {loss.item():.4f} depthL1rel {dl.item():.3f} N {len(params["means"])} testPSNR {np.mean(ps):.2f} t {time.time()-t0:.0f}s')
    if step > 0 and step % 20000 == 0:
        torch.save({k: v.detach().cpu() for k, v in params.items()}, W('splat', 'splat_params.pt'))
torch.save({k: v.detach().cpu() for k, v in params.items()}, W('splat', 'splat_params.pt'))
with torch.no_grad():
    rows = []
    for j in sorted(test)[3::max(1, len(test) // 6)][:6]:
        o, _ = render(j, SHD); o = (o[..., :3].clamp(0, 1) * 255).byte().cpu().numpy(); g = imgs[j].numpy()
        rows.append(np.hstack([cv2.resize(g, (324, 432)), cv2.resize(o, (324, 432))]))
    cv2.imwrite(W('renders', 'splat_test_views.jpg'), cv2.cvtColor(np.vstack([np.hstack(rows[:3]), np.hstack(rows[3:6])]) if len(rows) >= 6 else rows[0], cv2.COLOR_RGB2BGR))
log('done')
