"""Stage 15 (objects): lift SAM 3 masks onto the clean LiDAR cloud and split them into object instances.
For each photo, the clean points within 6 m are projected into the pinhole image (pose from cameras.json), occluded points
dropped by a z-buffer, and each visible point gets one vote for the label of the SAM 3 detection under it (weighted by score)
plus one 'seen' count. A point is labelled when votes for its best label are >= 35 % of its views and >= 3 views.
Labelled points are clustered per label (connected components on a 5 cm grid); instances need >= MIN_PTS points and a
plausible size; structural surfaces (walls / floors / ceilings of s12) are excluded.
Out: objects/instances.npz (point index into hq_colored, instance id), objects/instances.json (label, n, bbox, center)"""
import os, sys, json
import numpy as np, torch, cv2
from scipy import ndimage
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log, removed_mask

dev = 'cuda:' + os.environ.get('BH_GPU', '1')
d = np.load(W('color', 'hq_colored.npz')); keep = np.flatnonzero(d['has'] & ~removed_mask()); P = d['xyz'][keep]; Nr = d['nrm'][keep]
N = len(P); log('points', N)
cams = json.load(open(W('images', 'cameras.json'))); K = np.array(cams['K']); Wd, Hd = cams['W'], cams['H']
SAMD = W('objects', 'sam3', 'x')[:-2]
labels = sorted({d_['label'] for f in os.listdir(SAMD) if f.endswith('.json') for d_ in json.load(open(os.path.join(SAMD, f)))})
L = {l: i for i, l in enumerate(labels)}; NL = len(labels); log('labels', labels)
# spatial blocks for per-camera preselection
BS = 3.0; key = np.floor(P / BS).astype(np.int64); kmin = key.min(0); key -= kmin; dm = key.max(0) + 1
lin = (key[:, 0] * dm[1] + key[:, 1]) * dm[2] + key[:, 2]; order = np.argsort(lin, kind='stable')
ub, st = np.unique(lin[order], return_index=True); en = np.r_[st[1:], N]; bc = np.stack(np.unravel_index(ub, dm), 1) * BS + kmin * BS + BS / 2
og = torch.tensor(order, device=dev); Pg = torch.tensor(P, device=dev, dtype=torch.float32)
votes = torch.zeros(N, NL, device=dev); seen = torch.zeros(N, device=dev)
S = 2; Hs, Ws = Hd // S, Wd // S
for n, f in enumerate(cams['frames']):
    b = os.path.join(SAMD, f['file'][:-4])
    if not os.path.exists(b + '.json'): continue
    dets = json.load(open(b + '.json')); idx = cv2.imread(b + '.png', cv2.IMREAD_UNCHANGED)
    c2w = np.array(f['c2w']); cpos = c2w[:3, 3]; nb = np.flatnonzero(np.linalg.norm(bc - cpos, axis=1) < 6 + 2.6)
    if not len(nb): continue
    sel = og[torch.tensor(np.concatenate([np.arange(st[i], en[i]) for i in nb]), device=dev)]
    w2c = torch.tensor(np.linalg.inv(c2w), device=dev, dtype=torch.float32)
    Pc = Pg[sel] @ w2c[:3, :3].T + w2c[:3, 3]; z = Pc[:, 2]
    u = (K[0, 0] * Pc[:, 0] / z.clamp(min=1e-3) + K[0, 2]) / S; v = (K[1, 1] * Pc[:, 1] / z.clamp(min=1e-3) + K[1, 2]) / S
    ok = (z > 0.3) & (z < 6) & (u >= 0) & (u < Ws - 1) & (v >= 0) & (v < Hs - 1); ii = ok.nonzero().squeeze(1)
    if not len(ii): continue
    pix = v[ii].long() * Ws + u[ii].long(); zz = z[ii]
    zb = torch.full((Hs * Ws,), 1e9, device=dev).scatter_reduce(0, pix, zz, 'amin')
    for dy in (-Ws, Ws):
        for dx in (-1, 1): zb.scatter_reduce_(0, (pix + dy + dx).clamp(0, Hs * Ws - 1), zz, 'amin')
    vis = zz <= zb[pix] * 1.02 + 0.03; ii = ii[vis]; pix = pix[vis]
    seen.index_add_(0, sel[ii], torch.ones(len(ii), device=dev))
    if dets:
        di = torch.tensor(idx.astype(np.int64), device=dev).view(-1)[pix]; m = di > 0
        if m.any():
            lab = torch.tensor([L[x['label']] for x in dets], device=dev); sc = torch.tensor([x['score'] for x in dets], device=dev)
            k_ = di[m] - 1; flat = sel[ii[m]] * NL + lab[k_]
            votes.view(-1).index_add_(0, flat, sc[k_])
    if n % 1000 == 0: log('photo', n, 'of', len(cams['frames']))
best = votes.max(1); frac = best.values / seen.clamp(min=1)
lab = torch.where((frac >= 0.35) & (seen >= 3), best.indices, torch.full_like(best.indices, -1)).cpu().numpy()
log('labelled points', int((lab >= 0).sum()), {labels[i]: int((lab == i).sum()) for i in range(NL)})
# exclude structural surfaces: points within 4 cm of a wall/floor/ceiling rectangle of s12
S12 = json.load(open(W('mesh', 'structure.json')))['faces']
struct = np.zeros(N, bool); cand = np.flatnonzero(lab >= 0); Qg = torch.tensor(P[cand], device=dev, dtype=torch.float32)
hitg = torch.zeros(len(cand), dtype=torch.bool, device=dev)
for fc in S12:
    o, a, bvec, n_ = (torch.tensor(fc[k_], device=dev, dtype=torch.float32) for k_ in ('o', 'u', 'v', 'n'))
    rel = Qg - o; s_ = rel @ a; t_ = rel @ bvec; h_ = rel @ n_
    hitg |= (h_.abs() < 0.04) & (s_ > 0) & (s_ < fc['su']) & (t_ > 0) & (t_ < fc['sv'])
struct[cand[hitg.cpu().numpy()]] = True
lab[struct] = -1; log('after removing structural surfaces', int((lab >= 0).sum()))
# instances: connected components per label on a 5 cm grid
MIN_PTS = int(os.environ.get('BH_OBJ_MIN', 400)); inst = np.full(N, -1, np.int32); objs = []
for li in range(NL):
    ii = np.flatnonzero(lab == li)
    if len(ii) < MIN_PTS: continue
    g = np.floor(P[ii] / 0.05).astype(np.int64); g0 = g.min(0); g -= g0; dims = g.max(0) + 1
    if np.prod(dims) > 3e9: continue
    grid = np.zeros(dims, bool); grid[g[:, 0], g[:, 1], g[:, 2]] = True
    cc, nc = ndimage.label(grid, structure=np.ones((3, 3, 3))); pl = cc[g[:, 0], g[:, 1], g[:, 2]]; cnt = np.bincount(pl)
    for c in np.flatnonzero(cnt >= MIN_PTS):
        if c == 0: continue
        jj = ii[pl == c]; X = P[jj]; lo, hi = np.percentile(X, 1, 0), np.percentile(X, 99, 0); ext = hi - lo
        if ext.max() > 6 or ext.max() < 0.15: continue          # not furniture-sized
        inst[jj] = len(objs)
        objs.append(dict(id=len(objs), label=labels[li], n=int(len(jj)), center=np.round((lo + hi) / 2, 3).tolist(), size=np.round(ext, 3).tolist(),
                         lo=np.round(lo, 3).tolist(), hi=np.round(hi, 3).tolist(), mean_votes=round(float(frac[torch.tensor(jj, device=dev)].mean()), 2)))
log('instances', len(objs), {l: sum(o['label'] == l for o in objs) for l in labels})
np.savez(W('objects', 'instances.npz'), point=keep[inst >= 0], inst=inst[inst >= 0])
json.dump(dict(labels=labels, objects=objs), open(W('objects', 'instances.json'), 'w'), indent=1)
