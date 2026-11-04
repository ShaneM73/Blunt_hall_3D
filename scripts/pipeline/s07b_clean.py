"""Stage 7b: remove clutter that is not real structure.
1) Ghosts: LiDAR reflections off glossy floors, glass and metal appear behind real surfaces (below the floor, behind walls).
   For every 3rd frame, a spherical z-buffer (0.4 deg bins) of the static cloud itself is built around the scanner; a point
   is 'front' when it is the nearest surface in its bin (within 4 cm + 1 % of range). Real points are front from the frames
   that captured them; a point never front in >= 3 frames that have it in range is a ghost.
2) Floaters: connected components at 6 cm; components smaller than MIN_PTS points (and under 0.5 m across) are dropped.
Out: clouds/clutter_rm.npy (bool per hq_colored point), renders/clean_check.jpg"""
import os
import numpy as np, torch
from scipy import ndimage
from common import W, traj, log, detect_floors

dev = 'cuda:' + os.environ.get('BH_GPU', '1')
d = np.load(W('color', 'hq_colored.npz')); P = d['xyz']; N = len(P)
rm0 = np.load(W('clouds', 'person_rm.npy')); T = traj()
key = np.floor(P / 4.0).astype(np.int64); kmin = key.min(0); key -= kmin; dims = key.max(0) + 1
lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]; order = np.argsort(lin, kind='stable')
ub, st = np.unique(lin[order], return_index=True); en = np.r_[st[1:], N]
bc = np.stack(np.unravel_index(ub, dims), 1) * 4.0 + kmin * 4.0 + 2.0
Pg = torch.tensor(P[order], device=dev)
front = torch.zeros(N, device=dev, dtype=torch.int32); inr = torch.zeros(N, device=dev, dtype=torch.int32)
AZ, EL = 900, 450
for f in range(0, len(T), 3):
    o = T[f, 1:4]; nb = np.flatnonzero(np.linalg.norm(bc - o, axis=1) < 10 + 3.5)
    if not len(nb): continue
    sel = torch.tensor(np.concatenate([np.arange(st[i], en[i]) for i in nb]), device=dev)
    X = Pg[sel] - torch.tensor(o, device=dev, dtype=torch.float32); r = X.norm(dim=1)
    az = torch.atan2(X[:, 1], X[:, 0]); el = torch.asin((X[:, 2] / r.clamp(min=1e-6)).clamp(-1, 1))
    b = ((az + np.pi) / (2 * np.pi) * AZ).long().clamp(0, AZ - 1) * EL + ((el + np.pi / 2) / np.pi * EL).long().clamp(0, EL - 1)
    ok = (r > 0.4) & (r < 10)
    zb = torch.full((AZ * EL,), 1e9, device=dev).scatter_reduce(0, b[ok], r[ok], 'amin', include_self=False)
    fr = ok & (r <= zb[b] + 0.04 + 0.01 * r)
    front[sel] += fr.int(); inr[sel] += ok.int()
    if f % 1500 == 0: log('frame', f, 'of', len(T))
front_s = front.cpu().numpy(); inr_s = inr.cpu().numpy()
fr_ = np.empty(N, np.int32); fr_[order] = front_s; ir_ = np.empty(N, np.int32); ir_[order] = inr_s
ghost = (fr_ == 0) & (ir_ >= 3)
log('ghost points (never the front surface): %d (%.2f%%)' % (ghost.sum(), ghost.mean() * 100))

# slab test on raw returns: a beam cannot pass through a measured floor (or ceiling). A return below the scanner's floor
# (above its ceiling) whose beam crosses that plane over measured floor (ceiling) is a reflection.
Nr = d['nrm']; floors = detect_floors(P[~rm0], Nr[~rm0], T); log('floors', [(f['z'], f['ceiling_z']) for f in floors])
G = 0.10; gx0, gy0 = P[:, 0].min() - 1, P[:, 1].min() - 1; gnx, gny = int(np.ptp(P[:, 0]) / G) + 25, int(np.ptp(P[:, 1]) / G) + 25
def grid(mask):
    g = np.zeros((gnx, gny), bool); g[((P[mask, 0] - gx0) / G).astype(int), ((P[mask, 1] - gy0) / G).astype(int)] = True
    return ndimage.binary_dilation(ndimage.binary_closing(g, iterations=2), iterations=1)
fgrid = [grid(~rm0 & (Nr[:, 2] > 0.8) & (np.abs(P[:, 2] - f['z']) < 0.06)) for f in floors]
cgrid = [grid(~rm0 & (Nr[:, 2] < -0.8) & (np.abs(P[:, 2] - f['ceiling_z']) < 0.08)) for f in floors]
fz = np.array([f['z'] for f in floors]); cz = np.array([f['ceiling_z'] for f in floors])
Wr = np.load(W('decoded', 'world_xyz.npy'), mmap_mode='r'); WF = np.load(W('decoded', 'world_frame.npy'), mmap_mode='r')
ofl = np.array([np.flatnonzero(fz <= z - 0.5).max() if (fz <= z - 0.5).any() else 0 for z in T[:, 3]])   # scanner's floor
gk_all, ng_all = [], []
for s0 in range(0, len(Wr), 10_000_000):
    X = np.asarray(Wr[s0:s0 + 10_000_000], np.float64); fr = np.asarray(WF[s0:s0 + 10_000_000]); O = T[fr, 1:4]; fl = ofl[fr]
    bad = np.zeros(len(X), bool)
    for side, zp, grids in (('floor', fz[fl] , fgrid), ('ceil', cz[fl], cgrid)):
        cross = (X[:, 2] < zp - 0.10) if side == 'floor' else (X[:, 2] > zp + 0.10)
        ii = np.flatnonzero(cross & (np.abs(X[:, 2] - O[:, 2]) > 1e-3))
        a = (zp[ii] - O[ii, 2]) / (X[ii, 2] - O[ii, 2]); C = O[ii, :2] + a[:, None] * (X[ii, :2] - O[ii, :2])
        gi = ((C[:, 0] - gx0) / G).astype(int); gj = ((C[:, 1] - gy0) / G).astype(int)
        ok = (gi >= 0) & (gi < gnx) & (gj >= 0) & (gj < gny); hit = np.zeros(len(ii), bool)
        for f in np.unique(fl[ii]):
            m = ok & (fl[ii] == f); hit[m] = grids[f][gi[m], gj[m]]
        bad[ii[hit]] = True
    k3 = np.floor(X / 0.03).astype(np.int64)
    gk_all.append(k3[bad]); ng_all.append(k3[~bad][::4])            # non-ghost support subsampled (x4 weight below)
GK = np.vstack(gk_all); NG = np.vstack(ng_all); log('raw returns flagged as reflections: %d' % len(GK))
def keyof(k): return (k[:, 0] + 10 ** 5) * 10 ** 10 + (k[:, 1] + 10 ** 5) * 10 ** 5 + (k[:, 2] + 10 ** 5)
gu, gc = np.unique(keyof(GK), return_counts=True); nu, nc = np.unique(keyof(NG), return_counts=True)
pk = keyof(np.floor(P / 0.03).astype(np.int64))
gcnt = np.zeros(len(P)); ix = np.searchsorted(gu, pk); ok = (ix < len(gu)) & (gu[np.minimum(ix, len(gu) - 1)] == pk); gcnt[ok] = gc[ix[ok]]
ncnt = np.zeros(len(P)); ix = np.searchsorted(nu, pk); ok = (ix < len(nu)) & (nu[np.minimum(ix, len(nu) - 1)] == pk); ncnt[ok] = nc[ix[ok]] * 4
slab = (gcnt >= 2) & (gcnt > 2 * ncnt)
log('reflection points (behind a measured floor/ceiling): %d (%.2f%%)' % (slab.sum(), slab.mean() * 100))
ghost |= slab

# floaters: connected components on a 6 cm occupancy grid of the remaining points
MIN_PTS = int(os.environ.get('BH_MIN_CLUSTER', 3000))
keep = ~(rm0 | ghost); Q = P[keep]
g = np.floor((Q - Q.min(0)) / 0.06).astype(np.int64); gd = g.max(0) + 1
occ_lin, inv = np.unique((g[:, 0] * gd[1] + g[:, 1]) * gd[2] + g[:, 2], return_inverse=True)
cells = np.stack(np.unravel_index(occ_lin, gd), 1)
# label in chunks of the bounding grid would be huge; label on a compact grid via sparse dilation: use ndimage on cropped dense grid
lab = np.zeros(len(occ_lin), np.int64)
dense = np.zeros(gd, bool) if np.prod(gd) < 4e9 else None
if dense is None: raise SystemExit('grid too large for dense labelling')
dense[cells[:, 0], cells[:, 1], cells[:, 2]] = True
L, nl = ndimage.label(dense, structure=np.ones((3, 3, 3))); del dense
lab = L[cells[:, 0], cells[:, 1], cells[:, 2]]; del L
pl = lab[inv]; cnt = np.bincount(pl)
ext = np.zeros(len(cnt));
for ax in range(3):
    mx = np.full(len(cnt), -1e9); mn = np.full(len(cnt), 1e9); np.maximum.at(mx, pl, Q[:, ax]); np.minimum.at(mn, pl, Q[:, ax]); ext = np.maximum(ext, mx - mn)
small = (cnt[pl] < MIN_PTS) & (ext[pl] < 0.5)
log('components', nl, 'floater points removed', small.sum(), 'in', int(((cnt < MIN_PTS) & (ext < 0.5)).sum()), 'components')
rm = ghost.copy(); idx = np.flatnonzero(keep); rm[idx[small]] = True

# sparse streaks: far surfaces seen only by a few beam rings (through atria, down corridors). Density = points in the
# 3x3x3 block of 4 cm cells around a point (a well-scanned surface has ~100+); entrance range boxes are exempt.
MIN_DENS = int(os.environ.get('BH_MIN_DENS', 25))
keep2 = np.flatnonzero(~(rm | rm0)); Q = P[keep2]; c = np.floor(Q / 0.04).astype(np.int64)
def ck(c): return (c[:, 0] + 10 ** 5) * 10 ** 10 + (c[:, 1] + 10 ** 5) * 10 ** 5 + (c[:, 2] + 10 ** 5)
cu, cc = np.unique(ck(c), return_counts=True); dens = np.zeros(len(Q), np.int32)
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            k = ck(c + [dx, dy, dz]); ix = np.searchsorted(cu, k); ix = np.minimum(ix, len(cu) - 1); m = cu[ix] == k; dens[m] += cc[ix[m]]
import json as _j
RB = [r['range_box'] for r in (_j.load(open(W('regions.json'))) if os.path.exists(W('regions.json')) else []) if 'range_box' in r]
exempt = np.zeros(len(Q), bool)
for r in (_j.load(open(W('regions.json'))) if os.path.exists(W('regions.json')) else []):
    if 'range_box' in r: b = r['box']; exempt |= (Q[:, 0] >= b[0]) & (Q[:, 0] < b[1]) & (Q[:, 1] >= b[2]) & (Q[:, 1] < b[3])
sparse = (dens < MIN_DENS) & ~exempt
rm[keep2[sparse]] = True
log('sparse-streak points removed: %d (%.2f%%), density median %d' % (sparse.sum(), sparse.mean() * 100, np.median(dens)))
np.save(W('clouds', 'clutter_rm.npy'), rm)
log('clutter removed total %d (%.2f%%)' % (rm.sum(), rm.mean() * 100))

# check image: side view (x along building axis) of removed vs kept, subsampled
import cv2
ang = np.deg2rad(-55); R2 = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
def view(mask, col, img, s=8):
    X = P[mask][::s]; u = X[:, :2] @ R2[0]; z = X[:, 2]
    x = ((u - umin) / 0.06).astype(int); y = ((zmax - z) / 0.06).astype(int); ok = (x >= 0) & (x < img.shape[1]) & (y >= 0) & (y < img.shape[0])
    img[y[ok], x[ok]] = col
u_all = P[::50, :2] @ R2[0]; umin, umax = u_all.min(), u_all.max(); zmax, zmin = P[:, 2].max(), P[:, 2].min()
img = np.full((int((zmax - zmin) / 0.06) + 1, int((umax - umin) / 0.06) + 1, 3), 255, np.uint8)
view(~rm & ~rm0, (150, 150, 150), img, 10); view(rm, (0, 0, 230), img, 1)
cv2.imwrite(W('renders', 'clean_check.jpg'), img)
