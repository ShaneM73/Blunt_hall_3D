"""Stage 12: SketchUp-style structural model per floor from the clean cloud.
Frame: building axes (dominant wall direction, Manhattan). Per floor:
- walls: for each of the 4 wall-face orientations, points with that normal (+-12 deg) are swept in 0.25 m columns along
  the wall; c-histogram peaks (>= MINPTS) are chained across columns (|dc| < 3 cm, gaps < 0.5 m) into straight faces;
  each face spans floor -> local top; door/opening gaps (no points below 2.0 m over >= 0.6 m) become openings with a header
- floor / ceiling: occupancy of up-/down-facing points on a 5 cm grid, closed, decomposed into rectangles (ceiling height
  per rectangle = median of its points)
Out: mesh/structure.json (faces as rectangles: origin, u-axis, v-axis, size, normal, kind, floor), renders/structure_f*.png"""
import os, json
import numpy as np, cv2
from scipy import ndimage
from common import W, traj, log, removed_mask, detect_floors

d = np.load(W('color', 'hq_colored.npz')); rm = removed_mask(); k = d['has'] & ~rm
P = d['xyz'][k].astype(np.float64); Nr = d['nrm'][k]; T = traj()
REG = json.load(open(W('regions.json'))) if os.path.exists(W('regions.json')) else []
STAIR = [r['box'] for r in REG if r.get('stairs')]
in_stair = np.zeros(len(P), bool)
for b in STAIR: in_stair |= (P[:, 0] >= b[0]) & (P[:, 0] < b[1]) & (P[:, 1] >= b[2]) & (P[:, 1] < b[3])
log('points in stair boxes (kept as measured mesh)', in_stair.sum())
floors = detect_floors(P, Nr, T)
th = float(np.load(W('calib', 'manhattan_angle.npy'))) if os.path.exists(W('calib', 'manhattan_angle.npy')) else 0.0
ex = np.array([np.cos(th), np.sin(th)]); ey = np.array([-np.sin(th), np.cos(th)])
U = P[:, :2] @ ex; V = P[:, :2] @ ey; nu = Nr[:, :2] @ ex; nv = Nr[:, :2] @ ey
CW, MINPTS, DC = 0.25, 60, 0.03
faces = []

def rect(o2, a2, z0, z1, t0, t1, n2, kind, fl):
    """vertical rectangle: base line point o2 (u,v frame -> world) along a2 from t0..t1, z0..z1"""
    p0 = o2 + a2 * t0; L = t1 - t0
    w0 = p0[0] * ex + p0[1] * ey; wa = a2[0] * ex + a2[1] * ey; wn = n2[0] * ex + n2[1] * ey
    faces.append(dict(kind=kind, floor=fl, o=[float(w0[0]), float(w0[1]), float(z0)], u=[float(wa[0]), float(wa[1]), 0.0], v=[0.0, 0.0, 1.0],
                      su=float(L), sv=float(z1 - z0), n=[float(wn[0]), float(wn[1]), 0.0]))

for j, f in enumerate(floors):
    fz, cz = f['z'], f['ceiling_z']; top = f['zmax'] + 0.3 - 0.02
    band = (P[:, 2] > fz + 0.05) & (P[:, 2] < top) & ~in_stair
    vert = band & (np.abs(Nr[:, 2]) < 0.25)
    nw = 0
    for axis, sgn in ((0, 1), (0, -1), (1, 1), (1, -1)):
        nn = nu if axis == 0 else nv; c_all = U if axis == 0 else V; t_all = V if axis == 0 else U
        m = vert & (sgn * nn > np.cos(np.deg2rad(12)))
        c, t, z = c_all[m], t_all[m], P[m, 2]
        if len(c) < MINPTS: continue
        col = np.floor(t / CW).astype(int); order = np.argsort(col); c, t, z, col = c[order], t[order], z[order], col[order]
        ucol, st = np.unique(col, return_index=True); en = np.r_[st[1:], len(col)]
        peaks = []                                        # (col, c, zmin, zmax, n)
        for cc, a, b in zip(ucol, st, en):
            cs = c[a:b]
            if len(cs) < MINPTS: continue
            h, e = np.histogram(cs, bins=np.arange(cs.min() - 0.02, cs.max() + 0.03, 0.01))
            hs = np.convolve(h, [1, 2, 3, 2, 1], 'same')
            for i in np.flatnonzero((hs >= np.roll(hs, 1)) & (hs > np.roll(hs, -1)) & (hs >= 3 * MINPTS // 2)):
                sel = np.abs(cs - (e[i] + 0.005)) < 0.025
                if sel.sum() < MINPTS: continue
                zz = z[a:b][sel]; peaks.append((cc, float(np.median(cs[sel])), float(np.percentile(zz, 1)), float(np.percentile(zz, 99)), int(sel.sum()), t[a:b][sel], zz))
        # chain peaks across columns
        peaks.sort(key=lambda p: (p[0], p[1])); used = np.zeros(len(peaks), bool); bycol = {}
        for i, p in enumerate(peaks): bycol.setdefault(p[0], []).append(i)
        for i in range(len(peaks)):
            if used[i]: continue
            chain = [i]; used[i] = True; cur = peaks[i]
            while True:
                nxt = None
                for g in (1, 2):                                   # allow one empty column (gap <= 0.5 m)
                    for q in bycol.get(cur[0] + g, []):
                        if not used[q] and abs(peaks[q][1] - cur[1]) < DC: nxt = q; break
                    if nxt is not None: break
                if nxt is None: break
                chain.append(nxt); used[nxt] = True; cur = peaks[nxt]
            if len(chain) < 2: continue                            # >= 0.5 m long
            pts_t = np.concatenate([peaks[q][5] for q in chain]); pts_z = np.concatenate([peaks[q][6] for q in chain])
            cval = float(np.average([peaks[q][1] for q in chain], weights=[peaks[q][4] for q in chain]))
            t0, t1 = np.percentile(pts_t, 0.5), np.percentile(pts_t, 99.5)
            ztop = min(float(np.percentile(pts_z, 99)) + 0.02, top)
            if ztop - fz < 1.0: continue                           # low objects (furniture) are not walls
            # openings: 0.1 m bins along t with no points below 2.0 m but points above (doors) -> header only
            b = np.arange(t0, t1 + 0.1, 0.1); low = np.histogram(pts_t[pts_z < fz + 2.0], b)[0] > 0
            hi = np.histogram(pts_t[pts_z > fz + 2.15], b)[0] > 0
            lab, nl = ndimage.label(~low)
            o2 = np.zeros(2); a2 = np.zeros(2); n2 = np.zeros(2)
            if axis == 0: o2[:] = (cval, 0); a2[:] = (0, 1); n2[:] = (sgn, 0)
            else: o2[:] = (0, cval); a2[:] = (1, 0); n2[:] = (0, sgn)
            cuts = [t0]; heads = []
            for L in range(1, nl + 1):
                ii = np.flatnonzero(lab == L)
                if len(ii) * 0.1 >= 0.6 and ii[0] > 0 and ii[-1] < len(low) - 1:
                    a_, b_ = b[ii[0]], b[ii[-1] + 1]; cuts += [a_, b_]; heads.append((a_, b_, hi[ii].mean() > 0.3))
            cuts.append(t1)
            for s0, s1 in zip(cuts[0::2], cuts[1::2]):
                if s1 - s0 > 0.05: rect(o2, a2, fz, ztop, s0, s1, n2, 'wall', j); nw += 1
            for a_, b_, has_head in heads:
                if has_head and ztop > fz + 2.15: rect(o2, a2, fz + 2.1, ztop, a_, b_, n2, 'wall', j)
    # floor and ceiling rectangles on a 5 cm grid
    G = 0.05
    def rects(mask, z_of, kind, nz):
        if mask.sum() < 100: return 0
        u, v = U[mask], V[mask]; u0, v0 = u.min() - 0.5, v.min() - 0.5
        gi = ((u - u0) / G).astype(int); gj = ((v - v0) / G).astype(int); img = np.zeros((gi.max() + 11, gj.max() + 11), np.uint8)
        img[gi, gj] = 1; img = ndimage.binary_closing(img, np.ones((15, 15)), iterations=1); img = ndimage.binary_fill_holes(img); img = ndimage.binary_opening(img, np.ones((3, 3)))
        lab, nl = ndimage.label(img); sizes = ndimage.sum(img, lab, range(1, nl + 1)); img = np.isin(lab, 1 + np.flatnonzero(sizes * G * G > 0.5))
        zgrid = np.full(img.shape, np.nan); zsum = np.zeros(img.shape); zc = np.zeros(img.shape); np.add.at(zsum, (gi, gj), z_of); np.add.at(zc, (gi, gj), 1)
        zgrid[zc > 0] = zsum[zc > 0] / zc[zc > 0]
        n = 0; img = img.copy()
        while img.any():                                    # greedy maximal rectangles (row runs extended downwards)
            i, jj = np.argwhere(img)[0]; j1 = jj
            while j1 + 1 < img.shape[1] and img[i, j1 + 1]: j1 += 1
            i1 = i
            while i1 + 1 < img.shape[0] and img[i1 + 1, jj:j1 + 1].all(): i1 += 1
            blk = zgrid[i:i1 + 1, jj:j1 + 1]; zz = float(np.nanmedian(blk)) if np.isfinite(blk).any() else (fz if kind == 'floor' else cz)
            img[i:i1 + 1, jj:j1 + 1] = False
            ua, ub, va, vb = u0 + i * G, u0 + (i1 + 1) * G, v0 + jj * G, v0 + (j1 + 1) * G
            if (ub - ua) * (vb - va) < 0.02: continue
            w0 = ua * ex + va * ey
            faces.append(dict(kind=kind, floor=j, o=[float(w0[0]), float(w0[1]), zz], u=[float(ex[0]), float(ex[1]), 0.0], v=[float(ey[0]), float(ey[1]), 0.0],
                              su=float(ub - ua), sv=float(vb - va), n=[0.0, 0.0, nz])); n += 1
        return n
    fm = (Nr[:, 2] > 0.85) & (np.abs(P[:, 2] - fz) < 0.06) & ~in_stair
    cm = (Nr[:, 2] < -0.85) & (P[:, 2] > fz + 2.0) & (P[:, 2] < top) & ~in_stair
    nf = rects(fm, P[fm, 2], 'floor', 1.0); nc = rects(cm, P[cm, 2], 'ceiling', -1.0)
    log(f"floor {j} z {fz}: wall faces {nw}, floor rects {nf}, ceiling rects {nc}")
    # plan check
    sel = [q for q in faces if q['floor'] == j]
    allp = np.array([q['o'][:2] for q in sel]); mn = allp.min(0) - 3; mx = allp.max(0) + 3; S = 20
    img = np.full((int((mx[1] - mn[1]) * S) + 1, int((mx[0] - mn[0]) * S) + 1, 3), 255, np.uint8)
    px = lambda p: (int((p[0] - mn[0]) * S), int((mx[1] - p[1]) * S))
    for q in sel:
        o = np.array(q['o'][:2]); a = np.array(q['u'][:2]) * q['su']; b = np.array(q['v'][:2]) * q['sv']
        if q['kind'] == 'floor': cv2.fillPoly(img, [np.array([px(o), px(o + a), px(o + a + b), px(o + b)])], (225, 225, 225))
    for q in sel:
        if q['kind'] == 'wall':
            o = np.array(q['o'][:2]); a = np.array(q['u'][:2]) * q['su']; cv2.line(img, px(o), px(o + a), (30, 30, 30) if q['o'][2] < fz + 0.5 else (0, 120, 255), 2)
    cv2.imwrite(W('renders', f'structure_f{j}.png'), img)
json.dump(dict(angle=th, floors=floors, faces=faces, stairs=STAIR), open(W('mesh', 'structure.json'), 'w'))
log('faces', len(faces), 'area m2: walls %.0f floors %.0f ceilings %.0f' % tuple(sum(q['su'] * q['sv'] for q in faces if q['kind'] == kd) for kd in ('wall', 'floor', 'ceiling')))
