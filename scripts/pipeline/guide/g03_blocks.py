"""Guide stage 3: block ("积木") model for the wayfinding page, from the structural faces of s12 (mesh/structure.json).
Everything is turned into the building frame (walls along x / y, origin at the plan centre, metres, z up, z=0 = 1st floor).
Per floor:
- walls: full-height wall faces drawn 15 cm thick behind the face on a 5 cm grid, gaps < 25 cm closed, pieces < 0.5 m
  dropped, then split into axis-aligned boxes; door headers (faces starting above 1.7 m) become lintels
- slab: floor faces + walls, closed by 0.6 m, holes filled -> outline polygons (outer ring + holes)
Out: guide/blocks.json {frame:{angle, origin, z0}, floors:[{name, z, h, walls:[[x0,y0,x1,y1,z0,z1]], slab:[[ring, hole..]]}]}"""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log

S = json.load(open(W('mesh', 'structure.json'))); A = S['angle']; R = np.array([[np.cos(-A), -np.sin(-A)], [np.sin(-A), np.cos(-A)]])
FL = S['floors'][:4]; Z0 = FL[0]['z']; RES = 0.05; TH = 0.15
NAMES = ['1st floor', '2nd floor', '3rd floor', '4th floor']
xy = []
for f in S['faces']:
    o, u, v = np.array(f['o']), np.array(f['u']), np.array(f['v'])
    xy += [o[:2], (o + u * f['su'])[:2], (o + v * f['sv'])[:2], (o + u * f['su'] + v * f['sv'])[:2]]
xy = np.array(xy) @ R.T; lo, hi = np.percentile(xy, 0.2, 0) - 2, np.percentile(xy, 99.8, 0) + 2; ORG = (lo + hi) / 2
lo, hi = lo - ORG, hi - ORG; NX, NY = (np.ceil((hi - lo) / RES)).astype(int) + 1


def to_b(p): return (np.asarray(p)[..., :2] @ R.T) - ORG                         # world xy -> building xy
def px(p): return np.round((np.asarray(p) - lo) / RES).astype(int)


def boxes(mask):
    """split a binary mask into axis-aligned rectangles (row runs merged down identical columns)"""
    out, open_ = [], {}
    for r in range(mask.shape[0] + 1):
        runs = set()
        if r < mask.shape[0]:
            row = np.r_[0, mask[r].astype(np.int8), 0]; d = np.flatnonzero(np.diff(row)); runs = set(zip(d[::2], d[1::2]))
        for k in list(open_):
            if k not in runs: out.append((k[0], open_.pop(k), k[1], r))
        for k in runs:
            if k not in open_: open_[k] = r
    return out                                                                   # (c0, r0, c1, r1) exclusive


def rect_px(f, img, thick, val=1):
    o, u, v, n = map(np.array, (f['o'], f['u'], f['v'], f['n']))
    a, b = to_b(o), to_b(o + u * f['su']); nb = n[:2] @ R.T
    q = np.array([a, b, b - nb * thick, a - nb * thick]); cv2.fillPoly(img, [px(q)], val)


out = dict(frame=dict(angle=A, origin=ORG.tolist(), z0=Z0, res=RES), floors=[])
for i, fl in enumerate(FL):
    z, h = fl['z'], fl['ceiling'] - fl['z']
    wall = np.zeros((NY, NX), np.uint8); lint = np.zeros_like(wall); flo = np.zeros_like(wall)
    for f in S['faces']:
        if f['floor'] != i: continue
        if f['kind'] == 'wall':
            z0 = f['o'][2] - z; z1 = z0 + f['sv']
            if z0 < 0.5 and z1 > 1.6: rect_px(f, wall, TH)
            elif z0 > 1.7: rect_px(f, lint, TH)
        elif f['kind'] == 'floor':
            o, u, v = map(np.array, (f['o'], f['u'], f['v'])); q = np.array([o, o + u * f['su'], o + u * f['su'] + v * f['sv'], o + v * f['sv']])
            cv2.fillPoly(flo, [px(to_b(q))], 1)
    # close small gaps along each axis only (keeps door gaps >= 25 cm), drop crumbs
    g = int(0.25 / RES)
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((1, g), np.uint8)); wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((g, 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(wall, connectivity=8)
    keep = np.r_[False, np.maximum(st[1:, 2], st[1:, 3]) * RES >= 0.5]; wall = keep[lab].astype(np.uint8)
    lint = lint & (1 - wall)
    W_ = [[lo[0] + c0 * RES, lo[1] + r0 * RES, lo[0] + c1 * RES, lo[1] + r1 * RES, 0.0, round(h, 2)] for c0, r0, c1, r1 in boxes(wall)]
    L_ = [[lo[0] + c0 * RES, lo[1] + r0 * RES, lo[0] + c1 * RES, lo[1] + r1 * RES, 2.1, round(h, 2)] for c0, r0, c1, r1 in boxes(lint)
          if max(c1 - c0, r1 - r0) * RES >= 0.6]
    sl = cv2.morphologyEx(np.maximum(flo, wall), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(sl); keep = np.r_[False, st[1:, 4] * RES * RES >= 4]; sl = keep[lab].astype(np.uint8)
    cs, hier = cv2.findContours(sl, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE); polys = []
    for k, c in enumerate(cs):
        if hier[0][k][3] >= 0: continue
        ring = [cv2.approxPolyDP(c, 2, True)[:, 0]]
        ch = hier[0][k][2]
        while ch >= 0:
            if cv2.contourArea(cs[ch]) * RES * RES > 2: ring.append(cv2.approxPolyDP(cs[ch], 2, True)[:, 0])
            ch = hier[0][ch][0]
        polys.append([(lo + r * RES).round(3).tolist() for r in ring])
    r2 = lambda b: [round(float(x), 3) for x in b]
    out['floors'].append(dict(name=NAMES[i], z=round(z - Z0, 3), h=round(h, 2), walls=[r2(b) for b in W_], lintels=[r2(b) for b in L_], slab=polys))
    log(f'floor {i}: {len(W_)} wall boxes, {len(L_)} lintels, {len(polys)} slab parts')
os.makedirs(W('guide'), exist_ok=True); json.dump(out, open(W('guide', 'blocks.json'), 'w'))
