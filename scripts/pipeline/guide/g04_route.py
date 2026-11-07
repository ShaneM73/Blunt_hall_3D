"""Guide stage 4: route from the front entrance to a room, stairs as blocks, and the page data file.
- route: shortest path over the scan walking path (samples every 0.25 m, linked when < 0.8 m apart and < 0.4 m in
  height), from just outside the front entrance (BH_START_T s into the scan) to the walking-path point nearest the
  room's door sign (g02); feet = device height above the floor, simplified to 5 cm
- stairs: the route's climbing parts become 17 cm steps 1.3 m wide (landings stay flat)
- room: a block behind the sign's wall (3.6 m along the wall, up to the next parallel wall or 4.5 m deep)
- other rooms: every sign on the floors becomes a label
Usage: BH_ROOM=318 python g04_route.py     Out: data/<scan date>/guide.json (blocks + route + signs, building frame)"""
import os, sys, json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log, traj

ROOM = os.environ.get('BH_ROOM', '318'); START_T = float(os.environ.get('BH_START_T', 712))
OUT = os.environ.get('BH_GUIDE_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', '2026-11-04', 'guide.json'))
B = json.load(open(W('guide', 'blocks.json'))); fr = B['frame']; A, ORG, Z0 = fr['angle'], np.array(fr['origin']), fr['z0']
Rm = np.array([[np.cos(-A), -np.sin(-A)], [np.sin(-A), np.cos(-A)]])
FL = json.load(open(W('mesh', 'structure.json')))['floors'][:4]; FZ = np.array([f['z'] for f in FL])
def to_b(p): p = np.atleast_2d(p); return np.c_[p[:, :2] @ Rm.T - ORG, p[:, 2] - Z0]
signs = json.load(open(W('guide', 'signs.json')))

# ---- walking path as feet positions (device ~1.45 m above the floor)
T = traj(); t = T[:, 0] - T[0, 0]; P = T[:, 1:4].copy()
lvl = np.array([np.argmin(abs(z - 1.45 - FZ)) for z in P[:, 2]])
H = np.median(P[:, 2] - FZ[lvl]); feet = P.copy(); feet[:, 2] -= H
keep = [0]
for i in range(1, len(P)):
    if np.linalg.norm(feet[i] - feet[keep[-1]]) >= 0.25: keep.append(i)
keep = np.array(keep); F = feet[keep]; tk = t[keep]
tr = cKDTree(F[:, :2]); pairs = np.array(sorted(tr.query_pairs(0.8)))
dz = abs(F[pairs[:, 0], 2] - F[pairs[:, 1], 2]); pairs = pairs[dz < 0.4]
wgt = np.linalg.norm(F[pairs[:, 0]] - F[pairs[:, 1]], axis=1) + 1e-3
seq = np.c_[np.arange(len(F) - 1), np.arange(1, len(F))]; ws = np.linalg.norm(F[1:] - F[:-1], axis=1) + 1e-3
I = np.r_[pairs[:, 0], seq[:, 0]]; J = np.r_[pairs[:, 1], seq[:, 1]]; Wt = np.r_[wgt, ws]
G = coo_matrix((Wt, (I, J)), shape=(len(F), len(F))).tocsr()

# ---- target sign (estimated from same-parity neighbours when the sign itself was not read)
tg = [s for s in signs if s['num'] == ROOM]; est = False
if tg: tg = max(tg, key=lambda s: s['views'])
else:
    fl = int(ROOM[0]) - 1; n = int(ROOM); same = [s for s in signs if s['floor'] == fl and int(s['num']) % 2 == n % 2]
    a = max([s for s in same if int(s['num']) < n], key=lambda s: int(s['num'])); b = min([s for s in same if int(s['num']) > n], key=lambda s: int(s['num']))
    w = (n - int(a['num'])) / (int(b['num']) - int(a['num'])); tg = dict(num=ROOM, xyz=(np.array(a['xyz']) * (1 - w) + np.array(b['xyz']) * w).tolist(),
                                                                         floor=fl, views=0, estimated_from=[a['num'], b['num']]); est = True
    log('room sign not read; estimated between', a['num'], b['num'])
sx = np.array(tg['xyz']); onfl = np.flatnonzero(abs(F[:, 2] - FZ[tg['floor']]) < 0.5)
end = onfl[np.argmin(np.linalg.norm(F[onfl, :2] - sx[:2], axis=1))]
start = int(np.argmin(abs(tk - START_T)))
dist, pred = dijkstra(G, directed=False, indices=start, return_predecessors=True)
path = [end]
while path[-1] != start: path.append(pred[path[-1]])
path = path[::-1]; R3 = to_b(F[path]); log('route', round(dist[end], 1), 'm,', len(path), 'samples')

# smooth (keep z exact on the flat parts), simplify
from scipy.ndimage import uniform_filter1d
Rs = R3.copy(); Rs[:, :2] = uniform_filter1d(R3[:, :2], 5, axis=0, mode='nearest')
def rdp(p, eps):
    if len(p) < 3: return p
    a, b = p[0], p[-1]; ab = b - a; L = np.linalg.norm(ab) + 1e-9
    d = np.linalg.norm(np.cross(p - a, ab), axis=1) / L; k = int(np.argmax(d))
    return np.vstack([rdp(p[:k + 1], eps)[:-1], rdp(p[k:], eps)]) if d[k] > eps else np.vstack([a, b])
route = rdp(Rs, 0.05)

# ---- stairs as blocks: climbing stretches of the route
steps = []; zf = R3[:, 2]
for i in range(len(R3) - 1):
    a, b = R3[i], R3[i + 1]; dzz = b[2] - a[2]; d2 = np.linalg.norm(b[:2] - a[:2])
    if d2 < 1e-3 or abs(dzz) / d2 < 0.25: continue
    n = max(1, int(round(abs(dzz) / 0.17))); u = (b[:2] - a[:2]) / d2
    for k in range(n):
        c = a[:2] + u * d2 * (k + 0.5) / n; top = min(a[2], b[2]) + abs(dzz) * (k + 1) / n if dzz > 0 else a[2] - abs(dzz) * k / n
        steps.append(dict(c=c.round(3).tolist(), dir=u.round(3).tolist(), len=round(d2 / n, 3), top=round(float(top), 3)))
log('stair steps', len(steps))

# ---- room block behind the sign's wall
fb = B['floors'][tg['floor']]; sb = to_b(sx)[0]; corr = route[-1]
walls = np.array(fb['walls']); cx = (walls[:, 0] + walls[:, 2]) / 2; cy = (walls[:, 1] + walls[:, 3]) / 2
def dist_box(w, p): return np.hypot(max(w[0] - p[0], 0, p[0] - w[2]), max(w[1] - p[1], 0, p[1] - w[3]))
wi = int(np.argmin([dist_box(w, sb) for w in walls])); w = walls[wi]; along = 0 if (w[2] - w[0]) >= (w[3] - w[1]) else 1; ax = 1 - along
wc = (w[ax] + w[ax + 2]) / 2; side = np.sign(wc - corr[ax]) or 1.0; face = wc + side * (w[ax + 2] - w[ax]) / 2
depth = 4.5
for v in walls:
    vlen = v[along + 2] - v[along]; vc = (v[ax] + v[ax + 2]) / 2
    if vlen > 1.5 and v[along] < sb[along] < v[along + 2] and 1.5 < side * (vc - face) < depth: depth = side * (vc - face)
lo_a, hi_a = sb[along] - 1.8, sb[along] + 1.8; a0, a1 = sorted([face, face + side * depth])
box = [0, 0, 0, 0]; box[along], box[along + 2], box[ax], box[ax + 2] = lo_a, hi_a, a0, a1
room = dict(num=ROOM, floor=tg['floor'], box=[round(float(x), 3) for x in box], sign=sb.round(3).tolist(), estimated=est,
            views=tg.get('views', 0), est_from=tg.get('estimated_from'))

# ---- page data
lab = [dict(num=s['num'], floor=s['floor'], p=to_b(s['xyz'])[0].round(2).tolist(), views=s['views']) for s in signs
       if (s['views'] >= 2 and s['num'][0] == str(s['floor'] + 1)) or s['num'] == ROOM]          # a sign's first digit is its floor
cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
data = dict(room=room, route=route.round(3).tolist(), route_len=round(float(cum[-1]), 1), stairs=steps, labels=lab,
            floors=[dict(name=f['name'], z=f['z'], h=f['h'], walls=f['walls'], lintels=f['lintels'], slab=f['slab']) for f in B['floors']],
            entrance=route[0].round(2).tolist(), frame=dict(angle=A, origin=ORG.tolist(), z0=Z0))
json.dump(data, open(OUT, 'w'), separators=(',', ':')); log('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')
