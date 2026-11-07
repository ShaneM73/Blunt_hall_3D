"""Guide stage 4: route from the front entrance to a room, stairs as blocks, and the page data file.
- route: shortest path over the scan walking path (samples every 0.25 m, linked when < 0.8 m apart and < 0.4 m in
  height), from just outside the front entrance (BH_START_T s into the scan) to the walking-path point nearest the
  room's door sign (g02); feet = device height above the floor, simplified to 5 cm
- stairs: the route's climbing parts become straight flights on the building axes (17 cm steps, 1.3 m wide); the flat\n  parts between flights become landing plates that span both flight ends
- room: a block behind the sign's wall (3.6 m along the wall, up to the next parallel wall or 4.5 m deep)
- other rooms: every sign on the floors becomes a label
Usage: BH_ROOM=319 BH_ROOM_REF=323:2 python g04_route.py   (REF: the room is 2 doors from the 323 sign towards the stairs)
           Out: data/<scan date>/guide.json (blocks + route + signs, building frame)"""
import os, sys, json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log, traj

ROOM = os.environ.get('BH_ROOM', '319'); START_T = float(os.environ.get('BH_START_T', 712))
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

def from_ref(ref, k):
    """sign position of the room k doors from the read sign `ref`, towards the stairs, on the same wall.
    Doors are the 0.8-1.0 m recessed wall pieces along the sign's wall line; a sign sits just past its own door."""
    s = max([s for s in signs if s['num'] == ref], key=lambda s: (s['views'], s['votes'].get(ref, 0))); fl = s['floor']
    walls = np.array(B['floors'][fl]['walls']); sb = to_b(s['xyz'])[0]
    near = walls[np.argmin([np.hypot(max(w[0] - sb[0], 0, sb[0] - w[2]), max(w[1] - sb[1], 0, sb[1] - w[3])) for w in walls])]
    al = 0 if near[2] - near[0] >= near[3] - near[1] else 1; ax = 1 - al; wc = (near[ax] + near[ax + 2]) / 2
    ln = walls[(abs((walls[:, ax] + walls[:, ax + 2]) / 2 - wc) < 0.5) & (abs((walls[:, al] + walls[:, al + 2]) / 2 - sb[al]) < 15)]
    cs = to_b(F[np.argmin(np.linalg.norm(F - np.array(s['xyz']), axis=1))])[0]; out = -np.sign(cs[ax] - wc)        # away from the corridor
    face = out * np.where(out > 0, ln[:, ax], ln[:, ax + 2]); deep = face.max()           # corridor-side face; doors are set back
    doors = sorted(((w[al] + w[al + 2]) / 2, (w[al + 2] - w[al])) for w, b in zip(ln, face) if 0.8 <= w[al + 2] - w[al] <= 1.0 and b > deep - 0.05)
    log('wall line', round(wc, 2), 'corridor side', -out, 'doors at', [round(d[0], 2) for d in doors])
    st = json.load(open(W('mesh', 'structure.json')))['stairs'][0]
    stb = to_b([(st[0] + st[1]) / 2, (st[2] + st[3]) / 2, 0])[0]; toward = np.sign(stb[al] - sb[al])     # direction of the stairs
    own = min(doors, key=lambda d: abs(sb[al] - (d[0] - toward * (d[1] / 2 + 0.3))))                   # sign is past its door
    seq = sorted([d for d in doors if toward * (d[0] - own[0]) > 0], key=lambda d: toward * (d[0] - own[0]))
    pitch = np.median(np.diff([d[0] for d in doors])); tgt = seq[k - 1]
    log(f'room {ROOM}: {k} doors from {ref} towards the stairs (door pitch {pitch:.2f} m), door centre {tgt[0]:.2f}')
    pb = sb.copy(); pb[al] += tgt[0] - own[0]
    xy = (pb[:2] + ORG) @ Rm; return dict(num=ROOM, xyz=[xy[0], xy[1], pb[2] + Z0], floor=fl, views=0, ref=ref, ref_doors=k, door=float(tgt[0]), pitch=float(pitch))


# ---- target sign: read directly, or counted in doors from a read sign (BH_ROOM_REF=323:2), or interpolated
REF = os.environ.get('BH_ROOM_REF')
tg = [s for s in signs if s['num'] == ROOM]; est = False
if REF: tg = from_ref(REF.split(':')[0], int(REF.split(':')[1])); est = True
elif tg: tg = max(tg, key=lambda s: (s['views'], s['votes'].get(ROOM, 0)))          # most views, then strongest reading
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

# ---- stairs as blocks: straight flights on the building axes + flat landings between them
# climbing samples (height gain > 0.25 m per m walked, smoothed) form flights; the flat runs between flights are landings
cum2 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(R3[:, :2], axis=0), axis=1))]
zs = uniform_filter1d(R3[:, 2], 3, mode='nearest'); slope = np.gradient(zs, cum2 + np.arange(len(cum2)) * 1e-6)
up = slope > 0.25; idx = np.flatnonzero(up)
runs = []                                                              # [start, end] of climbing / flat runs inside the stairs
if len(idx):
    i = idx[0]
    while i <= idx[-1]:
        j = i
        while j + 1 <= idx[-1] and up[j + 1] == up[i]: j += 1
        runs.append([bool(up[i]), i, j]); i = j + 1
# merge tiny flat gaps (< 0.6 m walked and < 0.25 m height change) into the flight around them
m = []
for r in runs:
    if m and not r[0] and cum2[r[2]] - cum2[r[1]] < 0.6 and abs(R3[r[2], 2] - R3[r[1], 2]) < 0.25: m[-1][2] = r[2]; continue
    if m and r[0] and m[-1][0]: m[-1][2] = r[2]; continue
    m.append(r)
flights = []
for k, (is_up, i, j) in enumerate(m):
    if is_up:
        i0, j0 = max(i - 1, 0), min(j + 1, len(R3) - 1); a_, b_ = R3[i0], R3[j0]
        al = int(np.argmax(abs(b_[:2] - a_[:2]))); lat = float(np.median(R3[i0:j0 + 1, 1 - al]))
        z0, z1 = float(a_[2]), float(b_[2]); n = max(2, int(round((z1 - z0) / 0.17)))
        flights.append(dict(axis=al, lat=round(lat, 3), s0=round(float(a_[al]), 3), s1=round(float(b_[al]), 3), z0=round(z0, 3), z1=round(z1, 3), n=n, w=1.3))
# flights meet their landings: a landing is level with the flight below it and spans both flight ends (1.3 m wide)
for fa, fb in zip(flights[:-1], flights[1:]):
    zl = fa['z1']; fb['z0'] = zl
    ends = [(fa['s1'], fa['lat'], fa['axis']), (fb['s0'], fb['lat'], fb['axis'])]
    xy = np.array([[s_, l_] if ax_ == 0 else [l_, s_] for s_, l_, ax_ in ends])
    lo_, hi_ = xy.min(0) - 0.65, xy.max(0) + 0.65
    for e, (s_, l_, ax_) in enumerate(ends):                        # the landing reaches 1.3 m past each flight end
        d_ = 1.3 * (1 if e == 0 else -1) * np.sign((fa['s1'] - fa['s0']) if e == 0 else (fb['s1'] - fb['s0']))
        p_ = np.array([s_ + d_, l_] if ax_ == 0 else [l_, s_ + d_]); lo_, hi_ = np.minimum(lo_, p_), np.maximum(hi_, p_)
    fa['landing'] = [round(float(v), 3) for v in (lo_[0], lo_[1], hi_[0], hi_[1])] + [round(zl, 3)]
stairs = dict(flights=flights, landings=[f.pop('landing') for f in flights if 'landing' in f])
# route on the stairs: along each flight's centre line (half a rise above the nosings), across the landings with square turns
def P(ax_, s_, l_, z_): return [s_, l_, z_] if ax_ == 0 else [l_, s_, z_]
sp = []
for k, f in enumerate(flights):
    rise = (f['z1'] - f['z0']) / f['n']; d_ = np.sign(f['s1'] - f['s0'])
    sp += [P(f['axis'], f['s0'], f['lat'], f['z0'] + rise * 0.5), P(f['axis'], f['s1'], f['lat'], f['z1'] + rise * 0.5)]
    if k + 1 < len(flights):
        g_ = flights[k + 1]; sm = f['s1'] + d_ * 0.65
        sp += [P(f['axis'], sm, f['lat'], f['z1']), P(f['axis'], sm, g_['lat'], f['z1'])]
if flights:
    ia = int(np.flatnonzero(route[:, 2] > flights[0]['z0'] + 0.1)[0]) - 1
    ib = int(np.flatnonzero(route[:, 2] > flights[-1]['z1'] - 0.05)[0])
    route = np.vstack([route[:ia + 1], np.array(sp), route[ib:]])
log('stairs:', len(flights), 'flights', [f"{f['z0']:.2f}->{f['z1']:.2f} ({f['n']} steps)" for f in flights], len(stairs['landings']), 'landings')

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
lo_a, hi_a = (tg['door'] - tg['pitch'] / 2, tg['door'] + tg['pitch'] / 2) if 'door' in tg else (sb[along] - 1.8, sb[along] + 1.8); a0, a1 = sorted([face, face + side * depth])
box = [0, 0, 0, 0]; box[along], box[along + 2], box[ax], box[ax + 2] = lo_a, hi_a, a0, a1
room = dict(num=ROOM, floor=tg['floor'], box=[round(float(x), 3) for x in box], sign=sb.round(3).tolist(), estimated=est,
            views=tg.get('views', 0), est_from=tg.get('estimated_from'), ref=tg.get('ref'), ref_doors=tg.get('ref_doors'))

# ---- page data
lab = [dict(num=s['num'], floor=s['floor'], p=to_b(s['xyz'])[0].round(2).tolist(), views=s['views']) for s in signs
       if (s['views'] >= 2 and s['num'][0] == str(s['floor'] + 1)) or s['num'] == ROOM]          # a sign's first digit is its floor
cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
data = dict(room=room, route=route.round(3).tolist(), route_len=round(float(cum[-1]), 1), stairs=stairs, labels=lab,
            floors=[dict(name=f['name'], z=f['z'], h=f['h'], walls=f['walls'], lintels=f['lintels'], slab=f['slab']) for f in B['floors']],
            entrance=route[0].round(2).tolist(), frame=dict(angle=A, origin=ORG.tolist(), z0=Z0))
json.dump(data, open(OUT, 'w'), separators=(',', ':')); log('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')
