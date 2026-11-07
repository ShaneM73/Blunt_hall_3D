"""Guide stage 2: room-number signs -> 3D positions.
1. candidates: g01 text boxes that contain a digit
2. re-read: each box (+40 % margin) upscaled to 96 px high and read again with digits only; keep 3-digit numbers
3. place: ray from the camera through the box centre, marched against the clean LiDAR cloud (first point within 4 cm)
4. merge: readings within 0.6 m are one sign; the number is the confidence-weighted vote
Out: guide/signs.json  [{num, votes, views, conf, xyz, floor, frame, box}]  + renders/signs_contact.jpg"""
import os, sys, json, glob, re
import numpy as np, cv2, easyocr
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log, removed_mask

FL = json.load(open(W('mesh', 'structure.json')))['floors'][:4]
cams = json.load(open(W('images', 'cameras.json'))); K = np.array(cams['K']); FR = {f['file']: f for f in cams['frames']}
ocr = {}
for p in sorted(glob.glob(W('guide', 'ocr', 'shard*.json'))): ocr.update(json.load(open(p)))
sam = {}
for p in sorted(glob.glob(W('guide', 'sam3_signs', 'shard*.json'))): sam.update(json.load(open(p)))
cand = [(f, np.array(b[:4]).reshape(2, 2), '', b[4], 'sam') for f, L in sam.items() for b in L if b[3] - b[1] >= 12]
inside = lambda f, c: any(b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3] for b in sam.get(f, []))
cand += [(f, np.array(b), t, c, 'ocr') for f, L in ocr.items() for b, t, c in L
         if re.search(r'\d', t) and np.ptp(np.array(b)[:, 1]) >= 7 and not inside(f, np.array(b).mean(0))]
log('candidates', len(cand), '(SAM 3 sign boxes', sum(c[4] == 'sam' for c in cand), ')')

rd = easyocr.Reader(['en'], gpu=True, verbose=False); reads = []
for f, b, t, c, src in cand:
    im = cv2.imread(W('images', 'pinhole', f)); (x0, y0), (x1, y1) = b.min(0), b.max(0)
    m = 0.15 if src == 'sam' else 0.4; mx, my = m * (x1 - x0), m * (y1 - y0)
    x0, y0, x1, y1 = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(im.shape[1], x1 + mx)), int(min(im.shape[0], y1 + my))
    cr = im[y0:y1, x0:x1]
    if min(cr.shape[:2]) < 4: continue
    s = min(4.0, (240 if src == 'sam' else 96) / cr.shape[0]); cr = cv2.resize(cr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    rs = [(np.ptp(np.array(bb)[:, 1]), tt, cc) for bb, tt, cc in rd.readtext(cr, allowlist='0123456789', text_threshold=0.5, low_text=0.3)]
    rs += [(np.ptp(np.array(bb)[:, 1]), re.sub(r'\D', '', tt), cc) for bb, tt, cc in rd.readtext(cr, text_threshold=0.5, low_text=0.3)
           if re.fullmatch(r'\D*\d{3}\D*', tt)]
    if t: rs.append((0, re.sub(r'\D', '', t), c * 0.8))
    rs = [r for r in rs if re.fullmatch(r'[1-4]\d\d', r[1])]
    if not rs: continue
    hmax = max(r[0] for r in rs); rs = [r for r in rs if r[0] >= 0.6 * hmax]            # the room number is the big text
    best = max(rs, key=lambda r: r[2])
    reads.append(dict(frame=f, box=[x0, y0, x1, y1], num=best[1], conf=round(float(best[2]), 3), src=src))
log('3-digit readings', len(reads))

d = np.load(W('color', 'hq_colored.npz')); P = d['xyz'][d['has'] & ~removed_mask()]
v = np.unique(np.floor(P / 0.02).astype(np.int64), axis=0, return_index=True)[1]; P = P[v]; tree = cKDTree(P); log('cloud', len(P))
for r in reads:
    f = FR[r['frame']]; c2w = np.array(f['c2w']); x0, y0, x1, y1 = r['box']; u, vv = (x0 + x1) / 2, (y0 + y1) / 2
    ray = c2w[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], 1.0]); ray /= np.linalg.norm(ray)
    ts = np.arange(0.3, 10, 0.02); q = c2w[:3, 3] + ts[:, None] * ray; dd, _ = tree.query(q, distance_upper_bound=0.04)
    hit = np.flatnonzero(np.isfinite(dd)); r['xyz'] = (q[hit[0]].round(3).tolist() if len(hit) else None); r['dist'] = float(ts[hit[0]]) if len(hit) else None
reads = [r for r in reads if r['xyz'] is not None and r['dist'] < 7]

# merge readings of the same physical sign
X = np.array([r['xyz'] for r in reads]); par = list(range(len(X)))
def root(i):
    while par[i] != i: par[i] = par[par[i]]; i = par[i]
    return i
for i, j in cKDTree(X).query_pairs(0.6): par[root(i)] = root(j)
groups = {}
for i in range(len(X)): groups.setdefault(root(i), []).append(i)
signs = []
for g in groups.values():
    vote = {}
    for i in g: vote[reads[i]['num']] = vote.get(reads[i]['num'], 0) + reads[i]['conf']
    num = max(vote, key=vote.get); best = max((i for i in g if reads[i]['num'] == num), key=lambda i: reads[i]['conf'])
    z = np.median(X[g, 2]); fl = int(np.argmin([abs(z - f['z'] - 1.5) for f in FL]))
    signs.append(dict(num=num, votes={k: round(v, 2) for k, v in sorted(vote.items(), key=lambda kv: -kv[1])}, views=len(g),
                      conf=round(vote[num] / sum(vote.values()), 2), xyz=np.median(X[g], 0).round(3).tolist(), floor=fl,
                      frame=reads[best]['frame'], box=reads[best]['box']))
signs.sort(key=lambda s: (s['floor'], s['num']))
json.dump(signs, open(W('guide', 'signs.json'), 'w'), indent=1)
for s in signs: log('sign', s['num'], 'floor', s['floor'] + 1, 'views', s['views'], 'agree', s['conf'], s['votes'])

tiles = []
for s in signs:
    im = cv2.imread(W('images', 'pinhole', s['frame'])); x0, y0, x1, y1 = s['box']; cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cr = im[max(0, cy - 90):cy + 90, max(0, cx - 120):cx + 120]; cr = cv2.resize(cr, (240, 180))
    cv2.putText(cr, f"{s['num']} x{s['views']}", (4, 20), 0, 0.6, (0, 0, 255), 2); tiles.append(cr)
tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 8)
cv2.imwrite(W('renders', 'signs_contact.jpg'), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
