"""Comparison stage 5: data for the comparison web page (compare.html).
- plan_<method>.png: the scored wall centre lines at 5 cm (red channel = on a measured wall, green = no measured wall
  within the tolerance), north up; plan_ref.png: LiDAR wall points (alpha)
- compare.json: scores of every run, the SceneScript tile-size sweep, doors per method (plan boxes) and the door check
  (8 known office doors on the Room 319 wall), the grid that maps image pixels to metres
Run after c03_score.py.  Out: data/compare/"""
import os, sys, json, glob
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log

FI = int(os.environ.get('BH_FLOOR', 2)); D = W('compare', os.environ.get('BH_CMP_SET', f'f{FI}'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'compare'); os.makedirs(OUT, exist_ok=True)
SHOW = ['ours', 'cloud2bim', 'scenescript', 'scenescript-t8-20k']
LABEL = {'ours': 'Ours · LiDAR block model', 'cloud2bim': 'Cloud2BIM', 'scenescript': 'SceneScript · 28 m tiles',
         'scenescript-t8-20k': 'SceneScript · 8 m tiles'}
KNOWN = [[-1.19, y] for y in (42.2, 44.95, 47.65, 50.4, 53.15, 55.9, 58.62, 61.38)]       # recessed doors, Room 319 wall

R = np.load(os.path.join(D, 'ref_plan.npz')); lo, res = R['lo'], float(R['res'])


def pool(a):                                       # 2.5 cm -> 5 cm, keep any hit
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))


ref = pool(R['ref']); H, Wd = ref.shape
cv2.imwrite(os.path.join(OUT, 'plan_ref.png'), np.dstack([np.zeros_like(ref, np.uint8)] * 3 + [ref.astype(np.uint8) * 255])[::-1])
k3 = np.ones((2, 2), np.uint8)
for m in SHOW:
    P = np.load(os.path.join(D, f'{m}_plan.npz')); g, b = pool(P['good']), pool(P['bad'])
    g = cv2.dilate(g.astype(np.uint8), k3) > 0; b = (cv2.dilate(b.astype(np.uint8), k3) > 0) & ~g
    img = np.zeros((H, Wd, 3), np.uint8); img[..., 2] = g * 255; img[..., 1] = b * 255       # BGR: red = good, green = bad
    cv2.imwrite(os.path.join(OUT, f'plan_{m}.png'), img[::-1])

scores = json.load(open(os.path.join(D, 'scores.json')))
doors, check = {}, {}
for m in scores:
    z = np.load(os.path.join(D, f'{m}_walls.npz')); d = z['doors'] if 'doors' in z.files else np.zeros((0, 6))
    c = (d[:, :2] + d[:, 3:5]) / 2 if len(d) else np.zeros((0, 2))
    check[m] = int(sum(len(c) and np.linalg.norm(c - np.array(k), axis=1).min() < 1.0 for k in KNOWN))
    if m in SHOW: doors[m] = [[round(float(v), 2) for v in (r[0], r[1], r[3], r[4])] for r in d]
sweep = []
for f in sorted(glob.glob(os.path.join(D, 'scenescript*_tiles.json'))):
    t = json.load(open(f)); name = os.path.basename(f)[:-11]
    if name not in scores: continue
    sweep.append(dict(run=name, tile_m=t['tile'], stride_m=t['stride'], max_points=t.get('max_points', 0) or None,
                      tiles=len(t['tiles']), failed=sum('failed' in x for x in t['tiles']),
                      empty=sum(not x.get('entities') for x in t['tiles'] if 'failed' not in x), **scores[name]))
sweep.sort(key=lambda r: (r['tile_m'], r['max_points'] or 1e9, r['run']))
data = dict(floor='3rd floor', grid=dict(lo=[round(float(v), 3) for v in lo], res=res * 2, w=int(Wd), h=int(H)),
            show=SHOW, label=LABEL, scores=scores, doors=doors, door_check=check, known_doors=KNOWN, sweep=sweep,
            c2b=json.load(open(os.path.join(D, 'cloud2bim_summary.json'))))
json.dump(data, open(os.path.join(OUT, 'compare.json'), 'w'), separators=(',', ':'))
log('compare page data ->', OUT, sorted(os.listdir(OUT)))
