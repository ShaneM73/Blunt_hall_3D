"""Stage 18 (objects): pick, for each object instance, the photo + SAM 3 mask that SAM 3D should reconstruct it from.
A view is scored by (projected size of the object's LiDAR points) x (fraction of them that land on one SAM 3 detection of the
same label); the object must be fully inside the image and 0.8-5 m from the camera.
Usage: s18_pick_views.py [id ...]   (default: every object in objects/meshes/index.json)
Out: objects/sam3d_in/<id>/{image.png, mask.png, view.json}   view.json has the frame name, c2w, K, detection score"""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log

idx = json.load(open(W('objects', 'meshes', 'index.json')))['objects']
want = set(int(a) for a in sys.argv[1:]); objs = [o for o in idx if not want or o['id'] in want]
I = np.load(W('objects', 'instances.npz')); P = np.load(W('color', 'hq_colored.npz'))['xyz'][I['point']]; inst = I['inst']
cams = json.load(open(W('images', 'cameras.json'))); K = np.array(cams['K']); Wd, Hd = cams['W'], cams['H']
frames = cams['frames']; C = np.array([np.array(f['c2w'])[:3, 3] for f in frames])
SAMD = W('objects', 'sam3'); S = 2
for o in objs:
    X = P[inst == o['id']]; X = X[np.random.default_rng(0).choice(len(X), min(len(X), 3000), replace=False)]
    cand = np.flatnonzero(np.linalg.norm(C - np.array(o['center']), axis=1) < 5)
    best = None
    for fi in cand:
        f = frames[fi]; w2c = np.linalg.inv(np.array(f['c2w'])); Pc = X @ w2c[:3, :3].T + w2c[:3, 3]; z = Pc[:, 2]
        if (z < 0.8).any() or np.median(z) > 5: continue
        u = K[0, 0] * Pc[:, 0] / z + K[0, 2]; v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
        if u.min() < 8 or v.min() < 8 or u.max() > Wd - 8 or v.max() > Hd - 8: continue
        b = os.path.join(SAMD, f['file'][:-4]); dets = json.load(open(b + '.json'))
        if not dets: continue
        im = cv2.imread(b + '.png', cv2.IMREAD_UNCHANGED); di = im[(v / S).astype(int).clip(0, im.shape[0] - 1), (u / S).astype(int).clip(0, im.shape[1] - 1)]
        ok = [k for k in np.unique(di) if k > 0 and dets[k - 1]['label'] == o['label']]
        if not ok: continue
        k = max(ok, key=lambda k: (di == k).sum()); hit = (di == k).mean()
        # the detection must not be much larger than the object (it would then be a merged group)
        area = (im == k).sum() * S * S; parea = (u.max() - u.min()) * (v.max() - v.min())
        if area > 2.5 * parea: continue
        sc = parea * hit ** 2
        if best is None or sc > best[0]: best = (sc, fi, k, hit, dets[k - 1]['score'], im == k)
    if best is None: log('no view', o['id'], o['label']); continue
    sc, fi, k, hit, dsc, m = best; f = frames[fi]
    od = W('objects', 'sam3d_in', f'{o["id"]:03d}'); os.makedirs(od, exist_ok=True)
    img = cv2.imread(W('images', 'pinhole', f['file'])); cv2.imwrite(os.path.join(od, 'image.png'), img)
    m = cv2.resize(m.astype(np.uint8) * 255, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST); cv2.imwrite(os.path.join(od, 'mask.png'), m)
    json.dump(dict(id=o['id'], label=o['label'], frame=f['file'], c2w=f['c2w'], K=K.tolist(), hit=round(float(hit), 2), det_score=dsc), open(os.path.join(od, 'view.json'), 'w'), indent=1)
    log('view', o['id'], o['label'], f['file'], 'hit', round(float(hit), 2), 'px', int(sc))
