"""Guide stage 1: read text (room-number signs) in every scan photo with EasyOCR.
Usage: BH_SHARD=k BH_NSHARD=n python g01_ocr_signs.py      (resumable; one json per shard)
Out: guide/ocr/shard<k>.json  {frame: [[box 4x2 px, text, conf], ...]}  - only frames with text"""
import os, sys, json, time
import numpy as np, cv2, easyocr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log

k, n = int(os.environ.get('BH_SHARD', 0)), int(os.environ.get('BH_NSHARD', 1))
cams = json.load(open(W('images', 'cameras.json'))); frames = [f['file'] for f in cams['frames']][k::n]
os.makedirs(W('guide', 'ocr'), exist_ok=True); fp = W('guide', 'ocr', f'shard{k}.json')
res = json.load(open(fp)) if os.path.exists(fp) else {}; done = set(json.load(open(fp + '.done'))) if os.path.exists(fp + '.done') else set()
rd = easyocr.Reader(['en'], gpu=True, verbose=False); t0 = time.time()
for i, f in enumerate(frames):
    if f in done: continue
    im = cv2.imread(W('images', 'pinhole', f))
    out = rd.readtext(im, mag_ratio=1.5, text_threshold=0.6, low_text=0.3)
    out = [[np.asarray(b).round(1).tolist(), t, round(float(c), 3)] for b, t, c in out if c > 0.2 and any(ch.isalnum() for ch in t)]
    if out: res[f] = out
    done.add(f)
    if i % 100 == 0 or i == len(frames) - 1:
        json.dump(res, open(fp, 'w')); json.dump(sorted(done), open(fp + '.done', 'w'))
        log(f'shard {k}: {i + 1}/{len(frames)}  {time.time() - t0:.0f}s  frames with text {len(res)}')
