"""Guide stage 1b: SAM 3 boxes of room-number signs in every photo (finds the small signs the OCR detector misses).
Run with the SAM 3 environment:  BH_SHARD=k BH_NSHARD=n ~/.conda/envs/sam3/bin/python g01b_sam3_signs.py   (needs BH_WORK)
Out: guide/sam3_signs/shard<k>.json  {frame: [[x0, y0, x1, y1, score], ...]}  full-resolution pixel boxes   - resumable"""
import os, json, time
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

WORK = os.environ['BH_WORK']; OUT = os.path.join(WORK, 'guide', 'sam3_signs'); os.makedirs(OUT, exist_ok=True)
PROMPTS = os.environ.get('BH_PROMPTS', 'room number sign|door sign').split('|')
K, N = int(os.environ.get('BH_SHARD', 0)), int(os.environ.get('BH_NSHARD', 1))
fp = os.path.join(OUT, f'shard{K}.json'); res = json.load(open(fp)) if os.path.exists(fp) else {}
model = build_sam3_image_model(device='cuda'); proc = Sam3Processor(model, confidence_threshold=0.3, device='cuda')
frames = json.load(open(os.path.join(WORK, 'images', 'cameras.json')))['frames'][K::N]; t0 = time.time()
for n, f in enumerate(frames):
    if f['file'] in res: continue
    st = proc.set_image(Image.open(os.path.join(WORK, 'images', 'pinhole', f['file'])).convert('RGB')); boxes = []
    for p in PROMPTS:
        o = proc.set_text_prompt(prompt=p, state=st)
        boxes += [[round(float(x), 1) for x in b.tolist()] + [round(float(s), 3)] for b, s in zip(o['boxes'], o['scores'])]
    res[f['file']] = boxes
    if n % 200 == 0 or n == len(frames) - 1:
        json.dump(res, open(fp, 'w')); print(f'{n + 1}/{len(frames)}  {time.time() - t0:.0f}s  boxes {sum(map(len, res.values()))}', flush=True)
