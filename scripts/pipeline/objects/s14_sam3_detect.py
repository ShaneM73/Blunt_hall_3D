"""Stage 14 (objects): SAM 3 text-prompted segmentation of furniture / objects in every undistorted photo.
Run with the SAM 3 environment:  ~/.conda/envs/sam3/bin/python s14_sam3_detect.py   (needs BH_WORK)
Per photo: one image encoding, one grounding pass per prompt (confidence >= 0.4). Overlapping masks keep the highest score.
Out: objects/sam3/<image>.png  (uint16 index map at half resolution, 0 = none, k = detection k)
     objects/sam3/<image>.json ([{label, score, box}] for k = 1..n)   - resumable"""
import os, sys, json, time
import numpy as np, cv2, torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

WORK = os.environ['BH_WORK']; OUT = os.path.join(WORK, 'objects', 'sam3'); os.makedirs(OUT, exist_ok=True)
PROMPTS = os.environ.get('BH_PROMPTS', 'chair|table|desk|sofa|cabinet|shelf|trash can|vending machine|potted plant|'
                          'computer monitor|whiteboard|bench|filing cabinet|bookcase|stool|counter').split('|')
CONF = float(os.environ.get('BH_SAM3_CONF', 0.4)); STEP = int(os.environ.get('BH_SAM3_STEP', 1))
SHARD, NSHARD = int(os.environ.get('BH_SHARD', 0)), int(os.environ.get('BH_NSHARD', 1))
model = build_sam3_image_model(device='cuda'); proc = Sam3Processor(model, confidence_threshold=CONF, device='cuda')
frames = json.load(open(os.path.join(WORK, 'images', 'cameras.json')))['frames'][::STEP][SHARD::NSHARD]
t0 = time.time(); done = 0
for n, f in enumerate(frames):
    base = os.path.join(OUT, f['file'][:-4])
    if os.path.exists(base + '.json'): continue
    im = Image.open(os.path.join(WORK, 'images', 'pinhole', f['file'])).convert('RGB'); H, Wd = im.size[1] // 2, im.size[0] // 2
    mk = cv2.imread(os.path.join(WORK, 'images', 'masks', f['file'].replace('.jpg', '.png')), 0)
    valid = cv2.resize(mk, (Wd, H), interpolation=cv2.INTER_NEAREST) > 0 if mk is not None else np.ones((H, Wd), bool)
    st = proc.set_image(im); idx = np.zeros((H, Wd), np.uint16); best = np.zeros((H, Wd), np.float32); dets = []
    for p in PROMPTS:
        o = proc.set_text_prompt(prompt=p, state=st)
        for m, s, b in zip(o['masks'], o['scores'], o['boxes']):
            m = cv2.resize(m.squeeze().float().cpu().numpy(), (Wd, H), interpolation=cv2.INTER_LINEAR) > 0.5
            m &= valid                                         # people / outside-fisheye pixels never count
            if m.sum() < 150: continue
            s = float(s); dets.append(dict(label=p, score=round(s, 3), box=[round(float(x), 1) for x in b.tolist()]))
            take = m & (s > best); idx[take] = len(dets); best[take] = s
    cv2.imwrite(base + '.png', idx); json.dump(dets, open(base + '.json', 'w'))
    done += 1
    if done % 200 == 0: print(f'{n + 1}/{len(frames)} images, {(time.time() - t0) / done:.2f} s/image', flush=True)
print('done', len(frames), flush=True)
