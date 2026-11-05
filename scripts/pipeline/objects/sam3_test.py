import sys, json, time, numpy as np, cv2, torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
W = sys.argv[1]; out = sys.argv[2]
PROMPTS = ['chair', 'table', 'desk', 'sofa', 'cabinet', 'shelf', 'trash can', 'vending machine', 'potted plant', 'computer monitor', 'whiteboard', 'bench']
model = build_sam3_image_model(device='cuda'); proc = Sam3Processor(model, confidence_threshold=0.4, device='cuda')
fr = json.load(open(f'{W}/images/cameras.json'))['frames']
C = np.array([np.array(f['c2w'])[:3, 3] for f in fr])
pick = [i for i in range(len(fr)) if 15 < C[i, 0] < 30 and -15 < C[i, 1] < -5 and C[i, 2] > -0.5][::40][:4]
rows = []
for i in pick:
    im = Image.open(f"{W}/images/pinhole/{fr[i]['file']}").convert('RGB'); t = time.time()
    st = proc.set_image(im); vis = np.array(im)[:, :, ::-1].copy(); found = []
    for p in PROMPTS:
        o = proc.set_text_prompt(prompt=p, state=st)
        for m, s in zip(o['masks'], o['scores']):
            m = m.squeeze().cpu().numpy() > 0; col = np.random.randint(60, 255, 3)
            vis[m] = (vis[m] * 0.45 + col * 0.55).astype(np.uint8); ys, xs = np.where(m)
            if len(xs): cv2.putText(vis, f'{p} {float(s):.2f}', (int(xs.mean()) - 40, int(ys.mean())), 0, 1.1, (255, 255, 255), 3); found.append(p)
    print(fr[i]['file'], 'time %.2fs' % (time.time() - t), {p: found.count(p) for p in set(found)})
    rows.append(cv2.resize(vis, (486, 648)))
cv2.imwrite(out, np.hstack(rows))
