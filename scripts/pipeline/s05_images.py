"""Stage 5: undistort fisheye photos to pinhole 1458x1944 (f = 675 px) with refined poses, and make masks
(valid fisheye area minus dilated Mask R-CNN people, detected at 2/3 scale).
Out: images/pinhole/*.jpg  images/masks/*.png  images/cameras.json"""
import os, json
import numpy as np, cv2, torch
from concurrent.futures import ThreadPoolExecutor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from common import W, CAL, Rcb, Poser, traj, image_index, read_image, log

Wo, Ho, f = 1458, 1944, 675.0
Kn = np.array([[f, 0, Wo / 2], [0, f, Ho / 2], [0, 0, 1]])
pose = Poser(traj()); recs = image_index()
maps, valid = {}, {}
for cam in (1, 2, 3):
    K = np.array([[CAL[cam]['K'][0], 0, CAL[cam]['K'][2]], [0, CAL[cam]['K'][1], CAL[cam]['K'][3]], [0, 0, 1]]); D = np.array(CAL[cam]['D'])
    maps[cam] = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kn, (Wo, Ho), cv2.CV_16SC2)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kn, (Wo, Ho), cv2.CV_32FC1)
    valid[cam] = cv2.erode(((m1 >= 1) & (m1 < 1943) & (m2 >= 1) & (m2 < 2591)).astype(np.uint8), np.ones((13, 13), np.uint8))
os.makedirs(W('images', 'pinhole', 'x'), exist_ok=True); os.makedirs(W('images', 'masks', 'x'), exist_ok=True)
def work(r):
    cam, t = r['cam'], r['t']; Rwb, twb = pose(t)
    c2w = np.eye(4); c2w[:3, :3] = Rwb @ Rcb(cam).T; c2w[:3, 3] = twb + Rwb @ np.array(CAL[cam]['t'])
    name = f'c{cam}_{t:.6f}.jpg'; im = cv2.remap(read_image(r), *maps[cam], cv2.INTER_CUBIC)
    cv2.imwrite(W('images', 'pinhole', name), im, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return dict(file=name, cam=cam, t=t, c2w=c2w.tolist()), im
model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).cuda(int(os.environ.get('BH_GPU', '0'))).eval()
frames, npers = [], 0
with ThreadPoolExecutor(24) as ex, torch.no_grad():
    for s in range(0, len(recs), 16):
        batch = list(ex.map(work, recs[s:s + 16]))
        ims = [torch.tensor(cv2.cvtColor(cv2.resize(im, (972, 1296), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().cuda(int(os.environ.get('BH_GPU', '0'))) / 255 for _, im in batch]
        for (fr, _), o in zip(batch, model(ims)):
            keep = (o['labels'] == 1) & (o['scores'] > 0.5)
            pm = np.zeros((Ho, Wo), np.uint8)
            if keep.any():
                npers += 1
                pm = cv2.resize((o['masks'][keep, 0] > 0.3).any(0).cpu().numpy().astype(np.uint8), (Wo, Ho), interpolation=cv2.INTER_NEAREST)
                pm = cv2.dilate(pm, np.ones((37, 37), np.uint8))
            cv2.imwrite(W('images', 'masks', fr['file'].replace('.jpg', '.png')), valid[fr['cam']] * (1 - pm) * 255)
            frames.append(fr)
        if s % 800 == 0: log('images', s, 'of', len(recs))
json.dump(dict(W=Wo, H=Ho, K=Kn.tolist(), frames=frames), open(W('images', 'cameras.json'), 'w'))
log('images', len(frames), 'with people', npers)
