"""Stage 11: export the trained splat to 32-byte .splat files (pos f32x3, scale f32x3, rgba u8x4, quat u8x4 wxyz) in
data/<id>/, keeping opacity > 0.03 and scale < 1 m, most important first, at most MAXN in total, split by floor into
files of <= PART Gaussians (GitHub's 100 MB file limit).  Writes meta['splat'] with a start view per floor.
Usage: s11_splat_export.py <id> [MAXN]"""
import os, sys, json
import numpy as np, torch
from common import W, REPO, log

DID = sys.argv[1]; MAXN = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000; PART = 2_900_000
OUT = os.path.join(REPO, 'data', DID); meta = json.load(open(os.path.join(OUT, 'meta.json')))
p = torch.load(W('splat', 'splat_params.pt'))
xyz = p['means'].numpy(); sc = np.exp(p['scales'].numpy()); op = 1 / (1 + np.exp(-p['opacities'].numpy()))
q = p['quats'].numpy(); q /= np.linalg.norm(q, axis=1, keepdims=True)
rgb = np.clip(0.5 + 0.28209479 * p['sh0'].numpy()[:, 0, :], 0, 1)
imp = (sc.prod(1) ** (1 / 3)) * op
keep = (op > 0.03) & (sc.max(1) < 1.0) & (imp > 0)
idx = np.flatnonzero(keep); idx = idx[np.argsort(-imp[idx])][:MAXN]
fl = meta['floors']; files = []
for j, f in enumerate(fl):
    sel = idx[(xyz[idx, 2] >= f['zmin']) & (xyz[idx, 2] < f['zmax'])]
    for k, part in enumerate(np.array_split(sel, max(1, int(np.ceil(len(sel) / PART))))):
        buf = np.zeros(len(part), dtype=[('p', '<f4', 3), ('s', '<f4', 3), ('c', 'u1', 4), ('r', 'u1', 4)])
        buf['p'] = xyz[part]; buf['s'] = sc[part]; buf['c'][:, :3] = (rgb[part] * 255).round(); buf['c'][:, 3] = (op[part] * 255).round()
        buf['r'] = np.clip(q[part] * 128 + 128, 0, 255).round()
        fn = f'splat_f{j}_{k}.splat'; buf.tofile(os.path.join(OUT, fn)); files.append(dict(file=fn, floor=j, n=len(part)))
        log(f['name'], fn, len(part), round(os.path.getsize(os.path.join(OUT, fn)) / 1e6, 1), 'MB')
# start view per floor: from the camera position nearest the floor's path median, looking along the walking direction
cams = json.load(open(W('images', 'cameras.json')))['frames']; C = np.array([np.array(c['c2w'])[:3, 3] for c in cams])
Fw = np.array([np.array(c['c2w'])[:3, 2] for c in cams])
views = {}
for j, f in enumerate(fl):
    on = np.flatnonzero((C[:, 2] - f['z'] > 0.8) & (C[:, 2] - f['z'] < 1.8))
    if not len(on): continue
    med = np.median(C[on], 0); i = on[np.argmin(np.linalg.norm(C[on] - med, axis=1))]
    fw = Fw[i].copy(); fw[2] = 0; fw /= max(np.linalg.norm(fw), 1e-6)
    views[j] = dict(cam_pos=(C[i] - 0.3 * fw).round(2).tolist(), cam_look=(C[i] + 4 * fw + [0, 0, -0.3]).round(2).tolist())
meta['splat'] = dict(files=files, n=int(sum(f['n'] for f in files)), views=len(cams), views_by_floor={str(k): v for k, v in views.items()},
                     **views.get(meta['main_floor'], {}),
                     note=f'3D Gaussian Splatting (gsplat) trained on {len(cams):,} undistorted fisheye photos at 1458×1944 with SLAM-refined poses, initialized from the people-free 1 cm LiDAR cloud near the path plus a 2 cm cloud farther out. People were masked out of the photos and the splat depth was supervised by the carved LiDAR cloud. Each floor is a separate file; best viewed from inside the corridors near the walking path.')
json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
log('exported', meta['splat']['n'], 'of', len(xyz))
