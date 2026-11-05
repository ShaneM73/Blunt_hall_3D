"""Stage 19 (objects): SAM 3D Objects reconstruction of each picked view (s18).
Run in the SAM 3D env from the sam-3d-objects checkout:
  cd ~/apps/sam-3d-objects && BH_WORK=... ~/.conda/envs/sam3d-objects/bin/python <repo>/scripts/pipeline/objects/s19_sam3d.py [id ...]
Out per object: objects/sam3d_out/<id>/{mesh.glb, splat.ply, pose.json (rotation quat wxyz, translation, scale in the
SAM 3D camera frame)}   - resumable"""
import os, sys, json, time
import numpy as np, torch, trimesh
from scipy.spatial import cKDTree
from PIL import Image
sys.path.append('notebook')
from inference import Inference

WORK = os.environ['BH_WORK']; IN = os.path.join(WORK, 'objects', 'sam3d_in'); OUT = os.path.join(WORK, 'objects', 'sam3d_out')
ids = sys.argv[1:] or sorted(os.listdir(IN))
inf = Inference('checkpoints/hf/pipeline.yaml', compile=False); pipe = inf._pipeline
for d in ids:
    d = f'{int(d):03d}'; od = os.path.join(OUT, d)
    if os.path.exists(os.path.join(od, 'pose.json')): continue
    os.makedirs(od, exist_ok=True); t0 = time.time()
    img = np.array(Image.open(os.path.join(IN, d, 'image.png')).convert('RGB')); m = np.array(Image.open(os.path.join(IN, d, 'mask.png'))) > 0
    out = pipe.run(inf.merge_mask_to_rgba(img, m), None, 42, stage1_only=False, with_mesh_postprocess=True, with_texture_baking=False,
                   with_layout_postprocess=False, use_vertex_color=True, pointmap=None)
    g = out['glb']
    if g is not None:
        # vertex colours from SAM 3D's own Gaussians (same object frame; the glb was rotated z-up -> y-up by to_glb)
        gs = out['gs']; xyz = gs.get_xyz.detach().cpu().numpy(); op = gs.get_opacity.detach().cpu().numpy().ravel()
        rgb = (0.5 + 0.28209479 * gs._features_dc.detach().cpu().numpy().reshape(len(xyz), -1)[:, :3]).clip(0, 1)
        keep = op > 0.1; tree = cKDTree(xyz[keep])
        v = np.asarray(g.vertices) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
        _, nn = tree.query(v, k=8); w = op[keep][nn]; col = (rgb[keep][nn] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        g.visual = trimesh.visual.ColorVisuals(g, vertex_colors=np.c_[(col * 255).round().astype(np.uint8), np.full(len(v), 255, np.uint8)])
        g.export(os.path.join(od, 'mesh.glb'))
    if os.environ.get('BH_SAM3D_SPLAT') == '1' and out.get('gs') is not None: out['gs'].save_ply(os.path.join(od, 'splat.ply'))
    pose = {k: out[k].detach().float().cpu().numpy().reshape(-1).tolist() for k in ('rotation', 'translation', 'scale') if k in out and torch.is_tensor(out[k])}
    pose['keys'] = sorted(out.keys()); pose['seconds'] = round(time.time() - t0, 1)
    json.dump(pose, open(os.path.join(od, 'pose.json'), 'w'), indent=1)
    print('done', d, pose['seconds'], 's', pose.get('scale'), flush=True)
