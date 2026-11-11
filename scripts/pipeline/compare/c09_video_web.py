"""Comparison stage 9: data for the "From a video" section of compare.html (LingBot-Map run, c07-c08, video set c07c).
- plan_v-<method>.png: the video set's scored wall centre lines, same grid and colour code as c05's plan_<method>.png
- video_mesh.bin: the aligned video mesh, decimated: header uint32 [nv, nf], int16 xyz in cm from 'origin', uint8 rgb,
  uint8 error (cm to the nearest LiDAR point of the storey, capped at 255), uint32 triangle indices
- video.json: run facts, alignment, error along the walk, walked-area scores (10 and 25 cm tolerance) for the LiDAR and
  the video models, camera path (plan, 0.5 m steps), labels
  BH_WORK=... LB_DIR=... LB_TAG=fine python c09_video_web.py        Out: data/compare/"""
import os, sys, json
import numpy as np, cv2, open3d as o3d
from scipy.spatial import cKDTree

WORK = os.environ['BH_WORK']; FI = int(os.environ.get('BH_FLOOR', 2)); LB = os.environ['LB_DIR']; TAG = os.environ.get('LB_TAG', 'fine')
DL = os.path.join(WORK, 'compare', f'f{FI}'); DV = os.path.join(WORK, 'compare', f'f{FI}-video')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'compare'); os.makedirs(OUT, exist_ok=True)
VSHOW = ['ours-pts', 'cloud2bim', 'scenescript-t8-20k']
LABEL = {'ours': 'Ours · LiDAR block model', 'ours-pts': 'Block method', 'cloud2bim': 'Cloud2BIM', 'scenescript-t8-20k': 'SceneScript · 8 m tiles'}
NTRI = int(os.environ.get('LB_WEB_TRIS', 220000))

# plan images (same 2.5 cm grid as the LiDAR set: both use the LiDAR wall reference) -> 5 cm, as c05
R = np.load(os.path.join(DL, 'ref_plan.npz')); RV = np.load(os.path.join(DV, 'ref_plan.npz'))
assert np.allclose(R['lo'], RV['lo']) and R['ref'].shape == RV['ref'].shape


def pool(a):
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))


H, Wd = pool(R['ref']).shape; k3 = np.ones((2, 2), np.uint8)
for m in VSHOW:
    P = np.load(os.path.join(DV, f'{m}_plan.npz')); g, b = pool(P['good']), pool(P['bad'])
    g = cv2.dilate(g.astype(np.uint8), k3) > 0; b = (cv2.dilate(b.astype(np.uint8), k3) > 0) & ~g
    img = np.zeros((H, Wd, 3), np.uint8); img[..., 2] = g * 255; img[..., 1] = b * 255
    cv2.imwrite(os.path.join(OUT, f'plan_v-{m}.png'), img[::-1])

# decimated aligned mesh with per-vertex error against the whole storey's LiDAR
mesh = o3d.io.read_triangle_mesh(os.path.join(LB, f'{TAG}_aligned_mesh.ply'))
mesh = mesh.simplify_quadric_decimation(NTRI); mesh.remove_unreferenced_vertices()
V = np.asarray(mesh.vertices); C = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8); F = np.asarray(mesh.triangles).astype(np.uint32)
L = np.load(os.path.join(DL, 'points.npz'))['xyz']; L = L[np.unique(np.floor(L / 0.03).astype(np.int64), axis=0, return_index=True)[1]]
err = cKDTree(L).query(V)[0]; E8 = np.minimum(np.round(err * 100), 255).astype(np.uint8)
meta3d = json.load(open(os.path.join(OUT, 'compare3d.json'))); org = np.array(meta3d['origin'])
q = np.round((V - org) * 100); assert np.abs(q).max() < 32767
with open(os.path.join(OUT, 'video_mesh.bin'), 'wb') as f:
    f.write(np.array([len(V), len(F)], np.uint32).tobytes()); f.write(q.astype(np.int16).tobytes()); f.write(C.tobytes())
    f.write(E8.tobytes()); f.write(b'\0' * ((-(len(V) * 10)) % 4)); f.write(F.tobytes())

# run facts
A = json.load(open(os.path.join(LB, f'{TAG}_align.json'))); S = json.load(open(os.path.join(LB, f'{TAG}_stats.json')))
cam = np.load(os.path.join(LB, f'{TAG}_cams_building.npy'))[:, :2]
step = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cam, axis=0), axis=1))]; keep = [0]
for i in range(1, len(cam)):
    if step[i] - step[keep[-1]] >= 0.5: keep.append(i)
path = np.round(cam[keep], 2).tolist()
def walked(set_dir, tag):
    d = json.load(open(os.path.join(set_dir, f'scores_{tag}.json')))['scores']
    return {k: {kk: d[k][kk] for kk in ('completeness', 'precision', 'f_score', 'wall_length_m')} for k in d}
scores = dict(lidar=dict(t10=walked(DL, 'walked'), t25=walked(DL, 'walked-tol25')),
              video=dict(t10=walked(DV, 'walked'), t25=walked(DV, 'walked-tol25')),
              video_full=json.load(open(os.path.join(DV, 'scores.json'))))
c2b = json.load(open(os.path.join(DV, 'cloud2bim_summary.json'))); ss = json.load(open(os.path.join(DV, 'scenescript-t8-20k_tiles.json')))
data = dict(video=dict(file='DJI_20261110204154_0036_D.MP4', seconds=215.3, resolution='1920 × 1080', fps=23.976, frames_used=2582,
                       inference_s=282.2, gpu_peak_gb=25.1, model='lingbot-map-long', mode='windowed, 128-keyframe windows, keyframe every 2nd frame'),
            mesh=dict(fuse_s=S['seconds'], triangles=S['mesh_triangles'], web_triangles=int(len(F)), voxel_cm=round(S['voxel'] * A['scale_m_per_unit'] * 100, 1)),
            align={k: A[k] for k in ('lidar_storey_height', 'scale_guess', 'scale_m_per_unit', 'camera_height_m', 'walk_length_m',
                                     'median_error_cm', 'within_10cm', 'within_25cm', 'seconds')} | dict(rot_deg=A['plan_search']['rot_deg']),
            along=[[round(a, 1), round(b * 100, 1), round(c, 3)] for a, b, c in A['along_walk']],
            vertex_error_cm=dict(median=float(np.median(err) * 100), p90=float(np.percentile(err, 90) * 100)),
            scores=scores, show=VSHOW, label=LABEL, path=path, reach_m=3.0,
            runs=dict(cloud2bim=dict(walls=c2b['counts']['IfcWall'], seconds=scores['video_full']['cloud2bim']['seconds']),
                      scenescript=dict(tiles=len(ss['tiles']), failed=sum('failed' in t for t in ss['tiles']), seconds=scores['video_full']['scenescript-t8-20k']['seconds'])))
json.dump(data, open(os.path.join(OUT, 'video.json'), 'w'), separators=(',', ':'), default=float)
print('video page data:', {f: os.path.getsize(os.path.join(OUT, f)) // 1024 for f in os.listdir(OUT) if 'v' in f and (f.startswith('plan_v') or f.startswith('video'))},
      'vertex error median/p90 cm', round(data['vertex_error_cm']['median'], 1), round(data['vertex_error_cm']['p90'], 1))
