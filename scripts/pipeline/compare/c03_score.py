"""Comparison stage 3: score every method's wall model against the measured LiDAR of the same storey, on the floor plan.
Input per method: compare/f<k>/<method>_walls.npz with V (n,3) / F (m,3) triangles of the walls in the building frame
(+ optional doors (d,6) boxes, seconds). Methods: ours (g03 blocks), cloud2bim (IFC walls), scenescript (wall planes).
The walls are compared in plan (2.5 cm grid) so thick walls (blocks, IFC solids) and thin wall planes are treated alike:
- footprint: horizontal faces filled + vertical faces drawn as lines; centre line = skeleton of the footprint
- reference: LiDAR points on vertical surfaces 0.3-2.0 m above the floor
- completeness: share of reference points within 10 cm of the modelled footprint
- precision: share of modelled centre line within 10 cm + half the local wall thickness of a reference point
- F-score; wall length = centre-line length; doors; run time
Out: compare/f<k>/scores.json + renders/compare_f<k>_plan.png"""
import os, sys, json, glob
import numpy as np, trimesh
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log

FI = int(os.environ.get('BH_FLOOR', 2)); TOL = 0.10; D = W('compare', f'f{FI}')
ref = np.load(os.path.join(D, 'walls_ref.npz')); fz = float(ref['floor_z'])
R = ref['xyz']; R = R[np.unique(np.floor(R / 0.05).astype(np.int64), axis=0, return_index=True)[1]]       # 5 cm grid
allp = np.load(os.path.join(D, 'points.npz'))['xyz']; allp = allp[(allp[:, 2] > fz + 0.3) & (allp[:, 2] < fz + 2.0)]
allp = allp[np.unique(np.floor(allp / 0.03).astype(np.int64), axis=0, return_index=True)[1]]; tall = cKDTree(allp)
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
LABEL = {'ours': 'Ours (LiDAR block model)', 'cloud2bim': 'Cloud2BIM', 'scenescript': 'SceneScript, 28 m tiles', 'scenescript-t8-20k': 'SceneScript, 8 m tiles, 20k pts'}
RES = 0.025; Rw = R[(R[:, 2] > fz + 0.3) & (R[:, 2] < fz + 2.0)]
lo = Rw[:, :2].min(0) - 3; hi = Rw[:, :2].max(0) + 3; NX, NY = ((hi - lo) / RES).astype(int) + 1
def px(p): return np.round((np.asarray(p)[..., :2] - lo) / RES).astype(np.int32)
refimg = np.zeros((NY, NX), bool); q = px(Rw); refimg[q[:, 1].clip(0, NY - 1), q[:, 0].clip(0, NX - 1)] = True
dist_to_ref = ndimage.distance_transform_edt(~refimg) * RES
files = sorted(glob.glob(os.path.join(D, '*_walls.npz')))
PLOT = [x for x in os.environ.get('BH_CMP_PLOT', '').split(',') if x]          # methods to draw (default: all), in this order
pfiles = [os.path.join(D, f'{x}_walls.npz') for x in PLOT] if PLOT else files
scores = {}; fig, axs = plt.subplots(1, len(pfiles), figsize=(5.2 * len(pfiles), 15), squeeze=False)
axmap = dict(zip(pfiles, axs[0]))
for f in files:
    ax = axmap.get(f)
    name = os.path.basename(f)[:-10]; m = np.load(f); V, F = m['V'], m['F']
    mask = np.zeros((NY, NX), np.uint8)
    for t in F:
        T = V[t]; nrm = np.cross(T[1] - T[0], T[2] - T[0]); L = np.linalg.norm(nrm)
        if L < 1e-9: continue
        if abs(nrm[2]) / L > 0.7: cv2.fillPoly(mask, [px(T)], 1)                      # top / bottom face: footprint
        else:
            P2 = px(T); cv2.polylines(mask, [P2], True, 1, 1)                          # vertical face: its trace
    mask = mask.astype(bool)
    dist_to_model = ndimage.distance_transform_edt(~mask) * RES
    comp = float((dist_to_model[q[:, 1].clip(0, NY - 1), q[:, 0].clip(0, NX - 1)] <= TOL).mean())
    sk = skeletonize(mask); half = ndimage.distance_transform_edt(mask) * RES
    ok = dist_to_ref[sk] <= TOL + half[sk]; prec = float(ok.mean()) if sk.any() else 0.0
    fs = 2 * comp * prec / max(1e-9, comp + prec)
    scores[name] = dict(completeness=round(comp, 3), precision=round(prec, 3), f_score=round(fs, 3),
                        wall_length_m=round(float(sk.sum()) * RES, 1), median_thickness_cm=round(float(np.median(half[sk])) * 200, 1) if sk.any() else None,
                        doors=int(len(m['doors'])) if 'doors' in m.files else None,
                        seconds=None if 'seconds' not in m.files or not np.isfinite(m['seconds']) else round(float(m['seconds']), 1))
    log(name, scores[name])
    if ax is None: continue
    ax.imshow(refimg, origin='lower', cmap='Greys', alpha=0.35, extent=[lo[0], hi[0], lo[1], hi[1]])
    ys, xs = np.nonzero(sk); ax.scatter(lo[0] + xs * RES, lo[1] + ys * RES, s=0.2, c=np.where(ok, '#1f6f43', '#c0392b'))
    if 'doors' in m.files:
        for d in m['doors']: ax.add_patch(plt.Rectangle((d[0], d[1]), d[3] - d[0], d[4] - d[1], fill=False, ec='#1f4fbf', lw=1.2))
    ax.set_aspect('equal'); ax.set_title(f"{LABEL.get(name, name)}\ncompleteness {comp:.0%} · precision {prec:.0%} · F {fs:.2f}", fontsize=11)
json.dump(scores, open(os.path.join(D, 'scores.json'), 'w'), indent=1)
fig.savefig(W('renders', f'compare_f{FI}_plan.png'), dpi=70, bbox_inches='tight')
