"""Stage 16 (objects): one coloured mesh per object instance, cut out of the clean measured mesh (s08 tiles).
A triangle belongs to an object when its centroid lies within CUT of that object's labelled LiDAR points.
Slivers (second-largest extent < 0.2 m, e.g. a door-frame edge called 'vending machine') are dropped.
Out: objects/meshes/obj_<id>_<label>.ply (Z-up building coordinates, vertex colours) and objects/meshes/index.json"""
import os, sys, json
import numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log

CUT = float(os.environ.get('BH_OBJ_CUT', 0.03))
J = json.load(open(W('objects', 'instances.json'))); I = np.load(W('objects', 'instances.npz'))
P = np.load(W('color', 'hq_colored.npz'))['xyz'][I['point']]; inst = I['inst']
OUT = W('objects', 'meshes'); os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT): os.remove(os.path.join(OUT, f))
keep = []
for o in J['objects']:
    s = sorted(o['size'], reverse=True)
    if s[1] < 0.2: log('drop sliver', o['id'], o['label'], o['size']); continue
    keep.append(o)
tree = {o['id']: cKDTree(P[inst == o['id']]) for o in keep}
lo = {o['id']: np.array(o['lo']) - CUT for o in keep}; hi = {o['id']: np.array(o['hi']) + CUT for o in keep}
parts = {o['id']: ([], [], []) for o in keep}
for t in json.load(open(W('mesh', 'tiles.json')))['tiles']:
    M = np.load(W('mesh', 'tiles', t['file'])); V, F, C = M['v'], M['f'], M['c']; cen = V[F].mean(1)
    for o in keep:
        k = o['id']; box = np.all((cen >= lo[k]) & (cen <= hi[k]), 1)
        if not box.any(): continue
        fi = np.flatnonzero(box); d, _ = tree[k].query(cen[fi], distance_upper_bound=CUT); fi = fi[np.isfinite(d)]
        if len(fi): parts[k][0].append(V); parts[k][1].append(F[fi]); parts[k][2].append(C)
index = []
for o in keep:
    Vs, Fs, Cs = parts[o['id']]
    if not Fs: log('no mesh', o['id'], o['label']); continue
    vv, ff, cc, base = [], [], [], 0
    for V, F, C in zip(Vs, Fs, Cs):
        uq, inv = np.unique(F, return_inverse=True); vv.append(V[uq]); cc.append(C[uq]); ff.append(inv.reshape(-1, 3) + base); base += len(uq)
    m = trimesh.Trimesh(np.vstack(vv), np.vstack(ff), vertex_colors=np.c_[np.vstack(cc), np.full(base, 255, np.uint8)], process=True)
    cs = m.split(only_watertight=False)                                         # drop stray triangle islands (< 5 % of the faces)
    if len(cs) > 1: m = trimesh.util.concatenate([c for c in cs if len(c.faces) >= 0.05 * len(m.faces)])
    if len(m.faces) < 50: log('tiny mesh', o['id'], o['label'], len(m.faces)); continue
    fn = f"obj_{o['id']:03d}_{o['label'].replace(' ', '_')}.ply"; m.export(os.path.join(OUT, fn))
    index.append(dict(o, file=fn, faces=int(len(m.faces))))
json.dump(dict(objects=index), open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
log('object meshes', len(index), 'of', len(J['objects']), 'faces', sum(x['faces'] for x in index))
