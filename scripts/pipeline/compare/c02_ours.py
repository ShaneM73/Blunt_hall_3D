"""Comparison stage 2a: our block model (g03 / guide.json walls of one storey) as a wall mesh for scoring.
Out: compare/f<k>/ours_walls.npz (V, F, doors = door openings found by s12/g04 not stored -> lintels used as doors)"""
import os, sys, json
import numpy as np, trimesh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log

FI = int(os.environ.get('BH_FLOOR', 2))
g = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', '2026-11-04', 'guide.json')))
f = g['floors'][FI]; z = f['z']; boxes = []
for w in f['walls']:
    b = trimesh.creation.box(extents=[w[2] - w[0], w[3] - w[1], f['h']]); b.apply_translation([(w[0] + w[2]) / 2, (w[1] + w[3]) / 2, z + f['h'] / 2]); boxes.append(b)
m = trimesh.util.concatenate(boxes)
doors = np.array([[l[0], l[1], z, l[2], l[3], z + 2.1] for l in f['lintels']]).reshape(-1, 6)
np.savez(W('compare', f'f{FI}', 'ours_walls.npz'), V=m.vertices, F=m.faces, doors=doors, seconds=np.nan)
log('ours:', len(f['walls']), 'wall boxes,', len(doors), 'door headers')
