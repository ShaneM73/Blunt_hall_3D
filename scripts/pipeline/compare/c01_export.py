"""Comparison stage 1: export the clean LiDAR of one storey for Cloud2BIM and SceneScript, in the guide's building
frame (walls along x / y, metres, z = 0 at the 1st floor) so every method is scored in the same coordinates.
Usage: BH_FLOOR=2 python c01_export.py
Out: compare/f<k>/cloud2bim.xyz   ASCII "x y z r g b" with a header line, 1.5 cm voxel grid (Cloud2BIM input)
     compare/f<k>/points.npz      xyz (1.5 cm) + nrm, the reference for scoring
     compare/f<k>/walls_ref.npz   points on vertical surfaces 0.3-2.0 m above the floor (wall reference)"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import W, log, removed_mask

FI = int(os.environ.get('BH_FLOOR', 2)); VOX = 0.015
G = json.load(open(W('guide', 'blocks.json')))['frame']; A, ORG, Z0 = G['angle'], np.array(G['origin']), G['z0']
Rm = np.array([[np.cos(-A), -np.sin(-A)], [np.sin(-A), np.cos(-A)]])
fl = json.load(open(W('mesh', 'structure.json')))['floors'][FI]
d = np.load(W('color', 'hq_colored.npz')); k = d['has'] & ~removed_mask()
z = d['xyz'][:, 2]; k &= (z > fl['zmin'] - 0.25) & (z < fl['zmax'])                    # storey incl. its floor slab surface
P = d['xyz'][k].astype(np.float64); N = d['nrm'][k]; C = d['rgb'][k]
_, i = np.unique(np.floor(P / VOX).astype(np.int64), axis=0, return_index=True); P, N, C = P[i], N[i], C[i]
B = np.c_[P[:, :2] @ Rm.T - ORG, P[:, 2] - Z0]; Nb = np.c_[N[:, :2] @ Rm.T, N[:, 2]]
out = W('compare', f'f{FI}'); os.makedirs(out, exist_ok=True)
rgb = (C * 255).round() if C.max() <= 1.0 else C
np.savetxt(os.path.join(out, 'cloud2bim.xyz'), np.c_[B, rgb], fmt='%.3f %.3f %.3f %d %d %d', header='//X Y Z R G B', comments='')
np.savez_compressed(os.path.join(out, 'points.npz'), xyz=B.astype(np.float32), nrm=Nb.astype(np.float32))
fz = fl['z'] - Z0; wsel = (abs(Nb[:, 2]) < 0.3) & (B[:, 2] > fz + 0.3) & (B[:, 2] < fz + 2.0)
np.savez_compressed(os.path.join(out, 'walls_ref.npz'), xyz=B[wsel].astype(np.float32), floor_z=fz)
log(f'floor {FI}: {len(B)} points ({VOX * 100:.1f} cm grid), {wsel.sum()} wall reference points -> {out}')
