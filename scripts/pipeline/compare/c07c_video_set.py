"""Comparison stage 7c: the LingBot-Map video mesh (aligned by c08) as an input set for the wall methods, like c01 for LiDAR.
The aligned mesh is sampled on its surface (2 cm), cut to the storey and put on the 1.5 cm grid c01 uses, so Cloud2BIM,
SceneScript and the block method read it exactly as they read the LiDAR.
The LiDAR wall reference is copied so the video models are scored against the same measured walls.
'Walked area': plan cells within LB_REACH m (default 3) of the camera path on this storey; scores restricted to it compare
every method, LiDAR or video, on the part of the floor the video could see.
  BH_WORK=... LB_DIR=... LB_TAG=fine python c07c_video_set.py
Out: compare/f<k>-video/points.npz (xyz, nrm), cloud2bim.xyz, walls_ref.npz (LiDAR copy), region.npz (camera path, reach)"""
import os, sys, json, shutil
import numpy as np, open3d as o3d

WORK = os.environ['BH_WORK']; FI = int(os.environ.get('BH_FLOOR', 2)); LB = os.environ['LB_DIR']; TAG = os.environ.get('LB_TAG', 'fine')
REACH = float(os.environ.get('LB_REACH', 3.0)); VOX = 0.015
SRC = os.path.join(WORK, 'compare', f'f{FI}'); OUT = os.path.join(WORK, 'compare', f'f{FI}-video'); os.makedirs(OUT, exist_ok=True)
fz = float(np.load(os.path.join(SRC, 'walls_ref.npz'))['floor_z'])
S = json.load(open(os.path.join(WORK, 'mesh', 'structure.json'))); fl = S['floors'][FI]; Z0 = S['floors'][0]['z']
zlo, zhi = fl['zmin'] - Z0 - 0.25, fl['zmax'] - Z0
m = o3d.io.read_triangle_mesh(os.path.join(LB, f'{TAG}_aligned_mesh.ply'))
area = m.get_surface_area(); pc = m.sample_points_uniformly(int(area / 0.02 ** 2), use_triangle_normal=True)
P = np.asarray(pc.points); N = np.asarray(pc.normals); C = np.asarray(pc.colors)
k = (P[:, 2] > zlo) & (P[:, 2] < zhi); P, N, C = P[k], N[k], C[k]
_, i = np.unique(np.floor(P / VOX).astype(np.int64), axis=0, return_index=True); P, N, C = P[i], N[i], C[i]
np.savez_compressed(os.path.join(OUT, 'points.npz'), xyz=P.astype(np.float32), nrm=N.astype(np.float32))
np.savetxt(os.path.join(OUT, 'cloud2bim.xyz'), np.c_[P, (C * 255).round().clip(0, 255)], fmt='%.3f %.3f %.3f %d %d %d', header='//X Y Z R G B', comments='')
shutil.copy(os.path.join(SRC, 'walls_ref.npz'), os.path.join(OUT, 'walls_ref.npz'))
cam = np.load(os.path.join(LB, f'{TAG}_cams_building.npy')); on = (cam[:, 2] > fz + 0.5) & (cam[:, 2] < fz + 2.5)
np.savez(os.path.join(OUT, 'region.npz'), path=cam[on, :2].astype(np.float32), reach=REACH)
print(f'video set: {len(P)} points ({VOX * 100:.1f} cm grid, z {zlo:.2f}..{zhi:.2f}), {on.sum()} of {len(cam)} camera poses on the storey -> {OUT}')
