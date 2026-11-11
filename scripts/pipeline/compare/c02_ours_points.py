"""Comparison stage 2c: our block method straight from a point set (no s12 face fitting), so it can run on the video.
Same rules as guide/g03_blocks.py, with wall faces replaced by wall points:
- wall points: normals within 17 deg of horizontal, 0.3-2.0 m above the floor
- a 5 cm plan cell is a wall face when its wall points span >= 1.0 m of that height band (full-height surfaces,
  not furniture tops or door headers)
- each face cell is drawn 15 cm thick behind the surface (against the normal), gaps < 25 cm closed along x and y,
  pieces < 0.5 m dropped, the rest split into axis-aligned boxes of storey height
Run on the LiDAR set (compare/f2, as a control for the fitted-face model 'ours') and on the video set (compare/f2-video).
  BH_WORK=... BH_CMP_SET=f2-video python c02_ours_points.py
Out: compare/<set>/ours-pts_walls.npz (V, F, doors (none), seconds)"""
import os, sys, json, time
import numpy as np, cv2, trimesh

WORK = os.environ['BH_WORK']; FI = int(os.environ.get('BH_FLOOR', 2)); SET = os.environ.get('BH_CMP_SET', f'f{FI}')
D = os.path.join(WORK, 'compare', SET); RES, TH, SPAN = 0.05, 0.15, 1.0; t0 = time.time()
fz = float(np.load(os.path.join(D, 'walls_ref.npz'))['floor_z'])
S = json.load(open(os.path.join(WORK, 'mesh', 'structure.json'))); fl = S['floors'][FI]; H = fl['ceiling'] - fl['z']
z = np.load(os.path.join(D, 'points.npz')); P, N = z['xyz'].astype(np.float64), z['nrm'].astype(np.float64)
k = (np.abs(N[:, 2]) < 0.3) & (P[:, 2] > fz + 0.3) & (P[:, 2] < fz + 2.0); P, N = P[k], N[k]
lo = P[:, :2].min(0) - 1; NX, NY = (np.ceil((P[:, :2].max(0) + 1 - lo) / RES)).astype(int) + 1
c = np.floor((P[:, :2] - lo) / RES).astype(int); cid = c[:, 1] * NX + c[:, 0]
zmin = np.full(NX * NY, np.inf); zmax = np.full(NX * NY, -np.inf)
np.minimum.at(zmin, cid, P[:, 2]); np.maximum.at(zmax, cid, P[:, 2])
face = (zmax - zmin) >= SPAN                                            # full-height wall surface in this cell
sel = face[cid]; Pf, Nf = P[sel], N[sel]
nh = Nf[:, :2] / np.maximum(np.linalg.norm(Nf[:, :2], axis=1, keepdims=True), 1e-9)
wall = np.zeros((NY, NX), np.uint8)
for d in np.arange(0, TH + 1e-6, RES / 2):                              # 15 cm behind the surface
    q = np.floor((Pf[:, :2] - nh * d - lo) / RES).astype(int); ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < NX) & (q[:, 1] < NY)
    wall[q[ok, 1], q[ok, 0]] = 1
g = int(0.25 / RES)
wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((1, g), np.uint8)); wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((g, 1), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(wall, connectivity=8)
keep = np.r_[False, np.maximum(st[1:, 2], st[1:, 3]) * RES >= 0.5]; wall = keep[lab].astype(np.uint8)


def boxes(mask):                                                         # as g03: row runs merged down identical columns
    out, open_ = [], {}
    for r in range(mask.shape[0] + 1):
        runs = set()
        if r < mask.shape[0]:
            row = np.r_[0, mask[r].astype(np.int8), 0]; d = np.flatnonzero(np.diff(row)); runs = set(zip(d[::2], d[1::2]))
        for kk in list(open_):
            if kk not in runs: out.append((kk[0], open_.pop(kk), kk[1], r))
        for kk in runs:
            if kk not in open_: open_[kk] = r
    return out


B = []
for c0, r0, c1, r1 in boxes(wall):
    b = trimesh.creation.box(extents=[(c1 - c0) * RES, (r1 - r0) * RES, H])
    b.apply_translation([lo[0] + (c0 + c1) / 2 * RES, lo[1] + (r0 + r1) / 2 * RES, fz + H / 2]); B.append(b)
m = trimesh.util.concatenate(B)
np.savez(os.path.join(D, 'ours-pts_walls.npz'), V=m.vertices, F=m.faces, doors=np.zeros((0, 6)), seconds=round(time.time() - t0, 1))
print(f'{SET}: {len(P)} wall points, {int(face.sum())} face cells -> {len(B)} wall boxes, {time.time() - t0:.1f} s')
