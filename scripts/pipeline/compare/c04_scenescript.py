"""Comparison stage 4: Meta SceneScript (facebookresearch/scenescript, CC BY-NC) on one storey of the Blunt Hall LiDAR.
SceneScript reads a semi-dense point cloud of at most a 32 m cube (640 bins = 5 cm) and writes a scene language:
make_wall (corners a, b + height), make_door / make_window (centre, width, height, parent walls).
The storey is longer than 32 m, so it is cut into square tiles (BH_SS_TILE, default 28 m; stride BH_SS_STRIDE, default
20 m); each tile's points (floor to ceiling, on the 5 cm grid the model uses, optionally a random BH_SS_MAXPTS subset)
go through the network; a wall / door is kept by the tile whose central stride x stride core holds its midpoint, so
overlapping tiles do not double it. Best on Blunt Hall: 8 m tiles, stride 5 m, 20k points (room-sized, like its training data).
Tiles where the decoder fails are recorded and skipped; BH_SS_TAG names the output (scenescript-<tag>_walls.npz).
Run in the SceneScript env from the scenescript checkout:
  cd ~/apps/cmp/scenescript && BH_WORK=... BH_FLOOR=2 BH_SS_CKPT=... ~/.conda/envs/scenescript/bin/python <repo>/scripts/pipeline/compare/c04_scenescript.py
Out: compare/f<k>/scenescript_walls.npz (V, F wall quads, doors, seconds), scenescript_tiles.json (raw entities per tile)"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from src.networks.scenescript_model import SceneScriptWrapper

WORK = os.environ['BH_WORK']; FI = int(os.environ.get('BH_FLOOR', 2)); D = os.path.join(WORK, 'compare', os.environ.get('BH_CMP_SET', f'f{FI}'))
CKPT = os.environ['BH_SS_CKPT']; NUC = float(os.environ.get('BH_SS_NUCLEUS', 0.05))
TILE, STRIDE = float(os.environ.get('BH_SS_TILE', 28)), float(os.environ.get('BH_SS_STRIDE', 20))   # tile / core size (m)
MAXP = int(os.environ.get('BH_SS_MAXPTS', 0))           # >0: random subset per tile, closer to Aria semi-dense density
TAG = os.environ.get('BH_SS_TAG', ''); NAME = 'scenescript' + (f'-{TAG}' if TAG else '')
rng = np.random.default_rng(0)
pts = np.load(os.path.join(D, 'points.npz'))['xyz'].astype(np.float32)
fz = float(np.load(os.path.join(D, 'walls_ref.npz'))['floor_z'])
pts = pts[(pts[:, 2] > fz - 0.1) & (pts[:, 2] < fz + 2.7)]                                    # floor to ceiling
pts = pts[np.unique(np.floor(pts / 0.05).astype(np.int64), axis=0, return_index=True)[1]]      # 5 cm, as the model bins
model = SceneScriptWrapper.load_from_checkpoint(CKPT).cuda()
lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
xs = np.arange(lo[0] - (TILE - STRIDE) / 2, max(lo[0] + 1, hi[0] - STRIDE + (TILE - STRIDE) / 2) + 1e-3, STRIDE)
ys = np.arange(lo[1] - (TILE - STRIDE) / 2, max(lo[1] + 1, hi[1] - STRIDE + (TILE - STRIDE) / 2) + 1e-3, STRIDE)
walls, doors, tiles = [], [], []; t0 = time.time()
for x0 in xs:
    for y0 in ys:
        k = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + TILE) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + TILE)
        if k.sum() < 3000: continue
        c0, c1 = np.array([x0, y0]) + (TILE - STRIDE) / 2, np.array([x0, y0]) + (TILE + STRIDE) / 2      # core
        core = lambda p: c0[0] <= p[0] < c1[0] and c0[1] <= p[1] < c1[1]
        t1 = time.time(); q = pts[k]
        if MAXP and len(q) > MAXP: q = q[rng.choice(len(q), MAXP, replace=False)]
        try:
            with torch.no_grad():
                seq = model.run_inference(torch.as_tensor(q), nucleus_sampling_thresh=NUC, verbose=False)
        except Exception as ex:                                          # the decoder emitted an invalid token sequence
            tiles.append(dict(x0=float(x0), y0=float(y0), n_points=int(len(q)), failed=repr(ex))); print(f'tile ({x0:.0f}, {y0:.0f}): failed {ex!r}', flush=True); continue
        val = lambda v: v if isinstance(v, str) else float(v.item() if hasattr(v, 'item') else v)    # class names stay strings
        ents = [dict(cmd=e.COMMAND_STRING, **{p: val(v) for p, v in e.params.items()}) for e in seq.entities]
        tiles.append(dict(x0=float(x0), y0=float(y0), n_points=int(k.sum()), seconds=round(time.time() - t1, 1), entities=ents))
        nw = nd = 0
        for e in ents:
            need = {'make_wall': ('a_x', 'a_y', 'b_x', 'b_y', 'a_z', 'height'), 'make_door': ('position_x', 'position_y', 'position_z', 'width', 'height')}.get(e['cmd'], ())
            if any(x not in e for x in need): continue                  # entity cut off at the token limit
            if e['cmd'] == 'make_wall':
                a, b = np.array([e['a_x'], e['a_y']]), np.array([e['b_x'], e['b_y']])
                if core((a + b) / 2): walls.append([*a, *b, e['a_z'], e['height']]); nw += 1
            elif e['cmd'] == 'make_door':
                p = np.array([e['position_x'], e['position_y']])
                if core(p): doors.append([e['position_x'], e['position_y'], e['position_z'], e['width'], e['height']]); nd += 1
        print(f'tile ({x0:.0f}, {y0:.0f}): {len(q)} points, {len(ents)} entities, kept {nw} walls {nd} doors, {tiles[-1]["seconds"]} s', flush=True)
sec = time.time() - t0
# wall planes -> two triangles each (thin planes, as SceneScript models them); doors -> boxes for the plan overlay
V, F = [], []
for ax, ay, bx, by, z, h in walls:
    i = len(V); V += [[ax, ay, z], [bx, by, z], [bx, by, z + h], [ax, ay, z + h]]; F += [[i, i + 1, i + 2], [i, i + 2, i + 3]]
dbox = np.array([[x - w / 2, y - w / 2, z - h / 2, x + w / 2, y + w / 2, z + h / 2] for x, y, z, w, h in doors]).reshape(-1, 6)
np.savez(os.path.join(D, f'{NAME}_walls.npz'), V=np.array(V, float).reshape(-1, 3), F=np.array(F, int).reshape(-1, 3), doors=dbox, seconds=sec)
json.dump(dict(tile=TILE, stride=STRIDE, max_points=MAXP, nucleus=NUC, ckpt=os.path.basename(CKPT), tiles=tiles), open(os.path.join(D, f'{NAME}_tiles.json'), 'w'))
print(f'SceneScript {NAME}: {len(walls)} walls, {len(doors)} doors, {len(tiles)} tiles ({sum("failed" in t for t in tiles)} failed, {sum(not t.get("entities") for t in tiles if "failed" not in t)} empty), {sec:.0f} s', flush=True)
