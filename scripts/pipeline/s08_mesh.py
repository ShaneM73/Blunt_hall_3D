"""Stage 8: tiled screened-Poisson mesh of the people-free colored cloud.
Tiles of TILE m in X/Y (all heights) with 1 m overlap; Poisson depth chosen for ~9 mm cells; vertices farther than 3 cm
from real points removed, small islands dropped, each tile cut back to its own footprint, decimated to TRI_PER_M2
triangles per m^2 of covered area, vertex colors from the 6 nearest points.
Out: mesh/tiles/tile_<i>_<j>.npz (v, f, c) and mesh/tiles.json"""
import os, json
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from common import W, log

TILE = float(os.environ.get('BH_TILE', 16)); OV = 1.0; TRI_PER_M2 = float(os.environ.get('BH_TRI_M2', 6000))
d = np.load(W('color', 'hq_colored.npz')); rm = np.load(W('clouds', 'person_rm.npy')); k = ~rm & d['has']
P = d['xyz'][k]; Nr = d['nrm'][k]; C = d['rgb'][k]; log('mesh input points', len(P))
lo = np.floor(P[:, :2].min(0) / TILE) * TILE
ij = np.floor((P[:, :2] - lo) / TILE).astype(int)
tiles = sorted({tuple(x) for x in np.unique(ij, axis=0)})

def run(t):
    i, j = t; x0, y0 = lo + np.array([i, j]) * TILE
    m = (P[:, 0] > x0 - OV) & (P[:, 0] < x0 + TILE + OV) & (P[:, 1] > y0 - OV) & (P[:, 1] < y0 + TILE + OV)
    own = (P[:, 0] >= x0) & (P[:, 0] < x0 + TILE) & (P[:, 1] >= y0) & (P[:, 1] < y0 + TILE)
    if own.sum() < 20000: return None
    Q = P[m].astype(np.float64); Qn = Nr[m].astype(np.float64); Qc = C[m]
    ext = float(np.ptp(Q, 0).max()); depth = int(np.clip(np.ceil(np.log2(ext * 1.05 / 0.009)), 8, 12))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q)); pc.normals = o3d.utility.Vector3dVector(Qn)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, n_threads=8, scale=1.05)
    tree = cKDTree(Q); dist, _ = tree.query(np.asarray(mesh.vertices), workers=8)
    mesh.remove_vertices_by_mask(dist > 0.03); mesh.remove_unreferenced_vertices()
    cl, n, _ = mesh.cluster_connected_triangles(); cl = np.asarray(cl); n = np.asarray(n)
    mesh.remove_triangles_by_mask(n[cl] < 3000); mesh.remove_unreferenced_vertices()
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles); cen = V[F].mean(1)
    mesh.remove_triangles_by_mask(~((cen[:, 0] >= x0) & (cen[:, 0] < x0 + TILE) & (cen[:, 1] >= y0) & (cen[:, 1] < y0 + TILE)))
    mesh.remove_unreferenced_vertices()
    area_cells = len(np.unique(np.floor(P[own][:, :2] / 0.25), axis=0)) * 0.0625
    target = int(max(20000, TRI_PER_M2 * area_cells))
    if len(mesh.triangles) > target: mesh = mesh.simplify_quadric_decimation(target)
    mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles)
    dist, jj = tree.query(V, k=6, workers=8); w = 1 / (dist + 0.003) ** 2
    col = (Qc[jj] * w[..., None]).sum(1) / w.sum(1)[:, None]
    name = f'tile_{i}_{j}.npz'
    np.savez(W('mesh', 'tiles', name), v=V.astype(np.float32), f=F.astype(np.int32), c=col.astype(np.uint8))
    return dict(file=name, i=int(i), j=int(j), x0=float(x0), y0=float(y0), depth=depth, tris=len(F), verts=len(V), area_m2=round(area_cells, 1))

if __name__ == '__main__':
    log('tiles', len(tiles))
    with ProcessPoolExecutor(int(os.environ.get('BH_PROCS', 10))) as ex:
        res = [r for r in ex.map(run, tiles) if r]
    for r in res: log(r)
    json.dump(dict(tile=TILE, lo=lo.tolist(), tiles=res), open(W('mesh', 'tiles.json'), 'w'), indent=1)
    log('total tris', sum(r['tris'] for r in res))
