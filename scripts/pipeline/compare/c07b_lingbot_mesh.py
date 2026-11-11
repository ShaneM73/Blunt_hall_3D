"""Comparison stage 7b: LingBot-Map predictions (c07_lingbot_run.py npz) -> cleaned point cloud + TSDF mesh.
Floaters in the raw back-projection come from low-confidence depth (object edges, glass, sky, blur) and from frames
whose depths disagree; three filters remove them before fusion:
  1. confidence: keep pixels above the LB_CONF-th percentile of that frame's depth confidence (default 50)
  2. sky: drop pixels the skyseg ONNX model calls sky (outdoor clips only, LB_SKY=1)
  3. depth edges: drop pixels whose depth jumps > 5 % to a neighbour (flying pixels between foreground and background)
then TSDF fusion (Open3D tensor VoxelBlockGrid on the GPU; trunc LB_TRUNC voxels, a surface needs LB_WMIN views) keeps only surfaces seen consistently from several views, and small mesh islands are removed.
Units are the model's own (arbitrary scale); voxel = LB_VOX x scene size.
  PATH=~/.conda/envs/lingbot-map/bin:$PATH python c07b_lingbot_mesh.py pred.npz out_prefix
Out: <out>_mesh.ply, <out>_clean.ply, <out>_stats.json"""
import os, sys, json, time
import numpy as np, open3d as o3d, cv2

src, out = sys.argv[1], sys.argv[2]
CONF = float(os.environ.get('LB_CONF', 50)); SKY = os.environ.get('LB_SKY', '1') == '1'; VOXF = float(os.environ.get('LB_VOX', 1 / 500))
z = np.load(src); D = z['depth'][..., 0]; C = z['depth_conf']; E = z['extrinsic']; K = z['intrinsic']
I = z['images']; I = I.transpose(0, 2, 3, 1) if I.shape[1] == 3 else I
N, H, W = D.shape; t0 = time.time()
sess = None
if SKY:
    import onnxruntime as ort
    sys.path.insert(0, os.path.expanduser('~/apps/cmp/lingbot-map'))
    from lingbot_map.vis.sky_segmentation import segment_sky_from_array
    sess = ort.InferenceSession(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'work', 'models', 'lingbot-map', 'skyseg.onnx'),
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
masks, kept = [], 0
for i in range(N):
    d = D[i]; m = (C[i] >= np.percentile(C[i], CONF)) & (d > 0)
    if sess is not None: m &= segment_sky_from_array(I[i], sess, H, W) > 0.5
    k = np.ones((3, 3), np.uint8); rel = (cv2.dilate(d, k) - cv2.erode(d, k)) / np.maximum(d, 1e-6)
    m &= rel < 0.05
    masks.append(m); kept += m.sum()
cams = np.array([-e[:, :3].T @ e[:, 3] for e in E])
# scene size from the confident points of every 10th frame
pts = []
for i in range(0, N, 10):
    v, u = np.nonzero(masks[i]); d = D[i][v, u]
    x = np.stack([(u - K[i, 0, 2]) / K[i, 0, 0] * d, (v - K[i, 1, 2]) / K[i, 1, 1] * d, d], -1)
    pts.append((x - E[i][:, 3]) @ E[i][:, :3])
P = np.vstack(pts); lo, hi = np.percentile(P, [1, 99], axis=0); size = float(np.linalg.norm(hi - lo))
vox = size * VOXF; dmax = float(np.percentile(np.concatenate([D[i][masks[i]] for i in range(0, N, 10)]), float(os.environ.get('LB_DMAXP', 98))))
import open3d.core as o3c
dev = o3c.Device('CUDA:0') if o3c.cuda.is_available() else o3c.Device('CPU:0'); TR = float(os.environ.get('LB_TRUNC', 4))
vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'), attr_dtypes=(o3c.float32,) * 3,
                                    attr_channels=((1), (1), (3)), voxel_size=vox, block_resolution=16, block_count=400000, device=dev)
for i in range(N):                                                   # extrinsic = world-to-camera (OpenCV axes)
    d = o3d.t.geometry.Image(np.where(masks[i], D[i], 0).astype(np.float32)).to(dev)
    c = o3d.t.geometry.Image(np.ascontiguousarray(I[i].astype(np.float32))).to(dev)
    ext = np.eye(4); ext[:3] = E[i]; Kt, Et = o3c.Tensor(K[i].astype(np.float64)), o3c.Tensor(ext)
    bc = vbg.compute_unique_block_coordinates(d, Kt, Et, 1.0, dmax, trunc_voxel_multiplier=TR)
    vbg.integrate(bc, d, c, Kt, Et, 1.0, dmax, trunc_voxel_multiplier=TR)
WMIN = float(os.environ.get('LB_WMIN', 3))                           # a surface must be seen in >= WMIN frames
mesh = vbg.extract_triangle_mesh(weight_threshold=WMIN).to_legacy()
lab, cnt, _ = map(np.asarray, mesh.cluster_connected_triangles())
mesh.remove_triangles_by_mask(cnt[lab] < max(200, 0.002 * len(lab))); mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
pc = vbg.extract_point_cloud(weight_threshold=WMIN).to_legacy(); pc, _ = pc.remove_statistical_outlier(20, 2.0)
o3d.io.write_triangle_mesh(out + '_mesh.ply', mesh); o3d.io.write_point_cloud(out + '_clean.ply', pc)
st = dict(frames=N, pixels_kept=float(kept / D.size), scene_size=size, voxel=vox, depth_trunc=dmax,
          mesh_vertices=len(mesh.vertices), mesh_triangles=len(mesh.triangles), clean_points=len(pc.points), seconds=round(time.time() - t0, 1))
json.dump(st, open(out + '_stats.json', 'w'), indent=1); print(st)
