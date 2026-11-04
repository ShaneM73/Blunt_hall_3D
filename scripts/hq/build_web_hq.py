import os
import numpy as np, json, open3d as o3d
OUT=os.path.join(os.path.dirname(__file__),'..','..','data')+'/'
d=np.load('hq_colored.npz'); rm=np.load('hq_person_rm.npy'); k=d['has']&~rm
P=d['xyz'][k].astype(np.float64); C=d['rgb'][k]; I=d['inten'][k]
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc.colors=o3d.utility.Vector3dVector(np.c_[C/255.])
# 1.5 cm web cloud; carry intensity via a parallel cloud
v,_,idx=pc.voxel_down_sample_and_trace(0.015,pc.get_min_bound(),pc.get_max_bound())
Pv=np.asarray(v.points); Cv=(np.asarray(v.colors)*255).round().astype(np.uint8)
first=np.array([ix[0] for ix in idx]); Iv=I[first]
o=Pv.min(0); sc=float((Pv.max(0)-o).max()/65535); q=np.round((Pv-o)/sc).astype(np.uint16)
with open(OUT+'points.bin','wb') as f:
    f.write(b'BHPC'); f.write(np.array([len(Pv),1],np.uint32).tobytes()); f.write(np.array([*o,sc],np.float32).tobytes())
    f.write(q.tobytes()); f.write(Iv.astype(np.uint8).tobytes()); f.write(Cv.tobytes())
print('web points',len(Pv))
M=np.load('hq_mesh.npz'); V,F,Cm=M['v'],M['f'],M['c']
hdr=f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n"
vb=np.zeros(len(V),dtype=[('p','<f4',3),('c','u1',3)]); vb['p']=V; vb['c']=Cm
fb=np.zeros(len(F),dtype=[('n','u1'),('i','<i4',3)]); fb['n']=3; fb['i']=F
with open(OUT+'model.ply','wb') as f: f.write(hdr.encode()); f.write(vb.tobytes()); f.write(fb.tobytes())
T=np.load('traj_refined.npy'); cams=np.load('cam_positions.npy')
json.dump({'t':(T[::2,0]-T[0,0]).round(2).tolist(),'p':T[::2,1:4].round(3).tolist(),'cams':cams[:,2:5].round(3).tolist()},open(OUT+'trajectory.json','w'))
m=json.load(open(OUT+'meta.json'))
m.update(voxel_cm=1.5,rgb_frac=1.0,crop_m=5,
  rgb_note='Cropped to the well-scanned zone (within 5 m of the walking path). Photo exposure and white balance were equalized per image before blending; people were removed with photo masks and LiDAR free-space carving.')
m['model']=dict(file='data/model.ply',short='Screened Poisson (d12)',
  method=f'Screened Poisson reconstruction (octree depth 12, ~6 mm) of the denoised, people-free 1 cm LiDAR cloud, trimmed to within 3 cm of real points and decimated to {len(F):,} triangles; vertex colors from exposure-equalized photos.')
json.dump(m,open(OUT+'meta.json','w'),indent=1)
