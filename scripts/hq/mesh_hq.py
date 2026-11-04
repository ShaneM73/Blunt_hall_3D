import os
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
d=np.load('hq_colored.npz'); rm=np.load('hq_person_rm.npy'); k=~rm&d['has']
P=d['xyz'][k].astype(np.float64); N=d['nrm'][k].astype(np.float64); C=d['rgb'][k]
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc.normals=o3d.utility.Vector3dVector(N)
m,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc,depth=12,n_threads=96,scale=1.05)
print('poisson',len(m.triangles),flush=True)
tree=cKDTree(P); dist,_=tree.query(np.asarray(m.vertices),workers=96)
m.remove_vertices_by_mask(dist>0.03); m.remove_unreferenced_vertices()
cl,n,_=m.cluster_connected_triangles(); cl=np.asarray(cl); n=np.asarray(n); m.remove_triangles_by_mask(n[cl]<3000); m.remove_unreferenced_vertices()
print('trimmed',len(m.triangles),flush=True)
m=m.simplify_quadric_decimation(1_800_000); m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
V=np.asarray(m.vertices); dist,j=tree.query(V,k=6,workers=96); w=1/(dist+0.003)**2
col=(C[j]*w[...,None]).sum(1)/w.sum(1)[:,None]
F=np.asarray(m.triangles)
print('final tris',len(F),'verts',len(V))
np.savez('hq_mesh.npz',v=V.astype(np.float32),f=F.astype(np.int32),c=col.astype(np.uint8))
