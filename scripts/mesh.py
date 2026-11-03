import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
W=np.load('world_xyz.npy').astype(np.float64); I=np.load('world_int.npy')
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(W)); pc.colors=o3d.utility.Vector3dVector(np.repeat(I[:,None]/255.,3,1))
v=pc.voxel_down_sample(0.02); print('voxel 2cm',len(v.points),flush=True)
v,_=v.remove_statistical_outlier(20,2.0); print('after SOR',len(v.points),flush=True)
o3d.io.write_point_cloud('clean_2cm.ply',v)
v.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08,max_nn=30))
T=np.load('slam_optimized_trajectory.npy')[:,1:4]
P=np.asarray(v.points); N=np.asarray(v.normals); _,j=cKDTree(T).query(P); d=T[j]-P
N[(N*d).sum(1)<0]*=-1; v.normals=o3d.utility.Vector3dVector(N)
o3d.io.write_point_cloud('clean_2cm_normals.ply',v)
m,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(v,depth=11,n_threads=64)
dens=np.asarray(dens); m.remove_vertices_by_mask(dens<np.quantile(dens,0.08))
print('poisson',len(m.triangles),flush=True)
o3d.io.write_triangle_mesh('poisson_d11.ply',m)
