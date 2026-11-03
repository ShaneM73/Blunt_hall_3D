import numpy as np, open3d as o3d, json
from scipy.spatial import cKDTree
d=np.load('colored_2cm_static.npz'); P=d['xyz'].astype(np.float64); C=d['rgb']
v=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
v.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08,max_nn=30))
T=np.load('slam_optimized_trajectory.npy')[:,1:4]
N=np.asarray(v.normals); _,j=cKDTree(T).query(P); N[(N*(T[j]-P)).sum(1)<0]*=-1; v.normals=o3d.utility.Vector3dVector(N)
m,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(v,depth=11,n_threads=64)
dens=np.asarray(dens); m.remove_vertices_by_mask(dens<np.quantile(dens,0.08))
tree=cKDTree(P); dist,_=tree.query(np.asarray(m.vertices),workers=64); m.remove_vertices_by_mask(dist>0.05); m.remove_unreferenced_vertices()
cl,n,_=m.cluster_connected_triangles(); cl=np.asarray(cl); n=np.asarray(n); m.remove_triangles_by_mask(n[cl]<500); m.remove_unreferenced_vertices()
m=m.simplify_quadric_decimation(700000); m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
V=np.asarray(m.vertices); dist,j=tree.query(V,k=4,workers=64); w=1/(dist+0.005); col=(C[j]*w[...,None]).sum(1)/w.sum(1)[:,None]
m.vertex_colors=o3d.utility.Vector3dVector(col/255.)
o3d.io.write_triangle_mesh('/home/sgf.edubear.net/xm7303/Blunt_hall_3D/data/model.ply',m,write_ascii=False,write_vertex_normals=False)
print('tris',len(m.triangles))
