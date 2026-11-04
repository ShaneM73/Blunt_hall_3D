import os
import numpy as np, open3d as o3d, sys, json
from scipy.spatial.transform import Rotation as Rot, Slerp
sys.path.insert(0,'.'); from read_tvgo import read_slam_bin
idx,p=read_slam_bin(os.environ.get('TVGO_RAW',os.path.expanduser('~/3DTemplehall'))+'/SLAM_DATA/TMP_00000.bin')
T=np.load('slam_optimized_trajectory.npy'); nF=len(T)
cs=np.r_[0,np.cumsum(idx[:,0])]
B=np.stack([p['x'],p['y'],p['z']],1).astype(np.float64); r=np.linalg.norm(B,axis=1)
def pose(i):
    M=np.eye(4); q=T[i,4:8]/np.linalg.norm(T[i,4:8]); M[:3,:3]=Rot.from_quat([q[1],q[2],q[3],q[0]]).as_matrix(); M[:3,3]=T[i,1:4]; return M
Pw=[pose(i) for i in range(nF)]
L=int(sys.argv[1]) if len(sys.argv)>1 else 30
subs=[]; anchors=[]
for s0 in range(0,nF,L):
    fr=range(s0,min(s0+L,nF)); A=Pw[s0]; Ai=np.linalg.inv(A); pts=[]
    for i in fr:
        q=B[cs[i]:cs[i+1]]; q=q[(r[cs[i]:cs[i+1]]>0.5)&(r[cs[i]:cs[i+1]]<15)]
        pts.append(q@(Ai@Pw[i])[:3,:3].T+(Ai@Pw[i])[:3,3])
    pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(pts))).voxel_down_sample(0.03)
    pc,_=pc.remove_statistical_outlier(15,2.0)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12,max_nn=30))
    subs.append(pc); anchors.append(A)
n=len(subs); print('submaps',n,flush=True)
pg=o3d.pipelines.registration.PoseGraph()
for A in anchors: pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(A))
crit=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
est=o3d.pipelines.registration.TransformationEstimationPointToPlane(o3d.pipelines.registration.TukeyLoss(k=0.05))
ctr=np.array([a[:3,3] for a in anchors]); nedge=0; stats=[]
for i in range(n):
    for j in range(i+1,n):
        if np.linalg.norm(ctr[i]-ctr[j])>5: continue
        init=np.linalg.inv(anchors[j])@anchors[i]          # maps submap i local -> submap j local
        reg=o3d.pipelines.registration.registration_icp(subs[i],subs[j],0.15,init,est,crit)
        reg=o3d.pipelines.registration.registration_icp(subs[i],subs[j],0.05,reg.transformation,est,crit)
        if reg.fitness<0.3 and j!=i+1: continue
        info=o3d.pipelines.registration.get_information_matrix_from_point_clouds(subs[i],subs[j],0.05,reg.transformation)
        dT=np.linalg.inv(init)@reg.transformation; stats.append((i,j,np.linalg.norm(dT[:3,3])*100,np.rad2deg(np.arccos(np.clip((np.trace(dT[:3,:3])-1)/2,-1,1))),reg.fitness,reg.inlier_rmse*100))
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(i,j,reg.transformation,info,uncertain=(j!=i+1))); nedge+=1
S=np.array(stats); print('edges',nedge,'ICP correction vs SLAM: median %.1f cm %.2f deg; p90 %.1f cm; rmse median %.1f cm'%(np.median(S[:,2]),np.median(S[:,3]),np.percentile(S[:,2],90),np.median(S[:,5])),flush=True)
opt=o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=0.05,edge_prune_threshold=0.25,reference_node=0)
o3d.pipelines.registration.global_optimization(pg,o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),opt)
newA=[nd.pose for nd in pg.nodes]
# correction per submap, applied at submap mid-time, interpolated per frame
Cs=[newA[k]@np.linalg.inv(anchors[k]) for k in range(n)]
mid=np.array([T[min(k*L+L//2,nF-1),0] for k in range(n)])
rots=Rot.from_matrix(np.array([c[:3,:3] for c in Cs])); tr=np.array([c[:3,3] for c in Cs])
sl=Slerp(mid,rots); tq=np.clip(T[:,0],mid[0],mid[-1])
Rc=sl(tq).as_matrix(); tc=np.stack([np.interp(tq,mid,tr[:,k]) for k in range(3)],1)
T2=T.copy()
for i in range(nF):
    M=Pw[i].copy(); M[:3,:3]=Rc[i]@M[:3,:3]; M[:3,3]=Rc[i]@M[:3,3]+tc[i]
    q=Rot.from_matrix(M[:3,:3]).as_quat(); T2[i,1:4]=M[:3,3]; T2[i,4:8]=[q[3],q[0],q[1],q[2]]
np.save('traj_refined.npy',T2)
print('max correction %.1f cm, mean %.1f cm'%(np.abs(T2[:,1:4]-T[:,1:4]).max()*100,np.linalg.norm(T2[:,1:4]-T[:,1:4],axis=1).mean()*100))
