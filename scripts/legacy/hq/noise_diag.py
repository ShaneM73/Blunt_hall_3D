import os
import numpy as np, open3d as o3d
W=np.load('world_xyz.npy'); WT=np.load('world_t.npy')
d=np.load('colored_2cm_static.npz'); P=d['xyz']; T=np.load('slam_optimized_trajectory.npy')
from scipy.spatial import cKDTree
dist,_=cKDTree(T[:,1:4]).query(P,workers=64); fl=-1.01
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[(dist<3)&(P[:,2]>fl+0.3)&(P[:,2]<fl+2.0)].astype(np.float64)))
rest=pc
for k in range(5):
    plane,inl=rest.segment_plane(0.03,3,2000); n=np.array(plane[:3]); Q=np.asarray(rest.points)[inl]; rest=rest.select_by_index(inl,invert=True)
    if abs(n[2])>0.2 or len(inl)<20000: continue
    ctr=Q.mean(0); ext=np.ptp(Q[:,:2],axis=0).max()/2
    dW=W@n+plane[3]; near=(np.abs(dW)<0.12)&(np.linalg.norm(W[:,:2]-ctr[:2],axis=1)<ext)&(W[:,2]>fl+0.3)&(W[:,2]<fl+2)
    r=dW[near]; ft=WT[near]
    # range of each point from scanner pose at that frame
    fi=np.searchsorted(T[:,0],ft); fi=np.clip(fi,0,len(T)-1); rng=np.linalg.norm(W[near]-T[fi,1:4],axis=1)
    within=[];med=[]
    for f in np.unique(ft):
        m=ft==f
        if m.sum()>80: within.append(np.std(r[m])); med.append(np.median(r[m]))
    mad=lambda x:1.4826*np.median(np.abs(x-np.median(x)))
    print(f'plane {len(inl)} pts: ROBUST total {mad(r)*100:.1f} cm, robust within-frame {np.median([mad(r[ft==f]) for f in np.unique(ft) if (ft==f).sum()>80])*100:.1f} cm, robust per-frame offset spread {mad(np.array(med))*100:.1f} cm | total std {np.std(r)*100:.1f} cm | within-frame std median {np.median(within)*100:.1f} cm | std of per-frame offsets {np.std(med)*100:.1f} cm | n frames {len(med)}')
    for a,b in [(0,1.5),(1.5,3),(3,6),(6,15)]:
        m=(rng>=a)&(rng<b)
        if m.sum()>500: print(f'    range {a}-{b} m: std {np.std(r[m])*100:.1f} cm (n={m.sum()})')
