import os
import numpy as np, open3d as o3d, cv2, glob, os, json
from scipy.spatial import cKDTree
T=np.load('slam_optimized_trajectory.npy'); tp=T[:,1:4]
d=np.load('colored_2cm_static.npz'); P=d['xyz']; C=d['rgb'].astype(float); has=d['has']
fl=-1.01
print('=== 1. Distance from scanner path ===')
dist,_=cKDTree(tp).query(P,workers=64)
for a,b in [(0,2),(2,4),(4,7),(7,12),(12,1e9)]:
    m=(dist>=a)&(dist<b); print(f' {a:>2}-{b if b<1e9 else "inf":>3} m: {m.mean()*100:5.1f}% of points, colored {has[m].mean()*100:5.1f}%')
print('=== 2. Point density on surfaces (pts per 10x10 cm cell, 2cm voxels: max ~25) ===')
# local density: neighbours within 5 cm
tree=cKDTree(P); sub=np.random.default_rng(0).choice(len(P),200000,replace=False)
nn=np.array([len(x) for x in tree.query_ball_point(P[sub],0.05,workers=64)])
for a,b in [(0,2),(2,4),(4,7),(7,1e9)]:
    m=(dist[sub]>=a)&(dist[sub]<b); print(f' {a}-{b if b<1e9 else "inf"} m: median neighbours(5cm) {np.median(nn[m]):.0f}')
print('=== 3. Geometric noise & drift (raw points) ===')
W=np.load('world_xyz.npy'); WT=np.load('world_t.npy'); t0=WT.min()
# corridor south wall: find dominant vertical plane near path via RANSAC on a slab
rng=np.random.default_rng(1)
res=[]
for name,box in [('corridor wall A',None)]:
    pass
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[(dist<3)&(P[:,2]>fl+0.3)&(P[:,2]<fl+2.0)].astype(np.float64)))
rest=pc
for k in range(6):
    plane,inl=rest.segment_plane(0.03,3,2000)
    n=np.array(plane[:3]); 
    Q=np.asarray(rest.points)[inl]
    if abs(n[2])<0.2 and len(inl)>20000:
        # raw points near this plane, split by time
        dW=W@n+plane[3]; ctr=Q.mean(0); near=(np.abs(dW)<0.15)&(np.linalg.norm(W[:,:2]-ctr[:2],axis=1)<np.ptp(Q[:,:2],axis=0).max()/2)&(W[:,2]>fl+0.3)&(W[:,2]<fl+2)
        r=dW[near]; tt=WT[near]-t0
        early=r[tt<80]; late=r[tt>160]
        res.append((len(inl),np.std(r[np.abs(r)<0.1])*100, (np.median(late)-np.median(early))*100 if len(early)>500 and len(late)>500 else np.nan, np.percentile(np.abs(r[np.abs(r)<0.15]),90)*100))
    rest=rest.select_by_index(inl,invert=True)
for r in res: print(f' wall plane: {r[0]} pts, raw-point thickness std {r[1]:.1f} cm, p90 |offset| {r[3]:.1f} cm, early-vs-late offset {r[2]:.1f} cm')
print('=== 4. Stray patches ===')
pcs=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
lab=np.array(pcs.cluster_dbscan(eps=0.06,min_points=5))
cnt=np.bincount(lab[lab>=0]); big=cnt.argsort()[::-1]
size=np.where(lab>=0,cnt[np.maximum(lab,0)],0)
print(f' clusters {len(cnt)}; points in clusters <500 pts: {(size<500).mean()*100:.1f}%  (<5000: {(size<5000).mean()*100:.1f}%)')
print(f' points >7 m from path seen only through doors/windows: {(dist>7).mean()*100:.1f}%')
print('=== 5. Color ===')
print(f' uncolored (intensity gray fallback): {(~has).mean()*100:.1f}%')
fns=sorted(glob.glob('img/Camera1/*.jpg'))[::5]+sorted(glob.glob('img/Camera2/*.jpg'))[::5]+sorted(glob.glob('img/Camera3/*.jpg'))[::5]
br={c:[] for c in '123'}
for f in fns:
    g=cv2.imread(f,cv2.IMREAD_REDUCED_GRAYSCALE_8); br[f.split('Camera')[1][0]].append(g[g.shape[0]//4:3*g.shape[0]//4,g.shape[1]//4:3*g.shape[1]//4].mean())
for c in br: print(f' camera {c}: image brightness mean {np.mean(br[c]):.0f}, std over time {np.std(br[c]):.0f} (auto-exposure swings)')
# color mismatch between adjacent points on same surface: local color variance
nb=tree.query(P[sub],k=8,workers=64)[1]; lv=C[nb].std(1).mean(1)
print(f' median local color std among 8 neighbours: {np.median(lv[has[sub]]):.1f} /255 (blotchiness)')
s=C.max(1)-C.min(1); print(f' median saturation {np.median(s):.0f}/255')
print('=== 6. Capture ===')
print(f' duration {T[-1,0]-T[0,0]:.0f} s, path {np.linalg.norm(np.diff(tp,axis=0),axis=1).sum():.1f} m, mean speed {np.linalg.norm(np.diff(tp,axis=0),axis=1).sum()/(T[-1,0]-T[0,0]):.2f} m/s')
ext=np.ptp(tp,axis=0); print(f' path bounding box {ext[0]:.1f} x {ext[1]:.1f} m; cloud bbox {np.ptp(P[:,0]):.0f} x {np.ptp(P[:,1]):.0f} m')
sp=np.linalg.norm(np.diff(tp,axis=0),axis=1)/np.diff(T[:,0]); print(f' time spent nearly stationary (<0.1 m/s): {(sp<0.1).mean()*100:.0f}%')
