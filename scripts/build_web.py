import os; RAW=os.environ.get("TVGO_RAW", os.path.expanduser("~/3DTemplehall"))
import numpy as np, json, open3d as o3d, cv2
from scipy.spatial import cKDTree
OUT=os.path.join(os.path.dirname(__file__),'..','data')+'/'
d=np.load('colored_2cm_dyn.npz'); P=d['xyz'].astype(np.float64); C=d['rgb']; I=d['inten']; has=d['has']
# --- points.bin
o=P.min(0); sc=float((P.max(0)-o).max()/65535)
q=np.round((P-o)/sc).astype(np.uint16)
with open(OUT+'points.bin','wb') as f:
    f.write(b'BHPC'); f.write(np.array([len(P),1],np.uint32).tobytes()); f.write(np.array([*o,sc],np.float32).tobytes())
    f.write(q.tobytes()); f.write(I.astype(np.uint8).tobytes()); f.write(C.astype(np.uint8).tobytes())
print('points',len(P),'scale mm',sc*1000)
# --- floor / ceiling
z=P[:,2]; h,e=np.histogram(z,bins=400,range=(-4,6)); c=(e[:-1]+e[1:])/2
floor=c[np.argmax(h*(c<0))]; ceil=c[np.argmax(h*(c>floor+2.0)*(c<floor+4))]; print('floor',floor,'ceil',ceil)
# --- trajectory
T=np.load('slam_optimized_trajectory.npy'); cams=np.load('cam_positions.npy')
path=float(np.linalg.norm(np.diff(T[:,1:4],axis=0),axis=1).sum())
json.dump({'t':(T[::2,0]-T[0,0]).round(2).tolist(),'p':T[::2,1:4].round(3).tolist(),'cams':cams[:,2:5].round(3).tolist()},open(OUT+'trajectory.json','w'))
# --- SLAM occupancy map from raw points
W=np.load('world_xyz.npy'); m=(W[:,2]>floor+0.3)&(W[:,2]<floor+1.8); W=W[m]
res=0.05; xmin,ymin=np.floor(np.percentile(W[:,:2],0.2,axis=0))-1; xmax,ymax=np.ceil(np.percentile(W[:,:2],99.8,axis=0))+1
nx,ny=int((xmax-xmin)/res),int((ymax-ymin)/res)
ix=((W[:,0]-xmin)/res).astype(int); iy=((ymax-W[:,1])/res).astype(int); ok=(ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)
H=np.zeros((ny,nx)); np.add.at(H,(iy[ok],ix[ok]),1)
# free space: floor points (seen floor = traversable/observed)
F=np.load('world_xyz.npy'); F=F[np.abs(F[:,2]-floor)<0.1]
fx=((F[:,0]-xmin)/res).astype(int); fy=((ymax-F[:,1])/res).astype(int); ok=(fx>=0)&(fx<nx)&(fy>=0)&(fy<ny)
Fl=np.zeros((ny,nx)); np.add.at(Fl,(fy[ok],fx[ok]),1)
img=np.full((ny,nx,3),(232,231,225),np.uint8)              # unknown
img[Fl>2]=(250,250,248)                                     # observed floor
occ=np.clip(np.log1p(H)/np.log1p(np.percentile(H[H>0],95)),0,1)
wall=occ>0.15; shade=(40+ (1-occ[wall])*120).astype(np.uint8)
img[wall]=np.stack([shade]*3,1)
cv2.imwrite(OUT+'slam_map.png',img[:,:,::-1])
loops=sum(1 for _ in open(RAW+'/SLAM_DATA/TMP_LoopSearch/loop_ends.txt') if _.strip())
foc=np.median(T[:,1:4],axis=0)
meta=dict(n_raw=19472263,n_frames=len(T),duration_s=round(float(T[-1,0]-T[0,0]),1),path_m=round(path,1),voxel_cm=2,
  n_images=1487,floor_z=round(float(floor),2),ceiling_z=round(float(ceil),2),focus=foc.round(2).tolist(),rgb_frac=round(float(has.mean()),3),
  rgb_note='Ghost-like smudges on some walls come from people walking alongside the scanner; they are baked into the photos.',
  loop_closures=loops,
  slam_map=dict(image='data/slam_map.png',res=res,xmin=float(xmin),ymax=float(ymax),slice=f'{0.3:.1f}–{1.8:.1f} m above the floor',
     note='Dark = walls/objects (LiDAR hits), white = floor seen by the scanner, beige = never observed. Grid 5 cm.'),
  model=dict(file='data/model.ply',method='',short=''))
json.dump(meta,open(OUT+'meta.json','w'),indent=1)
print(meta)
