"""Occupancy floor plan: a 5 cm cell is a wall if LiDAR hits persist (>=4 hits/voxel) over >=1.2 m of height
in a 0.15-2.2 m slice above the floor; 0.3-1.2 m = furniture; floor hits = observed free space. Run after build_web.py."""
import numpy as np, cv2, json, os
OUT=os.path.join(os.path.dirname(__file__),'..','data')+'/'
meta=json.load(open(OUT+'meta.json')); fl=meta['floor_z']; res=0.05
W=np.load('world_xyz.npy')
if os.path.exists('colored_2cm_static.npz'):   # drop raw returns not near a static (non-carved) point, e.g. people
    from scipy.spatial import cKDTree
    S=np.load('colored_2cm_static.npz')['xyz']; dd,_=cKDTree(S).query(W,distance_upper_bound=0.04,workers=64); W=W[np.isfinite(dd)]
xmin,ymin=np.floor(np.percentile(W[:,:2],0.2,axis=0))-1; xmax,ymax=np.ceil(np.percentile(W[:,:2],99.8,axis=0))+1
nx,ny=int((xmax-xmin)/res),int((ymax-ymin)/res)
Q=W[(W[:,2]>fl+0.15)&(W[:,2]<fl+2.2)]
ix=((Q[:,0]-xmin)/res).astype(int); iy=((ymax-Q[:,1])/res).astype(int); iz=((Q[:,2]-fl-0.15)/0.1).astype(int)
ok=(ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)
cnt=np.zeros((ny,nx,21),np.int32); np.add.at(cnt,(iy[ok],ix[ok],iz[ok]),1); vb=(cnt>=4).sum(2)
Fp=W[np.abs(W[:,2]-fl)<0.08]; fx=((Fp[:,0]-xmin)/res).astype(int); fy=((ymax-Fp[:,1])/res).astype(int); k=(fx>=0)&(fx<nx)&(fy>=0)&(fy<ny)
Fl=np.zeros((ny,nx)); np.add.at(Fl,(fy[k],fx[k]),1)
img=np.full((ny,nx,3),(225,229,233),np.uint8); img[Fl>3]=(252,252,250); img[(vb>=3)&(vb<12)]=(150,150,155); img[vb>=12]=(30,32,36)
cv2.imwrite(OUT+'slam_map.png',cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
meta['slam_map']=dict(image='data/slam_map.png',res=res/2,grid_cm=5,xmin=float(xmin),ymax=float(ymax),slice='0.15–2.2 m above the floor',
  note='Black = walls (occupied over ≥1.2 m of height), gray = furniture and low objects, white = floor seen by the scanner, blue-gray = never observed. Grid 5 cm.')
json.dump(meta,open(OUT+'meta.json','w'),indent=1)
