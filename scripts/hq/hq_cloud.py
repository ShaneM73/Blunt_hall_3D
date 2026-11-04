import os
import numpy as np, torch, sys, open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
sys.path.insert(0,'.'); from read_tvgo import read_slam_bin
idx,p=read_slam_bin(os.environ.get('TVGO_RAW',os.path.expanduser('~/3DTemplehall'))+'/SLAM_DATA/TMP_00000.bin')
T=np.load('traj_refined.npy'); cs=np.r_[0,np.cumsum(idx[:,0])]
B=np.stack([p['x'],p['y'],p['z']],1).astype(np.float64); r=np.linalg.norm(B,axis=1); I=p['a']
out=[];ints=[];tt=[]
for i in range(len(T)):
    s=slice(cs[i],cs[i+1]); m=(r[s]>0.5)&(r[s]<6.0)
    q=T[i,4:8]/np.linalg.norm(T[i,4:8]); R=Rot.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
    out.append(B[s][m]@R.T+T[i,1:4]); ints.append(I[s][m]); tt.append(np.full(m.sum(),T[i,0]))
W=np.vstack(out); WI=np.concatenate(ints); WT=np.concatenate(tt)
np.save('hq_world_xyz.npy',W.astype(np.float32)); np.save('hq_world_t.npy',WT)
dist,_=cKDTree(T[:,1:4]).query(W,workers=64); k=dist<5.0
W,WI=W[k],WI[k]; print('raw in crop',len(W),flush=True)
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(W)); pc.colors=o3d.utility.Vector3dVector(np.repeat(WI[:,None]/255.,3,1))
v=pc.voxel_down_sample(0.01); P=np.asarray(v.points); Iv=(np.asarray(v.colors)[:,0]*255).astype(np.uint8)
print('1cm voxels',len(P),flush=True)
# denoise: local planes from a 2.5 cm base grid (~8 cm neighbourhoods), applied to 1 cm points
base=np.asarray(pc.voxel_down_sample(0.025).points); bt=cKDTree(base); _,nb=bt.query(base,k=40,workers=64)
dev='cuda:1'; Bt=torch.tensor(base,device=dev,dtype=torch.float32)
X=Bt[torch.tensor(nb,device=dev)]; mu=X.mean(1); ev,evec=torch.linalg.eigh(torch.einsum('nki,nkj->nij',X-mu[:,None],X-mu[:,None])/40)
bn=evec[:,:,0]; blam=(ev[:,0]/ev.sum(1)); bsig=ev[:,0].clamp(min=0).sqrt(); bmu=mu
print('base',len(base),'planar(lam<0.02) %.0f%%'%((blam<0.02).float().mean()*100),'median sigma %.2f cm'%(bsig.median()*100),flush=True)
_,j=bt.query(P,k=1,workers=64); jt=torch.tensor(j,device=dev)
Pt=torch.tensor(P,device=dev,dtype=torch.float32)
n=bn[jt]; res=((Pt-bmu[jt])*n).sum(1); sig=bsig[jt]; lam=blam[jt]
planar=lam<0.02
ok=(~planar)|(res.abs()<torch.clamp(3*sig,min=0.012))
proj=torch.where(planar[:,None],Pt-res[:,None]*n*0.85,Pt)
keep=ok.cpu().numpy(); newP=proj.cpu().numpy(); nrm=n.cpu().numpy()
print('kept after denoise',keep.sum(),f'({keep.mean()*100:.1f}%)',flush=True)
P2=newP[keep]; N2=nrm[keep]; I2=Iv[keep]
# orient normals toward trajectory
_,j=cKDTree(T[:,1:4]).query(P2,workers=64); N2[(N2*(T[j,1:4]-P2)).sum(1)<0]*=-1
np.savez('hq_cloud.npz',xyz=P2.astype(np.float32),nrm=N2.astype(np.float32),inten=I2)
