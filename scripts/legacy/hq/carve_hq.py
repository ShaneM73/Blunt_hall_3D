import os
"""Free-space carving: mark cloud points that LiDAR beams passed through in other frames as transient."""
import numpy as np, torch, sys
dev='cuda:1'
d=np.load('hq_cloud.npz'); P=torch.tensor(d['xyz'],device=dev,dtype=torch.float32); N=len(P)
W=np.load('world_xyz.npy'); WT=np.load('world_t.npy'); T=np.load('traj_refined.npy')
from scipy.spatial.transform import Rotation as Rot
free=torch.zeros(N,device=dev,dtype=torch.int32); seen=torch.zeros(N,device=dev,dtype=torch.int32)
AZ,EL=720,360   # 0.5 deg bins
bounds=np.searchsorted(WT,T[:,0]-1e-6); bounds=np.r_[bounds,len(WT)]
for f in range(0,len(T)):
    a,b=bounds[f],bounds[f+1]
    if b-a<100: continue
    o=torch.tensor(T[f,1:4],device=dev,dtype=torch.float32)
    Q=torch.tensor(W[a:b],device=dev)-o
    def sph(X):
        r=X.norm(dim=1); az=torch.atan2(X[:,1],X[:,0]); el=torch.asin((X[:,2]/r.clamp(min=1e-6)).clamp(-1,1))
        return r,((az+np.pi)/(2*np.pi)*AZ).long().clamp(0,AZ-1)*EL+((el+np.pi/2)/np.pi*EL).long().clamp(0,EL-1)
    rq,pq=sph(Q)
    rng=torch.full((AZ*EL,),1e9,device=dev).scatter_reduce(0,pq,rq,'amin',include_self=False)  # nearest hit per bin
    X=P-o; rp,pp=sph(X); near=(rp<15)&(rp>0.3)
    hit=rng[pp]<1e8
    m=near&hit
    free+= (m&(rp<rng[pp]-(0.15+0.03*rng[pp]))).int()
    seen+= (m&((rp-rng[pp]).abs()<0.06)).int()
    if f%500==0: print(f,flush=True)
tr=((free>=4)&(free.float()>seen.float()*0.5)).cpu().numpy()
print('transient',tr.sum(),'of',N, tr.mean())
np.save('hq_transient.npy',tr)
