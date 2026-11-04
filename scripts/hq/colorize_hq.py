import os
import numpy as np, torch, cv2, glob, os, sys, json
sys.path.insert(0,'.'); from calib import CAL
from scipy.spatial.transform import Rotation as Rot, Slerp
dev='cuda:1'
d=np.load('hq_cloud.npz'); tr=np.load('hq_transient.npy')
P=d['xyz'][~tr].astype(np.float64); inten=d['inten'][~tr]; nrm=d['nrm'][~tr]
Pg=torch.tensor(P,device=dev); Ng=torch.tensor(nrm,device=dev,dtype=torch.float64); N=len(P); print('points',N,flush=True)
T=np.load('traj_refined.npy')
def pose_at(t):
    i=int(np.clip(np.searchsorted(T[:,0],t),1,len(T)-1)); a=(t-T[i-1,0])/(T[i,0]-T[i-1,0])
    R=Slerp([0,1],Rot.from_quat(T[[i-1,i],4:8][:,[1,2,3,0]]))([a])[0].as_matrix(); return R,(1-a)*T[i-1,1:4]+a*T[i,1:4]
RAWP=np.load('world_xyz.npy'); RT=np.load('world_t.npy')
GM=json.load(open('gs/cameras.json')); KN=np.array(GM['K']); MW,MH=GM['W'],GM['H']
S=4; H,Wd=2592//S,1944//S
imgs=[]
for cam in (1,2,3):
    for fn in sorted(glob.glob(f'img/Camera{cam}/*.jpg')): imgs.append((cam,fn,float(os.path.basename(fn)[:-4])))
# precompute visibility per image once (indices, pixel coords, weights) on GPU -> keep on CPU
vis_cache=[]
vcount=torch.zeros(N,device=dev); mcount=torch.zeros(N,device=dev)
for k,(cam,fn,t) in enumerate(imgs):
    Rcb=torch.tensor(np.load(f'Rcb_nom_cam{cam}.npy'),device=dev,dtype=torch.float64); tb=torch.tensor(CAL[cam]['t'],device=dev,dtype=torch.float64)
    fx,fy,cx,cy=CAL[cam]['K']; kk=CAL[cam]['D']
    Rwb,twb=pose_at(t); Rwb=torch.tensor(Rwb,device=dev); twb=torch.tensor(twb,device=dev)
    def proj(X):
        Pc=((X-twb)@Rwb-tb)@Rcb.T; x,y,z=Pc.unbind(1); r=torch.hypot(x,y)+1e-9; th=torch.atan2(r,z); t2=th*th
        thd=th*(1+kk[0]*t2+kk[1]*t2**2+kk[2]*t2**3+kk[3]*t2**4); return Pc,fx*thd*x/r+cx,fy*thd*y/r+cy,th
    Pc,u,v,th=proj(Pg); dd=Pc.norm(dim=1)
    ok=(th<1.2)&(u>=0)&(u<1944)&(v>=0)&(v<2592)&(dd<6)&(dd>0.3)
    # back-facing surfaces: normal must face camera
    cpos=twb+Rwb@tb; vd=(cpos-Pg); vd=vd/vd.norm(dim=1,keepdim=True); cosi=(vd*Ng).sum(1).abs()
    ok&=cosi>0.15
    idx=ok.nonzero().squeeze(1)
    pix=(v[idx]/S).long()*Wd+(u[idx]/S).long(); zb=torch.full((H*Wd,),1e9,device=dev,dtype=torch.float64).scatter_reduce(0,pix,dd[idx],'amin')
    a0,a1=np.searchsorted(RT,[t-0.3,t+0.3]); Rp=torch.tensor(RAWP[a0:a1],device=dev,dtype=torch.float64)
    Qc,qu,qv,qt=proj(Rp); qn=Qc.norm(dim=1); qo=(qt<1.2)&(qu>=0)&(qu<1944)&(qv>=0)&(qv<2592)&(qn>0.3)
    qpix=(qv[qo]/S).long()*Wd+(qu[qo]/S).long()
    for dy in (-Wd,0,Wd):
        for dx in (-1,0,1): zb.scatter_reduce_(0,(qpix+dy+dx).clamp(0,H*Wd-1),qn[qo],'amin')
    vis=dd[idx]<=zb[pix]*1.02+0.03; idx=idx[vis]
    mk=cv2.imread(f'gs/masks/c{cam}_{t:.6f}.png',0)
    if mk is not None:
        x,y,z=Pc[idx].unbind(1); zz=z.clamp(min=1e-3); pu=KN[0,0]*x/zz+KN[0,2]; pv=KN[1,1]*y/zz+KN[1,2]
        inside=(pu>=0)&(pu<MW)&(pv>=0)&(pv<MH); M=torch.tensor(mk>127,device=dev); good=torch.zeros_like(inside); ii=inside.nonzero().squeeze(1)
        good[ii]=M[pv[ii].long(),pu[ii].long()]
        if not (mk[MH//4:3*MH//4,MW//4:3*MW//4]==0).any(): good|=~inside
        vcount.index_add_(0,idx[inside],torch.ones(int(inside.sum()),device=dev)); mcount.index_add_(0,idx[inside&~good],torch.ones(int((inside&~good).sum()),device=dev))
        idx=idx[good]
    w=(cosi[idx]**1.5)/(dd[idx]**2+0.05)*torch.cos(th[idx])**2
    im=torch.tensor(cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB),device=dev,dtype=torch.float32)
    col=im[v[idx].long(),u[idx].long()]
    vis_cache.append((idx.cpu(),col.to(torch.uint8).cpu(),w.float().cpu()))
    if k%300==0: print('vis',k,flush=True)
torch.save(vis_cache,'hq_vis_cache.pt')
def blend(gains,ref=None,p=1.0,sigma=30.0):
    acc=torch.zeros(N,3,device=dev); ws=torch.zeros(N,device=dev)
    for k,(idx,col,w) in enumerate(vis_cache):
        idx=idx.to(dev); c=col.to(dev).float()*gains[k]; w=w.to(dev)**p
        if ref is not None: w=w*torch.exp(-((c-ref[idx])**2).sum(1)/(2*sigma**2))
        acc.index_add_(0,idx,c*w[:,None]); ws.index_add_(0,idx,w)
    return acc/ws.clamp(min=1e-12)[:,None], ws
gains=torch.ones(len(imgs),3,device=dev)
ref,ws=blend(gains)
for it in range(3):
    newg=torch.ones_like(gains)
    for k,(idx,col,w) in enumerate(vis_cache):
        if len(idx)<500: continue
        c=col.to(dev).float(); r=ref[idx.to(dev)]
        m=(c.min(1).values>15)&(c.max(1).values<240)
        if m.sum()<300: continue
        newg[k]=(torch.median(r[m]/c[m].clamp(min=1),dim=0).values).clamp(0.5,2.0)
    newg=newg/newg.mean(0,keepdim=True)   # keep overall exposure
    gains=newg; ref,ws=blend(gains,ref,p=1.0)
    g=gains.cpu().numpy(); print(f'iter {it}: gain spread (std) R {g[:,0].std():.3f} G {g[:,1].std():.3f} B {g[:,2].std():.3f}',flush=True)
final,ws=blend(gains,ref,p=3.0,sigma=25.0)
has=(ws>0).cpu().numpy()
rgb=final.clamp(0,255).byte().cpu().numpy()
print('colored',has.mean())
np.savez('hq_colored.npz',xyz=P.astype(np.float32),nrm=nrm,rgb=rgb,inten=inten,has=has)
np.save('hq_gains.npy',gains.cpu().numpy()); np.save('hq_person_frac.npy',(mcount/vcount.clamp(min=1)).cpu().numpy()); np.save('hq_vcount.npy',vcount.cpu().numpy())
