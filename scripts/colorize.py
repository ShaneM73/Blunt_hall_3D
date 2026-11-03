import os; RAW=os.environ.get("TVGO_RAW", os.path.expanduser("~/3DTemplehall"))
import numpy as np, torch, cv2, glob, os, sys, open3d as o3d
sys.path.insert(0,'.'); from calib import CAL, pose_at
dev='cuda:1'
import json
GM=json.load(open('gs/cameras.json')); KN=np.array(GM['K']); MW,MH=GM['W'],GM['H']
pc=o3d.io.read_point_cloud('clean_2cm.ply'); P=np.asarray(pc.points); inten=(np.asarray(pc.colors)[:,0]*255).astype(np.uint8)
Pg=torch.tensor(P,device=dev,dtype=torch.float64); N=len(P)
# Pass 1: no reference -> plain weighted mean, saved as colored_2cm.npz.
# Pass 2: re-run; samples far from the pass-1 color are down-weighted (robust to transient occluders).
PASS2=os.path.exists('colored_2cm.npz')
if PASS2:
    ref0=np.load('colored_2cm.npz'); REF=torch.tensor(ref0['rgb'],device=dev,dtype=torch.float64); HAS=torch.tensor(ref0['has'],device=dev)
else:
    REF=torch.zeros(N,3,device=dev,dtype=torch.float64); HAS=torch.zeros(N,dtype=torch.bool,device=dev)
acc=torch.zeros(N,3,device=dev,dtype=torch.float64); wsum=torch.zeros(N,device=dev,dtype=torch.float64)
S=4; H,Wd=2592//S,1944//S
RAWP=np.load('world_xyz.npy'); RT=np.load('world_t.npy')
poses_out=[]
for cam in (1,2,3):
    Rcb=torch.tensor(np.load(f'Rcb_nom_cam{cam}.npy'),device=dev); tb=torch.tensor(CAL[cam]['t'],device=dev)
    fx,fy,cx,cy=CAL[cam]['K']; k=CAL[cam]['D']
    fns=sorted(glob.glob(f'img/Camera{cam}/*.jpg'))
    for fn in fns:
        t=float(os.path.basename(fn)[:-4]); Rwb,twb=pose_at(t)
        Rwb=torch.tensor(Rwb,device=dev); twb=torch.tensor(twb,device=dev)
        Pb=(Pg-twb)@Rwb; Pc=(Pb-tb)@Rcb.T
        x,y,z=Pc.unbind(1); r=torch.hypot(x,y)+1e-9; th=torch.atan2(r,z); t2=th*th
        thd=th*(1+k[0]*t2+k[1]*t2**2+k[2]*t2**3+k[3]*t2**4)
        u=fx*thd*x/r+cx; v=fy*thd*y/r+cy; d=Pc.norm(dim=1)
        ok=(th<1.25)&(u>=0)&(u<1944)&(v>=0)&(v<2592)&(d<7)&(d>0.3)
        idx=ok.nonzero().squeeze(1)
        if len(idx)==0: continue
        ui=(u[idx]/S).long(); vi=(v[idx]/S).long(); pix=vi*Wd+ui; dd=d[idx]
        zb=torch.full((H*Wd,),1e9,device=dev,dtype=torch.float64).scatter_reduce(0,pix,dd,'amin')
        a0,a1=np.searchsorted(RT,[t-0.3,t+0.3]); Rp=torch.tensor(RAWP[a0:a1],device=dev,dtype=torch.float64)
        Qc=((Rp-twb)@Rwb-tb)@Rcb.T; qx,qy,qz=Qc.unbind(1); qr=torch.hypot(qx,qy)+1e-9; qt=torch.atan2(qr,qz); q2=qt*qt
        qd=qt*(1+k[0]*q2+k[1]*q2**2+k[2]*q2**3+k[3]*q2**4); qu=fx*qd*qx/qr+cx; qv=fy*qd*qy/qr+cy; qn=Qc.norm(dim=1)
        qo=(qt<1.25)&(qu>=0)&(qu<1944)&(qv>=0)&(qv<2592)&(qn>0.3)
        qpix=(qv[qo]/S).long()*Wd+(qu[qo]/S).long()
        # dilate dynamic hits 3x3 so sparse beams still block
        for dy in (-Wd,0,Wd):
            for dx in (-1,0,1):
                pp=(qpix+dy+dx).clamp(0,H*Wd-1); zb.scatter_reduce_(0,pp,qn[qo],'amin')
        vis=dd<=zb[pix]*1.03+0.04
        idx=idx[vis]
        # person mask (computed on the undistorted pinhole image): drop samples on detected people
        mk=cv2.imread(f'gs/masks/c{cam}_{t:.6f}.png',0)
        if mk is not None:
            haspeople=(mk[MH//4:3*MH//4,MW//4:3*MW//4]==0).any()
            zz=z[idx].clamp(min=1e-3); pu=(KN[0,0]*x[idx]/zz+KN[0,2]); pv=(KN[1,1]*y[idx]/zz+KN[1,2])
            inside=(pu>=0)&(pu<MW)&(pv>=0)&(pv<MH)
            M=torch.tensor(mk>127,device=dev)
            good=torch.zeros_like(inside); ii=inside.nonzero().squeeze(1)
            good[ii]=M[pv[ii].long(),pu[ii].long()]
            if not haspeople: good|=~inside
            idx=idx[good]
        im=torch.tensor(cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB),device=dev,dtype=torch.float64)
        col=im[v[idx].long(),u[idx].long()]
        w=1.0/(d[idx]**2+0.1)*torch.cos(th[idx])**2
        dc=((col-REF[idx])**2).sum(1); w=w*torch.where(HAS[idx],torch.exp(-dc/(2*35.0**2)),torch.ones_like(dc))
        acc.index_add_(0,idx,col*w[:,None]); wsum.index_add_(0,idx,w)
        poses_out.append((cam,t,*(twb.cpu().numpy()+Rwb.cpu().numpy()@tb.cpu().numpy())))
    print('cam',cam,'done',flush=True)
rgb=(acc/wsum.clamp(min=1e-12)[:,None]).clamp(0,255).byte().cpu().numpy(); has=(wsum>0).cpu().numpy()
rgb[~has]=np.repeat(inten[~has,None],3,1)
print('colored fraction',has.mean())
np.savez('colored_2cm_masked.npz' if PASS2 else 'colored_2cm_masked_p1.npz',xyz=P.astype(np.float32),rgb=rgb,inten=inten,has=has)
np.save('cam_positions.npy',np.array(poses_out))
