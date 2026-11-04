import numpy as np, cv2, glob, os, sys
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from scipy.optimize import minimize
sys.path.insert(0,'.'); from calib import CAL, pose_at
cam=int(sys.argv[1]); rng=np.random.default_rng(cam)
W=np.load('world_xyz.npy')[::4]; tree=cKDTree(W)
fns=sorted(glob.glob(f'img/Camera{cam}/*.jpg')); ts=np.array([float(os.path.basename(f)[:-4]) for f in fns])
fx,fy,cx,cy=CAL[cam]['K']; k=np.array(CAL[cam]['D'])
def proj(Pc):
    x,y,z=Pc.T; r=np.hypot(x,y)+1e-9; th=np.arctan2(r,z); th2=th*th
    thd=th*(1+k[0]*th2+k[1]*th2**2+k[2]*th2**3+k[3]*th2**4)
    u=fx*thd*x/r+cx; v=fy*thd*y/r+cy; ok=(th<1.4)&(u>=0)&(u<1944)&(v>=0)&(v<2592); return u,v,ok
S=8
imgs={}
def gray(i):
    if i not in imgs:
        g=cv2.imread(fns[i],0); g=cv2.GaussianBlur(cv2.resize(g,(1944//S,2592//S),interpolation=cv2.INTER_AREA),(3,3),0).astype(np.float32)
        imgs[i]=(g-g.mean())/(g.std()+1e-6)
    return imgs[i]
groups=[]
for gi in np.linspace(20,len(fns)-25,int(sys.argv[2]) if len(sys.argv)>2 else 10).astype(int):
    ids=[gi,gi+3,gi+6]   # 1.5 s spacing
    tc=ts[gi+3]; R0,t0=pose_at(tc)
    pts=W[tree.query_ball_point(t0,5.0)]; pts=pts[rng.choice(len(pts),min(6000,len(pts)),replace=False)]
    poses=[pose_at(ts[i]) for i in ids]
    groups.append((ids,pts,[(R,t,(pts-t)@R) for R,t in poses]))
def cost(Rcb,tb,dt=0.0,detail=False):
    tot=0;n=0
    for ids,pts,poses in groups:
        vals=[];oks=[]
        for (R,t,Pb),i in zip(poses,ids):
            Pc=(Pb-tb)@Rcb.T; u,v,ok=proj(Pc); ok&=Pc[:,2]>0.3
            g=gray(i); s=np.zeros(len(Pc),np.float32)
            s[ok]=g[(v[ok]/S).astype(int),(u[ok]/S).astype(int)]
            # crude occlusion: z-buffer
            D=np.full(g.shape,np.inf); ui=(u[ok]/S).astype(int); vi=(v[ok]/S).astype(int); d=np.linalg.norm(Pc[ok],axis=1)
            np.minimum.at(D,(vi,ui),d); vis=np.zeros(len(Pc),bool); vis[np.flatnonzero(ok)]=d<D[vi,ui]+0.15
            vals.append(s);oks.append(vis)
        V=np.array(vals);O=np.array(oks); m=O.all(0)
        if m.sum()<200: tot+=1.0*len(pts);n+=len(pts);continue
        tot+=V[:,m].var(0).sum()/ (2/3) + (len(pts)-m.sum())*1.0 ; n+=len(pts)
    return tot/n
if __name__=='__main__':
    t_nom=np.array(CAL[cam]['t'])
    cands=Rot.random(4000,random_state=cam).as_matrix()
    sc=np.array([cost(R,t_nom) for R in cands]); o=np.argsort(sc)
    print('coarse best',sc[o[:5]].round(3),'median',np.median(sc).round(3))
    best=[]
    for j in o[:6]:
        r0=Rot.from_matrix(cands[j]).as_rotvec()
        f=lambda p: cost(Rot.from_rotvec(p[:3]).as_matrix(),t_nom+p[3:6])
        res=minimize(f,np.r_[r0,0,0,0],method='Powell',options=dict(maxfev=1500,xtol=1e-3,ftol=1e-4))
        best.append((res.fun,res.x)); print(res.fun, res.x.round(4))
    b=min(best,key=lambda x:x[0]); np.save(f'extr_cam{cam}.npy',b[1])
    R=Rot.from_rotvec(b[1][:3]); print('cam',cam,'final',b[0],'euler xyz(body->cam) deg',R.as_euler('xyz',degrees=True).round(2),'inv',R.inv().as_euler('xyz',degrees=True).round(2),'t',b[1][3:])
