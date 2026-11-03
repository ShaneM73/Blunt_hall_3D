import torch, numpy as np, cv2, json, math, time, sys, torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from scipy.spatial import cKDTree
torch.manual_seed(0); dev='cuda'
ITERS=int(sys.argv[1]) if len(sys.argv)>1 else 30000; SHD=1
meta=json.load(open('gs/cameras.json')); W,H=meta['W'],meta['H']; K=torch.tensor(meta['K'],dtype=torch.float32,device=dev)
frames=meta['frames']; n=len(frames)
test=set(range(0,n,25)); train=[i for i in range(n) if i not in test]
print('loading images',flush=True)
imgs=torch.stack([torch.from_numpy(cv2.cvtColor(cv2.imread('gs/images/'+f['file']),cv2.COLOR_BGR2RGB)) for f in frames])  # uint8 cpu
masks=torch.stack([torch.from_numpy(cv2.imread('gs/masks/'+f['file'].replace('.jpg','.png'),0)>127) for f in frames])
imgs=imgs.to(dev); masks=masks.to(dev)
c2w=torch.tensor([f['c2w'] for f in frames],dtype=torch.float32,device=dev); w2c=torch.linalg.inv(c2w)
cpos=c2w[:,:3,3]; scene_scale=float((cpos-cpos.mean(0)).norm(dim=1).max())*1.1; print('scene_scale',scene_scale)
# init from colored LiDAR points
d=np.load('colored_2cm_static.npz'); P=d['xyz']; C=d['rgb']/255.
sel=np.random.default_rng(0).choice(len(P),min(len(P),1_200_000),replace=False); P=P[sel]; C=C[sel]
dist,_=cKDTree(P).query(P,k=4); s0=np.log(np.clip(dist[:,1:].mean(1),0.003,0.1))
N=len(P)
LP=torch.tensor(d['xyz'],dtype=torch.float32,device=dev)   # full clean cloud (2.78M)
DS=4; DW,DH=W//DS,H//DS
def lidar_depth(i):
    Pc=LP@w2c[i,:3,:3].T+w2c[i,:3,3]; z=Pc[:,2]; ok=z>0.2
    u=K[0,0]*Pc[:,0]/z.clamp(min=1e-3)+K[0,2]; v=K[1,1]*Pc[:,1]/z.clamp(min=1e-3)+K[1,2]
    ok&=(u>=0)&(u<W)&(v>=0)&(v<H)&(z<12); idx=ok.nonzero().squeeze(1)
    pix=(v[idx]/DS).long()*DW+(u[idx]/DS).long()
    zb=torch.full((DH*DW,),1e9,device=dev).scatter_reduce(0,pix,z[idx],'amin')
    vis=z[idx]<=zb[pix]*1.02+0.02; idx=idx[vis]
    return u[idx].long(),v[idx].long(),z[idx]
params=torch.nn.ParameterDict({
 'means':torch.nn.Parameter(torch.tensor(P,dtype=torch.float32,device=dev)),
 'scales':torch.nn.Parameter(torch.tensor(s0,dtype=torch.float32,device=dev)[:,None].repeat(1,3)),
 'quats':torch.nn.Parameter(torch.rand(N,4,device=dev)),
 'opacities':torch.nn.Parameter(torch.logit(torch.full((N,),0.1,device=dev))),
 'sh0':torch.nn.Parameter(((torch.tensor(C,dtype=torch.float32,device=dev)-0.5)/0.28209479)[:,None,:]),
 'shN':torch.nn.Parameter(torch.zeros(N,(SHD+1)**2-1,3,device=dev))})
lrs={'means':1.6e-4*scene_scale,'scales':5e-3,'quats':1e-3,'opacities':5e-2,'sh0':2.5e-3,'shN':2.5e-3/20}
opts={k:torch.optim.Adam([{'params':params[k],'lr':lrs[k],'name':k}],eps=1e-15) for k in params}
sched=torch.optim.lr_scheduler.ExponentialLR(opts['means'],gamma=0.01**(1/ITERS))
# per-image exposure (affine color)
expo=torch.nn.Parameter(torch.zeros(n,3,4,device=dev)); expo_opt=torch.optim.Adam([expo],lr=1e-3)
strategy=DefaultStrategy(refine_stop_iter=int(ITERS*0.5),reset_every=3000,refine_every=100,absgrad=True,grow_grad2d=0.0008,verbose=False)
strategy.check_sanity(params,opts); state=strategy.initialize_state(scene_scale=scene_scale)
def ssim(a,b):
    k=torch.tensor(cv2.getGaussianKernel(11,1.5),dtype=torch.float32,device=dev); k=(k@k.T)[None,None].repeat(3,1,1,1)
    f=lambda x: F.conv2d(x,k,padding=5,groups=3)
    mu1,mu2=f(a),f(b); s11=f(a*a)-mu1**2; s22=f(b*b)-mu2**2; s12=f(a*b)-mu1*mu2
    return ((2*mu1*mu2+1e-4)*(2*s12+9e-4))/((mu1**2+mu2**2+1e-4)*(s11+s22+9e-4))
def render(i,sh_deg):
    colors=torch.cat([params['sh0'],params['shN']],1)
    out,alpha,info=rasterization(params['means'],params['quats'],torch.exp(params['scales']),torch.sigmoid(params['opacities']),colors,
        w2c[i:i+1],K[None],W,H,sh_degree=sh_deg,packed=False,absgrad=True,near_plane=0.05,far_plane=60,render_mode='RGB+ED')
    return out[0],info
t0=time.time()
for step in range(ITERS):
    i=train[np.random.randint(len(train))]
    out4,info=render(i,min(SHD,step//1000)); out=out4[...,:3]; dep=out4[...,3]
    A=torch.eye(3,4,device=dev)+expo[i]; out=out@A[:,:3].T+A[:,3]
    gt=imgs[i].float()/255; m=masks[i][...,None].float()
    strategy.step_pre_backward(params,opts,state,step,info)
    l1=((out-gt).abs()*m).sum()/(m.sum()*3)
    ss=(ssim(out.permute(2,0,1)[None],gt.permute(2,0,1)[None])[0].permute(1,2,0)*m).sum()/(m.sum()*3)
    lu,lv,lz=lidar_depth(i); mk=masks[i][lv,lu]; lu,lv,lz=lu[mk],lv[mk],lz[mk]
    dl=((dep[lv,lu]-lz).abs()/lz).mean() if len(lz)>100 else torch.zeros((),device=dev)
    lam=0.5 if step<10000 else 0.2
    loss=0.8*l1+0.2*(1-ss)+lam*dl+1e-3*torch.sigmoid(params['opacities']).mean()
    loss.backward()
    strategy.step_post_backward(params,opts,state,step,info,packed=False)
    for o in opts.values(): o.step(); o.zero_grad(set_to_none=True)
    expo_opt.step(); expo_opt.zero_grad(); sched.step()
    if step%1000==0 or step==ITERS-1:
        with torch.no_grad():
            ps=[]
            for j in list(test)[::3]:
                o,_=render(j,SHD); o=o[...,:3]; A=torch.eye(3,4,device=dev)+expo[j-1 if j>0 else 1]; o=(o@A[:,:3].T+A[:,3]).clamp(0,1)
                g=imgs[j].float()/255; mm=masks[j][...,None]; mse=(((o-g)**2)*mm).sum()/(mm.sum()*3); ps.append(-10*math.log10(mse.item()))
        print(f'step {step} loss {loss.item():.4f} depthL1rel {dl.item():.3f} N {len(params["means"])} testPSNR {np.mean(ps):.2f} t {time.time()-t0:.0f}s',flush=True)
torch.save({k:v.detach().cpu() for k,v in params.items()},'gs/splat_params5.pt')
# preview renders of test views
with torch.no_grad():
    rows=[]
    for j in list(test)[5:50:9][:5]:
        o,_=render(j,SHD); o=(o[...,:3].clamp(0,1)*255).byte().cpu().numpy(); g=imgs[j].cpu().numpy()
        rows.append(np.hstack([cv2.resize(g,(324,432)),cv2.resize(o,(324,432))]))
    cv2.imwrite('gs_test_views5.jpg',cv2.cvtColor(np.vstack([np.hstack(rows[:3]) if len(rows)>=3 else rows[0]]),cv2.COLOR_RGB2BGR))
print('done')
