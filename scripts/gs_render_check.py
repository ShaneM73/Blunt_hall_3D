import torch, numpy as np, cv2, sys, json
from gsplat import rasterization
p=torch.load(sys.argv[1]); dev='cuda'
for k in p: p[k]=p[k].to(dev)
TH=float(sys.argv[5]) if len(sys.argv)>5 else 0
keep=torch.sigmoid(p['opacities'])>TH
for k in p: p[k]=p[k][keep]
print('kept',keep.sum().item(),'of',len(keep))
eye=np.array(json.loads(sys.argv[2])); tgt=np.array(json.loads(sys.argv[3])); up=np.array([0,0,1.])
f=tgt-eye; f/=np.linalg.norm(f); r=np.cross(f,up); r/=np.linalg.norm(r); u=np.cross(r,f)
R=np.stack([r,-u,f]); w2c=np.eye(4); w2c[:3,:3]=R; w2c[:3,3]=-R@eye
W,H=1000,640; fy=H/2/np.tan(np.deg2rad(50)/2); K=torch.tensor([[fy,0,W/2],[0,fy,H/2],[0,0,1]],dtype=torch.float32,device=dev)
sh=p['shN'].shape[1]; deg=int(np.sqrt(sh+1))-1
im,_,_=rasterization(p['means'],p['quats'],torch.exp(p['scales']),torch.sigmoid(p['opacities']),torch.cat([p['sh0'],p['shN']],1),
   torch.tensor(w2c,dtype=torch.float32,device=dev)[None],K[None],W,H,sh_degree=deg,near_plane=0.05)
cv2.imwrite(sys.argv[4],cv2.cvtColor((im[0].clamp(0,1)*255).byte().cpu().numpy(),cv2.COLOR_RGB2BGR))
