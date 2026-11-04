"""Export trained gsplat params to the 32-byte .splat format (pos f32x3, scale f32x3, rgba u8x4, quat u8x4 wxyz)."""
import torch, numpy as np, json, os, sys
OUT=os.path.join(os.path.dirname(__file__),'..','data')+'/'
p=torch.load('gs/splat_params.pt'); MAXN=int(sys.argv[1]) if len(sys.argv)>1 else 1_300_000
xyz=p['means'].numpy(); sc=np.exp(p['scales'].numpy()); op=1/(1+np.exp(-p['opacities'].numpy()))
q=p['quats'].numpy(); q/=np.linalg.norm(q,axis=1,keepdims=True)
rgb=np.clip(0.5+0.28209479*p['sh0'].numpy()[:,0,:],0,1)
keep=(op>0.03)&(sc.max(1)<1.0)
imp=(sc.prod(1)**(1/3))*op; keep&=imp>0
idx=np.flatnonzero(keep); idx=idx[np.argsort(-imp[idx])][:MAXN]
buf=np.zeros(len(idx),dtype=[('p','<f4',3),('s','<f4',3),('c','u1',4),('r','u1',4)])
buf['p']=xyz[idx]; buf['s']=sc[idx]; buf['c'][:,:3]=(rgb[idx]*255).round(); buf['c'][:,3]=(op[idx]*255).round()
buf['r']=np.clip(q[idx]*128+128,0,255).round()
buf.tofile(OUT+'scene.splat')
meta=json.load(open(OUT+'meta.json'))
cams=json.load(open('gs/cameras.json'))
meta['splat']=dict(file='data/scene.splat',n=int(len(idx)),views=len(cams['frames']),
  cam_pos=[-1.5,-2.8,0.3],cam_look=[6.0,-3.6,0.0],
  note=f'3D Gaussian Splatting (gsplat) trained on {len(cams["frames"])} undistorted fisheye photos with poses from the SLAM trajectory, initialized from the colored LiDAR points. People were masked out of the photos, and the splat depth was supervised by the people-free (carved) LiDAR cloud. Best viewed from inside the corridors near the walking path.')
json.dump(meta,open(OUT+'meta.json','w'),indent=1)
print('exported',len(idx),'of',len(xyz),os.path.getsize(OUT+'scene.splat')/1e6,'MB')
