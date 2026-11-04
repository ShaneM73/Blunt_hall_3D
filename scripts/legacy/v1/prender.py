import numpy as np, cv2
def look(eye,tgt,up=(0,0,1)):
    eye,tgt,up=map(np.asarray,(eye,tgt,up)); f=tgt-eye; f/=np.linalg.norm(f); r=np.cross(f,up); r/=np.linalg.norm(r); u=np.cross(r,f)
    return np.stack([r,-u,f]),eye
def render(P,C,eye,tgt,W=1400,H=900,fov=70,rad=1,up=(0,0,1)):
    R,e=look(eye,tgt,up); Q=(P-e)@R.T; m=Q[:,2]>0.1; Q=Q[m]; C=C[m]
    f=W/2/np.tan(np.deg2rad(fov)/2); u=(f*Q[:,0]/Q[:,2]+W/2).astype(int); v=(f*Q[:,1]/Q[:,2]+H/2).astype(int)
    img=np.full((H,W,3),255,np.uint8); zb=np.full((H,W),np.inf)
    o=np.argsort(-Q[:,2]); u,v,z,C=u[o],v[o],Q[o,2],C[o]
    for du in range(-rad,rad+1):
        for dv in range(-rad,rad+1):
            uu,vv=u+du,v+dv; ok=(uu>=0)&(uu<W)&(vv>=0)&(vv<H); img[vv[ok],uu[ok]]=C[ok]  # far-to-near painter
    return img
