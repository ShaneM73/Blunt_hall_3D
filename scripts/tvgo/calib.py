import numpy as np, cv2, glob, os, itertools
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
CAL={1:dict(K=(1016.642079,1016.642079,1014.348859,1316.41759),D=(0.3965564782,0.1679324775,-0.1793566642,0.09789566746),a=(3.1352,0.00196143,-1.56158),t=(0.0390904,0.00723959,-0.052974)),
     2:dict(K=(1019.375183,1019.375183,970.3474868,1350.530619),D=(0.3946503416,0.1669601288,-0.1654274236,0.08572290486),a=(3.12487,0.991133,-1.57634),t=(0.0220678,0.00821497,-0.0222534)),
     3:dict(K=(1016.980694,1016.980694,955.4598244,1305.261835),D=(0.3953330955,0.1685326108,-0.1820437284,0.0993139611),a=(3.12214,-0.994001,-1.54939),t=(0.0174704,0.00758442,-0.0886481))}
T=np.load('slam_optimized_trajectory.npy') if os.path.exists('slam_optimized_trajectory.npy') else None   # legacy scripts only
def pose_at(t):
    i=np.searchsorted(T[:,0],t); i=np.clip(i,1,len(T)-1); a=(t-T[i-1,0])/(T[i,0]-T[i-1,0])
    from scipy.spatial.transform import Slerp
    q=T[[i-1,i],4:8][:,[1,2,3,0]]; R=Slerp([0,1],Rot.from_quat(q))([a])[0]
    return R.as_matrix(), (1-a)*T[i-1,1:4]+a*T[i,1:4]
def Kmat(c): fx,fy,cx,cy=CAL[c]['K']; return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
def cam_from_body(c, conv):
    order,inv,tsign=conv; R=Rot.from_euler(order,CAL[c]['a']).as_matrix(); t=np.array(CAL[c]['t'])
    # conv: R maps cam->body (inv=False) or body->cam (inv=True); t = camera position in body
    Rcb = R if inv else R.T
    return Rcb, t*tsign
def project(c, Pb, Rcb, tb):
    Pc=(Pb-tb)@Rcb.T; m=Pc[:,2]>0.2
    uv,_=cv2.fisheye.projectPoints(Pc[m].reshape(-1,1,3).astype(np.float64),np.zeros(3),np.zeros(3),Kmat(c),np.array(CAL[c]['D']))
    uv=uv.reshape(-1,2); ok=(uv[:,0]>=0)&(uv[:,0]<1944)&(uv[:,1]>=0)&(uv[:,1]<2592)
    idx=np.flatnonzero(m)[ok]; return idx, uv[ok], Pc[idx,2]
