import os
import numpy as np, cv2, glob, os, json, sys
sys.path.insert(0,'.'); from calib import CAL
from scipy.spatial.transform import Rotation as Rot, Slerp
from concurrent.futures import ThreadPoolExecutor
T=np.load('traj_refined.npy')
def pose_at(t):
    i=int(np.clip(np.searchsorted(T[:,0],t),1,len(T)-1)); a=(t-T[i-1,0])/(T[i,0]-T[i-1,0])
    R=Slerp([0,1],Rot.from_quat(T[[i-1,i],4:8][:,[1,2,3,0]]))([a])[0].as_matrix(); return R,(1-a)*T[i-1,1:4]+a*T[i,1:4]
Wo,Ho,f=1458,1944,675.0
Knew=np.array([[f,0,Wo/2],[0,f,Ho/2],[0,0,1]]); frames=[]
for cam in (1,2,3):
    K=np.array([[CAL[cam]['K'][0],0,CAL[cam]['K'][2]],[0,CAL[cam]['K'][1],CAL[cam]['K'][3]],[0,0,1]])
    m1,m2=cv2.fisheye.initUndistortRectifyMap(K,np.array(CAL[cam]['D']),np.eye(3),Knew,(Wo,Ho),cv2.CV_16SC2)
    Rcb=np.load(f'Rcb_nom_cam{cam}.npy'); tb=np.array(CAL[cam]['t'])
    def work(fn):
        t=float(os.path.basename(fn)[:-4]); Rwb,twb=pose_at(t)
        c2w=np.eye(4); c2w[:3,:3]=Rwb@Rcb.T; c2w[:3,3]=twb+Rwb@tb
        name=f'c{cam}_{t:.6f}.jpg'
        cv2.imwrite('gs_hq/images/'+name,cv2.remap(cv2.imread(fn),m1,m2,cv2.INTER_CUBIC),[cv2.IMWRITE_JPEG_QUALITY,95])
        mk=cv2.imread('gs/masks/'+name.replace('.jpg','.png'),0); cv2.imwrite('gs_hq/masks/'+name.replace('.jpg','.png'),cv2.resize(mk,(Wo,Ho),interpolation=cv2.INTER_NEAREST))
        return dict(file=name,cam=cam,t=t,c2w=c2w.tolist())
    with ThreadPoolExecutor(32) as ex: frames+=list(ex.map(work,sorted(glob.glob(f'img/Camera{cam}/*.jpg'))))
json.dump(dict(W=Wo,H=Ho,K=Knew.tolist(),frames=frames),open('gs_hq/cameras.json','w')); print(len(frames))
