import numpy as np, cv2, glob, os, json, sys
sys.path.insert(0,'.'); from calib import CAL, pose_at
from concurrent.futures import ThreadPoolExecutor
OUT='gs/images'; os.makedirs(OUT,exist_ok=True)
Wo,Ho,f=972,1296,450.0
Knew=np.array([[f,0,Wo/2],[0,f,Ho/2],[0,0,1]])
frames=[]
for cam in (1,2,3):
    K=np.array([[CAL[cam]['K'][0],0,CAL[cam]['K'][2]],[0,CAL[cam]['K'][1],CAL[cam]['K'][3]],[0,0,1]]); D=np.array(CAL[cam]['D'])
    m1,m2=cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),Knew,(Wo,Ho),cv2.CV_16SC2)
    Rcb=np.load(f'Rcb_nom_cam{cam}.npy'); tb=np.array(CAL[cam]['t'])
    fns=sorted(glob.glob(f'img/Camera{cam}/*.jpg'))
    def work(fn):
        t=float(os.path.basename(fn)[:-4]); Rwb,twb=pose_at(t)
        Rwc=Rwb@Rcb.T; twc=twb+Rwb@tb
        c2w=np.eye(4); c2w[:3,:3]=Rwc; c2w[:3,3]=twc
        name=f'c{cam}_{t:.6f}.jpg'
        cv2.imwrite(os.path.join(OUT,name),cv2.remap(cv2.imread(fn),m1,m2,cv2.INTER_AREA if False else cv2.INTER_LINEAR),[cv2.IMWRITE_JPEG_QUALITY,95])
        return dict(file=name,cam=cam,t=t,c2w=c2w.tolist())
    with ThreadPoolExecutor(32) as ex: frames+=list(ex.map(work,fns))
json.dump(dict(W=Wo,H=Ho,K=Knew.tolist(),frames=frames),open('gs/cameras.json','w'))
print(len(frames))
