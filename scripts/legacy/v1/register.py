import os; RAW=os.environ.get("TVGO_RAW", os.path.expanduser("~/3DTemplehall"))
import numpy as np, sys, laspy; sys.path.insert(0,'.')
from read_tvgo import read_slam_bin
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
idx,p=read_slam_bin(RAW+'/SLAM_DATA/TMP_00000.bin')
T=np.load('slam_optimized_trajectory.npy')
print('time offset idx vs traj', np.abs(idx[:,1]-np.floor(T[:,0])).max())
cs=np.r_[0,np.cumsum(idx[:,0])]
def R(w,x,y,z):
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
N=len(p); out=np.empty((N,3),np.float64); gt=np.empty(N)
P=np.stack([p['x'],p['y'],p['z']],1).astype(np.float64)
for i in range(len(idx)):
    s=slice(cs[i],cs[i+1]); q=T[i,4:8]; q=q/np.linalg.norm(q)
    out[s]=P[s]@R(*q).T+T[i,1:4]; gt[s]=T[i,0]
r=np.linalg.norm(P,axis=1); keep=r>0.3   # drop operator/near-sensor hits
out,gt=out[keep],gt[keep]; inten=p['a'][keep]; ring=p['b'][keep]
print('kept',keep.sum(),'extent',out.min(0).round(1),out.max(0).round(1))
h=laspy.LasHeader(point_format=1,version='1.4'); h.offsets=out.min(0); h.scales=[0.001]*3
las=laspy.LasData(h); las.x,las.y,las.z=out.T; las.intensity=inten.astype(np.uint16)*256; las.gps_time=gt; las.point_source_id=ring
las.write('blunt_hall_slam.laz' if laspy.LazBackend.detect_available() else 'blunt_hall_slam.las')
sub=slice(None,None,4); x,y,z=out[sub].T
zf=np.percentile(z,2); print('floor~',zf)
m=(z>zf+0.3)&(z<zf+2.0)
fig,ax=plt.subplots(1,2,figsize=(20,10))
ax[0].hist2d(x[m],y[m],bins=1200,cmap='gray_r',vmax=60); ax[0].plot(T[:,1],T[:,2],'r-',lw=1); ax[0].set_aspect('equal'); ax[0].set_title('Plan view (slice 0.3–2 m above floor) + trajectory')
ax[1].hist2d(x,z,bins=1200,cmap='gray_r',vmax=100); ax[1].set_aspect('equal'); ax[1].set_title('Side view X–Z')
plt.tight_layout(); plt.savefig('blunt_hall_views.png',dpi=70)
