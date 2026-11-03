import numpy as np, sys, itertools
sys.argv=[sys.argv[0],sys.argv[1],'14']
import photo_calib as pc
from scipy.spatial.transform import Rotation as Rot
cam=int(sys.argv[1]); a=np.array(pc.CAL[cam]['a']); base=np.deg2rad([-180,-90,-90]); t=np.array(pc.CAL[cam]['t'])
orders=['xyz','xzy','yxz','yzx','zxy','zyx','XYZ','XZY','YXZ','YZX','ZXY','ZYX']
cands={}
for o in orders:
    A=Rot.from_euler(o,a)
    cands[(o,'A')]=A
    for ob in ['xyz','zyx','XYZ','ZYX']:
        B=Rot.from_euler(ob,base)
        cands[(o,ob,'BA')]=B*A; cands[(o,ob,'AB')]=A*B; cands[(o,ob,'BtA')]=B.inv()*A; cands[(o,ob,'ABt')]=A*B.inv()
res=[]
for k,R in cands.items():
    for inv in (0,1):
        M=(R.inv() if inv else R).as_matrix()   # M = Rcb (body->cam)
        for ts in (1,-1):
            res.append((pc.cost(M,ts*t),k,inv,ts))
res.sort(key=lambda r:r[0])
for r in res[:12]: print(round(r[0],4),r[1:])
print('median',np.median([r[0] for r in res]))
