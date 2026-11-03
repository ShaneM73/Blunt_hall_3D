import torch, torchvision, cv2, numpy as np, json, os
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
meta=json.load(open('gs/cameras.json')); os.makedirs('gs/masks',exist_ok=True)
# valid-area mask per camera
from calib import CAL
Wo,Ho=meta['W'],meta['H']; Kn=np.array(meta['K'])
valid={}
for cam in (1,2,3):
    K=np.array([[CAL[cam]['K'][0],0,CAL[cam]['K'][2]],[0,CAL[cam]['K'][1],CAL[cam]['K'][3]],[0,0,1]])
    m1,m2=cv2.fisheye.initUndistortRectifyMap(K,np.array(CAL[cam]['D']),np.eye(3),Kn,(Wo,Ho),cv2.CV_32FC1)
    v=((m1>=1)&(m1<1943)&(m2>=1)&(m2<2591)).astype(np.uint8); valid[cam]=cv2.erode(v,np.ones((9,9),np.uint8))
model=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).cuda().eval()
npers=0
with torch.no_grad():
    for i in range(0,len(meta['frames']),16):
        fr=meta['frames'][i:i+16]
        ims=[torch.tensor(cv2.cvtColor(cv2.imread('gs/images/'+f['file']),cv2.COLOR_BGR2RGB)).permute(2,0,1).float().cuda()/255 for f in fr]
        outs=model(ims)
        for f,o in zip(fr,outs):
            keep=(o['labels']==1)&(o['scores']>0.5)
            pm=(o['masks'][keep,0]>0.3).any(0).cpu().numpy().astype(np.uint8) if keep.any() else np.zeros((Ho,Wo),np.uint8)
            if keep.any(): npers+=1; pm=cv2.dilate(pm,np.ones((25,25),np.uint8))
            cv2.imwrite('gs/masks/'+f['file'].replace('.jpg','.png'),(valid[f['cam']]*(1-pm))*255)
print('images with people',npers,'of',len(meta['frames']))
