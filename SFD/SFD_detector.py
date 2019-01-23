# Runs SFD for an image and plots detections
# https://github.com/clcarwin/SFD_pytorch
# Implementation of Single Shot Scale-invariant Face Detector, ICCV, 2017 (based on SSD)

from SFD_pytorch import net_s3fd, detect
from SFD_pytorch.bbox import *
import torch
import cv2

filename = 'test.jpg'
input_size = 512 # 640 # Multiscale approach, can try different sizes
net ='s3fd'
model = "SFD_pytorch/data/s3fd_convert.pth"
threshold = 0
use_cuda = torch.cuda.is_available()

def resize_image(img, size):
    h, w = img.shape[0], img.shape[1]
    ar = float(w)/h
    if h < w:
        new_h = size
        new_w = int(size * ar)
    else:
        new_w = size
        new_h = int(size / ar)
    img = cv2.resize(img, (new_w, new_h))
    return img

img = cv2.imread(filename)
img = resize_image(img, input_size)
net = getattr(net_s3fd,net)()
net.load_state_dict(torch.load(model))
net.cuda()
net.eval()

bboxlist = detect.detect(net,img)
keep = nms(bboxlist,0.3)
bboxlist = bboxlist[keep,:]

# Draw result
for b in bboxlist:
    x1,y1,x2,y2,s = b
    if s < threshold: continue
    cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
cv2.imwrite(filename.split('.')[-2].split('/')[-1] + '_' + model.split('/')[-1].strip('.pth') + '_predictions.jpg',img)
cv2.imshow("img", img)
k = cv2.waitKey(0) # 0==wait forever