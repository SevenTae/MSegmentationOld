import torch
from PIL import Image
import numpy as np
import cv2 as cv

import os
rotpath = r"D:\MSegmentation\data\VOCdevkit\VOC2012\SegmentationClass"
imglsit = os.listdir(rotpath)

for  i in imglsit:
    imgpath = os.path.join(rotpath,i)
    name = i.split(".")[0]
    img = Image.open(imgpath)
    imgarr = np.array(img)
    print(name,":",np.unique(imgarr))

