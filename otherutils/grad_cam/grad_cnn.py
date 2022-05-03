

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
import os
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms


'''废了废了'''
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


device      = torch.device('cpu')


target_layers =[model.vgg.features]


# ----------数据处理区域--------------#
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# load image
img_path = "bike.jpeg"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path).convert('RGB')
img=img.resize((512,512))
img = np.array(img, dtype=np.uint8)
# [C, H, W]
img_tensor = data_transform(img)
# expand batch dimension
# [C, H, W] -> [N, C, H, W]
input_tensor = torch.unsqueeze(img_tensor, dim=0)
# ----------数据处理区域--------------#


cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)  # 把你的模型 taget_layer传进来 #这玩意当黑盒知道怎么用算了
target_category = 1  # #需要指定一下你感兴趣的类别
# target_category = 254  # pug, pug-dog

grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

grayscale_cam = grayscale_cam[0, :]  # 0就是取你第一张图 如果你有多个batch的话
visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 画图展示
                                  grayscale_cam,
                                  use_rgb=True)
plt.imshow(visualization)
plt.show()

