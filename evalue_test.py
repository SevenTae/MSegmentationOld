##在测试集上进行评估miou 版本1 和验证集的计算miou一样
#

from nets.unet.unet_model import UNet
from utils.utils_metrics import evaluatemiou,evaluateiou
from torch.utils.data import DataLoader
from dataloaders.datasets import cityscapesmy
import argparse
import torch
import tqdm
'''注意啊注意city数据集没有测试集！！所以说测试还是得在val上'''
from nets.unet.unet_model import UNet#网络
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#1.准备数据
parser2 = argparse.ArgumentParser()
args2 = parser2.parse_args()
args2.base_size = 256  # 这玩意干啥的
args2.crop_size = 256
args2.resize = (512, 256)  #把它缩放成原图训练时候的大小

batch_size=2 #测试的时候batch直接等于图片综述
cityscapes_test = cityscapesmy.CityscapesSegmentation(args2, split='test')
test_loader=DataLoader(cityscapes_test,shuffle=False,num_workers=2,pin_memory=True,batch_size=batch_size)
n_test =cityscapes_test.__len__()
#2.搭建网络

model_path = ""
model_dict = torch.load(model_path)
model= UNet(n_channels=3,n_classes=20)
model.load_state_dict(model_dict)

#3.评估

miou= evaluatemiou(model, test_loader, device,num_classes=20)
iou = evaluateiou(model, test_loader, device,num_classes=20)

print("数据集总体的miou为：",miou)
print("iou为：",iou)

