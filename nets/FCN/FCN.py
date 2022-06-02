import torch
from torch import nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(np.array(weight))


# 加载预训练的resnet34网络
model_root = "./model/resnet34-333f7ec4.pth"
pretrained_net = models.resnet34(pretrained=False)
# pre = torch.load(model_root)
# pretrained_net.load_state_dict(pre)
# 分类的总数
num_classes = 19



class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()

        # 第一段，通道数为128，输出特征图尺寸为28*28
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        # 第二段，通道数为256，输出特征图尺寸为14*14
        self.stage2 = list(pretrained_net.children())[-4]
        # 第三段，通道数为512，输出特征图尺寸为7*7
        self.stage3 = list(pretrained_net.children())[-3]

        # 三个1*1的卷积操作，各个通道信息融合
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        # 将特征图尺寸放大八倍
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel
        # 这是放大了两倍，下同
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 224/8 = 28

        x = self.stage2(x)
        s2 = x  # 224/16 = 14

        x = self.stage3(x)
        s3 = x  # 224/32 = 7

        s3 = self.scores1(s3)  # 将各通道信息融合
        s3 = self.upsample_2x(s3)  # 上采样
        s2 = self.scores2(s2)
        s2 = s2 + s3  # 14*14

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)  # 上采样，变成28*28
        s = s1 + s2  # 28*28

        s = self.upsample_8x(s2)  # 放大八倍，变成224*224
        return s  # 返回特征图

#
# inp = torch.randn([1,3,512,512])
# mod =fcn(num_classes=19)
# out= mod(inp)
# print(out.shape)