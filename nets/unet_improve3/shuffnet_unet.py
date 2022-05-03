'''以整个shuffnet为主干'''
from nets.unet_improve3.unet_parts import *
import torch
import torch.nn as nn
from nets.unet_improve3.shuffnetv2 import shufflenet_v2_x0_5

import  numpy as np
""" Full assembly of the parts to form the complete network """
'''
描述: 原unet第一层不变还是用doble卷2，3，4，5换成shufnet2 s=2的模块

'''
#第一种写法
class shuff_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(shuff_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #[64, 128, 256, 512, 1024]
        self.inc = DoubleConv(n_channels, 64)#输出通道64

        self.down1 = InvertedResidual(input_c=64,output_c=128,stride=2)#128
        self.down2 =InvertedResidual(input_c=128,output_c=256,stride=2)#96
        self.down3 =InvertedResidual(input_c=256,output_c=512,stride=2)#192
        factor = 2 if bilinear else 1#如果用一般的双线性插值的话就是2 用转置卷积的话就是1
        self.down4 = InvertedResidual(input_c=512,output_c=1024,stride=2) #1024

        self.up1 = Up(1024, 512, bilinear,first_up=True)
        self.up2 = Up(512, 256, bilinear,first_up=False)
        self.up3 = Up(256, 128, bilinear,first_up=False)
        self.up4 = Up(128, 64, bilinear,first_up=False)
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


#model te

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)



inp = torch.randn([1,3,512,512])
mod= shuff_UNet(n_channels=3,n_classes=19)
out= mod(inp)
print(out.shape)

