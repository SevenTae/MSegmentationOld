'''以整个shuffnet为主干'''
from nets.unet_improve2.unet_parts import *
import torch
import torch.nn as nn
from nets.unet_improve2.shuffnetv2 import shufflenet_v2_x0_5
from nets.unet_improve2.shuffnetv2_my import shufflenet_v2_x0_5
import  numpy as np
""" Full assembly of the parts to form the complete network """

#第一种写法
# class shuff_UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False,pretrain=True):
#         super(shuff_UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         pretrain_path =r"D:\MSegmentation\pretrain\shufflenetv2_x0.5.pth"
#         backbone= shufflenet_v2_x0_5()
#         if pretrain:
#             backbone.load_state_dict(torch.load(pretrain_path))
#             print("权重已加载")
#         else:
#             print("未使用预训练权重")
#         backbone.__delattr__('fc')  # 去掉最后的fc
#
#         #[24, 48, 96, 192, 1024]
#         self.inc = backbone.conv1 #输出通道24
#         self.down1 = backbone.stage2#48
#         self.down2 = backbone.stage3#96
#         self.down3 =backbone.stage4#192
#         factor = 2 if bilinear else 1#如果用一般的双线性插值的话就是2 用转置卷积的话就是1
#         self.down4 = backbone.conv5 #1024
#
#         self.up1 = Up(1024, 192, bilinear,first_up=True)
#         self.up2 = Up(192, 96, bilinear,first_up=False)
#         self.up3 = Up(96, 48, bilinear,first_up=False)
#         self.up4 = Up(48, 24, bilinear,first_up=False)
#         self.up5 = Up_last(in_channels=24,out_channels=24)
#         self.outc = OutConv(24, n_classes)
#
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         # x5 = self.attention(x5)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         #到这里只完成了上采样到原来的二分之一还得再上一次
#         x = self.up5(x)
#         logits = self.outc(x)
#         return logits

#第二种写法
class shuff_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,pretrain=True):
        super(shuff_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        pretrain_path =r"D:\MSegmentation\pretrain\shufflenetv2_x0.5.pth"
        self.backbone= shufflenet_v2_x0_5()
        if pretrain:
            model_dict = self.backbone.state_dict()
            pretrained_dict = torch.load(pretrain_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict)
            print("权重已加载")
        else:
            print("未使用预训练权重")


        #[24, 48, 96, 192, 1024]
         #输出通道24

        self.up1 = Up(1024, 192, bilinear,first_up=True)
        self.up2 = Up(192, 96, bilinear,first_up=False)
        self.up3 = Up(96, 48, bilinear,first_up=False)
        self.up4 = Up(48, 24, bilinear,first_up=False)
        self.up5 = Up_last(in_channels=24,out_channels=24)
        self.outc = OutConv(24, n_classes)


    def forward(self, x):
        [feat1, feat2, feat3, feat4, feat5] =self.backbone.forward(x)
        x1 = self.backbone
        x2 = self.backbone
        x3 = self.backbone
        x4 = self.backbone
        x5 = self.backbone
        # x5 = self.attention(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #到这里只完成了上采样到原来的二分之一还得再上一次
        x = self.up5(x)
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


# backbone =shufflenet_v2_x0_5(num_classes=1000)
# backbone.__delattr__('fc') #去掉最后的fc
# # print(backbone)
# # down1 =nn.Sequential(
# #   backbone.conv1,
# #   backbone.maxpool
# # )
# down1 = backbone.conv1
# print(down1)
# down2 =backbone.stage2
# print(down2)
# down3=backbone.stage3
# print(down3)
# down4=backbone.stage4
# print(down4)
# down5=backbone.conv5
# print(down5)
#
inp = torch.randn([1,3,512,512])
mod= shuff_UNet(n_channels=3,n_classes=19)
out= mod(inp)
print(out.shape)
#
# # inp = torch.tensor([[1,2,3,4],[5,4,6,5]])
# # print(inp)
# # out = nn.Identity()(inp)
# # print(out)
# pretrain = r"D:\MSegmentation\pretrain\shufflenetv2_x0.5.pth"
# mod= shuff_UNet(n_channels=3,n_classes=19,pretrain=True)
# out = mod(inp)


