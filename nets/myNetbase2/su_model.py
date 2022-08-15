# from nets3.SUnet4_0.su_part import *
from nets.myNetbase2.su_part2 import *
from torch import  nn
from  nets.myNetbase2.shufflenet_d import shufflenet_v2_x1_0
import torch
from torchsummary import summary
from thop import profile

#shufflev2去掉最后的全连接 ，第一个卷积不让他下采样了 下采样4次 16倍

#shufflev2去掉最后的全连接 ，第一个卷积不让他下采样了 下采样4次 16倍
class SUNet(nn.Module):
    #用反卷积
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.backbon= shufflenet_v2_x1_0(pretrained=False)

        #[24, 116, 232, 464, 1024 （1.0的]
        # self.attention1 =SpatialAttention2(kernel_size=3) #空间
        self.attention2 = SpatialAttention2(kernel_size=3) #空间
        # self.attention3 = ECAAttention(channel=232)#通道/空间
        # self.attention4 = ECAAttention(channel=464)#通道/空间
        self.attention4  = sa_layer(channel=232,groups=4)

        factor = 2 if bilinear else 1#如果用一般的双线性插值的话就是2 用转置卷积的话就是1

        self.up1 = Up(464,232, bilinear=False)
        self.up2 = Up(232, 116, bilinear)
        # #这里的x出来和x1差了4被 ：1.直接反卷积上采样4倍，2.先用双线性上采样2倍再和之前一样的操作
        self.bili = Upbi(in_channels=116,out_channels=116,bilinear=True)
        self.up3 = Up2(116, 24,doublein=48,mid=24, bilinear=False)
        self.outc = OutConv(24, n_classes)

    def forward(self, input):

        #24(256),24(128)  116(64), 232(32),
        x1, x2, x3, x4 = self.backbon(input)

        x3 =self.attention2(x3)

        x = self.attention4(x4)

        # x =   self.up1(x5,x4)
        x= self.up2(x,x3)
        x = self.bili(x)
        x = self.up3(x, x1)
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
# #
# input = torch.rand([1,3,256,256])
# moed = SUNet(n_channels=3,n_classes=6)
# out = moed(input)
# print(out.shape)

# model= SUNet(n_channels=3,n_classes=6)
# import  numpy as np
# model_path = r"F:\MSegmentation\pretrain\shufflenetv2_x1.pth"
# device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_dict = model.state_dict()
# pretrained_dict = torch.load(model_path, map_location=device)
# load_key, no_load_key, temp_dict = [], [], {}
# for k, v in pretrained_dict.items():
#     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#         temp_dict[k] = v
#         load_key.append(k)
#     else:
#         no_load_key.append(k)
# model_dict.update(temp_dict)
# model.load_state_dict(model_dict)
# print("加载成功")