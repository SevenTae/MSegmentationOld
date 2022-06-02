from nets3.SUnet.su_part import *
from torch import  nn
import torch
from torchsummary import summary
# from thop import profile


#shufflev2去掉最后的全连接 ，第一个卷积不让他下采样了 下采样4次 16倍
class SUNet(nn.Module):
    #用反卷积
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.backbon= shufflenet_v2_x0_5(pretrained=False)
        # self.midchanel0.5 = [24, 24 48, 96, 192, ]
  #     #1.0[24, 116, 232, 464, 1024]
        factor = 2 if bilinear else 1#如果用一般的双线性插值的话就是2 用转置卷积的话就是1
        self.attention=cbam_block(channel=192)
        self.up1 = Up(192,96, bilinear=False)
        self.up2 = Up(96, 48, bilinear)
        self.up3 = Up(48, 24, bilinear)
        self.outc = OutConv(24, n_classes)

    def forward(self, input):
        x1, x2, x3, x4, x5 = self.backbon(input)
        x5 =self.attention(x5)
        x =   self.up1(x5,x4)
        x= self.up2(x,x3)
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

# input = torch.rand([1,3,512,512])
# moed = SUNet(n_channels=3,n_classes=19)
# out = moed(input)
# print(out.shape)

# model= SUNet(n_channels=3,n_classes=19)
# import  numpy as np
# model_path = "D:\MSegmentation\pretrain\shufflenetv2_x0.5.pth"
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