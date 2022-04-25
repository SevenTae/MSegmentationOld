'''
计算网络或者model的参数量和FLOPS（貌似是计算量）
'''
'''cr：https://blog.csdn.net/qq_35407318/article/details/109359006
      http://t.csdn.cn/prmSk

'''
import torch

import torch
from torchsummary import summary

import torch
from torch import  nn
from nets.unet_improve1 import unet_model


# if __name__ == "__main__":
#     # 需要使用device来指定网络在GPU还是CPU运行
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = unet_model.SEAttention(channel=64).to(device)
#     # model= unet_model.ECAAttention(channel=64)
#     # model= unet_model.cbam_block(channel=64)
#     summary(model, input_size=(64, 32, 32))



import torch
from torchsummary import summary
from thop import profile



if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = unet_model.CBAM_ECAAttention(channel=64).to(device)
    input = torch.randn(1, 64, 32, 32).to(device)
    flops, params = profile(model, inputs=(input,))
    print("FLOPS:",flops)
    print("params:",params)