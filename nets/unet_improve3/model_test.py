
import torch
from torch import  nn
from nets.unet_improve1 import unet_model


inp = torch.randn(1,3,512,512)
# model = unet_model.SEAttention(channel=64)
# model= unet_model.ECAAttention(channel=64)
# model= unet_model.cbam_block(channel=64)
# model= unet_model.CBAM_ECAAttention(channel=64)
#
# out = model(inp)
# print(out.shape)

from nets.unet_improve2.shuffnetv2_my import shufflenet_v2_x0_5

# model =shufflenet_v2_x0_5(num_classes=1000)
# model.__delattr__('fc')

# class mymoedel(nn.Module):
#     def __init__(self):
#         super(mymoedel, self).__init__()
#         base= shufflenet_v2_x0_5(num_classes=1000)
#         self.backbone =base
#         print(self.backbone)
#         self. class_t = nn.Conv2d(1024,19,kernel_size=1)
#     def forward(self,x):
#         ou = self.backbone.conv1(x)
#         print("d")
#         out= self.class_t(ou)
#         return out
#
# input =torch.randn([1,3,224,224])
# mode = mymoedel()
# out =mode(input)
#
# print(out.shape)

model = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1,groups=3)
input = torch.randn([1,3,512,512])
out =model(input)
print(out.shape)