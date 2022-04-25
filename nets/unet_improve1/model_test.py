
import torch
from torch import  nn
from nets.unet_improve1 import unet_model


inp = torch.randn(1,64,32,32)
# model = unet_model.SEAttention(channel=64)
# model= unet_model.ECAAttention(channel=64)
# model= unet_model.cbam_block(channel=64)
model= unet_model.CBAM_ECAAttention(channel=64)

out = model(inp)
print(out.shape)