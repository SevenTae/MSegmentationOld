'''
计算网络或者model的参数量和FLOPS（貌似是计算量）
'''
'''cr：https://blog.csdn.net/qq_35407318/article/details/109359006
      http://t.csdn.cn/prmSk

# '''
import torch
import torch
from torchsummary import summary
import torch
from torch import nn
# from nets3.SUnetbase.su_model import SUNet
from  nets.myNetbase.su_model import SUNet
import torch
from torchsummary import summary
from thop import profile
#
'''计算网络的参数量和计算量'''

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SUNet(n_channels=3, n_classes=6).to(device)
    input = torch.randn(1, 3, 256, 256).to(device)
    flops, params = profile(model, inputs=(input,))
    print("FLOPS:", flops)
    print("params:", params)

    print("Total FLOPS: %.2fM" % (flops / 1e6))
    print("Total params: %.2fM" % (params / 1e6))

    # torch.save(model.state_dict(),"test.pth")

    '''
    原unet：Total FLOPS: 219033.63M
            Total params: 31.04M （上采样用了反卷积的）


    这个：FLOPS: 7842267136.0
                params: 567811.0
                Total FLOPS: 7842.27M
                Total params: 0.57M

    '''
