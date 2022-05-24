import torch
import time
from torch import nn
import torchvision.models as models
from nets2.unet_dds.unet_model import UNet
inputd = torch.randn([1,3,224,224],dtype=torch.float32)
model = UNet(n_channels=3,n_classes=20)
cuda =True

'''废了废了 待完善'''


if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


input = inputd.to(device)
model = model.to(device)
#推理单个（未进行cuda预热）
# model.eval()
# time0=time.time()
# with torch.no_grad():
    # output = model(input)
# tim1=time.time()
# print("推理时间为:",tim1-time0)

#批量推理
model.eval()
tim3 = time.time()
with torch.no_grad():
    for i in range(1000):
        out=model(input)
tim4 = time.time()
print("批量推理的时间:",tim4-tim3)

'''
cr:http://t.csdn.cn/0Pd3P
需要克服GPU异步执行和GPU预热两个问题，
下面例子使用 Efficient-net-b0，在进行任何时间测量之前，
我们通过网络运行一些虚拟示例来进行“GPU 预热”。
这将自动初始化 GPU 并防止它在我们测量时间时进入省电模式。
接下来，我们使用 tr.cuda.event 来测量 GPU 上的时间。
在这里使用 torch.cuda.synchronize() 至关重要。
这行代码执行主机和设备（即GPU和CPU）之间的同步，
因此只有在GPU上运行的进程完成后才会进行时间记录。这克服了不同步执行的问题。
'''

