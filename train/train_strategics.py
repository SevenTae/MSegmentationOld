'''一些训练策略的学习'''
import torch
from torch import nn
import numpy as np
import torch.nn.functional  as F
import  torch.optim as optim
from nets.unet.unet_model import UNet
net = UNet(n_channels=3,n_classes=20)
'''优化器的选择'''
init_lr = 1e-3
'''1.adam:
常用参数：
Adam（params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0)
betas：平滑常数β1和β2;epsilon :加在分母上防止除0;一般这三个默认
weight_decay的作用是L2正则化，和Adam并无直接关系。

'''
optimizer = optim.Adam(net.parameters(), lr=init_lr)
'''2.sgd
常用参数
SGD(param,lr=<objectobject>,momentum=0,dampening=0,weight_decay=0,nesterov=False)
lr:学习率，默认值为1e*-3
momentum:动量因子，用于动量梯度的下降算法，默认为0
dampening:抑制因子，用于动量算法，默认为0
weight_decay:权值衰减系数，L2参数，默认为0
'''
optimizer = optim.SGD(net.parameters(),lr=init_lr,momentum=0,weight_decay=0)

'''学习率调整策略'''
#cr:http://t.zoukankan.com/jfdwd-p-11241201.html
'''
1.StepLR
等间隔调整学习率 StepLR
等间隔调整学习率，调整倍数为 gamma 倍，
调整间隔为 step_size。间隔单位是step。需要注意的是， step 通常是指 epoch，不要弄成 iteration 了,貌似在iterate里边也是按epoch来的

step_size(int)- 学习率下降间隔数，若为 30，则会在 30、 60、 90…个 step 时，将学习率调整为 lr*gamma。
gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
last_epoch(int)- 上一个 epoch 数，这个变量用来指示学习率是否需要调整。当last_epoch 符合设定的间隔时，就会对学习率进行调整。当为-1 时，学习率设置为初始值。
'''
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.98)
'''2.
CosineAnnealingLR
 余弦退火调整学习率 CosineAnnealingLR
以余弦函数为周期，并在每个周期最大值时重新设置学习率。
以初始学习率为最大学习率，以 2∗Tmax 2*Tmax2∗Tmax 为周期，在一个周期内先下降，后上升。 
T_max(int)- 一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率。T_max是周期的四分之一
eta_min(float)- 最小学习率，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0。
last_epoch （int）：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始。
'''
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1) #这是最大最小，
#和他比较相似的一个是余弦重启，上一个是下降到最小后余弦的缓慢上到最大，这个是余弦下降到最小后没有缓冲直接恢复
scheduler=optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1,eta_min=0) #这个是余弦，当经过xx后学习率又恢复到原来初始的学习率
#cr：http://t.csdn.cn/4Tq1W
#T_0就是初始restart的epoch数目，T_mult就是重启之后因子，默认是1。我觉得可以这样理解，每个restart后，T_0 = T_0 * T_mult。
#当T_mult设置为2时，当epoch=5时重启依次，下一次T_0 = T_0 * T_mul此时T_0等于10，在第16次重启，下一阶段，T_0 = T_0 * T_mult 此时T_0等于20再20个epcoh重启。所以曲线重启越来越缓慢，依次在第5，5+5*2=15，15+10*2=35，35+20 * 2=75次时重启。


'''3.
从别人复现的unet看到的
cr：http://t.csdn.cn/4ooFe
ReduceLROnPlateau：根据测试指标调整学习率
常用的参数：
mode：'min' ,'max'.'min’模式检测metric是否不再减小，'max’模式检测metric是否不再增大；
patience:不再减小（或增大）的累计次数；

'''
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
#注意这个在后边执行.step的时候要scheduler.step(val_score) 把需要监督的指标放进去







'''固定一下吧：
优化器选择adam 学习率选择step以每个epoch去下调整一次
'''
'''注意用step调整学习率的时候是按照epoch！！'''
mydam =optim.Adam(net.parameters(),lr=0.001)
le_sche = optim.lr_scheduler.StepLR(mydam,step_size=5,gamma=0.99,last_epoch=-1)
epoch = 20
itera = 5
for i in range(epoch):
    for j in range(itera):
        mydam.zero_grad()
        mydam.step()
        print("第%d个itera的学习率：%f" % (j+1, mydam.param_groups[0]['lr']))
        le_sche.step()
    print("第%d个epoch的学习率：%f" % (i + 1, mydam.param_groups[0]['lr'])) #到这里学习率已经调整了
