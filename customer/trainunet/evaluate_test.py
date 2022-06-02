##在测试集上进行评估miou 版本1 和验证集的计算miou一样

from torch import nn
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.miou import SegmentationMetric
from  tqdm import  tqdm


from torch.utils.data import DataLoader
from dataloaders.datasets import cityscapesmy,customer,pascal_customer

import argparse
import torch
from  tqdm import tqdm


from nets.unet.unet_model import UNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#1.准备数据
parser2 = argparse.ArgumentParser()
args2 = parser2.parse_args()
args2.base_size = 256  # 这玩意干啥的
args2.crop_size = 256
args2.resize = (256, 256)  #把它缩放成原图训练时候的大小

batch_size=4#测试的时候batch直接等于图片综述
test = pascal_customer.Customer_VOCSegmentation(args2, split='test')
test_loader=DataLoader(test,shuffle=False,pin_memory=True,batch_size=batch_size)
n_test =test.__len__()
#2.搭建网络

model_path = r"F:\MSegmentation\customer\trainunet\checkpoints\best.pth"
model_dict = torch.load(model_path)
model= UNet(n_channels=3,n_classes=6)
model.load_state_dict(model_dict['net'])
print("权重加载")

#3.评估

def evaluateiou(net, dataloader, device, num_classes=20,ignoreindex=100):

    net.eval()
    num_val_batches = len(dataloader)
    iou_score=torch.zeros(num_classes)

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation iou round', unit='batch',
                      leave=False):  # 迭代玩所有的验证集累积所有batch的miou 最后除以验证集的batch的长度
        image, mask_true = batch['image'], batch['label']
        mask_true = mask_true - 1  # 这一条是
        # move images and labelss to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            # mask_pred = net(image)[0]  # 取他元组的第一个

            # convert to one-hot format
            if num_classes == 1:  # 类别是1的时候 ，因为我们跑的是city所以不用管
                pass
            else:  # 多类别的时候，计算miou
                # 得到输出后不能直接用，先softmax然后argmax
                mask_pred = nn.Softmax(dim=1)(mask_pred).argmax(dim=1)
                ignore_labels = [ignoreindex]  # 改成20类后就没有ignore了
                metric = SegmentationMetric(num_classes)  # 3表示有3个分类，有几个分类就填几, 0也是1个分类
                # 从GPU里拿出来
                mask_pred = mask_pred.cuda().data.cpu()
                mask_true = mask_true.cuda().data.cpu()
                hist = metric.addBatch(mask_pred, mask_true, ignore_labels)
                IoU = metric.IntersectionOverUnion()#它返回的莪是一个tensor的类别列表
                iou_score += IoU  # batch  的miou累计 # 这一个batch中的iou是怎么算的？一张图一张图算完的和再除以所以图片数还是，这个bacthsize的图片都弄成一个大矩阵？  #但是反正是这一个batch

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return iou_score
    # print("iou:", iou_score/num_val_batches )
    return iou_score/num_val_batches   # 然后再区batch的平均



def evaluatemiou(net, dataloader, device, num_classes=20,ignoreindex=100):

    net.eval()
    net.to(device)
    num_val_batches = len(dataloader)
    miou_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation miou round', unit='batch',
                      leave=False):  # 迭代玩所有的验证集累积所有batch的miou 最后除以验证集的batch的长度
        image, mask_true = batch['image'], batch['label']
        mask_true = mask_true - 1  # 这一条是
        # move images and labelss to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            # mask_pred = net(image)[0]  # 取他元组的第一个

            # convert to one-hot format
            if num_classes == 1:  # 类别是1的时候 ，因为我们跑的是city所以不用管
                pass
            else:  # 多类别的时候，计算miou
                # 得到输出后不能直接用，先softmax然后argmax
                mask_pred = nn.Softmax(dim=1)(mask_pred).argmax(dim=1)
                ignore_labels = [ignoreindex]  # 改成20类后就没有ignore了
                metric = SegmentationMetric(num_classes)  # 3表示有3个分类，有几个分类就填几, 0也是1个分类
                # 从GPU里拿出来
                mask_pred = mask_pred.cuda().data.cpu()
                mask_true = mask_true.cuda().data.cpu()
                hist = metric.addBatch(mask_pred, mask_true, ignore_labels)
                # IoU = metric.IntersectionOverUnion()
                mIoU = metric.meanIntersectionOverUnion()
                # print('hist is :\n', hist)
                # print('IoU is : ', IoU)
                miou_score += mIoU  # batch  的miou累计

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return miou_score
    # print("miou:", miou_score / num_val_batches)
    return miou_score / num_val_batches  # 然后再区batch的平均




miou= evaluatemiou(model, test_loader, device,num_classes=6)
iou = evaluateiou(model, test_loader, device,num_classes=6,)

print("数据集总体的miou为：",miou)
print("iou为：",iou)

#测评dice



