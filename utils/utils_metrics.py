from torch import nn
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.miou import SegmentationMetric
import tqdm
import torch
'''在测试集上的评价'''

def evaluatemiou(net, dataloader, device, num_classes=20):
    net.eval()
    num_val_batches = len(dataloader)
    miou_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation miou round', unit='batch',
                      leave=False):  # 迭代玩所有的验证集累积所有batch的miou 最后除以验证集的batch的长度
        image, mask_true = batch['image'], batch['label']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:  # 类别是1的时候 ，因为我们跑的是city所以不用管
                pass
            else:  # 多类别的时候，计算miou
                # 得到输出后不能直接用，先softmax然后argmax
                mask_pred = nn.Softmax(dim=1)(mask_pred).argmax(dim=1)
                ignore_labels = [100]  # 改成20类后就没有ignore了
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
    print("miou:", miou_score / num_val_batches)
    return miou_score / num_val_batches  # 然后再区batch的平均

def evaluateiou(net, dataloader, device, num_classes=20):
    net.eval()
    num_val_batches = len(dataloader)
    iou_score = []

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation miou round', unit='batch',
                      leave=False):  # 迭代玩所有的验证集累积所有batch的miou 最后除以验证集的batch的长度
        image, mask_true = batch['image'], batch['label']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:  # 类别是1的时候 ，因为我们跑的是city所以不用管
                pass
            else:  # 多类别的时候，计算miou
                # 得到输出后不能直接用，先softmax然后argmax
                mask_pred = nn.Softmax(dim=1)(mask_pred).argmax(dim=1)
                ignore_labels = [100]  # 改成20类后就没有ignore了
                metric = SegmentationMetric(num_classes)  # 3表示有3个分类，有几个分类就填几, 0也是1个分类
                # 从GPU里拿出来
                mask_pred = mask_pred.cuda().data.cpu()
                mask_true = mask_true.cuda().data.cpu()
                hist = metric.addBatch(mask_pred, mask_true, ignore_labels)
                IoU = metric.IntersectionOverUnion()#它返回的莪是一个tensor的类别列表
                iou_score += IoU  # batch  的miou累计

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return iou_score
    print("iou:", iou_score / num_val_batches)
    return iou_score / num_val_batches  # 然后再区batch的平均

