import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.miou import SegmentationMetric


#这个evalue是原版
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['label']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:

                SegmentationMetric.meanIntersectionOverUnion()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


'''上边这个evla貌似不太行的样子。得自己写一个miou的评估'''


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

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return miou_score
    print("miou:", miou_score / num_val_batches)
    return miou_score / num_val_batches  # 然后再区batch的平均


def evaluateloss(net, dataloader, device, ignore_index=100):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation loss  round', unit='batch',
                      leave=False):  # 迭代完所有的验证集，输出整个验证集的loss

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
            else:  # 计算验证集的损失
                #
                criterion = nn.CrossEntropyLoss()

                loss = criterion(mask_pred, mask_true)  # 计算完成一个batch的了
        val_loss += loss.item()

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return val_loss
    print("valloss(total val):", val_loss / num_val_batches)
    return val_loss / num_val_batches  # 然后再区batch的平均



