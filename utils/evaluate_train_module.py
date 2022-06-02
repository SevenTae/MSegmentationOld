import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn

'''以前的版本全部作废 之前miou的计算方法错了'''
'''只是个模板'''

def evaluateloss(net, dataloader, device,numclass=20, ignoreindex=100):

    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation loss  round', unit='batch',
                      leave=False):  # 迭代完所有的验证集，输出整个验证集的loss

        image, mask_true = batch['image'], batch['label']
        # mask_true =mask_true-1 #
        # move images and labelss to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if numclass == 1:  # 类别是1的时候 ，因为我们跑的是city所以不用管
                pass
            else:  # 计算验证集的损失
                #
                criterion = nn.CrossEntropyLoss(ignore_index=ignoreindex)

                loss = criterion(mask_pred, mask_true)  # 计算完成一个batch的了
        val_loss += loss.item()


    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return val_loss
    # print("valloss(total val):", val_loss / num_val_batches)
    return val_loss / num_val_batches  # 然后再区batch的平均





import utils.utils_iou_miou as  utils


#我有个问题他这怎么计算带有ignor的标签
def evalue_iou_miou(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 100, header):
            image, target = batch['image'], batch['label']
            image = image.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.long)
            # ！！注意有的数据集标签可能从1开开始 要把它弄成从0
            model = model.to(device)
            # target = target - 1
            output = model(image)

            confmat.update(target.flatten(), output.argmax(1).flatten())

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.compute()
    return confmat.re_zhib()  #返回acc_global, acc, iu,miou