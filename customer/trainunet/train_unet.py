import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
# from utils.evaluate_train import evaluatemiou, evaluateloss
from customer.trainunet.evaluate_train import evaluatemiou, evaluateloss

from nets.unet.unet_model import UNet,weights_init
import numpy  as np

'''WHDLD数据集'''

# 加了tensorboard记录
dir_checkpoint = Path('./checkpoints/')
tensorboarddir= Path('./TensorboardLog/')


##cityscpes 我直接改成20类了19类原来的+1类背景（原来255位置的）
def train_net(net,
              device,
              resume=False,
              isPretrain=False,
              epochs: int = 5,
              batch_size: int = 2,
              learning_rate: float = 1e-3,
              save_checkpoint: bool = True,
              ignoreindex: int = 100

              ):
    # 1. Create dataset
    # myself数据加载
    from dataloaders.datasets import cityscapesmy, pascal_customer
    parser2 = argparse.ArgumentParser()
    args2 = parser2.parse_args()
    args2.base_size = 256  # 这玩意干啥的
    args2.crop_size = 256  # 数据增强用的裁剪的尺寸大小
    args2.resize = (256, 256)  # 输入是w h(长和高)

    # train_d = cityscapesmy.CityscapesSegmentation(args2, split='train',isAugTrain=True)
    train_d = pascal_customer.Customer_VOCSegmentation(args2, split='train',isAug=False)
    val_d = pascal_customer.Customer_VOCSegmentation(args2, split='val')

    # val_d = cityscapesmy.CityscapesSegmentation(args2, split='val')
    n_val = val_d.__len__()
    n_train = train_d.__len__()

    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False,drop_last=True)
    val_loader = DataLoader(val_d, shuffle=True, num_workers=2, pin_memory=False, batch_size=batch_size)

    # (Initialize logging)
    experiment = wandb.init(project='unetWH', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  save_checkpoint=save_checkpoint
                                  ))

    # 初始化tensorboard
    Path(tensorboarddir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tensorboarddir)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}


    ''')
    net = net.to(device)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98, )
    criterion = nn.CrossEntropyLoss(ignore_index=ignoreindex)
    start_epoch = 0
    #是否有预训练
    if isPretrain:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(isPretrain, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        net.load_state_dict(model_dict)
        print("加载成功")
        logging.info(f'Model loaded from {isPretrain}')
    else:#没有预训练使用pyroch的一些权重初始化方法
        strweight='normal'
        weights_init(net,init_type=strweight)
        logging.info(f'没有预训练权重，{strweight}权重初始化完成')
    #是否使用断点训练
    if resume:
        path_checkpoint = resume  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        # 加载优化器参数
        start_epoch = checkpoint['epoch'] +1 # 设置开始的epoch
        scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler
    else:
        start_epoch= 0


    # 5. Begin training
    epochs_score = []  # 记录每个epoh的miou
    best_miou = 0.0  # 记录最好的那个
    for epoch in range(start_epoch, epochs + 1):
        current_miou = 0.0
        total_train_loss = 0
        net.train()
        print('Start Train')
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for iteration, batch in enumerate(train_loader):  # batchsize=2 一共1487.5个batch
                images = batch['image']
                true_masks = batch['label']
                true_masks =true_masks-1#这一条是针对有的8位彩色图索引从1开始
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                out = net(images)
                loss = criterion(out, true_masks)

                '''1.loss 2.梯度清零，3.反向传播。backward 4optomizer更新.'''

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update()  # 用来记录一次更新多少的
                total_train_loss += loss.item()
                #
                experiment.log({
                    'train batch loss': loss.item(),
                    'step': iteration + 1,
                    'epoch': epoch
                })

                pbar.set_postfix(**{'loss (batch)': loss.item()})

        print("Finish Train")

        print('start Validation')
        # Evaluation round  #每个epoch评估一次
        isevalue = True
        if isevalue == True:
            val_score = evaluatemiou(net, val_loader, device, args.classes, ignoreindex=ignoreindex)
            val_loss = evaluateloss(net, val_loader, device, numclass=args.classes, ignoreindex=ignoreindex)
            logging.info('Validation miou score: {}'.format(val_score))
            logging.info('Validation loss score: {}'.format(val_loss))
            current_miou = val_score
            epochs_score.append(val_score)
        print('Finish Validation')
        scheduler.step()  # 这个地方是按照迭代来调整学习率的
        # 一整个训练验证结束后记录一下信息
        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'train_total_loss': total_train_loss / iteration + 1,
            'val_loss': val_loss,
            'validation miou': val_score,
            'images': wandb.Image(images[0].cpu()),
            'masks': {
                'true': wandb.Image(true_masks[0].float().cpu()),
                'pred': wandb.Image(torch.softmax(out, dim=1).argmax(dim=1)[0].float().cpu()),
            },
            'epoch': epoch,
            'best_epoch_index': epochs_score.index(max(epochs_score))+1  # 记录验证集上最好的那个epoch

        })

        # tensorboard jilu
        writer.add_scalar("train_total_loss", total_train_loss / iteration + 1, epoch)
        writer.add_scalar("valloss", val_loss, epoch)
        writer.add_scalar("valmiou", val_score, epoch)
        # 保存最好的miou和最新的
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'lr_schedule': scheduler.state_dict()
            }
            torch.save(checkpoint, str(dir_checkpoint / 'last.pth'))  # 保存最新的
            # 保存最好的
            if current_miou >= best_miou:
                best_miou = current_miou
                torch.save(checkpoint, str(dir_checkpoint / 'best.pth'.format(best_miou)))
    writer.close()
    logging.info("训练完成")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default="F:/MSegmentation/pretrain/unet_carvana_scale1.0_epoch2.pth", help='Load model from a .pth file')  # 有没有预训练。。
    parser.add_argument('--ignore-index', '-i', type=int, dest='ignore_index', default=255,
                        help='ignore index defult 100')  # 有没有预训练。。
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--resume', '-r', type=str, default=False, help='is use Resume')
    parser.add_argument('--classes', '-c', type=int, default=6, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=args.classes)

    try:
        train_net(net=net,
                  resume=args.resume,
                  epochs=args.epochs,
                  isPretrain=args.load,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  ignoreindex=args.ignore_index
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
