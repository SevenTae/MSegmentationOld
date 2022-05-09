import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.evaluate_train import evaluatemiou,evaluateloss

from nets.bisegnet.bisenetv1 import BiSeNetV1
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

##cityscpes 我直接改成20类了19类原来的+1类背景（原来255位置的）
def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 2,
              learning_rate: float = 1e-3,
              save_checkpoint: bool = True,

              amp: bool = False):
    # 1. Create dataset
    #myself数据加载
    from dataloaders.datasets import cityscapesmy
    parser2 = argparse.ArgumentParser()
    args2 = parser2.parse_args()
    args2.base_size = 256  # 这玩意干啥的
    args2.crop_size = 256
    args2.resize = (512, 256)

    cityscapes_train = cityscapesmy.CityscapesSegmentation(args2, split='train')

    cityscapes_val = cityscapesmy.CityscapesSegmentation(args2, split='val')
    n_val =cityscapes_val.__len__()
    n_train =cityscapes_train.__len__()

    train_loader = DataLoader(cityscapes_train, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True)
    val_loader=DataLoader(cityscapes_val,shuffle=False,num_workers=2,pin_memory=True,batch_size=batch_size)



    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                   save_checkpoint=save_checkpoint,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}

        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate )
    # scheduler=optim.lr_scheduler.CosineAnnealingWarmRestarts #这个是遇险重启，当经过xx后学习率又恢复到原来初始的学习率
    # scheduler = optim.lr_scheduler.CosineAnnealingLR #这是最大最小，
    # scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.98)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score  #当网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能。所使用的类
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #amp：Automatic mixed precision，自动混合精度，可以在神经网络推理过程中，针对不同的层，采用不同的数据精度进行计算，从而实现节省显存和加快速度的目的。这里就不适用了，以后再看
    criterion = nn.CrossEntropyLoss()
    global_step = 0  #记录迭代的次数

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.aux_mode='train'
        net.train()
        epoch_loss = 0
        iter_loss=0 #每次迭代500次或者多少次输出一次loss
        iter_record = 500 #每500迭代输出一次loss
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:#batchsize=2 一共1487.5个batch
                images = batch['image']
                true_masks = batch['label']


                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)


                with torch.cuda.amp.autocast(enabled=amp):
                    out, out16, out32 = net(images)
                    loss = criterion(out,true_masks)+ criterion(out16,true_masks)+criterion(out32, true_masks)

                    #
                    '''1.loss 2.梯度清零，3.反向传播。backward 4optomizer更新.'''

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1#记录迭代次数
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,

                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})  # 他这个loss是按照batch输出的一个batch弄完后输出一个batch的

                #这是迭代xx次的平均loss（迭代一次=跑一个batch）
                iter_loss+=loss.item()
                if global_step % iter_record ==0:
                    logging.info(f'iter_{iter_record}的loss为{iter_loss/iter_record}!')

                # Evaluation round  #每迭代500个batch验证一次
                division_step = 500#(n_train // (10 * batch_size))#148
                if division_step > 0:
                    if global_step % division_step == 0:#没1000验证一次
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluatemiou(net, val_loader, device,args.classes)
                        val_loss =evaluateloss(net,val_loader,device)
                        scheduler.step(val_score)  #学习率是在这里监控miou的 ,#这是用那个监督策略的
                        # scheduler.step() #这个地方是按照迭代来调整学习率的

                        logging.info('Validation miou score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'train_iter_loss':iter_loss/iter_record,
                            'val_loss':val_loss,
                            'validation miou': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(out, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file') #有没有预训练。。
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images') #貌似没用
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=20, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = BiSeNetV1(n_classes=args.classes)


    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,


                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
