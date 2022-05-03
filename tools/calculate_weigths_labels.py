import os
from tqdm import tqdm
import numpy as np
from dataloaders.mypath import Path

'''
cr：http://t.csdn.cn/C9pWg
计算类别权重，前提是你的数据个是是example的格式
'''
def calculate_weigths_labels( dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    # classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    # np.save(classes_weights_path, ret)

    return ret

#EXample
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import argparse
# parser = argparse.ArgumentParser()
# from dataloaders.datasets.customer import CustomerSegmentation
# args = parser.parse_args()
# args.base_size = 512#这玩意干啥的
# args.crop_size = 512
# args.resize=(999,960)
#
# customer_train = CustomerSegmentation(args, split='train')
# print(len(customer_train))
# dataloader = DataLoader(customer_train, batch_size=1, shuffle=False)
# num_classes = 2
# re = calculate_weigths_labels(dataloader,num_classes)
# print(re)
