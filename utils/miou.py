'''计算miou'''
"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
from torch import nn
import torch
import cv2
import numpy as np
__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean() # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean()# 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask] #这是干啥
        count = torch.bincount(label, minlength=self.numClass ** 2) #貌似是记录每一项出现的频次
        confusionMatrix = count.view(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    # mask返回的是一个布尔型数据
    # imgLabel = tensor([[0, 1, 255],
    #  					 [1, 1, 2]])
    # mask = tensor([[ True,  True, False],
    #                [ True,  True,  True]])

    # 利用mask只返回True对应的元素，用于计算


    def addBatch(self, imgPredict, imgLabel, ignore_labels):

        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))
'''
from nets.unet.unet_model import UNet
from PIL import  Image
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader,Dataset
import os
class mydata(Dataset):
    def __init__(self,img_root,label_root):
        self.imgpath=img_root
        self.labelpath = label_root
        self.img=os.listdir(img_root)
        self.label =os.listdir(label_root)
    def __getitem__(self, index):
        img= self.img[index]
        label =   self.label [index]
        pathi=os.path.join(self.imgpath,img)
        image= Image.open(pathi)
        pathl= os.path.join(self.labelpath,label)
        mask = Image.open(pathl)
        return image,mask
    def __len__(self):
        return len(self.img)
'''
#
# imgp=r"F:\Q3\BaseUNet\Pytorch-UNet\data\imgs"
# labep= r"F:\Q3\BaseUNet\Pytorch-UNet\data\masks"
# datam = mydata(imgp,labep)
# loads= DataLoader(datam,batch_size=2)
# for batch in loads:  # batchsize=2 一共1487.5个batch
#     # images = batch['image']
#     # true_masks = batch['label']
#     print(type(batch))



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# img=Image.open(r"F:\Q3\BaseUNet\Pytorch-UNet\data\cityscapes\leftImg8bit\train\aachen\aachen_000000_000019_leftImg8bit.png")
#
# label = Image.open(r"F:\Q3\BaseUNet\Pytorch-UNet\data\cityscapes\gtFine\train\aachen\aachen_000000_000019_gtFine_labelTrainIds.png")
#
# img = img.resize((512,512), Image.BILINEAR)
# label = label.resize((512,512), Image.NEAREST)
# img = transforms.ToTensor()(img).unsqueeze(0).to(dtype=torch.float32)
# label = transforms.ToTensor()(label).to( dtype=torch.long)
#
# #
# #
# input=img
# real_label=label
# model=UNet(n_channels=3,n_classes=19)
#
# preidct=  model(input)
# print(preidct.shape)
#
# #然后把这个predic他取出最大的
# #经过softmax 再取出最大的来
# mask = nn.Softmax(dim=1)(preidct).argmax(dim=1)
# print(mask.shape)
# ignore_labels = [255]
# metric = SegmentationMetric(19) # 3表示有3个分类，有几个分类就填几, 0也是1个分类
# hist = metric.addBatch(mask, real_label,ignore_labels)
#
# IoU = metric.IntersectionOverUnion()
# mIoU = metric.meanIntersectionOverUnion()
# # print('hist is :\n', hist)
# print('IoU is : ', IoU)
# print('mIoU is : ', mIoU)
#
#


# # 测试内容
# if __name__ == '__main__':
# 	imgPredict = torch.tensor([[0,1,2],[2,1,1]]).long()  # 可直接换成预测图片
# 	imgLabel = torch.tensor([[0,1,255],[1,1,2]]).long() # 可直接换成标注图片
# 	ignore_labels = [255]
# 	metric = SegmentationMetric(3) # 3表示有3个分类，有几个分类就填几, 0也是1个分类
# 	hist = metric.addBatch(imgPredict, imgLabel,ignore_labels)
#
# 	IoU = metric.IntersectionOverUnion()
# 	mIoU = metric.meanIntersectionOverUnion()
# 	print('hist is :\n', hist)
# 	print('IoU is : ', IoU)
# 	print('mIoU is : ', mIoU)
#
# ##output
# # hist is :
# # tensor([[1., 0., 0.],
# #        [0., 2., 1.],
# #        [0., 1., 0.]])
# # PA is : 0.600000
# # cPA is : tensor([1.0000, 0.6667, 0.0000])
# # mPA is : 0.555556
# # IoU is :  tensor([1.0000, 0.5000, 0.0000])
# # mIoU is :  tensor(0.5000)


#第二版计算miou
