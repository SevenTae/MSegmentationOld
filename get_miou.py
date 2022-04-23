# 待完善
import os
from PIL import Image
from tqdm import tqdm

from predict_model.model_predict import   UnetP
from utils.utils_metrics import compute_mIoU, show_results
from dataloaders.datasets.cityscapes import CityscapesSegmentation

'''注意啊注意啊!这city数据集测试集官方没有给标签，所以验证的话得还是得用验证集
那个测试集留着测评可视化结果吧
'''

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 21
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
    # name_classes    = ["_background_","cat","dog"]
    # -------------------------------------------------------#
    #   指向测试数据集所在的文件夹
    #   默
    # -------------------------------------------------------#
    img_path = r'D:\MSegmentation\data\cityscapes\leftImg8bit\val' #
    label_path = r"D:\MSegmentation\data\cityscapes\gtFine\val"
    images_ =CityscapesSegmentation.recursive_glob(img_path,suffix='.png')
    gts_ =CityscapesSegmentation.recursive_glob(label_path,'.png')
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        unet = UnetP()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)