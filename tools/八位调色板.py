##用来可视化一些黑乎乎的8位灰度图
import numpy as np
import torchvision.transforms as transforms
from collections import namedtuple  #namedtuple创建一个和tuple类似的对象，而且对象拥有可访问的属性
import matplotlib.pyplot as  plt


from  PIL import Image
Cls = namedtuple('cls', ['name', 'id', 'color'])
# Clss = [
#     Cls('building', 0, (255, 0, 0)),
#     Cls('road', 1, (255, 255, 0)),
#     Cls('pavement', 2, (192, 192, 0)),
#     Cls('vegetation', 3, (0, 255, 0)),
#     Cls('bare soil', 4, (128, 128, 128)),
#     Cls('water', 5, (0, 0, 255)),
#
# ]
Clss = [
    Cls('c1', 0, (0, 0, 0)),
    Cls('c2', 1, (128, 0, 0)),
    Cls('c3', 2, (0, 128, 0)),
    Cls('c4', 3, (128, 128, 0)),
    Cls('c5', 4, (0, 0, 128)),
    Cls('c6', 5, (128, 0, 128)),
    Cls('c7', 6, (0, 128, 128)),
    Cls('c8', 7, (128, 128, 128)),
    Cls('c9', 8, (64, 0, 128)),
    Cls('c10', 9, (192, 0, 128)),
    Cls('c11', 10, (64, 128, 0)),
    Cls('c12', 11, (192, 128, 0)),
    Cls('c13', 12, (64, 0, 128))

]



def get_putpalette(Clss, color_other=[0, 0, 0]):
    '''
    灰度图转8bit彩色图
    :param Clss:颜色映射表
    :param color_other:其余颜色设置
    :return:
    '''
    putpalette = []
    for cls in Clss:
        putpalette += list(cls.color)
    putpalette += color_other * (255 - len(Clss))
    return putpalette



if __name__ == '__main__':


    img_path = r"F:\一些数据集\Tree\VOCdevkit\VOC2012\SegmentationClass\1_21-180j.tif"
    img=Image.open(img_path)

    imarr =np.array(img)

    dst = Image.fromarray(np.uint8(imarr), 'P')
    bin_colormap = get_putpalette(Clss)
    dst.putpalette(bin_colormap)
    dst.save("dd.png")
