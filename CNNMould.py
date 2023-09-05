'''
@Author ：YZX
@Date ：2023/9/4 10:18 
@Python-Version ：3.8
'''

# 构建CNN模型
from PIL import Image
import numpy as np
# 导入pytorch库
import torch
# 导入torch.nn模块
from torch import nn
# nn.functional：(一般引入后改名为F)有各种功能组件的函数实现，如：F.conv2d
import torch.nn.functional as F


# 定义AlexNet网络模型
# MyLeNet5（子类）继承 nn.Module（父类）
class MyAlexNet(nn.Module):
    # 子类继承中重新定义Module类的__init__()和forward()函数
    # init()：进行初始化，申明模型中各层的定义
    def __init__(self):
        # super：引入父类的初始化方法给子类进行初始化
        super(MyAlexNet, self).__init__()

        # 使用ReLU作为激活函数
        self.ReLU = nn.ReLU()

        # 第一个卷积层，输入通道为4(对应RGBA图像)，输出为32，卷积核为3
        # 步长为4，即每次卷积核在水平和垂直方向上移动4个像素
        # 边缘像素为2，以便在卷积操作中保持输入和输出的尺寸相似
        # 计算公式：N = (输入大小-卷积核大小+2*填充值的大小)/步长大小 + 1
        self.c1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        # 第一个最大池化层，池化核为2，步长为2
        # 计算公式：H = (输入大小-卷积核大小)/步长大小 + 1
        self.s1 = nn.MaxPool2d(kernel_size=2, stride=1)

        # 第二个卷积层，输入通道为32，输出为64，卷积核为3，步长为1，扩充边缘为2
        self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 第二个最大池化层，池化核为2，步长为2
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Flatten()：将张量（多维数组）平坦化处理，神经网络中第0维表示的是batch_size，所以Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()

        # 全连接层，第一个参数表示样本的大小，第二个参数表示样本输出的维度大小
        self.f3 = nn.Linear(6400,512)
        # softmax输出层，此时第二个参数代表了该全连接层的神经元个数(或者说分类个数)
        self.f4 = nn.Linear(512, 2)


    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, imageTensor):
        # 第一次卷积-激活
        imageTensor = self.ReLU(self.c1(imageTensor))
        # 第一次池化
        imageTensor = self.s1(imageTensor)

        # 第二次卷积-激活
        imageTensor = self.ReLU(self.c2(imageTensor))
        # 第二次池化
        imageTensor = self.s2(imageTensor)

        # 平坦化处理
        imageTensor = self.flatten(imageTensor)

        # 全连接
        imageTensor = self.f3(imageTensor)
        # Dropout：随机地将输入中50%的神经元激活设为0，即去掉了一些神经节点，防止过拟合
        # “失活的”神经元不再进行前向传播并且不参与反向传播，这个技术减少了复杂的神经元之间的相互影响
        imageTensor = F.dropout(imageTensor, p=0.5)
        # 输出
        imageTensor = self.f4(imageTensor)
        return imageTensor


# 每个python模块（python文件）都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）
# 如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）
# “__main__” 始终指当前执行模块的名称（包含后缀.py）
# if确保只有单独运行该模块时，此表达式才成立，才可以进入此判断语法，执行其中的测试代码，反之不行
if __name__ == '__main__':
    # 图像文件路径
    image_path = "C:/Users/29973/Desktop/论文/深度强化学习/论文复现/Image/1.png"
    # 使用Pillow库加载图像
    image = Image.open(image_path)
    # 裁剪图像
    image = image.resize((12,12))
    # 定义图像转换操作，将图像转化为张量
    image_array = np.array(image)
    # 这是对张量的维度进行重新排列的操作
    # permute(2, 0, 1)表示将原始图像张量的维度从（高度，宽度，通道数）重新排列为（通道数，高度，宽度）
    # unsqueeze(0)表示在张量的第0维度（批次维度）上添加一个维度，将图像张量转化为一个批次张量
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    x = image_tensor.to(torch.float32)
    # 模型实例化
    model = MyAlexNet()
    y = model.forward(x)
