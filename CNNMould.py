'''
@Author ：YZX
@Date ：2023/9/4 10:18 
@Python-Version ：3.8
'''

# 导入torch.nn模块
from torch import nn
# nn.functional：(一般引入后改名为F)有各种功能组件的函数实现，如：F.conv2d
import torch.nn.functional as F


# 定义AlexNet网络模型
# MyLeNet5（子类）继承 nn.Module（父类）
class MyAlexNet(nn.Module):
    # 子类继承中重新定义Module类的__init__()和forward()函数
    # init()：进行初始化，申明模型中各层的定义
    def __init__(self,out):
        # super：引入父类的初始化方法给子类进行初始化
        super(MyAlexNet, self).__init__()

        # 使用ReLU作为激活函数
        self.ReLU = nn.ReLU()

        # 第一个卷积层，输入通道为4(对应RGBA图像)，输出为32，卷积核为3
        # 步长为1，即每次卷积核在水平和垂直方向上移动1个像素
        # 边缘像素为1，以便在卷积操作中保持输入和输出的尺寸相似
        # 计算公式：特征图大小 = ((输入大小-卷积核大小+2*填充值的大小)/步长大小) + 1
        self.c1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # 第一个最大池化层，池化核为2，步长为2
        # 计算公式：特征图大小 = (输入大小-卷积核大小)/步长大小 + 1
        self.s1 = nn.MaxPool2d(kernel_size=2, stride=1)

        # 第二个卷积层，输入通道为32，输出为64，卷积核为3，步长为1，扩充边缘为1
        self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 第二个最大池化层，池化核为2，步长为2
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Flatten()：将张量（多维数组）平坦化处理，神经网络中第0维表示的是batch_size，所以Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()

        # 全连接层，第一个参数表示样本的大小，第二个参数表示样本输出的维度大小
        self.f3 = nn.Linear(6400,512)
        # softmax输出层，此时第二个参数代表了该全连接层的神经元个数(或者说分类个数)，此时需要实现5种分类
        self.f4 = nn.Linear(512, out)


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

