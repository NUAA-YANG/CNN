'''
@Author ：YZX
@Date ：2023/9/5 11:20 
@Python-Version ：3.8
'''
# 训练模型
import torch
from torch import nn
from CNNMould import MyAlexNet
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# 解决中文显示问题
# 运行配置参数中的字体（font）为黑体（SimHei）
plt.rcParams['font.sans-serif'] = ['simHei']
# 运行配置参数总的轴（axes）正常显示正负号（minus）
plt.rcParams['axes.unicode_minus'] = False

# 训练集和测试集的位置
ROOT_TRAIN = "C:/Users/29973/Desktop/论文/深度强化学习/论文复现/TestImage"
#ROOT_TEST = 'D:/pycharm/AlexNet/data/val'


# Compose()：将多个transforms的操作整合在一起
train_transform = transforms.Compose([
    # Resize()：把给定的图像随机裁剪到指定尺寸
    transforms.Resize((12, 12)),
    # ToTensor()：数据转化为Tensor格式
    transforms.ToTensor()])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()])

# 加载训练数据集
# ImageFolder：假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名，其构造函数如下：
# ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
# root：在root指定的路径下寻找图像，transform：对输入的图像进行的转换操作
train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
# batch_size=512表示每个batch加载多少个样本(默认: 1)
# shuffle=True表示在每个epoch重新打乱数据(默认: False)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 加载训练数据集
# val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
# val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# 如果有NVIDA显卡，可以转到GPU训练，否则用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化，将模型转到device
model = MyAlexNet().to(device)

# 定义损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

# 定义优化器(随机梯度下降法)
# params(iterable)：要训练的参数，一般传入的是model.parameters()
# lr(float)：learning_rate学习率，也就是步长
# momentum(float, 可选)：动量因子（默认：0），矫正优化率
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 学习率每隔10轮变为原来的0.5
# StepLR：用于调整学习率，一般情况下会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果
# optimizer （Optimizer）：更改学习率的优化器
# step_size（int）：每训练step_size个epoch，更新一次参数
# gamma（float）：更新lr的乘法因子
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 定义训练函数(数据加载器、CNN模型、损失函数[现成的]、优化器[现成的])
def train(dataloader, model, loss_fn, optimizer):
    # 损失值，准确率，训练次数
    loss, current, n = 0.0, 0.0, 0
    # dataloader: 传入数据（数据包括：训练数据和标签）
    # enumerate()：用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中
    # enumerate返回值有两个：一个是序号，一个是数据（包含训练数据和标签）
    # x：训练数据（inputs）(tensor类型的），y：标签（labels）(tensor类型）
    for batch, (x, y) in enumerate(dataloader):
        # 前向传播
        image, y = x.to(device), y.to(device)
        # 计算训练值
        output = model(image)
        # 计算观测值（label）与训练值的损失函数
        cur_loss = loss_fn(output, y)
        # torch.max(input, dim)函数
        # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        _, pred = torch.max(output, axis=1)
        # 计算每批次的准确率
        # output.shape[0]为该批次的多少，output的一维长度
        # torch.sum()对输入的tensor数据的某一维度求和
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 反向传播
        # 清空过往梯度
        optimizer.zero_grad()
        # 反向传播，计算当前梯度
        cur_loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        # item()：得到元素张量的元素值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    # 计算训练的错误率
    print('train_loss==' + str(train_loss))
    # 计算训练的准确率
    print('train_acc' + str(train_acc))
    return train_loss, train_acc


# 定义验证函数
# def val(dataloader, model, loss_fn):
#     loss, current, n = 0.0, 0.0, 0
#     # eval()：如果模型中有Batch Normalization和Dropout，则不启用，以防改变权值
#     model.eval()
#     with torch.no_grad():
#         for batch, (x, y) in enumerate(dataloader):
#             # 前向传播
#             image, y = x.to(device), y.to(device)
#             output = model(image)
#             cur_loss = loss_fn(output, y)
#             _, pred = torch.max(output, axis=1)
#             cur_acc = torch.sum(y == pred) / output.shape[0]
#             loss += cur_loss.item()
#             current += cur_acc.item()
#             n = n + 1
#
#     val_loss = loss / n
#     val_acc = current / n
#     # 计算验证的错误率
#     print('val_loss=' + str(val_loss))
#     # 计算验证的准确率
#     print('val_acc=' + str(val_acc))
#     return val_loss, val_acc


# 定义画图函数
# 错误率
def matplot_loss(train_loss, val_loss):
    # 参数label = ''传入字符串类型的值，也就是图例的名称
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    # loc代表了图例在整个坐标轴平面中的位置（一般选取'best'这个参数值）
    plt.legend(loc='best')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.title("训练集和验证集的loss值对比图")
    plt.show()


# 准确率
def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.xlabel('acc')
    plt.ylabel('epoch')
    plt.title("训练集和验证集的acc值对比图")
    plt.show()


# 开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

# 训练次数
epoch = 20
# 用于判断最佳模型
# min_acc = 0
for t in range(epoch):
    lr_scheduler.step()
    print(f"epoch{t + 1}\n----------")
    # 训练模型
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    # 验证模型
    # val_loss, val_acc = val(val_dataloader, model, loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    # loss_val.append(val_loss)
    # acc_val.append(val_acc)

    # 保存最好的模型权重
    # if val_acc > min_acc:
    #     folder = 'save_model'
    #     # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
    #     if not os.path.exists(folder):
    #         # os.mkdir() 方法用于以数字权限模式创建目录
    #         os.mkdir('save_model')
    #     min_acc = val_acc
    #     print(f"save best model，第{t + 1}轮")
    #     # torch.save(state, dir)：保存模型等相关参数，dir表示保存文件的路径+保存文件名
    #     # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
    #     torch.save(model.state_dict(), 'save_model/best_model.pth')

    # 保存最后一轮权重
    if t == epoch - 1:
        torch.save(model.state_dict(), 'Mould/best_model.pth')

matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)

print('done')