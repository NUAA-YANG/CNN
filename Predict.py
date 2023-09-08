'''
@Author ：YZX
@Date ：2023/9/6 17:23 
@Python-Version ：3.8
'''

# 用模型预测

import torch
from CNNMould import MyAlexNet
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


ROOT_TEST = "C:/Users/29973/Desktop/论文/深度强化学习/论文复现/PredictImage"


val_transform = transforms.Compose([
    transforms.Resize((12, 12)),
    transforms.ToTensor()])

# 加载训练数据集
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
# batch_size=16表示每一轮预测都放入16张图片
val_dataloader = DataLoader(val_dataset,batch_size=16, shuffle=True)

# 如果有NVIDA显卡，转到GPU训练，否则用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化，将模型转到device
model = MyAlexNet(5).to(device)

# 加载train.py里训练好的模型
model.load_state_dict(torch.load('Mould/best_model.pth'))
model.eval()


# 批量测试
def predictBatch():
    # 定义损失函数（交叉熵损失）
    loss_fn = nn.CrossEntropyLoss()
    # 最终损失值，最终准确率，训练次数
    loss, current, n = 0.0, 0.0, 0
    # 每个类别的准确率
    # 进入验证阶段
    for batch, (image, label) in enumerate(val_dataloader):
        # 前向传播
        image, label = image.to(device), label.to(device)
        # 计算训练值
        output = model(image)
        # 计算观测值（label）与训练值的损失函数
        cur_loss = loss_fn(output, label)
        # torch.max(input, dim)函数
        # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        _, pred = torch.max(output, axis=1)
        # 计算每批次的准确率
        # output.shape[0]为该批次的多少，output的一维长度
        # torch.sum()对输入的tensor数据的某一维度求和
        cur_acc = torch.sum(label == pred) / output.shape[0]

        # item()：得到元素张量的元素值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

        if batch % 400 == 0 :
            print("处理完毕%s张图片" % str((batch*16)))
    # 计算训练的错误率
    print('train_loss ==' + str(loss / n))
    # 计算训练的准确率
    print('train_acc ==' + str(current / n))


# 单个测试
def predictOneByOne():
    # 结果类型
    classes = ["Dos","Normal","Probe","R2L","U2R"]
    # 记录每种结果的预测概率
    preList = [0,0,0,0,0]
    actList = [0,0,0,0,0]
    # 最终总的结果预测概率
    succeed = 0
    lengthData = len(val_dataset)
    print("======================预测速度======================")
    for i in tqdm(range(lengthData)):
        x, y = val_dataset[i][0], val_dataset[i][1]
        # torch.unsqueeze(input, dim)，input(Tensor)：输入张量，dim (int)：插入维度的索引，最终扩展张量维度为4维
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
        with torch.no_grad():
            pred = model(x)
            # 返回预测值所在的索引编号
            # argmax(input)：返回指定维度最大值的序号
            predIndex = torch.argmax(pred[0])
            # 预测值和实际值所在索引编号加一
            preList[predIndex] += 1
            actList[y] += 1
            if predIndex == y:
                succeed += 1
    for i in range(5):
        if preList[i] <= actList[i]:
            print(f"{classes[i]}的预测精准率为：{str(preList[i] / actList[i])}")
        else:
            print(f"{classes[i]}的预测精准率为：{str(actList[i] / preList[i])}")
    print(f"AlexNet-CNN模型综合预测准确率为：{str(succeed/lengthData)}")


predictOneByOne()
