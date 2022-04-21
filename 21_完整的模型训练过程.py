import torch
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

from model import *
# 1.准备数据集
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 2.利用dataloader来加载数据集
train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)

# 3.搭建神经网络，一般单独创建一个文件再引入
# model 见 model.py
nn6 = nn6()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# 0.01 = 1e-2 = 1X(10)^(-2)
learning_rate = 1e-2
optimizer = torch.optim.SGD(nn6.parameters(), lr=learning_rate)

# 网络训练的参数设置
# 训练的次数
total_train_step = 0
# 测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
# 添加tensorboard
writer = SummaryWriter("./logs/nn6")

for i in range(epoch):
    print("________第{}轮训练开始_________".format(i+1))
    # 调用train表示训练步骤开始，这里没有特殊含义
    nn6.train()
    # start to train
    for data in train_dataloader:
        imgs, tags = data
        outputs = nn6(imgs)
        loss = loss_fn(outputs, tags)
        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step%100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # test
    # 调用eval表示测试步骤开始，在这里没有特殊作用
    nn6.eval()
    total_test_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, tags = data
            outputs = nn6(imgs)
            loss = loss_fn(outputs, tags)
            total_test_loss = total_test_loss + loss.item()
            # 分类问题计算正确率
            accuracy = (outputs.argmax(1) == tags).sum()
            total_acc = total_acc + accuracy
    print("测试集上的loss为：{}".format(total_test_loss))
    print("测试集上的accuracy为：{}".format(total_acc/test_data_size))
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    writer.add_scalar("test_accuracy", total_acc/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存模型
    torch.save(nn6, "nn6_{}.pth".format(i))
    print("模型{}已保存！".format(i))

writer.close()

