# 优化器例子：
# for input, target in dataset:
#     optimizer.zero_grad()
#     output = model(input)
#     loss = loss_fn(output, target)
#     loss.backward()
#     optimizer.step()
import torch
import torchvision as torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from torch.utils.data import DataLoader

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)


class nn5(nn.Module):
    def __init__(self) -> None:
        super(nn5, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


nn5 = nn5()
loss_cross = CrossEntropyLoss()
optim = torch.optim.SGD(nn5.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in data_loader:
        imgs, targets = data
        outputs = nn5(imgs)
        # print(outputs)
        # print(targets)
        result = loss_cross(outputs, targets)
        optim.zero_grad()
        # 计算里面参数的梯度grad，为后面的优化做好准备
        result.backward()
        # 开始优化,优化前需要计算grad
        optim.step()
        running_loss = running_loss + result
    print("epoch: ".format(epoch), running_loss)


