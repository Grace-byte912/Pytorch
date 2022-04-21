# torch.nn.Conv2d
# (in_channels:输入图像的通道数, out_channels：输出图像的通道数, kernel_size：卷积核大小,
# stride=1：步进大小, padding=0：对周围进行填充,
# dilation=1：空洞卷积使用, groups=1, bias=True,
# padding_mode='zeros', device=None, dtype=None)

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


class nn1(nn.Module):
    def __init__(self):
        super(nn1, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x


nn1 = nn1()
print(nn1)

writer = SummaryWriter("logs/nn1")
step = 0
for data in data_loader:
    imgs,tags = data
    output = nn1(imgs)
    print(imgs.shape)
    # [64,3,32,32]
    writer.add_images("input",imgs,step)
    print(output.shape)
    # [64,6,30,30]
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1

