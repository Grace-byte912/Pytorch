# kernel_size – 最大的窗口大小
# stride——窗口的步幅。默认值为kernel_size
# padding – 要在两边添加隐式零填充
# dilation – 控制窗口中元素步幅的参数
# return_indices - 如果True，将返回最大索引以及输出。torch.nn.MaxUnpool2d以后有用
# ceil_mode – 如果为 True，将使用ceil而不是floor来计算输出形状
#     ceil:向上输出值，floor:向下输出值

# 池化层作用：重点采样，保留数据特征，检索数据量

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input, (-1, 1, 5, 5))
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


class nn1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

nn1 = nn1()
# output = nn1(input)
# print(output)

writer = SummaryWriter("logs/nn1")
step = 0
for data in data_loader:
    imgs, tags = data
    writer.add_images("max_pool_in1", imgs, step)
    output = nn1(imgs)
    writer.add_images("max_pool_out1", output, step)
    step = step + 1
writer.close()