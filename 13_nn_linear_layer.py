# 线性层
# [Xn] -> f -> [Hn]  h = kx + b
import torch
import torch as torch
import torchvision as torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


class nn3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


writer = SummaryWriter("logs/nn3")
step = 0
nn3 = nn3()
for data in data_loader:
    imgs, tags = data
    # 可以展开为一维
    # input = torch.flatten(imgs)
    input = torch.reshape(imgs, (1, 1, 1, -1))
    print(input.shape)
    writer.add_images("linear_in", input, step)
    output = nn3(input)
    writer.add_images("linear_out1", output, step)
    step = step + 1

writer.close()
