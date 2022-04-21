import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Sequential, Linear


# 搭建神经网络
class nn6(nn.Module):
    def __init__(self) -> None:
        super(nn6, self).__init__()
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


if __name__ == '__main__':
    nn6 = nn6()
    input1 = torch.ones((64, 3, 32, 32))
    output = nn6(input1)
    print(output.shape)