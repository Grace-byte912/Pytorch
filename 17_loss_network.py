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
for data in data_loader:
    imgs, targets = data
    outputs = nn5(imgs)
    print(outputs)
    print(targets)
    result = loss_cross(outputs, targets)
    # 计算梯度grad，为后面的优化做好准备
    result.backward()
    print(result)
