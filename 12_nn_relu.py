# 非线性激活函数:提升模型的泛化能力
# relu（隐藏层），sigmoid（多分类输出层）
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input1 = torch.tensor([[1, -0.5], [-1, 3]])
output = torch.reshape(input1, (-1, 1, 2, 2))
print(output.shape)


class nn2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # inplace = False 是否在原地进行操作即是否修改input的值。
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return output


nn2 = nn2()
output = nn2(input1)
print(output)

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("logs/nn2")
step = 0
for data in data_loader:
    imgs, tags = data
    writer.add_images("sigmoid_in1", imgs, step)
    output = nn2(imgs)
    writer.add_images("sigmoid_out1", output, step)
    step = step + 1
writer.close()
