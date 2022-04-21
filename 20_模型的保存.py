# 模型的保存与使用
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Sequential

vgg16 = torchvision.models.vgg16(pretrained=False)
# 1.保存方式1,模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")
# 加载模型
model = torch.load("vgg16_method1.pth")

# 2.保存方式2，模型参数
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
# 加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))


class nn5(nn.Module):
    def __init__(self) -> None:
        super(nn5, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
        )

    def forward(self, x):
        x = self.model(x)
        return x


nn5 = nn5()
torch.save(nn5, "nn5_menthod1.pth")
model = torch.load("nn5_method1.pth")
print(model)
