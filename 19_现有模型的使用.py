import torchvision
from torch import nn

vgg16_f = torchvision.models.vgg16(pretrained=False)
vgg16_t = torchvision.models.vgg16(pretrained=True)
print("ok")
print(vgg16_t)
# 输出为10分类的
vgg16_t.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_t)

print(vgg16_f)
# 修改classifier第7个网络的结构
vgg16_f.classifier[6] = nn.Linear(4096, 10)
print(vgg16_f)

