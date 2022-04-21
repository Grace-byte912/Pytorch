import torch
import torchvision.transforms
from PIL import Image
from model import *

img_path = "./images/dog.jpg"
img = Image.open(img_path)
print(img)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
# 如果加载的是GPU训练的模型需要加一个个参数 map_location
# model = torch.load("nn6_0_gpu.pth", map_location=torch.device("cpu"))
model = torch.load("nn6_0.pth")
img = transform(img)
print(model)
# 添加batch_size
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)

print(output.argmax(1))