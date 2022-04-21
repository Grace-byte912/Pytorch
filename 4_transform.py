from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms

# transforms.py 工具箱
# 工具：totensor resize 来处理图片

# 1.transform如何使用 1)ToTensor
img_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

# 2.为什么需要Tensor数据类型
writer = SummaryWriter("logs")
writer.add_image("test",tensor_img)
writer.close()