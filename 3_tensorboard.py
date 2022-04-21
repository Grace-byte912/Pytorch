# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np

# 表示 out_dir = "./logs"
writer = SummaryWriter("logs")

# writer.add_scalar()的使用
# y = x
for i in range(100):
    writer.add_scalar("y=3x", 3*i, i)

# writer.add_image()的使用
img_path = "C:\\Users\\14590\\PycharmProjects\\Pytorch\\hymenoptera_data\\train\\bees\\16838648_415acd9e3f.jpg"
img_path = "C:\\Users\\14590\\PycharmProjects\\Pytorch\\hymenoptera_data\\train\\bees\\39672681_1302d204d1.jpg"
img = Image.open(img_path)
print(type(img))
# 转换为np格式
img_array = np.array(img)
print(type(img_array))
writer.add_image("train", img_array, 2, dataformats='HWC')
writer.close()
