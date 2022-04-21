from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms


class Person:
    # __ccc__函数的作用是把类名变成函数名，可以直接用类名调用该函数
    def __call__(self, name):
        print("hello:" + name)


writer = SummaryWriter("logs")
img = Image.open("images/pytorch.jpg")
# print(img)
# ToTensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)

# Normalize
print(img_tensor[0][3][0])
trans_norm = transforms.Normalize([1, 0.5, 0.5], [5, 6, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][3][0])
writer.add_image("Normalize", img_norm)

# resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_tensor(img_resize)
writer.add_image("resize", img_resize, 0)
# print(img_resize)

# compose - resize -2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> Tensor
trans_compose = transforms.Compose([trans_resize_2, trans_tensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize", img_resize_2, 1)

# randomcrop

