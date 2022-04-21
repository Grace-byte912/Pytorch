import torchvision
# 准备测试集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

img, tag = test_set[0]
print(img.shape)
print(tag)

writer = SummaryWriter("logs/DataLoader")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, tags = data
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step = step + 1
        # print(imgs.shape)
        # print(tags)

writer.close()