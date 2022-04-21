# 损失函数 L1loss：
#
#
import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)
# inputs = torch.reshape(inputs, (1, 1, 1, 3))
# targets = torch.reshape(targets, (1, 1, 1, 3))
# reduction (string, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
loss = L1Loss(reduction="mean")
result = loss(inputs, targets)

print(result)

# 平方差
loss_mse = MSELoss()
result = loss_mse(inputs, targets)
print(result)

# 交叉熵（分类问题中常见的损失函数）
# loss(x, class) = -x[class] + log( Σ exp(x[j]))

x = torch.tensor([[0.1], [0.2], [0.3]])
print(x.shape)
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = CrossEntropyLoss()
result = loss_cross(x, y)
print(result)