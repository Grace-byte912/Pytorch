# argmax()函数 参数为1横着看，为0竖着看，选出分数最大的某一行/列
import torch

outputs = torch.tensor([[0.1, 0.2],
                        [0.3, 0.4]])

print(outputs.argmax(1))

outputs = torch.tensor([[0.1, 0.2],
                        [0.05, 0.4]])

print(outputs.argmax(0))