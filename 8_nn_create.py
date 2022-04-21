# NN:
# input --> forward --> output
import torch
from torch import nn


class New_nn(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 9
        return output


nn1 = New_nn()
x = torch.tensor(1.0)
output = nn1(x)
print(output)
