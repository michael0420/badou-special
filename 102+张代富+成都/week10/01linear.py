import torch
from torch.nn import Parameter
import torch.nn.functional as f


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(out_features))

    def forward(self, _x):
        return f.linear(_x, self.weight, self.bias)


if __name__ == '__main__':
    x = torch.randn((2, 3))
    net = Linear(3, 2)
    print(net.forward(x))
