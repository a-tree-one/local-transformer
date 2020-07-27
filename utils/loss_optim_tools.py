import os
import torch.nn as nn
import torch
import copy
import numpy as np


def adjust_learning_rate(optimizer, epoch):
    if epoch < 10000:
        lr = 1e-4 * (0.8 ** (epoch // 10))
    else:
        lr = 1e-4 * (0.1 ** ((epoch-50) // 10))
    # x = epoch
    # CYCLE = 8000
    # LR_INIT = 0.1
    # LR_MIN = 0.001
    # scheduler = lambda x: ((LR_INIT - LR_MIN) / 2) * (np.cos(np.pi * (np.mod(x - 1, CYCLE) / (CYCLE))) + 1) + LR_MIN
    # lr = scheduler

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Gauss_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gauss_Net, self).__init__()
        self._init_data = torch.Tensor([[1/273, 4/273, 7/273, 4/273, 1/273],
                                        [4/273, 16/273, 26/273, 16/273, 4/273],
                                        [7/273, 26/273, 41/273, 26/273, 7/273],
                                        [4/273, 16/273, 26 / 273, 16 / 273, 4 / 273],
                                        [1/273, 4/273, 7 / 273, 4 / 273, 1 / 273]])
        self.gauss_kernel = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self._initialize_weights()
        for p in self.parameters():
            p.requires_grad = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data = self._init_data.expand_as(m.weight.data)
                for ii in range(m.weight.data.size(0)):
                    for jj in range(m.weight.data.size(1)):

                        if ii == jj:
                            m.weight.data[ii, jj, :, :] = self._init_data
                            # print(self._init_data)
                        else:
                            m.weight.data[ii, jj, :, :] = 0
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.gauss_kernel(x)
        return x


if __name__ == '__main__':
    gauss_net = Gauss_Net(3, 3)
    x = torch.randn(1, 3, 224, 224)
    out = gauss_net(x)
    print(out)
