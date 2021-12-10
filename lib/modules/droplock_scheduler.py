import numpy as np
from torch import nn


class DropLockScheduler(nn.Module):
    def __init__(self, dropblock):
        super(DropLockScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = None

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]
        self.i += 1

    def init_para(self, start=0.0, stop=0.1, block_size=5, nr_steps=5000):
        self.i = 0
        self.drop_values = np.linspace(start=start, stop=stop, num=int(nr_steps))
        self.dropblock.drop_prob = start
        self.dropblock.block_size = block_size
