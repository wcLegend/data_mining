
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import numpy as np

class mlp(nn.Module):
    def __init__(self, batch_size,hidden_size, common_size):
        super(mlp, self).__init__()

        self.batch_size = batch_size

        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),              #  pytorch默认使用kaiming正态分布初始化卷积层参数
            nn.Linear(hidden_size // 2, common_size)
        )

        for layer in self.linear:
            if isinstance(layer, nn.Linear): # 判断是否是线性层
                #print('old',layer.weight)
                nn.init.xavier_uniform_(layer.weight)
                #print(layer.weight)

        self.softmax = nn.LogSoftmax(dim =1)

    def forward(self, x):

        print('mlp_input_size',x.size())
        out = self.linear(x)
        #print(out)
        output = self.softmax(out)  #加负号?
        #print(output)
        return output

    def test(self, x):

        print('mlp_input_size',x.size())
        out = self.linear(x)
        #print(out)
        output = F.softmax(out)  #加负号?
        #print(output)
        return output

