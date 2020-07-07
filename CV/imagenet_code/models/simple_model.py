# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 23:24:35 2019

@author: ZML
"""
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.fc1(x.view(x.size(0), -1))
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        return out



def get_Net(num_classes=10):
    
    return Net(num_classes=num_classes)