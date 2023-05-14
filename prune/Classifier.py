import torch

from collections import OrderedDict
from functools import reduce
import re
import copy
import numpy as np
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self,c):
        super(Classifier, self).__init__()
        self.laysers = nn.Sequential()

        # self._layers =  nn.Conv2d(48, 17, kernel_size=(1, 1), stride=(1, 1))
        if(c!=0):
            self.laysers.append(nn.Upsample(scale_factor=(2.0**(c+1)), mode='nearest'))

            self.laysers.append(nn.Conv2d(48*(2**c), 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False))

            self.laysers.append(nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True))
        self.laysers.append( nn.Conv2d(48, 17, kernel_size=(1, 1), stride=(1, 1)))

    def forward(self, inputs):
        for layer in self.laysers:
            inputs = layer(inputs)
        return inputs

class upSample(nn.Module):
    def __init__(self,c):
        super(upSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(48*c, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.Upsample(scale_factor=(2.0*c), mode='nearest')
        )

    def forward(self, inputs):

        return self.conv(inputs)

if __name__ == '__main__':


    a = torch.ones(48,96,72).cuda()
    print("before:", a.shape)
    classifer = Classifier(0).cuda()
    a = classifer(a)
    print("after:", a.shape)

    a = torch.ones(96, 48, 36).cuda()
    classifer = Classifier(1).cuda()
    print(classifer)
    print("before:", a.shape)
    a = classifer(a)
    print("after:", a.shape)

    a = torch.ones(1,192, 24, 18).cuda()
    classifer = Classifier(2).cuda()
    print(classifer)
    print("before:", a.shape)
    a = classifer(a)
    print("after:", a.shape)

    a = torch.ones(1,384, 12, 9).cuda()
    classifer = Classifier(3).cuda()
    print(classifer)
    print("before:", a.shape)
    a = classifer(a)
    print("after:", a.shape)

