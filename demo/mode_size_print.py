import os
import sys
import torch
from torchstat import stat
sys.path.insert(1, os.getcwd())
from models_.hrnet import HRNet


def main(EP_list):
    device = torch.device('cuda:0')
    model = HRNet(c=32, nof_joints=17,bn_momentum=0.01,retain_block = EP_list)
    stat(model, (3, 256, 192))  # 格式为 stat(网络名称，（波段数，图像大小）)

if __name__ == '__main__':
    retain_block = []
    for i in range(3):
        l2 = []
        for j in range(4):
            l3 = [1 for _ in range(4)]
            l2.append(l3)
        retain_block.append(l2)
    main(retain_block)