import torch.nn.functional as F
from .separable_unet_parts import *

class Separable_UNet_1(nn.Module):
    def __init__(self, class_num):
        super(Separable_UNet_1, self).__init__()
        self.inp = nn.Conv2d(3, 64, 1)
        self.block2 = Down(64, 128, expand=1)
        self.block3 = Down(128, 256, expand=2)
        self.block4 = Down(256, 512, expand=3)

        self.block5 = Up(512, 256, expand=1)
        self.block6 = Up(256, 128, expand=1)
        self.block7 = Up(128, 64, expand=1)
        self.out = nn.Conv2d(64, class_num, 1)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x1_use = self.inp(x)
        x1 = self.maxpool(x1_use)
        x2_use = self.block2(x1)
        x2 = self.maxpool(x2_use)
        x3_use = self.block3(x2)
        x3 = self.maxpool(x3_use)
        x4 = self.block4(x3)
        
        x5 = self.block5(x4, x3_use)
        x6 = self.block6(x5, x2_use)
        x7 = self.block7(x6, x1_use)
        out= self.out(x7)
        return F.sigmoid(out)