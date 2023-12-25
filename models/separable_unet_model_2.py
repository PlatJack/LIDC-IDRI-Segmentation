import torch.nn.functional as F
from .separable_unet_parts import *

class Separable_UNet_2(nn.Module):
    def __init__(self, class_num):
        super(Separable_UNet_2, self).__init__()
        self.inp = nn.Conv2d(3, 64, 1)
        self.block2 = Down(64, 128, expand=1)
        self.block3 = Down(128, 256, expand=2)
        self.block4 = Down(256, 512, expand=3)
        self.block5 = Down(512, 1024, expand=1)
        self.block6 = Up(1024, 512, expand=1)
        self.block7 = Up(512, 256, expand=1)
        self.block8 = Up(256, 128, expand=1)
        self.block9 = Up(128, 64, expand=1)
        self.out = nn.Conv2d(64, class_num, 1)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x1_use = self.inp(x)
        x1 = self.maxpool(x1_use)
        x2_use = self.block2(x1)
        x2 = self.maxpool(x2_use)
        x3_use = self.block3(x2)
        x3 = self.maxpool(x3_use)
        x4_use = self.block4(x3)
        x4 = self.maxpool(x4_use)
        x5 = self.block5(x4)

        x6 = self.block6(x5, x4_use)
        x7 = self.block7(x6, x3_use)
        x8 = self.block8(x7, x2_use)
        x9 = self.block9(x8, x1_use)
        out= self.out(x9)
        return F.sigmoid(out)