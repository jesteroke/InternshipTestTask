"""Human segmentation models"""
import torch  # type: ignore

from lib import *  # type: ignore


class HumanSegmentationModelUNet(torch.nn.Module):
    """U-Net model for human segmentation
    based on https://arxiv.org/pdf/1505.04597.pdf"""
    def __init__(self, config):
        super(HumanSegmentationModelUNet, self).__init__()

        n_channels = config['n_channels']
        n_classes = config['n_classes']

        self.inc = UNetInConv(n_channels, 64)

        self.down1 = UNetDown(64, 128)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512)
        self.down4 = UNetDown(512, 512)

        self.up1 = UNetUp(1024, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        self.up4 = UNetUp(128, 64)

        self.outc = UNetOutConv(64, n_classes)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)

        y = torch.nn.functional.sigmoid(x)

        return y
