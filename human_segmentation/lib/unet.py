"""UNet modules"""
import torch  # type: ignore


class UNetDoubleConv(torch.nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super(UNetDoubleConv, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv(x)

        return x


class UNetInConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetInConv, self).__init__()

        self.conv = UNetDoubleConv(in_ch, out_ch)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv(x)

        return x


class UNetDown(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetDown, self).__init__()

        self.mpconv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            UNetDoubleConv(in_ch, out_ch)
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.mpconv(x)

        return x


class UNetUp(torch.nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UNetUp, self).__init__()

        self.up = None
        if bilinear:
            self.up = torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
        else:
            self.up = torch.nn.ConvTranspose2d(
                in_ch // 2,
                in_ch // 2,
                2,
                stride=2
            )

        self.conv = UNetDoubleConv(in_ch, out_ch)

    def __call__(self, x1, x2):
        return self.forward(x1, x2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(
            x1,
            (
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2
            )
        )

        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)

        return x


class UNetOutConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetOutConv, self).__init__()

        self.conv = torch.nn.Conv2d(in_ch, out_ch, 1)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv(x)

        return x
