import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _upsample_like(src,tar):

    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')

    return src

class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()

        self.c1 = nn.Conv2d(input, 12, 3, 1, 1, bias=bias)

        self.conv1 = nn.Conv2d(12, 12, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(24,24, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(48, 48, 3, 1, 1, bias=bias)

        self.bn4 = nn.BatchNorm2d(output)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn1 = nn.BatchNorm2d(12)

        self.conv6 = nn.Conv2d(48, output, 3, 1, 1, bias=bias)

        self.prelu = nn.ReLU(inplace=True)
        self.prelu2 = nn.ReLU(inplace=True)
        self.prelu3 = nn.ReLU(inplace=True)
        self.prelu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        #
        x = self.c1(x)
        x1 =self.conv1(x)
        x1 = self.bn1(x1)

        x1 = self.prelu(x1)


        x2 =self.conv2(torch.cat((x, x1), 1))
        x2 = self.bn2(x2)

        x2 = self.prelu2(x2)

        x3 = self.conv3(torch.cat((x, x1,x2), 1))
        x3 = self.bn3(x3)

        x3= self.prelu3(x3)

        x5 = self.conv6(x3)
        x5 = self.bn4(x5)

        x5 = self.prelu4(x5)
        return x5



class LGIPB(nn.Module):
    """包含残差连接的深度可分离卷积块，用于替换原始的REBNCONV。"""
    def __init__(self, in_ch, out_ch, dirate=1):
        super(LGIPB, self).__init__()
        self.conv = nn.Sequential(
            ResidualDenseBlock_out(in_ch, out_ch),

        )

    def forward(self, x):
        return self.conv(x)  # 使用残差连接

class Encoder7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(Encoder7, self).__init__()
        self.c1 = LGIPB(in_ch, mid_ch, dirate=1)

        self.c3 = LGIPB(mid_ch, out_ch, dirate=1)

    def forward(self, x):
        x1 = self.c1(x)

        x3 = self.c3(x1)
        return x3


class Encoder6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(Encoder6, self).__init__()
        self.c1 = LGIPB(in_ch, mid_ch, dirate=1)

        self.c3 = LGIPB(mid_ch, out_ch, dirate=1)

    def forward(self, x):
        x1 = self.c1(x)

        x3 = self.c3(x1)
        return x3


class Encoder5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(Encoder5, self).__init__()

        self.c1 = LGIPB(in_ch, mid_ch, dirate=1)

        self.c3 = LGIPB(mid_ch, out_ch, dirate=1)


    def forward(self, x):
        x1 = self.c1(x)

        x3 = self.c3(x1)
        return x3

#