import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def f(x, y):
    # return 1 / 2 * (1 - x) * (1 - y) +  x * y
    return   (1 - x) * (1 - y) + 1 / 2 * x * y


Sobel = np.array([[-1,-2,-1],
                  [ 0, 0, 0],
                  [ 1, 2, 1]])
Robert = np.array([[0, 0],
                  [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)


# binarize the deep structures
class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg(x)
        x = torch.sign(x - y)
        out = (x + 1) / 2
        return out

def Conv(in_channels, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), stride=stride, bias=bias)

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16, bias=False):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
#                 nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y

# #   RCAB(ngf, kernel_size, reduction, bias=bias, act=act)

# class RCAB(nn.Module):
#     def __init__(self, n_feat, kernel_size, reduction, bias, act):
#         super(RCAB, self).__init__()
#         feats_cal = []
#         feats_cal.append(Conv(n_feat, n_feat, kernel_size, bias=bias))
#         feats_cal.append(act)
#         feats_cal.append(Conv(n_feat, n_feat, kernel_size, bias=bias))

#         self.SE = SELayer(n_feat, reduction, bias=bias)
#         self.feats_cal = nn.Sequential(*feats_cal)

#     def forward(self, x):
#         feats = self.feats_cal(x)
#         feats = self.SE(feats)
#         feats += x
#         return feats


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, 1, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class CPAB(nn.Module):
    def __init__(self, dim, kernel_size, bias):
        super(CPAB, self).__init__()
        self.conv1 = Conv(dim, dim, kernel_size, bias=bias)
        self.act1 = nn.PReLU()
        self.conv2 = Conv(dim, dim, kernel_size, bias=bias)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res



class Output(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, output_channel=3, residual=True):
        super(Output, self).__init__()
        self.conv = Conv(n_feat, output_channel, kernel_size, bias=bias)
        self.residual = residual

    def forward(self, x, x_img):
        x = self.conv(x)
        if self.residual:
            x += x_img
        return x


class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, atten):
        super(Encoder, self).__init__()
        self.atten = atten

        self.encoder_level1 =  CPAB(n_feat, kernel_size, bias=bias)
        self.encoder_level2 =  CPAB(n_feat*2, kernel_size, bias=bias)
        self.encoder_level3 =  CPAB(n_feat*4, kernel_size, bias=bias)
        # RCAB(n_feat,       kernel_size, reduction, bias=bias, act=act)

        self.down12  = DownSample(n_feat, n_feat*2)
        self.down23  = DownSample(n_feat*2, n_feat*4)

        if self.atten:  # feature attention
            self.atten_conv1 = Conv(n_feat, n_feat, 1, bias=bias)
            self.atten_conv2 = Conv(n_feat*2, n_feat*2, 1, bias=bias)
            self.atten_conv3 = Conv(n_feat*4, n_feat*4, 1, bias=bias)

    def forward(self, x, encoder_outs=None):
        if encoder_outs is None:
            enc1 = self.encoder_level1(x)
            x = self.down12(enc1)
            enc2 = self.encoder_level2(x)
            x = self.down23(enc2)
            enc3 = self.encoder_level3(x)

            return [enc1, enc2, enc3]
        else:
            # assert encoder_outs is not None

            enc1 = self.encoder_level1(x)
            enc1_fuse_nir = enc1 + self.atten_conv1(encoder_outs[0])
            x = self.down12(enc1_fuse_nir)
            enc2 = self.encoder_level2(x)
            enc2_fuse_nir = enc2 + self.atten_conv2(encoder_outs[1])
            x = self.down23(enc2_fuse_nir)
            enc3 = self.encoder_level3(x)
            enc3_fuse_nir = enc3 + self.atten_conv3(encoder_outs[2])

            return [enc1_fuse_nir, enc2_fuse_nir, enc3_fuse_nir]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, residual=True):
        super(Decoder, self).__init__()

        self.residual = residual

        self.decoder_level1 = CPAB(n_feat, kernel_size, bias=bias)
        self.decoder_level2 = CPAB(n_feat*2, kernel_size, bias=bias)
        self.decoder_level3 = CPAB(n_feat*4, kernel_size, bias=bias)
        #  RCAB(n_feat,      kernel_size, reduction, bias=bias, act=act)

        self.skip_conv_1 = Conv(n_feat, n_feat, kernel_size, bias=bias)
        self.skip_conv_2 = Conv(n_feat*2, n_feat*2, kernel_size, bias=bias)

        self.up21  = UpSample(n_feat*2, n_feat)
        self.up32  = UpSample(n_feat*4, n_feat*2)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3)
        if self.residual:
            x += self.skip_conv_2(enc2)
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2)
        if self.residual:
            x += self.skip_conv_1(enc1)
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]

 
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(DownSample, self).__init__()
        self.conv = Conv(in_channels, out_channel, 1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UpSample, self).__init__()
        self.conv = Conv(in_channels, out_channel, 1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x



class Edge(nn.Module):
    def __init__(self, channel, kernel='sobel'):
        super(Edge, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.kernel_x = Sobel.repeat(channel, 1, 1, 1)
        self.kernel_y = self.kernel_x.permute(0, 1, 3, 2)
        self.kernel_x = nn.Parameter(self.kernel_x, requires_grad=False)
        self.kernel_y = nn.Parameter(self.kernel_y, requires_grad=False)

    def forward(self, current):
        current = F.pad(current, (1,1,1,1), mode='reflect')
        gradient_x = torch.abs(F.conv2d(current, weight=self.kernel_x, groups=self.channel, padding=0))
        gradient_y = torch.abs(F.conv2d(current, weight=self.kernel_y, groups=self.channel, padding=0))
        out = gradient_x + gradient_y
        return out

