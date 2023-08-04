import torch.nn as nn
from .basic import CPAB, DownSample


def Conv(in_channels, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), stride=stride, bias=bias)


class fusion1(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, kernel_size, bias):
        super(fusion1, self).__init__()
        self.conv_1 = CPAB(n_feat, kernel_size, bias=bias)
        self.conv_2 = CPAB(n_feat, kernel_size, bias=bias)

    def forward(self, x, y):
        res = self.conv_2(self.conv_1(x))
        return res


class fusion2(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, bias):
        super(fusion2, self).__init__()
        self.conv_1 = Conv(n_feat, n_feat, 1, stride=1, bias=bias)
        self.conv_2 = Conv(n_feat, n_feat, 1, stride=1, bias=bias)
        self.downsample = DownSample(n_feat-scale_unetfeats, n_feat)

    def forward(self, x, y):
        x = self.conv_1(x)
        x += self.downsample(y)
        res = self.conv_2(x)
        return res


class DSFE(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(DSFE, self).__init__()

        relu = nn.PReLU()

        self.enc_1 = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), relu)
        self.enc_2 = nn.Sequential(nn.Conv2d(n_feat*2, n_feat*2, kernel_size=1, bias=bias), relu)
        self.enc_3 = nn.Sequential(nn.Conv2d(n_feat*4, n_feat*4, kernel_size=1, bias=bias), relu)

        self.dec_1 = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), relu)
        self.dec_2 = nn.Sequential(nn.Conv2d(n_feat*2, n_feat*2, kernel_size=1, bias=bias), relu)
        self.dec_3 = nn.Sequential(nn.Conv2d(n_feat*4, n_feat*4, kernel_size=1, bias=bias), relu)

        self.fs_1 = fusion1(n_feat, n_feat, kernel_size, bias)
        self.fs_2 = fusion2(n_feat*2, n_feat, bias)
        self.fs_3 = fusion2(n_feat*4, n_feat*2, bias)

        self.sigmoid_1 = nn.Sequential(Conv(n_feat, n_feat, kernel_size, bias=bias), nn.Sigmoid())
        self.sigmoid_2 = nn.Sequential(Conv(n_feat*2, n_feat*2, kernel_size, bias=bias), nn.Sigmoid())
        self.sigmoid_3 = nn.Sequential(Conv(n_feat*4, n_feat*4, kernel_size, bias=bias), nn.Sigmoid())

    def forward(self, encoder_feature, decoder_feature):
        enc1, enc2, enc3 = encoder_feature
        dec1, dec2, dec3 = decoder_feature

        feat1 = self.enc_1(enc1) + self.dec_1(dec1)
        stru1 = self.fs_1(feat1, None)
        feat2 = self.enc_2(enc2) + self.dec_2(dec2)
        stru2 = self.fs_2(feat2, stru1)
        feat3 = self.enc_3(enc3) + self.dec_3(dec3)
        stru3 = self.fs_3(feat3, stru2)

        Stru1 = self.sigmoid_1(stru1)
        Stru2 = self.sigmoid_2(stru2)
        Stru3 = self.sigmoid_3(stru3)


        return [Stru1, Stru2, Stru3]