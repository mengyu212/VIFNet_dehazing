import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import Conv, CPAB, Encoder, Decoder, f, Edge
from .DSFE import DSFE


class vifnet(nn.Module):
    def __init__(self, input_nc, output_nc, n_feat=64, kernel_size=3, reduction=4, bias=False):   # n_feat=80,reduction=4,scale_unetfeats=48,
        super(vifnet, self).__init__()
        
        self.inf_layer1 = nn.Sequential(Conv(input_nc, n_feat, kernel_size, bias=bias),
                                              CPAB(n_feat, kernel_size, bias),
                                              CPAB(n_feat, kernel_size, bias))
        self.rgb_layer1 = nn.Sequential(Conv(input_nc, n_feat, kernel_size, bias=bias),
                                              CPAB(n_feat, kernel_size, bias),
                                              CPAB(n_feat, kernel_size, bias))

        self.inf_encoder = Encoder(n_feat, kernel_size, bias, atten=False)
        self.inf_decoder = Decoder(n_feat, kernel_size, bias, residual=True)

        self.rgb_encoder = Encoder(n_feat, kernel_size, bias, atten=True)
        self.rgb_decoder = Decoder(n_feat, kernel_size, bias, residual=True)

        self.conv = Conv(n_feat, output_nc, kernel_size=1, bias=bias)

        # DSFE Module
        self.inf_structure = DSFE(n_feat, kernel_size, bias)
        self.rgb_structure = DSFE(n_feat, kernel_size, bias)

    def forward(self, rgb, inf):
        # infrared image feature extraction branch 
        inf_fea1 = self.inf_layer1(inf)
        inf_encode_feature = self.inf_encoder(inf_fea1)
        inf_decode_feature = self.inf_decoder(inf_encode_feature)
        inf_structure = self.inf_structure(inf_encode_feature, inf_decode_feature)

        # visible image feature extraction branch 
        rgb_fea1 = self.rgb_layer1(rgb)
        rgb_encode_feature = self.rgb_encoder(rgb_fea1)
        rgb_decode_feature = self.rgb_decoder(rgb_encode_feature)
        rgb_structure = self.rgb_structure(rgb_encode_feature, rgb_decode_feature)
        # rgb_structure[0]: torch.Size([1, 64, 480, 640])
        # rgb_structure[1]: torch.Size([1, 128, 240, 320])
        # rgb_structure[2]: torch.Size([1, 256, 120, 160])

        # coarse output
        rgb_fea2 = self.conv(rgb_decode_feature[0])
        rgb_out1 = rgb_fea2 + rgb

        # # single modality output
        # rgb_out2 = self.conv(rgb_structure[0])

        # Calculate inconsistency map
        incons_feature = []
        for i in range(len(rgb_structure)):
            incons_feature.append(f(rgb_structure[i], inf_structure[i]))


        inf_weight = [None for _ in range(3)]
        for i in range(3):
            inf_weight[i] = incons_feature[i] * inf_structure[i]
            # inf_weight[0]: torch.Size([1, 64, 480, 640])
            # inf_weight[1]: torch.Size([1, 128, 240, 320])
            # inf_weight[2]: torch.Size([1, 256, 120, 160])
        
        rgb_fea3 = self.rgb_layer1(rgb_out1)
        rgb_encode_feature_2 = self.rgb_encoder(rgb_fea3, inf_weight)
        rgb_decode_feature_2 = self.rgb_decoder(rgb_encode_feature_2)
        out = self.conv(rgb_decode_feature_2[0])

        return out, rgb_structure, inf_structure, incons_feature, inf_weight

