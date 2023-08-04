import torch.nn as nn
import torch
from metrics import ssim
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1
        self.balance = 1.1

    def forward(self, inputs, targets):
        n, c, h, w = inputs.size()

        input_flat = inputs.view(-1)
        target_flat = targets.view(-1)
        intersecion = input_flat * target_flat
        unionsection = input_flat.pow(2).sum() + target_flat.pow(2).sum() + self.smooth
        loss = unionsection/(2 * intersecion.sum() + self.smooth)
        loss = loss.sum()

        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, x, y):
        ssim1 = ssim(x, y).item()
        return 1-ssim1


# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))
        return sum(loss)/len(loss)