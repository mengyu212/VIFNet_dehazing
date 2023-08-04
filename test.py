import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data_utils.data_utils import *
from models.VIFnet import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from models.mssim import MSSSIM


loaders_ = {
    'RGB_train': RGB_train_loader,
    'RGB_test': RGB_test_loader,
    'RGB_val': RGB_val_loader
}
loader_train = loaders_[opt.valset]
rgb, ir, gt = next(iter(loader_train))


def tensorShow(tensors, titles=['haze']):
        fig=plt.figure()
        for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='rgbt', help='dataset')
parser.add_argument('--test_imgs', type=str, default='test_imgs', help='Test imgs folder')
opt = parser.parse_args()
dataset = opt.task

output_dir = f'pred_imgs_{dataset}/'
print("pred_dir:", output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir = f'trained_models/AECR.pk.best.best'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir, map_location=device)
net = vifnet(3, 3)
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()
net = net.to(device)
# print(net)
# gt = Image.open('/home/ym/Downloads/datasets/test/clear/01466D.png')
# gt = tfs.Compose([
#     tfs.ToTensor(),
#     tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
# ])(gt)[None, ::]

rgb = rgb.to(device)
ir = ir.to(device)
gt = gt.to(device)



with torch.no_grad():
    out, rgb_structure, inf_structure, incons_fea, inf_weight = net(rgb, ir)
    ssim1 = ssim(out, gt).item()
    psnr1 = psnr(out, gt)
    ts = torch.squeeze(out.clamp(0, 1).cpu())
    # print(rgb_structure[0].shape)
    # t_stru=np.squeeze(inf_weight[0].clamp(0, 1).cpu())
    # t_stru=np.transpose(t_stru,[1,2,0])
    # print(t_stru.shape)
    # ts = torch.squeeze(t_stru[:,:,10])

# plt.figure()
# for i in range(12):
#     ax=plt.subplot(3,4,i+1)
#     plt.imshow(t_stru[:,:,i],cmap='gray') 
# plt.show()


print('ssim:', ssim1)
print('psnr:', psnr1)
# tensorShow([rgb.cpu(), t_stru[:,:,32].cpu()], ['haze', 'pred'])
vutils.save_image(ts, output_dir+'pred.png')

# edge_detect = Edge(3)
# edge_detect = edge_detect.to(device)
# out_edge = edge_detect(out)
# gt_edge = edge_detect(y)
# tensorShow([out_edge.cpu().detach(), gt_edge.cpu().detach()], ['out', 'gt'])


# for im in os.listdir(img_dir):
#     print(f'\r {im}', end='', flush=True)
#     haze = Image.open(img_dir+im)
#     haze1 = tfs.Compose([
#         tfs.ToTensor(),
#         tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
#     ])(haze)[None, ::]
#     haze_no = tfs.ToTensor()(haze)[None, ::]
#     with torch.no_grad():
#         # pred, _, _, _ = net(haze1)
#         pred = net(haze1, gt)
#     ts = torch.squeeze(pred.clamp(0, 1).cpu())
#     tensorShow([haze_no, pred.clamp(0, 1).cpu()], ['haze', 'pred'])
#     vutils.save_image(ts, output_dir+im.split('.')[0]+'_fea.png')
