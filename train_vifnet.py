import torch, os, sys, torchvision, argparse
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
from option import opt
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from metrics import psnr, ssim
from data_utils.data_utils import *
from models.VIFnet import *
from models.losses import DiceLoss, CharbonnierLoss, SSIMLoss, LossNetwork
from torchvision.models import vgg16
from models.mssim import MSSSIM


warnings.filterwarnings('ignore')

models_={
	'VIFnet': vifnet(3, 3),
}

loaders_={
	'RGB_train': RGB_train_loader,
	'RGB_test': RGB_test_loader
}
edge_detect = Edge(3)
edge_detect = edge_detect.to(opt.device)

start_time = time.time()
start_time = time.time()
model_name = opt.model_name
steps = opt.eval_step * opt.epochs
T = steps


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
	lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
	return lr


def train(net, loader_train, loader_test, optim, criterion):
	losses = []
	start_step = 0
	max_ssim = 0
	max_psnr = 0
	ssims = []
	psnrs = []

	print(os.path.exists(opt.model_dir))
	if opt.resume and os.path.exists(opt.model_dir):
		if opt.pre_model != 'null':
			ckp = torch.load('./trained_models/'+opt.pre_model)
			print('pre_model exists.')
		else:
			ckp = torch.load(opt.model_dir)

		print(f'resume from {opt.model_dir}')
		losses = ckp['losses']
		net.load_state_dict(ckp['model'])
		optim.load_state_dict(ckp['optimizer'])
		start_step = ckp['step']
		max_ssim = ckp['max_ssim']
		max_psnr = ckp['max_psnr']
		psnrs = ckp['psnrs']
		ssims = ckp['ssims']
		print(f'max_psnr: {max_psnr} max_ssim: {max_ssim}')
		print(f'start_step:{start_step} start training ---')
	else:
		print('train from scratch *** ')

	for step in range(start_step+1, steps+1):
		net.train()
		lr = opt.lr
		if not opt.no_lr_sche:
			lr = lr_schedule_cosdecay(step,T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr
		rgb, nir, gt = next(iter(loader_train))    # [x, y] 10:10
		rgb = rgb.to(opt.device)
		nir = nir.to(opt.device)
		gt = gt.to(opt.device)
		net = net.to(opt.device)
		out, _, _, _, _ = net(rgb, nir)

		out = out.to(opt.device)
		out_edge = edge_detect(out)
		gt_edge = edge_detect(gt)


		loss = criterion[0](out, gt)
		if opt.perloss:
			loss2 = criterion[1](out, gt)  # perceptual loss
			loss = loss + 0.04 * loss2  # 0.96, 0.04
		# loss_ssim = criterion[1](out, gt)
		loss_rgb_stru = torch.sum(criterion_grad(out_edge, gt_edge)) / 1000   # np.sum
		loss_mssim = criterion[1](out, gt)
		
		loss = loss + 0.03*loss_rgb_stru + 0.1*(1-loss_mssim)    # adjust parameters weights according to reference,L_all

		# loss = 0.16*loss + 0.84*(1-loss_mssim)   # L_combination3
		# loss = 0.16*loss + loss_rgb_stru         # L_combination4
		# loss = 0.16 * loss

		loss.backward()
		
		optim.step()
		optim.zero_grad()
		losses.append(loss.item())

		print(f'\rloss:{loss.item():.5f} l2:{loss_rgb_stru:.5f}   '
			f'step :{step}/{steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time)/60 :.1f} min', end='', flush=True)


		if step % opt.eval_step == 0:
			epoch = int(step / opt.eval_step)

			save_model_dir = opt.model_dir
			with torch.no_grad():
				# ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)
				ssim_eval, psnr_eval = test(net, loader_test)

			log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'

			print(log)
			with open(f'logs_train/{opt.model_name}.txt', 'a') as f:
				f.write(log + '\n')

			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)

			if psnr_eval > max_psnr:
				max_ssim = max(max_ssim, ssim_eval)
				max_psnr = max(max_psnr, psnr_eval)
				save_model_dir = opt.model_dir + '.best'
				print(
					f'\n model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')

			torch.save({
				'epoch': epoch,
				'step': step,
				'max_psnr': max_psnr,
				'max_ssim': max_ssim,
				'ssims': ssims,
				'psnrs': psnrs,
				'losses': losses,
				'model': net.state_dict(),
				'optimizer': optim.state_dict()
			}, save_model_dir)

	np.save(f'./numpy_files/{model_name}_{steps}_losses.npy', losses)
	np.save(f'./numpy_files/{model_name}_{steps}_ssims.npy', ssims)
	np.save(f'./numpy_files/{model_name}_{steps}_psnrs.npy', psnrs)


def test(net, loader_test):
	net.eval()
	torch.cuda.empty_cache()
	ssims = []
	psnrs = []

	for i, (input_rgb, input_nir, targets) in enumerate(loader_test):
		input_rgb = input_rgb.to(opt.device)
		input_nir = input_nir.to(opt.device)
		targets = targets.to(opt.device)
		with torch.no_grad():
			pred, _, _, _, _ = net(input_rgb, input_nir)

		ssim1 = ssim(pred, targets).item()
		psnr1 = psnr(pred, targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)

	return np.mean(ssims), np.mean(psnrs)


def set_seed_torch(seed=2018):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True


criterion_grad = DiceLoss()


if __name__ == "__main__":

	set_seed_torch(666)

	if not opt.resume and os.path.exists(f'logs_train/{opt.model_name}.txt'):
		print(f'./logs_train/{opt.model_name}.txt 已存在，请删除该文件……')
		exit()
	# with open(f'./logs_train/args_{opt.model_name}.txt', 'w') as f:
	# 	json.dump(opt.__dict__, f, indent=2)

	loader_train = loaders_[opt.trainset]
	loader_test = loaders_[opt.testset]

	net = vifnet(3, 3)
	net = net.to(opt.device)
	epoch_size = len(loader_train)
	print("epoch_size: ", epoch_size)
	if opt.device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True
	pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total_params: ==> {}".format(pytorch_total_params))
	criterion=[]
	criterion.append(nn.L1Loss().to(opt.device))
	if opt.perloss:
		vgg_model = vgg16(pretrained=True).features[:16]
		vgg_model = vgg_model.to(opt.device)
		for param in vgg_model.parameters():
			param.requires_grad = False
		criterion.append(LossNetwork(vgg_model).to(opt.device))

	# criterion.append(SSIMLoss().to(opt.device))
	criterion.append(MSSSIM().to(opt.device))    # multiscale ssim loss

	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),
						   lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net, loader_train, loader_test, optimizer, criterion)
	

