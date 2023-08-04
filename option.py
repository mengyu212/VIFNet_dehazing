import torch,os,sys,torchvision,argparse
import torch,warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str,default='Automatic detection')
parser.add_argument('--resume', type=bool,default=True)
parser.add_argument('--epochs', type=int,default=100)  # 100
parser.add_argument('--eval_step', type=int,default=1000)    # 5000
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')   #0.0001
parser.add_argument('--trainset', type=str,default='RGB_train')
parser.add_argument('--testset', type=str,default='RGB_test')
parser.add_argument('--valset', type=str, default='RGB_val')
parser.add_argument('--net', type=str,default='vifnet')

parser.add_argument('--bs', type=int,default=1,help='batch size')     # 16
parser.add_argument('--crop', action='store_true')
parser.add_argument('--crop_size', type=int,default=240,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche', action='store_true',help='no lr cos schedule')

parser.add_argument('--transfer', type=bool,default=False)
parser.add_argument('--pre_model', type=str,default='vifnet.pk.best')



# parser.add_argument('--pre_train_epochs', type=int, default=10, help='train with l1 and fft')
# parser.add_argument('--lr_decay', type=bool, default=True)
# parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='lr decay rate')
# parser.add_argument('--lr_decay_win', type=int, default=4, help='lr decay windows: epoch')
#
# parser.add_argument('--eval_dataset', type=bool, default=False)
parser.add_argument('--model_dir', type=str, default='./trained_models/')
parser.add_argument('--perloss', action='store_true', default=False, help='perceptual loss')

opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'
# if not opt.transfer:
# 	opt.model_name = opt.trainset + '_' + opt.net.split('.')[0] + '_' + str(opt.model_name)
# else:
opt.model_name = 'vifnet'

opt.model_dir=opt.model_dir + opt.model_name + '.pk.best'
log_dir='logs/'+opt.model_name if not opt.transfer else 'logs/'+opt.model_name+'_transfer_' + opt.model_info

print(opt)
print('model_dir:',opt.model_dir)
print(f'log_dir: {log_dir}')


if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
	os.mkdir('numpy_files')
# if not os.path.exists('logs'):
# 	os.mkdir('logs')
# if not os.path.exists('samples'):
# 	os.mkdir('samples')
# if not os.path.exists(f"samples/{opt.model_name}"):
# 	os.mkdir(f'samples/{opt.model_name}')
# if not os.path.exists(log_dir):
# 	os.mkdir(log_dir)
