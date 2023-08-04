import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt


BS = opt.bs
print('batch size is:', BS)
crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size    # (240,240)
else:
    crop_size = 'whole_img'


def tensorShow(tensors, titles=None):
        fig = plt.figure()
        for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()


class RGB_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, input_h=480, input_w=640, format='.png'):
        super(RGB_Dataset, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        self.input_h = input_h
        self.input_w = input_w
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'haze'))   # haze
        self.haze_imgs = [os.path.join(path, 'haze', img) for img in self.haze_imgs_dir]   # haze
        self.clear_dir = os.path.join(path, 'clear')
        self.nir_dir = os.path.join(path, 'nir')

    def __getitem__(self, index):
        haze = np.asarray(Image.open(self.haze_imgs[index]).convert("RGB"))
        haze = np.asarray(Image.fromarray(haze).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        img = self.haze_imgs[index]
        id = img.split('/')[-1].split('_')[0]
        clear_name = id + '_rgb.png'
        clear = np.asarray(Image.open(os.path.join(self.clear_dir, clear_name)).convert("RGB"))
        clear = np.asarray(Image.fromarray(clear).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        nir_name = id + '_thermal_foggy_1.5.png'
        nir = np.asarray(Image.open(os.path.join(self.nir_dir, nir_name)).convert("RGB"))
        nir = np.asarray(Image.fromarray(nir).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        return torch.tensor(haze), torch.tensor(nir), torch.tensor(clear)

    def __len__(self):
        return len(self.haze_imgs)


class val_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, input_h=480, input_w=640, format='.png'):
        super(val_Dataset, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        self.input_h = input_h
        self.input_w = input_w
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'haze'))   # haze
        self.haze_imgs = [os.path.join(path, 'haze', img) for img in self.haze_imgs_dir]   # haze
        self.nir_dir = os.path.join(path, 'nir')

    def __getitem__(self, index):
        haze = np.asarray(Image.open(self.haze_imgs[index]).convert("RGB"))
        haze = np.asarray(Image.fromarray(haze).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        img = self.haze_imgs[index]
        id = img.split('/')[-1].split('_')[-1]
        nir_name = id
        nir = np.asarray(Image.open(os.path.join(self.nir_dir, nir_name)).convert("RGB"))
        nir = np.asarray(Image.fromarray(nir).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        return torch.tensor(haze), torch.tensor(nir)

    def __len__(self):
        return len(self.haze_imgs)


path = '/data/ym/multidata'           # path to your 'dataset' folder
RGB_train_loader = DataLoader(dataset=RGB_Dataset(path+'/train',
                               train=True, format='.png'), batch_size=BS, shuffle=True)
RGB_test_loader = DataLoader(dataset=RGB_Dataset(path+'/test',
                               train=False, size='whole img', format='.png'), batch_size=1, shuffle=False)
RGB_val_loader = DataLoader(dataset=RGB_Dataset(path+'/val',
                               train=False, size='whole img', format='.png'), batch_size=1, shuffle=False)


if __name__ == "__main__":
    pass
