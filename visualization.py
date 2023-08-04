import numpy as np
import matplotlib.pyplot as plt

loss_base = np.load('numpy_files/vifnet_100000_losses.npy')
psnr = np.load('numpy_files/RGBT_train_50000_psnrs.npy')
ssim = np.load('numpy_files/RGBT_train_50000_ssims.npy')

plt.subplot(211)
plt.title('ssim')
plt.plot(ssim)
plt.subplot(212)
plt.title('psnr')
plt.plot(psnr)
plt.show()
