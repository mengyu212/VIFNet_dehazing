# VIFNet: An End-to-end Visible-Infrared Fusion Network for Image Dehazing

![method](https://github.com/mengyu212/VIFNet_dehazing/blob/master/img/method.jpg)
## Performance
![performance](https://github.com/mengyu212/VIFNet_dehazing/blob/master/img/performance.jpg)

## Train and test
### dataset
Airsim-VID--We propose a foggy visible-infrared dataset based on AirSim , a high-fidelity simulation platform for autonomous vehicles, which can provide real-time ground truth and paired images under different degrees of fog conditions. Our dataset comprises 2,310 aligned hazy/clear/infrared image pairs, each corresponding to three different fog concentration coefficients.
You can download via https://pan.baidu.com/s/1HYMKXmGMJQ1x6JmBNxaABw (code: egc9) and modify the path of the dataset in data_utils.py. 
### training
```
cd your/path/to/VIFNet
python train_vifnet.py
```
The trained models will be placed in the /trained_models folder.
### testing
```
cd your/path/to/VIFNet
python test.py
```
The predicted image will be placed in the /pred_imgs_rgbt folder.

## Citation
```
@article{yu2024vifnet,
  title={VIFNet: An end-to-end visible-infrared fusion network for image dehazing},
  author={Yu, Meng and Cui, Te and Lu, Haoyang and Yue, Yufeng},
  journal={Neurocomputing},
  pages={128105},
  year={2024},
  publisher={Elsevier}
}
```
