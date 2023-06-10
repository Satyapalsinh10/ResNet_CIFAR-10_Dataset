# Implementing various optimization strategies on ResNet for CIFAR-10 Dataset 

This repository contains the code and instructions for training the ResNet model on the CIFAR-10 dataset using different optimization strategies, namely Stochastic Gradient Descent (SGD), SGD with momentum (with a momentum term of 0.9), and ADAM. The purpose of this experiment is to compare the performance of these optimization methods in terms of training and testing error and loss.

## Requirements
- Python 3.6+
- PyTorch 1.6.0+
- Numpy

## Paper Implemented
ResNets paper: https://arxiv.org/pdf/1512.03385.pdf

## Steps to Implement:
1. Train

```git clone https://github.com/drgripa1/resnet-cifar10.git```
mkdir path/to/checkpoint_dir
python train.py --n 3 --checkpoint_dir path/to/checkpoint_dir
```
`n` means the network depth, you can choose from {3, 5, 7, 9}, which means ResNet-{20, 32, 44, 56}.
For other options, please refer helps: `python train.py -h`.
When you run the code for the first time, the dataset will be downloaded automatically.

2. Test

When your training is done, the model parameter file `path/to/checkpoint_dir/model_final.pth` will be generated.
```
python test.py --n 3 --params_path path/to/checkpoint_dir/model_final.pth
```

## Note
If you want to specify GPU to use, you should set environment variable `CUDA_VISIBLE_DEVICES=0`, for example.

## References
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep residual learning for image recognition," In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.
