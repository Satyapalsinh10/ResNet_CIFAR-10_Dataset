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

```git clone https://github.com/Satyapalsinh10/ResNet_CIFAR-10_Dataset.git```

```mkdir resnet-cifar10/SGD_LR_0.1```

```python resnet-cifar10/train.py --n 3 --momentum 0.0 --checkpoint_dir /content/resnet-cifar10/SGD_LR_0.1```

`n` means the network depth, you can choose from {3, 5, 7, 9}, which means ResNet-{20, 32, 44, 56}.
For other options, please refer helps: `python train.py -h`.
When you run the code for the first time, the dataset will be downloaded automatically.


## Visualization:
```
import matplotlib.pyplot as plt
# load data from file
data_file = open('/content/resnet-cifar10/SGD_LR_0.1/loss_log.txt', 'r')
lines = data_file.readlines()
data_file.close()
# initialize lists to store iteration and loss values
iters = []
losses = []
# extract iteration and loss values from each line
for line in lines:
if 'iter:' in line:
iter_val = int(line.split(':')[1].split(',')[0].strip())
iters.append(iter_val)
loss_val = float(line.split(':')[2].strip())
losses.append(loss_val)
# plot the iter vs loss graph
plt.plot(iters, losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.show()
```

# result image:

2. Test

When your training is done, the model parameter file `/content/resnet-cifar10/SGD_LR_0.01/model_final.pth` will be generated.
```
python resnet-cifar10/test.py --n 3 --params_path /content/resnet-cifar10/SGD_LR_0.01/model_final.pth
```

## References
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep residual learning for image recognition," In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.
