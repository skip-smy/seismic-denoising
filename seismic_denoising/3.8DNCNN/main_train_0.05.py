# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to Train400 the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1. 
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default="data/Train", type=str, help='path of Train data')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level')
parser.add_argument('--epoch', default=85, type=int, help='number of Train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma

save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))
if not os.path.exists('models'):
    os.mkdir('models')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)




class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU())
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()
    if cuda:
        model = model.cuda()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))

    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    epoch_loss_history = []  # 每个epoch的平均loss
    # 加载历史loss（如果存在）
    loss_history_path = os.path.join(save_dir, 'epoch_loss_history.txt')
    if os.path.exists(loss_history_path) and initial_epoch > 0:
        epoch_loss_history = np.loadtxt(loss_history_path).tolist()

    for epoch in range(initial_epoch, n_epoch):

        
        xs = dg.datagenerator(data_dir=args.train_data)
        print(f'总patch数：{xs.shape[0]}')
        batch_num = xs.shape[0] // batch_size
        print(f'每个epoch的循环数：{batch_num}')
        xs = xs.astype('float32')
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, (batch_y,batch_x) in enumerate(DLoader):
                optimizer.zero_grad()
                if  cuda:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                if n_count == 0 and epoch == 0:
                    # batch_y：加噪数据（模型输入），batch_x：干净数据（标签）
                    print("加噪数据均值：", batch_y.mean().item())
                    print("干净数据均值：", batch_x.mean().item())
                    print("输入-标签的MSE：", F.mse_loss(batch_y, batch_x).item())  # 这是“理论最小Loss下限”
                    print("模型输出-标签的MSE：", F.mse_loss(model(batch_y), batch_x).item())
                #前向传播 + 计算损失 + 反向传播 + 优化
                loss = criterion(model(batch_y), batch_x)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            


                #批次打印损失
                if n_count % 10 == 0:
                    batch_avg_loss = loss.item() / batch_size
                    log(f'epoch {epoch + 1:4d}, batch {n_count:4d}/{batch_num:4d}, loss = {batch_avg_loss:.4f}')
        scheduler.step(epoch)  # 更新学习率
        #计算epoch平均损失
        elapsed_time = time.time() - start_time
        avg_loss = epoch_loss / (n_count + 1)  # 避免n_count为0的情况
        epoch_loss_history.append(avg_loss)
        # 打印epoch总结
        log(f'epoch = {epoch + 1:4d}, avg loss = {avg_loss:.4f}, time = {elapsed_time:.2f} s')
        #保存epoch级loss
        np.savetxt(loss_history_path, np.array(epoch_loss_history), fmt='%.6f')
        #保存模型
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        # 保存训练日志
        with open(os.path.join(save_dir, 'train_log.txt'), 'a') as f:
            f.write(f"{epoch + 1},{avg_loss:.6f},{elapsed_time:.2f}\n")


        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_loss_history) + 1), epoch_loss_history, 'b-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('DnCNN Training Loss Curve (sigma={})'.format(sigma))
        plt.legend()
        plt.grid(True)
        # 保存图片到模型目录
        loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()
        print('Loss曲线已保存到：', loss_curve_path)





# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to Train400 the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1.
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default="data/Train", type=str, help='path of Train data')
parser.add_argument('--sigma', default=0.2, type=float, help='noise level')
parser.add_argument('--epoch', default=85, type=int, help='number of Train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma

save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))
if not os.path.exists('models'):
    os.mkdir('models')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

class HardSwish(nn.Module):
    def forward(self,x):
        return x*F.relu6(x+3.0)/6.0


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        self.hard_swish = HardSwish()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(self.hard_swish)
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(self.hard_swish)
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()
    if cuda:
        model = model.cuda()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))

    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    epoch_loss_history = []  # 每个epoch的平均loss
    # 加载历史loss（如果存在）
    loss_history_path = os.path.join(save_dir, 'epoch_loss_history.txt')
    if os.path.exists(loss_history_path) and initial_epoch > 0:
        epoch_loss_history = np.loadtxt(loss_history_path).tolist()

    for epoch in range(initial_epoch, n_epoch):


        xs = dg.datagenerator(data_dir=args.train_data)
        print(f'总patch数：{xs.shape[0]}')
        batch_num = xs.shape[0] // batch_size
        print(f'每个epoch的循环数：{batch_num}')
        xs = xs.astype('float32')
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, (batch_y,batch_x) in enumerate(DLoader):
                optimizer.zero_grad()
                if  cuda:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                if n_count == 0 and epoch == 0:
                    # batch_y：加噪数据（模型输入），batch_x：干净数据（标签）
                    print("加噪数据均值：", batch_y.mean().item())
                    print("干净数据均值：", batch_x.mean().item())
                    print("输入-标签的MSE：", F.mse_loss(batch_y, batch_x).item())  # 这是“理论最小Loss下限”
                    print("模型输出-标签的MSE：", F.mse_loss(model(batch_y), batch_x).item())
                #前向传播 + 计算损失 + 反向传播 + 优化
                loss = criterion(model(batch_y), batch_x)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()



                #批次打印损失
                if n_count % 10 == 0:
                    batch_avg_loss = loss.item() / batch_size
                    log(f'epoch {epoch + 1:4d}, batch {n_count:4d}/{batch_num:4d}, loss = {batch_avg_loss:.4f}')
        scheduler.step(epoch)  # 更新学习率
        #计算epoch平均损失
        elapsed_time = time.time() - start_time
        avg_loss = epoch_loss / (n_count + 1)  # 避免n_count为0的情况
        epoch_loss_history.append(avg_loss)
        # 打印epoch总结
        log(f'epoch = {epoch + 1:4d}, avg loss = {avg_loss:.4f}, time = {elapsed_time:.2f} s')
        #保存epoch级loss
        np.savetxt(loss_history_path, np.array(epoch_loss_history), fmt='%.6f')
        #保存模型
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        # 保存训练日志
        with open(os.path.join(save_dir, 'train_log.txt'), 'a') as f:
            f.write(f"{epoch + 1},{avg_loss:.6f},{elapsed_time:.2f}\n")


        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_loss_history) + 1), epoch_loss_history, 'b-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('DnCNN Training Loss Curve (sigma={})'.format(sigma))
        plt.legend()
        plt.grid(True)
        # 保存图片到模型目录
        loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()
        print('Loss曲线已保存到：', loss_curve_path)





# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to Train400 the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1.
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default="data/Train", type=str, help='path of Train data')
parser.add_argument('--sigma', default=0.2, type=float, help='noise level')
parser.add_argument('--epoch', default=85, type=int, help='number of Train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma

save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))
if not os.path.exists('models'):
    os.mkdir('models')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

class HardSwish(nn.Module):
    def forward(self,x):
        return x*F.relu6(x+3.0)/6.0


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        self.hard_swish = HardSwish()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(self.hard_swish)
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(self.hard_swish)
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()
    if cuda:
        model = model.cuda()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))

    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    epoch_loss_history = []  # 每个epoch的平均loss
    # 加载历史loss（如果存在）
    loss_history_path = os.path.join(save_dir, 'epoch_loss_history.txt')
    if os.path.exists(loss_history_path) and initial_epoch > 0:
        epoch_loss_history = np.loadtxt(loss_history_path).tolist()

    for epoch in range(initial_epoch, n_epoch):


        xs = dg.datagenerator(data_dir=args.train_data)
        print(f'总patch数：{xs.shape[0]}')
        batch_num = xs.shape[0] // batch_size
        print(f'每个epoch的循环数：{batch_num}')
        xs = xs.astype('float32')
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, (batch_y,batch_x) in enumerate(DLoader):
                optimizer.zero_grad()
                if  cuda:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                if n_count == 0 and epoch == 0:
                    # batch_y：加噪数据（模型输入），batch_x：干净数据（标签）
                    print("加噪数据均值：", batch_y.mean().item())
                    print("干净数据均值：", batch_x.mean().item())
                    print("输入-标签的MSE：", F.mse_loss(batch_y, batch_x).item())  # 这是“理论最小Loss下限”
                    print("模型输出-标签的MSE：", F.mse_loss(model(batch_y), batch_x).item())
                #前向传播 + 计算损失 + 反向传播 + 优化
                loss = criterion(model(batch_y), batch_x)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()



                #批次打印损失
                if n_count % 10 == 0:
                    batch_avg_loss = loss.item() / batch_size
                    log(f'epoch {epoch + 1:4d}, batch {n_count:4d}/{batch_num:4d}, loss = {batch_avg_loss:.4f}')
        scheduler.step(epoch)  # 更新学习率
        #计算epoch平均损失
        elapsed_time = time.time() - start_time
        avg_loss = epoch_loss / (n_count + 1)  # 避免n_count为0的情况
        epoch_loss_history.append(avg_loss)
        # 打印epoch总结
        log(f'epoch = {epoch + 1:4d}, avg loss = {avg_loss:.4f}, time = {elapsed_time:.2f} s')
        #保存epoch级loss
        np.savetxt(loss_history_path, np.array(epoch_loss_history), fmt='%.6f')
        #保存模型
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        # 保存训练日志
        with open(os.path.join(save_dir, 'train_log.txt'), 'a') as f:
            f.write(f"{epoch + 1},{avg_loss:.6f},{elapsed_time:.2f}\n")


        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_loss_history) + 1), epoch_loss_history, 'b-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('DnCNN Training Loss Curve (sigma={})'.format(sigma))
        plt.legend()
        plt.grid(True)
        # 保存图片到模型目录
        loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()
        print('Loss曲线已保存到：', loss_curve_path)





# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to Train400 the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1.
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default="data/Train", type=str, help='path of Train data')
parser.add_argument('--sigma', default=0.2, type=float, help='noise level')
parser.add_argument('--epoch', default=85, type=int, help='number of Train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma

save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))
if not os.path.exists('models'):
    os.mkdir('models')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

class HardSwish(nn.Module):
    def forward(self,x):
        return x*F.relu6(x+3.0)/6.0


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        self.hard_swish = HardSwish()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(self.hard_swish)
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(self.hard_swish)
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()
    if cuda:
        model = model.cuda()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))

    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    epoch_loss_history = []  # 每个epoch的平均loss
    # 加载历史loss（如果存在）
    loss_history_path = os.path.join(save_dir, 'epoch_loss_history.txt')
    if os.path.exists(loss_history_path) and initial_epoch > 0:
        epoch_loss_history = np.loadtxt(loss_history_path).tolist()

    for epoch in range(initial_epoch, n_epoch):


        xs = dg.datagenerator(data_dir=args.train_data)
        print(f'总patch数：{xs.shape[0]}')
        batch_num = xs.shape[0] // batch_size
        print(f'每个epoch的循环数：{batch_num}')
        xs = xs.astype('float32')
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, (batch_y,batch_x) in enumerate(DLoader):
                optimizer.zero_grad()
                if  cuda:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                if n_count == 0 and epoch == 0:
                    # batch_y：加噪数据（模型输入），batch_x：干净数据（标签）
                    print("加噪数据均值：", batch_y.mean().item())
                    print("干净数据均值：", batch_x.mean().item())
                    print("输入-标签的MSE：", F.mse_loss(batch_y, batch_x).item())  # 这是“理论最小Loss下限”
                    print("模型输出-标签的MSE：", F.mse_loss(model(batch_y), batch_x).item())
                #前向传播 + 计算损失 + 反向传播 + 优化
                loss = criterion(model(batch_y), batch_x)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()



                #批次打印损失
                if n_count % 10 == 0:
                    batch_avg_loss = loss.item() / batch_size
                    log(f'epoch {epoch + 1:4d}, batch {n_count:4d}/{batch_num:4d}, loss = {batch_avg_loss:.4f}')
        scheduler.step(epoch)  # 更新学习率
        #计算epoch平均损失
        elapsed_time = time.time() - start_time
        avg_loss = epoch_loss / (n_count + 1)  # 避免n_count为0的情况
        epoch_loss_history.append(avg_loss)
        # 打印epoch总结
        log(f'epoch = {epoch + 1:4d}, avg loss = {avg_loss:.4f}, time = {elapsed_time:.2f} s')
        #保存epoch级loss
        np.savetxt(loss_history_path, np.array(epoch_loss_history), fmt='%.6f')
        #保存模型
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        # 保存训练日志
        with open(os.path.join(save_dir, 'train_log.txt'), 'a') as f:
            f.write(f"{epoch + 1},{avg_loss:.6f},{elapsed_time:.2f}\n")


        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_loss_history) + 1), epoch_loss_history, 'b-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('DnCNN Training Loss Curve (sigma={})'.format(sigma))
        plt.legend()
        plt.grid(True)
        # 保存图片到模型目录
        loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()
        print('Loss曲线已保存到：', loss_curve_path)





