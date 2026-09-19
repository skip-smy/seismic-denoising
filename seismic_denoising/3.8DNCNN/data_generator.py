# -*- coding: utf-8 -*-

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

# no need to run this code separately


import glob
import segyio
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch

patch_size, stride = 35, 10
aug_times = 1
scales = [1]
batch_size = 128


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]
        batch_x = batch_x.permute(1, 0, 2)
        noise = torch.randn_like(batch_x)*(self.sigma)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    vlim = np.percentile(np.abs(x),99)
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.fliplr(img)
    elif mode == 3:
        return np.flipud(np.fliplr(img))



def gen_patches(segy_path):
    # get multiscale patches from a single image
    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        # f.trace.raw[:] 直接读取所有道，形状为 (n_traces, n_samples)
        seismic = f.trace.raw[:].astype(np.float32)
    h, w = seismic.shape
    patches = []
    for s in scales:
        if s != 1:
            continue
        for i in range(0, h-patch_size+1, stride):
            for j in range(0, w-patch_size+1, stride):
                x = seismic[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    patch_aug = data_aug(x, mode=np.random.randint(0, 4))
                    patches.append(patch_aug)
    return np.array(patches, dtype=np.float32)





def datagenerator(data_dir, verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.segy') + glob.glob(data_dir+'/*.sgy')  # get name list of all .png files
    if not file_list:
        raise ValueError("未找到2D SEGY文件，请检查目录和后缀！")
    # initrialize
    data = []
    # generate patches
    for i, segy_file in enumerate(file_list):
        try:
            patches = gen_patches(segy_file)
            data.extend(patches)
            if verbose:
                print(f"{i + 1}/{len(file_list)} - {segy_file} 处理完成，生成 {len(patches)} 个2D补丁")
        except Exception as e:
            print(f"处理 {segy_file} 失败：{e}")
            continue
    data = np.array(data, dtype=np.float32)
    print(data.shape)
    print(data.ndim)
    data = np.expand_dims(data, axis=1)
    pmin, pmax = np.percentile(data, 1), np.percentile(data, 99)  # 按百分位数过滤异常值
    print(pmin)
    print(pmax)
    data = 2*(data - pmin) / (pmax - pmin) - 1  # 归一到[-1,1]

    # 3. 强制截断到 [-1, 1]，防止极端异常值超出范围
    data = np.clip(data, -1, 1)
    discard_n = len(data) - len(data) // batch_size * batch_size
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    print(data.ndim)
    print(data.min())
    print(data.max())
    return data


if __name__ == '__main__': 

    data = datagenerator(data_dir='data/Train',verbose = True)


#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')       