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

# run this to test the model

import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.io import imread, imsave
import torch.nn.functional as F
import segyio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Test'], help='directory of test dataset')
    parser.add_argument('--sigma', default=0.2, type=float, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_sigma0.2'), help='directory of the model')
    parser.add_argument('--model_name', default='model_040.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    parser.add_argument('--src_segy_path',default='data/Test/test.segy',type=str,help='directory of test dataset')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path, src_segy_path=None):
    path = path if path.find('.') != -1 else path + '.segy'
    ext = os.path.splitext(path)[-1].lower()

    # 确保结果是2D数组（SEGY要求道数×采样点数）
    if len(result.shape) != 2:
        raise ValueError(f"地震数据必须是2D数组（道数×采样点数），当前维度：{result.shape}")
    trace_count, samples_count = result.shape  # 道数、采样点数

    # 1. 核心：保存为标准SEGY格式（重点优化）
    if ext in ('.sgy', '.segy'):
        # 1.1 构建SEGY规格（spec）
        spec = segyio.spec()
        spec.samples = range(samples_count)  # 采样点索引
        spec.tracecount = trace_count  # 道数
        spec.format = 5  # 数据格式：5=IEEE浮点（现代地震数据主流）
        spec.endian = 'big'  # 字节序：big-endian（SEGY标准）

        # 1.2 创建SEGY文件并写入数据
        with segyio.create(path, spec) as f:
            # 写入数据道
            f.trace = result

            # 复用原始SEGY的头信息（如果提供了src_segy_path，生成的SEGY更标准）
            if src_segy_path and os.path.exists(src_segy_path):
                with segyio.open(src_segy_path, 'r', ignore_geometry=True) as src_f:
                    # 复制文本头（3200字节）
                    f.text[0] = src_f.text[0]
                    # 复制二进制头（400字节）
                    f.bin = src_f.bin
                    # 复制道头（Trace Header）
                    for i in range(trace_count):
                        if i < src_f.tracecount:
                            f.header[i] = src_f.header[i]
        print(f"地震数据已保存为标准SEGY格式: {path}")
        print(f"道数: {trace_count}, 采样点数: {samples_count}")
    else:
        default_path = os.path.splitext(path)[0] + '.segy'
        save_result(result, default_path, src_segy_path)
        print(f"⚠️ 不支持的格式 {ext}，已默认保存为SEGY: {default_path}")


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
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
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


if __name__ == '__main__':

    args = parse_args()
    torch.cuda.empty_cache()
    # 加载模型
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):

        model = torch.load(os.path.join(args.model_dir, 'model.pth'))
        # load weights into new model
        log('load trained model on Train400 dataset by kai')
    else:
        model = torch.load(os.path.join(args.model_dir, args.model_name),weights_only=False,map_location=torch.device('cpu'))
        log('load trained model')



    model.eval()  # evaluation mode
#    model.Train400()
    if torch.cuda.is_available():
        model = model.cuda()

    # 2. 计算初始基线（加噪数据 vs 干净数据）：PSNR + SSIM
    data_dir = os.path.join(args.set_dir, args.set_names[0])
    segy_file = [f for f in os.listdir(data_dir) if f.endswith(('.segy', '.sgy'))][0]  # 唯一的SEGY文件
    segy_full_path = os.path.join(data_dir, segy_file)

    # 读取并预处理干净地震数据
    with segyio.open(segy_full_path, 'r', ignore_geometry=True) as f:
        x_clean = f.trace.raw[:].astype(np.float32)
    pmin, pmax = np.percentile(x_clean, 1), np.percentile(x_clean, 99)
    x_clean = 2 * (x_clean - pmin) / (pmax - pmin) - 1
    x_clean = np.clip(x_clean, -1, 1)

    # 加噪声（和推理时的输入一致）
    np.random.seed(0)
    x_noisy = x_clean + np.random.normal(0, args.sigma, x_clean.shape)
    x_noisy = x_noisy.astype(np.float32)

    # 计算初始基线（关键：未去噪的加噪数据 vs 干净数据）
    initial_psnr = compare_psnr(x_clean, x_noisy, data_range=2.0)
    initial_ssim = compare_ssim(x_clean, x_noisy, data_range=2.0)
    print("=" * 50)
    print(f"【初始基线（加噪未去噪）】")
    print(f"PSNR：{initial_psnr:.2f} dB | SSIM：{initial_ssim:.4f}")
    print("=" * 50)

    # 3. 创建结果目录
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)



    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []
        ssims = []

        for im in os.listdir(data_dir):
            if im.endswith(('.segy', '.sgy')):
                with torch.no_grad():
                    # 模型推理
                    y_ = torch.from_numpy(x_noisy).unsqueeze(0).unsqueeze(0)
                    if torch.cuda.is_available():
                        y_ = y_.cuda()
                    start_time = time.time()
                    x_denoised = model(y_)  # 训练好的模型输出去噪结果
                    x_denoised = x_denoised.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
                    elapsed_time = time.time() - start_time

                # 计算去噪后PSNR/SSIM
                psnr_denoised = compare_psnr(x_clean, x_denoised, data_range=2.0)
                ssim_denoised = compare_ssim(x_clean, x_denoised, data_range=2.0)
                psnrs.append(psnr_denoised)
                ssims.append(ssim_denoised)

                # 输出单文件去噪结果（对比初始基线）
                print(f"\n【去噪结果】文件：{im} | 耗时：{elapsed_time:.4f} 秒")
                print(f"PSNR：{psnr_denoised:.2f} dB（相比初始提升：{psnr_denoised - initial_psnr:.2f} dB）")
                print(f"SSIM：{ssim_denoised:.4f}（相比初始提升：{ssim_denoised - initial_ssim:.4f}）")

                # 保存去噪结果
                if args.save_result:
                    name, _ = os.path.splitext(im)
                    save_result(x_denoised, path=os.path.join(args.result_dir, set_cur, name + '_dncnn.segy'),
                                src_segy_path=args.src_segy_path)
                    save_result(x_noisy, path=os.path.join(args.result_dir, set_cur, name + '_noisy.segy'),
                                src_segy_path=args.src_segy_path)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        if args.save_result:
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'),src_segy_path=args.src_segy_path)
        print("\n" + "=" * 50)
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))








