# -*- coding: utf-8 -*-
import argparse
import os
import re
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from data_generator import DenoisingDataset
import data_generator as dg
import segyio
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.cuda.amp import autocast
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data', type=str, help='dataset root')
    parser.add_argument('--set_name', default='Val', type=str, help='validation set folder name')
    parser.add_argument('--sigma', default=0.05, type=float, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models'), type=str, help='checkpoint folder')
    parser.add_argument('--result_dir', default='val_results', type=str, help='save validation results')
    parser.add_argument('--batch_size', default=128, type=int, help='inference batch size')
    parser.add_argument('--seed', default=0, type=int, help='random seed for adding noise')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


class sum_squared_error(nn.Module):
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, reduction='sum'):
        super(sum_squared_error, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        loss = F.mse_loss(input, target, reduction=self.reduction)
        return loss.div(2)


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, kernel_size=3):
        super(DnCNN, self).__init__()
        padding = 1
        layers = []
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def calc_snr(x_clean, x_result):
    noise = x_clean - x_result
    ps = np.sum(x_clean ** 2)
    pn = np.sum(noise ** 2)
    return 10 * np.log10(ps / (pn + 1e-12))


def extract_epoch(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1


def evaluate_one_model(model, dataloader, device):
    criterion = sum_squared_error()
    model.eval()

    total_loss = 0.0
    total_samples = 0

    clean_list = []
    noisy_list = []
    denoised_list = []

    start_time = time.time()

    with torch.no_grad():
        for n_count, (batch_y, batch_x) in enumerate(dataloader):
            # 一般 DenoisingDataset 返回 (noisy, clean) 或 (target, input)
            # 你原代码里写的是(batch_y, batch_x)，并且训练里通常 y=clean, x=noisy
            batch_y = batch_y.to(device)   # clean
            batch_x = batch_x.to(device)   # noisy

            if device.type == 'cuda':
                output = model(batch_y)
                loss = criterion(output, batch_x)
            else:
                output = model(batch_y)
                loss = criterion(output, batch_x)

            total_loss += loss.item()
            
            total_samples += batch_x.size(0)

            clean_list.append(batch_y.detach().cpu().numpy())
            noisy_list.append(batch_x.detach().cpu().numpy())
            denoised_list.append(output.detach().cpu().numpy())

    elapsed = time.time() - start_time
    avg_loss = total_loss / (n_count+1)
    x_clean = np.concatenate(clean_list, axis=0)
    x_noisy = np.concatenate(noisy_list, axis=0)
    x_denoised = np.concatenate(denoised_list, axis=0)

    # 展平成一维算整体指标
    x_clean_flat = x_clean.reshape(-1)
    x_noisy_flat = x_noisy.reshape(-1)
    x_denoised_flat = x_denoised.reshape(-1)

    init_snr = calc_snr(x_clean_flat, x_noisy_flat)
    init_psnr = compare_psnr(x_clean_flat, x_noisy_flat, data_range=2.0)
    init_ssim = compare_ssim(x_clean_flat, x_noisy_flat, data_range=2.0)
    init_rmse = np.sqrt(np.mean((x_noisy_flat - x_clean_flat) ** 2))

    snr = calc_snr(x_clean_flat, x_denoised_flat)
    psnr = compare_psnr(x_clean_flat, x_denoised_flat, data_range=2.0)
    ssim = compare_ssim(x_clean_flat, x_denoised_flat, data_range=2.0)
    rmse = np.sqrt(np.mean((x_denoised_flat - x_clean_flat) ** 2))

    return  init_snr, init_psnr, init_ssim, init_rmse, avg_loss, snr, psnr, ssim, rmse, elapsed


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log(f"使用GPU: {torch.cuda.get_device_name(0)}")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    data_dir = os.path.join(args.set_dir, args.set_name)
    segy_files = [f for f in os.listdir(data_dir) if f.endswith(('.segy', '.sgy'))]
    if len(segy_files) == 0:
        raise FileNotFoundError(f'{data_dir} 下没有找到 segy 文件')

    segy_path = os.path.join(data_dir, segy_files[0])

    # 这里沿用你原本 patch 生成方式
    xs = dg.datagenerator(data_dir=data_dir)
    print(f'总patch数：{xs.shape[0]}')

    xs = xs.astype('float32')
    xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))   # N,H,W,C -> N,C,H,W

    DDataset = DenoisingDataset(xs, args.sigma)
    DLoader = DataLoader(
        dataset=DDataset,
        num_workers=4,
        drop_last=False,
        batch_size=args.batch_size,
        shuffle=False
    )

    model_set = os.path.join(args.model_dir, f'DnCNN_sigma{args.sigma}')
    model_files = [f for f in os.listdir(model_set) if f.endswith('.pth')]
    if len(model_files) == 0:
        raise FileNotFoundError(f'{model_set} 下没有找到 .pth 模型文件')

    model_files = sorted(model_files, key=extract_epoch)

    best_val_loss = 1e18
    best_epoch = -1
    best_model_name = ''

    result_txt = os.path.join(args.result_dir, f'val_results_sigma{args.sigma}.txt')

    first_init = True

    with open(result_txt, 'w', encoding='utf-8') as fw:
        fw.write(f"Validation file: {segy_files[0]}\n")
        fw.write(f"Patch count: {xs.shape[0]}\n")
        fw.write("=" * 60 + "\n")

        for model_name in model_files:
            model_path = os.path.join(model_set, model_name)
            log(f'正在验证: {model_name}')

            model = torch.load(model_path, weights_only=False, map_location=device)
            model.eval()
            model.to(device)

            ( init_snr, init_psnr, init_ssim, init_rmse,
             avg_loss, snr, psnr, ssim, rmse, elapsed) = evaluate_one_model(
                model=model,
                dataloader=DLoader,
                device=device
            )

            if first_init:
                print("=" * 60)
                print(f"验证集文件: {segy_files[0]}")
                print(f"初始基线  | SNR={init_snr:.2f} dB | "
                      f"PSNR={init_psnr:.2f} dB | SSIM={init_ssim:.4f} | RMSE={init_rmse:.6f}")
                print("=" * 60)

                fw.write(f"Initial SNR: {init_snr:.4f} dB\n")
                fw.write(f"Initial SSIM: {init_ssim:.6f}\n")
                fw.write(f"Initial PSNR: {init_psnr:.4f} dB\n")
                fw.write(f"Initial RMSE: {init_rmse:.6f}\n")
                fw.write("=" * 60 + "\n")
                first_init = False

            epoch_num = extract_epoch(model_name)
            msg = (f"[{model_name}] Epoch={epoch_num:03d} | "
                   f"Loss={avg_loss:.6f} | SNR={snr:.2f} dB | PSNR={psnr:.2f} dB | "
                   f"SSIM={ssim:.4f} | RMSE={rmse:.6f} | Time={elapsed:.2f}s")
            print(msg)
            fw.write(msg + '\n')

            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_epoch = epoch_num
                best_model_name = model_name
                best_snr = snr
                best_psnr = psnr
                best_ssim = ssim
                best_rmse = rmse

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        fw.write("=" * 60 + "\n")
        fw.write(f"Best Epoch: {best_epoch}\n")
        fw.write(f"Best Model: {best_model_name}\n")
        fw.write(f"Best Val Loss: {best_val_loss:.6f}\n")
        fw.write(f"Best SNR: {best_snr:.4f} dB\n")
        fw.write(f"Best PSNR: {best_psnr:.4f} dB\n")
        fw.write(f"Best SSIM: {best_ssim:.6f}\n")
        fw.write(f"Best RMSE: {best_rmse:.6f}\n")

    print("=" * 60)
    print(f"最佳模型: {best_model_name}")
    print(f"最佳轮次: {best_epoch}")
    print(f"最佳Val Loss: {best_val_loss:.6f}")
    print(f"对应SNR: {best_snr:.2f} dB")
    print(f"对应PSNR: {best_psnr:.4f} dB")
    print(f"对应SSIM: {best_ssim:.4f}")
    print(f"对应RMSE: {best_rmse:.6f}")
    print(f"结果已保存到: {result_txt}")
    print("=" * 60)