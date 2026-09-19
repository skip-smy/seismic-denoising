# -*- coding: utf-8 -*-

import argparse
import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import segyio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.cuda.amp import autocast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Test'], help='directory of test dataset')
    parser.add_argument('--sigma', default=0.05, type=float, help='noise level')
    parser.add_argument('--model_1_dir', default=os.path.join('models'), help='directory of the model')
    parser.add_argument('--model_name', default='model_042.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='real_results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    parser.add_argument('--src_segy_path', default='data/P.sgy', type=str, help='source segy path')
    parser.add_argument('--batch_size', default=128, type=int, help='patch inference batch size')
    parser.add_argument('--patch_size', default=35, type=int, help='test patch size')
    parser.add_argument('--stride', default=10, type=int, help='test patch stride')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def calc_snr(x_clean, x_result):
    noise = x_clean - x_result
    ps = np.sum(x_clean ** 2)
    pn = np.sum(noise ** 2)
    return 10 * np.log10(ps / (pn + 1e-12))


def save_result(result, path, src_segy_path=None):
    path = path if path.find('.') != -1 else path + '.segy'
    ext = os.path.splitext(path)[-1].lower()

    if len(result.shape) != 2:
        raise ValueError(f"地震数据必须是2D数组（道数×采样点数），当前维度：{result.shape}")

    trace_count, samples_count = result.shape

    if ext in ('.sgy', '.segy'):
        spec = segyio.spec()
        spec.samples = range(samples_count)
        spec.tracecount = trace_count
        spec.format = 5
        spec.endian = 'big'

        with segyio.create(path, spec) as f:
            f.trace = result.astype(np.float32)

            if src_segy_path and os.path.exists(src_segy_path):
                with segyio.open(src_segy_path, 'r', ignore_geometry=True) as src_f:
                    try:
                        f.text[0] = src_f.text[0]
                    except Exception:
                        pass

                    # 不直接 f.bin = src_f.bin，避免 BinField 类型问题
                    try:
                        for key, value in src_f.bin.items():
                            try:
                                f.bin[int(key)] = value
                            except Exception:
                                pass
                    except Exception:
                        pass

                    for i in range(min(trace_count, src_f.tracecount)):
                        try:
                            f.header[i] = src_f.header[i]
                        except Exception:
                            pass

        print(f"地震数据已保存为标准SEGY格式: {path}")
        print(f"道数: {trace_count}, 采样点数: {samples_count}")
    else:
        default_path = os.path.splitext(path)[0] + '.segy'
        save_result(result, default_path, src_segy_path)
        print(f"不支持的格式 {ext}，已默认保存为SEGY: {default_path}")


class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.clamp((x + 3.0) / 6.0, 0.0, 1.0)


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        self.hard_swish = HardSwish()
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
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


def extract_patches_2d(img, patch_size=35, stride=10):
    """
    img: [H, W]
    return:
        patches: [N, patch_size, patch_size]
        positions: [(i, j), ...]
        out_shape: (H, W)
    """
    H, W = img.shape
    patches = []
    positions = []

    h_starts = list(range(0, H - patch_size + 1, stride))
    w_starts = list(range(0, W - patch_size + 1, stride))

    # 保证最后能覆盖到边缘
    if len(h_starts) == 0 or h_starts[-1] != H - patch_size:
        h_starts.append(H - patch_size)
    if len(w_starts) == 0 or w_starts[-1] != W - patch_size:
        w_starts.append(W - patch_size)

    h_starts = sorted(list(set(h_starts)))
    w_starts = sorted(list(set(w_starts)))

    for i in h_starts:
        for j in w_starts:
            patch = img[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
            positions.append((i, j))

    patches = np.array(patches, dtype=np.float32)
    return np.array(patches, dtype=np.float32), positions, (H, W)


def reconstruct_from_patches(patches, positions, out_shape, patch_size=35):
    """
    patches: [N, patch_size, patch_size]
    positions: [(i, j), ...]
    """
    H, W = out_shape
    output = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for patch, (i, j) in zip(patches, positions):
        output[i:i + patch_size, j:j + patch_size] += patch
        weight[i:i + patch_size, j:j + patch_size] += 1.0

    weight[weight == 0] = 1.0
    output = output / weight
    return output


def patch_inference(model, x_noisy, patch_size=35, stride=10, batch_size=128, device='cuda'):
    """
    x_noisy: [H, W]
    return x_denoised: [H, W]
    """
    patches, positions, out_shape = extract_patches_2d(
        x_noisy, patch_size=patch_size, stride=stride
    )

    denoised_patches = []

    for start in range(0, len(patches), batch_size):
        end = min(start + batch_size, len(patches))
        batch = patches[start:end]                      # [B, P, P]
        batch_tensor = torch.from_numpy(batch).unsqueeze(1).float().to(device)  # [B,1,P,P]

        with torch.no_grad():
            if device.startswith('cuda'):
                with autocast():
                    batch_out = model(batch_tensor)
            else:
                batch_out = model(batch_tensor)

        batch_out = batch_out.squeeze(1).detach().cpu().numpy()  # [B,P,P]
        denoised_patches.append(batch_out)

        del batch_tensor, batch_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    denoised_patches = np.concatenate(denoised_patches, axis=0)
    x_denoised = reconstruct_from_patches(
        denoised_patches, positions, out_shape, patch_size=patch_size
    )
    return x_denoised


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'Using device: {device}')

    # 1. 创建模型
    model = DnCNN(depth=17, n_channels=64, image_channels=1)
    model = model.to(device)

    # 2. 加载模型参数
    # 加载模型
    model_dir = os.path.join(args.model_1_dir, f"DnCNN_sigma{args.sigma}")
    model_path = os.path.join(model_dir, args.model_name)
    
    if not os.path.exists(model_path):
        model = torch.load(os.path.join(model_dir, 'model.pth'), map_location='cpu')
        log('load trained model on Train400 dataset by kai')
    else:
        model = torch.load(model_path, weights_only=False, map_location='cpu')
        log('load trained model')
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # 3. 读取并预处理干净地震数据
    segy_full_path = args.src_segy_path
    if not os.path.exists(segy_full_path):
        raise FileNotFoundError(f'SEGY文件不存在: {segy_full_path}')

    with segyio.open(segy_full_path, 'r', ignore_geometry=True) as f:
        x_clean = f.trace.raw[:].astype(np.float32)

    # 与你原来的 test 一样：整幅数据做百分位归一化
    pmin, pmax = np.percentile(x_clean, 1), np.percentile(x_clean, 99)
    x_clean = 2 * (x_clean - pmin) / (pmax - pmin + 1e-12) - 1
    x_clean = np.clip(x_clean, -1, 1).astype(np.float32)

    # 4. 加噪
    np.random.seed(0)
    x_noisy = x_clean + np.random.normal(0, args.sigma, x_clean.shape).astype(np.float32)
    x_noisy = x_noisy.astype(np.float32)

    # 5. 基线指标
    initial_rmse = np.sqrt(np.mean((x_noisy - x_clean) ** 2))
    initial_snr = calc_snr(x_clean, x_noisy)
    initial_psnr = compare_psnr(x_clean, x_noisy, data_range=2.0)
    initial_ssim = compare_ssim(x_clean, x_noisy, data_range=2.0, channel_axis=None)

    print("=" * 50)
    print("【初始基线（加噪未去噪）】")
    print(f"SNR: {initial_snr:.2f} | PSNR: {initial_psnr:.2f} dB | SSIM: {initial_ssim:.4f} | RMSE: {initial_rmse:.6f}")
    print("=" * 50)

    # 6. 创建结果目录
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    segy_name = os.path.splitext(os.path.basename(segy_full_path))[0]

    for set_cur in args.set_names:
        set_result_dir = os.path.join(args.result_dir, set_cur, f"sigma_{args.sigma}")
        os.makedirs(set_result_dir, exist_ok=True)

        log(f"开始推理数据集: {set_cur} | patch_size={args.patch_size} | stride={args.stride} | batch_size={args.batch_size}")
        start_time = time.time()

        # 7. patch 推理
        x_denoised = patch_inference(
            model,
            x_noisy,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device
        )

        total_time = time.time() - start_time
        log(f"推理完成，总耗时: {total_time:.2f} 秒")
        log(f"x_noisy - x_denoised:{x_noisy}-{x_denoised}")

        # 8. 去噪后指标
        psnr_denoised = compare_psnr(x_clean, x_denoised, data_range=2.0)
        ssim_denoised = compare_ssim(x_clean, x_denoised, data_range=2.0, channel_axis=None)
        snr_denoised = calc_snr(x_clean, x_denoised)
        rmse_denoised = np.sqrt(np.mean((x_clean - x_denoised) ** 2))

        print("\n" + "=" * 50)
        print("【去噪结果】")
        print(f"SNR: {snr_denoised:.2f} | PSNR: {psnr_denoised:.2f} dB | SSIM: {ssim_denoised:.4f} | RMSE: {rmse_denoised:.6f}")
        print("=" * 50)
        
        # 9. 保存结果
        if args.save_result:
            save_result(
                x_denoised,
                path=os.path.join(set_result_dir, segy_name + '_dncnn_denoised.segy'),
                src_segy_path=args.src_segy_path
            )
            save_result(
                x_noisy,
                path=os.path.join(set_result_dir, segy_name + '_noisy.segy'),
                src_segy_path=args.src_segy_path
            )
        
            np.save(os.path.join(set_result_dir, segy_name + '_clean.npy'), x_clean)
            np.save(os.path.join(set_result_dir, segy_name + '_dncnn_denoised.npy'), x_denoised)
        
            with open(os.path.join(set_result_dir, 'results.txt'), 'w', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"文件: {segy_full_path}\n")
                f.write(f"sigma: {args.sigma}\n")
                f.write(f"总耗时: {total_time:.4f} 秒\n")
                f.write(f"patch_size: {args.patch_size}\n")
                f.write(f"stride: {args.stride}\n")
                f.write(f"batch_size: {args.batch_size}\n")
                f.write(f"初始PSNR: {initial_psnr:.2f} dB\n")
                f.write(f"去噪PSNR: {psnr_denoised:.2f} dB\n")
                f.write(f"初始SNR: {initial_snr:.2f} dB\n")
                f.write(f"去噪SNR: {snr_denoised:.2f} dB\n")
                f.write(f"初始SSIM: {initial_ssim:.4f}\n")
                f.write(f"去噪SSIM: {ssim_denoised:.4f}\n")
                f.write(f"初始RMSE: {initial_rmse:.6f}\n")
                f.write(f"去噪RMSE: {rmse_denoised:.6f}\n")
        

        log(f"数据集: {set_cur:10s} | PSNR = {psnr_denoised:.2f} dB, SSIM = {ssim_denoised:.4f}, RMSE = {rmse_denoised:.6f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()