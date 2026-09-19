# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import time
import segyio

from skimage.metrics import (
    peak_signal_noise_ratio as calculate_psnr,
    structural_similarity as calculate_ssim,
    mean_squared_error as calculate_mse
)

from bm3d import bm3d


# ====================== 参数 ======================
def parse_args():
    parser = argparse.ArgumentParser(description='BM3D地震数据去噪基准方法')
    parser.add_argument('--segy_path', type=str, default='data/P.sgy', help='原始SEGY文件路径')
    parser.add_argument('--set_dir', type=str, default='bm3d', help='结果保存目录名')
    parser.add_argument('--log_file', type=str, default='log_test.txt', help='日志文件名')
    parser.add_argument('--sigma', type=float, default=0.05, help='归一化域中的高斯噪声标准差')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    return parser.parse_args()


# ====================== 保存SEGY ======================
def save_segy(data, original_segy_path, save_path):
    with segyio.open(original_segy_path, "r", ignore_geometry=True) as src:
        spec = segyio.spec()
        spec.sorting = src.sorting
        spec.format = src.format
        spec.samples = src.samples
        spec.tracecount = src.tracecount

        with segyio.create(save_path, spec) as dst:
            dst.bin = src.bin
            dst.header = src.header

            if len(data) != src.tracecount:
                print(f"警告：数据道数({len(data)})和原始SEGY道数({src.tracecount})不匹配，自动截断/补0")
                if len(data) > src.tracecount:
                    data = data[:src.tracecount]
                else:
                    data = np.pad(data, ((0, src.tracecount - len(data)), (0, 0)), mode='constant')

            for i in range(src.tracecount):
                dst.trace[i] = data[i]

    print(f"SEGY文件已成功保存到：{save_path}")


# ====================== 归一化 / 反归一化 ======================
def normalize_seismic(seismic):
    pmin, pmax = np.percentile(seismic, 1), np.percentile(seismic, 99)
    seismic_norm = 2 * (seismic - pmin) / (pmax - pmin + 1e-12) - 1
    seismic_norm = np.clip(seismic_norm, -1, 1).astype(np.float32)
    return seismic_norm, pmin, pmax


def denormalize(data_norm, min_val, max_val):
    data = (data_norm + 1.0) / 2.0
    data = data * (max_val - min_val) + min_val
    return data.astype(np.float32)


# ====================== 指标 ======================
def calculate_rmse(original, reconstructed):
    mse = calculate_mse(np.array(original, np.float32), np.array(reconstructed, np.float32))
    return np.sqrt(mse)


def calculate_snr(clean, test):
    noise = clean - test
    return 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-12))


# ====================== 主流程 ======================
if __name__ == "__main__":
    args = parse_args()

    print("===== 当前运行参数 =====")
    print(f"SEGY路径: {args.segy_path}")
    print(f"sigma: {args.sigma}")
    print(f"log文件: {args.log_file}")
    print("=======================\n")

    save_dir = os.path.join("real_results", f"{args.set_dir}_sigma{args.sigma}")
    os.makedirs("real_results", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, args.log_file)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"BM3D地震去噪测试日志 | sigma={args.sigma}\n")
        f.write(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=====================\n")

    # 1. 读取原始地震数据
    print("====读取原始地震数据====")
    with segyio.open(args.segy_path, "r", ignore_geometry=True) as f:
        seismic_original = f.trace.raw[:].astype(np.float32)

    print(f"原始数据形状: {seismic_original.shape}")

    # 2. 归一化到[-1,1]
    print("====归一化====")
    seismic_clean_norm, pmin, pmax = normalize_seismic(seismic_original)

    # 3. 添加高斯噪声
    print("====添加高斯噪声====")
    np.random.seed(args.seed)
    seismic_noisy_norm = seismic_clean_norm + np.random.normal(
        0, args.sigma, seismic_clean_norm.shape
    ).astype(np.float32)

    # 可选：裁剪到[-1,1]
    seismic_noisy_norm = np.clip(seismic_noisy_norm, -1, 1).astype(np.float32)

    # 4. BM3D去噪
    print("====BM3D去噪====")
    denoise_start = time.time()

    # bm3d 的 sigma_psd 就是噪声标准差；你的噪声是在归一化域里加的，所以直接传 args.sigma
    seismic_denoised_norm = bm3d(seismic_noisy_norm, sigma_psd=args.sigma)
    seismic_denoised_norm = np.clip(seismic_denoised_norm, -1, 1).astype(np.float32)

    total_denoise_time = time.time() - denoise_start
    print(f"BM3D去噪完成，总耗时: {total_denoise_time:.2f} 秒")

    # 5. 反归一化
    seismic_noisy = denormalize(seismic_noisy_norm, pmin, pmax)
    seismic_denoised = denormalize(seismic_denoised_norm, pmin, pmax)

    # 6. 保存结果
    print("====保存结果====")
    segy_noisy_norm_path = os.path.join(save_dir, f"bm3d_noisy_norm_sigma{args.sigma}.segy")
    segy_denoised_norm_path = os.path.join(save_dir, f"bm3d_denoised_norm_sigma{args.sigma}.segy")
    segy_noisy_path = os.path.join(save_dir, f"bm3d_noisy_original_sigma{args.sigma}.segy")
    segy_denoised_path = os.path.join(save_dir, f"bm3d_denoised_original_sigma{args.sigma}.segy")

    save_segy(seismic_noisy_norm, args.segy_path, segy_noisy_norm_path)
    save_segy(seismic_denoised_norm, args.segy_path, segy_denoised_norm_path)
    save_segy(seismic_noisy, args.segy_path, segy_noisy_path)
    save_segy(seismic_denoised, args.segy_path, segy_denoised_path)

    # 同时保存 npy，方便画图和后处理
    np.save(os.path.join(save_dir, f"clean_norm_sigma{args.sigma}.npy"), seismic_clean_norm)
    np.save(os.path.join(save_dir, f"noisy_norm_sigma{args.sigma}.npy"), seismic_noisy_norm)
    np.save(os.path.join(save_dir, f"bm3d_denoised_norm_sigma{args.sigma}.npy"), seismic_denoised_norm)

    # 7. 计算评价指标（整图，归一化域）
    print("====计算评价指标====")
    noisy_psnr = calculate_psnr(seismic_clean_norm, seismic_noisy_norm, data_range=2)
    noisy_ssim = calculate_ssim(seismic_clean_norm, seismic_noisy_norm, data_range=2, channel_axis=None)
    noisy_rmse = calculate_rmse(seismic_clean_norm, seismic_noisy_norm)
    noisy_snr = calculate_snr(seismic_clean_norm, seismic_noisy_norm)

    den_psnr = calculate_psnr(seismic_clean_norm, seismic_denoised_norm, data_range=2)
    den_ssim = calculate_ssim(seismic_clean_norm, seismic_denoised_norm, data_range=2, channel_axis=None)
    den_rmse = calculate_rmse(seismic_clean_norm, seismic_denoised_norm)
    den_snr = calculate_snr(seismic_clean_norm, seismic_denoised_norm)

    final_log = f"""
===== BM3D测试结果汇总 =====
测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}
噪声参数：sigma={args.sigma}
耗时统计：总去噪时间={total_denoise_time:.2f}s

【未去噪指标（Noisy vs Clean, 整图, data_range=2）】
PSNR：{noisy_psnr:.4f} dB
SSIM：{noisy_ssim:.4f}
RMSE：{noisy_rmse:.4f}
SNR ：{noisy_snr:.4f} dB

【去噪后指标（BM3D vs Clean, 整图, data_range=2）】
PSNR：{den_psnr:.4f} dB
SSIM：{den_ssim:.4f}
RMSE：{den_rmse:.4f}
SNR ：{den_snr:.4f} dB
=====================
"""

    print(final_log)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(final_log)

    txt_path = os.path.join(save_dir, f"bm3d_metrics_sigma{args.sigma}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(final_log)

    print(f"\n所有结果已保存到：{save_dir}")