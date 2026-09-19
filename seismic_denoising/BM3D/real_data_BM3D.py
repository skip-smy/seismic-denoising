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
    parser = argparse.ArgumentParser(description='BM3D地震数据去噪（已有 noisy.sgy + clean.npy）')
    parser.add_argument('--noisy_segy_path', type=str, default='data/P.sgy', help='含噪SEGY文件路径')
    parser.add_argument('--clean_npy_path', type=str, default='data/clean.npy', help='干净npy文件路径')
    parser.add_argument('--set_dir', type=str, default='bm3d_real_data', help='结果保存目录名')
    parser.add_argument('--log_file', type=str, default='log_test.txt', help='日志文件名')
    parser.add_argument('--sigma', type=float, default=0.074, help='归一化域中的噪声标准差（传给BM3D）')
    return parser.parse_args()


# ====================== 保存SEGY ======================
def save_segy(data, original_segy_path, save_path):
    data = data.astype(np.float32)

    with segyio.open(original_segy_path, "r", ignore_geometry=True) as src:
        spec = segyio.spec()
        spec.sorting = src.sorting
        spec.format = src.format
        spec.samples = src.samples
        spec.tracecount = src.tracecount

        with segyio.create(save_path, spec) as dst:
            dst.bin = src.bin
            dst.header = src.header

            if data.shape[0] != src.tracecount:
                print(f"警告：数据道数({data.shape[0]})和原始SEGY道数({src.tracecount})不匹配，自动截断/补0")
                if data.shape[0] > src.tracecount:
                    data = data[:src.tracecount]
                else:
                    data = np.pad(data, ((0, src.tracecount - data.shape[0]), (0, 0)), mode='constant')

            for i in range(src.tracecount):
                dst.trace[i] = data[i]

    print(f"SEGY文件已成功保存到：{save_path}")


# ====================== 读取SEGY为npy数组 ======================
def read_segy_to_npy(segy_path):
    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        data = f.trace.raw[:].astype(np.float32)
    return data


# ====================== 统一归一化 / 反归一化 ======================
def normalize_with_given_range(data, pmin, pmax):
    data_norm = (data - pmin) / (pmax - pmin + 1e-12) 
    data_norm = 2 * data_norm - 1
    data_norm = np.clip(data_norm, -1, 1).astype(np.float32)
    return data_norm


def denormalize(data_norm, pmin, pmax):
    data_norm = (data_norm +1.0)/2.0
    data = data_norm * (pmax - pmin) + pmin
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
    print(f"noisy SEGY路径: {args.noisy_segy_path}")
    print(f"clean npy路径 : {args.clean_npy_path}")
    print(f"sigma        : {args.sigma}")
    print(f"log文件       : {args.log_file}")
    print("=======================\n")

    save_dir = os.path.join("real_results", f"{args.set_dir}_sigma{args.sigma}")
    os.makedirs("real_results", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, args.log_file)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"BM3D地震去噪测试日志 | sigma={args.sigma}\n")
        f.write(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=====================\n")

    # 1. 读取 clean.npy
    print("====读取 clean.npy====")
    seismic_clean = np.load(args.clean_npy_path).astype(np.float32)
    print(f"clean shape: {seismic_clean.shape}")

    # 2. 读取 noisy.sgy，并先转成 noisy.npy
    print("====读取 noisy.sgy -> 转成 noisy.npy====")
    seismic_noisy = read_segy_to_npy(args.noisy_segy_path)
    print(f"noisy shape: {seismic_noisy.shape}")

    noisy_npy_path = os.path.join(save_dir, "noisy_from_segy.npy")
    np.save(noisy_npy_path, seismic_noisy)
    print(f"noisy.npy 已保存到: {noisy_npy_path}")

    # 3. 检查形状一致
    if seismic_clean.shape != seismic_noisy.shape:
        raise ValueError(f"clean.npy 与 noisy.sgy 数据形状不一致: clean={seismic_clean.shape}, noisy={seismic_noisy.shape}")

    # 4. 用 clean 的分位数做统一归一化
    print("====统一归一化到[-1,1]====")
    pmin, pmax = np.percentile(seismic_clean, 1), np.percentile(seismic_clean, 99)

    seismic_clean_norm = normalize_with_given_range(seismic_clean, pmin, pmax)
    seismic_noisy_norm = normalize_with_given_range(seismic_noisy, pmin, pmax)
    noise_norm = seismic_noisy_norm - seismic_clean_norm

    sigma_std = np.std(noise_norm)
    sigma_mad = np.median(np.abs(noise_norm - np.median(noise_norm))) / 0.6745
    print(f"sigma_std = {sigma_std:.6f}")
    print(f"sigma_mad = {sigma_mad:.6f}")

    # 保存归一化后的npy
    np.save(os.path.join(save_dir, "clean_norm.npy"), seismic_clean_norm)
    np.save(os.path.join(save_dir, "noisy_norm.npy"), seismic_noisy_norm)

    # 5. BM3D去噪
    print("====BM3D去噪====")
    denoise_start = time.time()

    seismic_denoised_norm = bm3d(seismic_noisy_norm, sigma_psd=args.sigma)
    seismic_denoised_norm = np.clip(seismic_denoised_norm, -1, 1).astype(np.float32)

    total_denoise_time = time.time() - denoise_start
    print(f"BM3D去噪完成，总耗时: {total_denoise_time:.2f} 秒")

    # 6. 反归一化
    print("====反归一化====")
    seismic_denoised = denormalize(seismic_denoised_norm, pmin, pmax)

    # 7. 保存结果
    print("====保存结果====")
    np.save(os.path.join(save_dir, "bm3d_denoised_norm.npy"), seismic_denoised_norm)
    np.save(os.path.join(save_dir, "bm3d_denoised.npy"), seismic_denoised)

    
    # 8. 计算评价指标（在归一化域）
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

【未去噪指标（Noisy vs Clean, 归一化域0-1, data_range=1）】
PSNR：{noisy_psnr:.4f} dB
SSIM：{noisy_ssim:.4f}
RMSE：{noisy_rmse:.4f}
SNR ：{noisy_snr:.4f} dB

【去噪后指标（BM3D vs Clean, 归一化域0-1, data_range=1）】
PSNR：{den_psnr:.4f} dB
SSIM：{den_ssim:.4f}
RMSE：{den_rmse:.4f}
SNR ：{den_snr:.4f} dB
=====================
"""
    print(final_log)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(final_log)

    txt_path = os.path.join(save_dir, "bm3d_metrics.txt")
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(final_log)

    print(f"\n所有结果已保存到：{save_dir}")