# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import time
import segyio
import gc
from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode


from skimage.metrics import (
    peak_signal_noise_ratio as calculate_psnr,
    structural_similarity as calculate_ssim,
    mean_squared_error as calculate_mse
)

from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode


# ====================== 参数 ======================
def parse_args():
    parser = argparse.ArgumentParser(description='K-SVD自适应字典地震数据去噪')
    parser.add_argument('--segy_path', type=str, default='data/P.sgy', help='原始SEGY文件路径')
    parser.add_argument('--set_dir', type=str, default='ksvd', help='结果保存目录名')
    parser.add_argument('--log_file', type=str, default='log_test.txt', help='日志文件名')
    parser.add_argument('--sigma', type=float, default=0.05, help='归一化域中的高斯噪声标准差')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')

    # K-SVD / patch 参数
    parser.add_argument('--patch_h', type=int, default=8, help='patch高度')
    parser.add_argument('--patch_w', type=int, default=8, help='patch宽度')
    parser.add_argument('--stride', type=int, default=5, help='patch滑动步长')
    parser.add_argument('--dict_size', type=int, default=256, help='字典原子数')
    parser.add_argument('--n_nonzero', type=int, default=4, help='OMP稀疏系数个数')
    parser.add_argument('--max_iter', type=int, default=20, help='字典学习迭代次数')
    parser.add_argument('--train_patches_limit', type=int, default=20000, help='最多用于训练字典的patch数')
    parser.add_argument('--batch_size', type=int, default=400, help='字典学习batch size')
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


# ====================== patch 提取 / 重建 ======================


def reconstruct_from_patches(patches, positions, data_shape, patch_h, patch_w):
    H, W = data_shape
    recon = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for patch_vec, (i, j) in zip(patches, positions):
        patch = patch_vec.reshape(patch_h, patch_w)
        recon[i:i + patch_h, j:j + patch_w] += patch
        weight[i:i + patch_h, j:j + patch_w] += 1.0

    recon /= (weight + 1e-12)
    return recon


def patch_generator_2d(data, patch_h, patch_w, stride, chunk_size=1000):
    """
    流式产生 patch，不一次性存全部 patch
    返回:
        patches_chunk: (B, patch_h*patch_w)
        positions_chunk: (B, 2)
    """
    H, W = data.shape
    patches = []
    positions = []

    for i in range(0, H - patch_h + 1, stride):
        for j in range(0, W - patch_w + 1, stride):
            patch = data[i:i + patch_h, j:j + patch_w].reshape(-1)
            patches.append(patch)
            positions.append((i, j))

            if len(patches) >= chunk_size:
                yield np.asarray(patches, dtype=np.float32), np.asarray(positions, dtype=np.int32)
                patches = []
                positions = []

    if len(patches) > 0:
        yield np.asarray(patches, dtype=np.float32), np.asarray(positions, dtype=np.int32)


def count_total_patches(data_shape, patch_h, patch_w, stride):
    H, W = data_shape
    n1 = (H - patch_h) // stride + 1
    n2 = (W - patch_w) // stride + 1
    return n1 * n2

#==================计算评价指标====================================
def calc_psnr_rmse_snr(clean, test, data_range=2.0):
    clean = clean.astype(np.float32, copy=False)
    test = test.astype(np.float32, copy=False)

    diff = clean - test
    mse = np.mean(diff * diff, dtype=np.float64)
    rmse = np.sqrt(mse)

    psnr = 10.0 * np.log10((data_range ** 2) / (mse + 1e-12))

    signal_power = np.sum(clean * clean, dtype=np.float64)
    noise_power = np.sum(diff * diff, dtype=np.float64)
    snr = 10.0 * np.log10(signal_power / (noise_power + 1e-12))

    return float(psnr), float(rmse), float(snr)

# ====================== 自适应字典 K-SVD 去噪 ======================
def ksvd_denoise_adaptive(
    noisy_norm,
    patch_h=8,
    patch_w=8,
    stride=5,
    dict_size=128,
    n_nonzero=3,
    max_iter=5,
    train_patches_limit=5000,
    batch_size=256,
    random_state=0,
    chunk_size=500
):
    """
    流式K-SVD近似版：
    1) 第一遍：随机抽样patch训练字典
    2) 第二遍：分块稀疏编码 + 分块重建
    """
    H, W = noisy_norm.shape
    patch_dim = patch_h * patch_w
    total_patches = count_total_patches(noisy_norm.shape, patch_h, patch_w, stride)
    print(f"总patch数: {total_patches}")

    rng = np.random.default_rng(random_state)

    # =========================
    # 第一遍：抽样训练字典
    # =========================
    sampled = []
    seen = 0

    for patches_chunk, _ in patch_generator_2d(
        noisy_norm, patch_h, patch_w, stride, chunk_size=chunk_size
    ):
        for patch in patches_chunk:
            seen += 1
            if len(sampled) < train_patches_limit:
                sampled.append(patch.copy())
            else:
                # reservoir sampling
                j = rng.integers(0, seen)
                if j < train_patches_limit:
                    sampled[j] = patch.copy()

        del patches_chunk
        gc.collect()

    train_patches = np.asarray(sampled, dtype=np.float32)
    print(f"用于训练字典的patch数: {len(train_patches)}")

    train_means = train_patches.mean(axis=1, keepdims=True)
    train_centered = train_patches - train_means

    dict_learner = MiniBatchDictionaryLearning(
        n_components=dict_size,
        alpha=1.0,
        max_iter=max_iter,
        batch_size=batch_size,
        transform_algorithm='omp',
        transform_n_nonzero_coefs=n_nonzero,
        random_state=random_state,
        verbose=False
    )
    dictionary = dict_learner.fit(train_centered).components_.astype(np.float32)
    print(f"字典形状: {dictionary.shape}")

    del sampled, train_patches, train_means, train_centered
    gc.collect()

    # =========================
    # 第二遍：流式编码 + 重建
    # =========================
    recon = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    processed = 0
    for patches_chunk, positions_chunk in patch_generator_2d(
        noisy_norm, patch_h, patch_w, stride, chunk_size=chunk_size
    ):
        means_chunk = patches_chunk.mean(axis=1, keepdims=True)
        centered_chunk = patches_chunk - means_chunk

        code_chunk = sparse_encode(
            centered_chunk,
            dictionary,
            algorithm='omp',
            n_nonzero_coefs=n_nonzero
        ).astype(np.float32)

        recon_centered_chunk = np.dot(code_chunk, dictionary).astype(np.float32)
        recon_patches_chunk = recon_centered_chunk + means_chunk

        for k in range(len(recon_patches_chunk)):
            i, j = positions_chunk[k]
            patch = recon_patches_chunk[k].reshape(patch_h, patch_w)
            recon[i:i + patch_h, j:j + patch_w] += patch
            weight[i:i + patch_h, j:j + patch_w] += 1.0

        processed += len(patches_chunk)
        print(f"编码进度: {processed}/{total_patches}")

        del patches_chunk, positions_chunk, means_chunk, centered_chunk
        del code_chunk, recon_centered_chunk, recon_patches_chunk
        gc.collect()

    denoised_norm = recon / (weight + 1e-12)
    denoised_norm = np.clip(denoised_norm, -1, 1).astype(np.float32)

    del recon, weight
    gc.collect()

    return denoised_norm, dictionary

# ====================== 主流程 ======================
if __name__ == "__main__":
    args = parse_args()

    print("===== 当前运行参数 =====")
    print(f"SEGY路径: {args.segy_path}")
    print(f"sigma: {args.sigma}")
    print(f"log文件: {args.log_file}")
    print(f"patch: ({args.patch_h}, {args.patch_w})")
    print(f"stride: {args.stride}")
    print(f"dict_size: {args.dict_size}")
    print(f"n_nonzero: {args.n_nonzero}")
    print("=======================\n")

    save_dir = os.path.join("real_results", f"{args.set_dir}_sigma{args.sigma}")
    os.makedirs("real_results", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, args.log_file)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"K-SVD自适应字典地震去噪测试日志 | sigma={args.sigma}\n")
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

    # 3. 添加高斯噪声（和BM3D脚本一致）
    print("====添加高斯噪声====")
    np.random.seed(args.seed)
    seismic_noisy_norm = seismic_clean_norm + np.random.normal(
        0, args.sigma, seismic_clean_norm.shape
    ).astype(np.float32)

    seismic_noisy_norm = np.clip(seismic_noisy_norm, -1, 1).astype(np.float32)

    # 4. K-SVD自适应字典去噪
    print("====K-SVD自适应字典去噪====")
    denoise_start = time.time()

    seismic_denoised_norm, dictionary = ksvd_denoise_adaptive(
        seismic_noisy_norm,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        stride=args.stride,
        dict_size=args.dict_size,
        n_nonzero=args.n_nonzero,
        max_iter=args.max_iter,
        train_patches_limit=args.train_patches_limit,
        batch_size=args.batch_size,
        random_state=args.seed
    )

    total_denoise_time = time.time() - denoise_start
    print(f"K-SVD去噪完成，总耗时: {total_denoise_time:.2f} 秒")

    # 5. 反归一化
    seismic_noisy = denormalize(seismic_noisy_norm, pmin, pmax)
    seismic_denoised = denormalize(seismic_denoised_norm, pmin, pmax)

    # 6. 保存结果
    print("====保存结果====")
    segy_noisy_norm_path = os.path.join(save_dir, f"ksvd_noisy_norm_sigma{args.sigma}.segy")
    segy_denoised_norm_path = os.path.join(save_dir, f"ksvd_denoised_norm_sigma{args.sigma}.segy")
    segy_noisy_path = os.path.join(save_dir, f"ksvd_noisy_original_sigma{args.sigma}.segy")
    segy_denoised_path = os.path.join(save_dir, f"ksvd_denoised_original_sigma{args.sigma}.segy")

    save_segy(seismic_noisy_norm, args.segy_path, segy_noisy_norm_path)
    save_segy(seismic_denoised_norm, args.segy_path, segy_denoised_norm_path)
    save_segy(seismic_noisy, args.segy_path, segy_noisy_path)
    save_segy(seismic_denoised, args.segy_path, segy_denoised_path)

    # 同时保存 npy
    np.save(os.path.join(save_dir, f"clean_norm_sigma{args.sigma}.npy"), seismic_clean_norm)
    np.save(os.path.join(save_dir, f"noisy_norm_sigma{args.sigma}.npy"), seismic_noisy_norm)
    np.save(os.path.join(save_dir, f"ksvd_denoised_norm_sigma{args.sigma}.npy"), seismic_denoised_norm)
    np.save(os.path.join(save_dir, f"ksvd_dictionary_sigma{args.sigma}.npy"), dictionary)

    # 7. 计算评价指标（整图，归一化域，和BM3D一致）
    del seismic_noisy
    del seismic_denoised
    gc.collect()
    print("====计算评价指标====")
    noisy_psnr, noisy_rmse, noisy_snr = calc_psnr_rmse_snr(
        seismic_clean_norm, seismic_noisy_norm, data_range=2.0
    )
    
    den_psnr, den_rmse, den_snr = calc_psnr_rmse_snr(
        seismic_clean_norm, seismic_denoised_norm, data_range=2.0
    )
    noisy_ssim = np.nan
    den_ssim = np.nan


    final_log = f"""
===== K-SVD自适应字典测试结果汇总 =====
测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}
噪声参数：sigma={args.sigma}
Patch参数：patch=({args.patch_h},{args.patch_w}), stride={args.stride}
字典参数：dict_size={args.dict_size}, n_nonzero={args.n_nonzero}, max_iter={args.max_iter}
耗时统计：总去噪时间={total_denoise_time:.2f}s

【未去噪指标（Noisy vs Clean, 整图, data_range=2）】
PSNR：{noisy_psnr:.4f} dB
SSIM：{noisy_ssim:.4f}
RMSE：{noisy_rmse:.4f}
SNR ：{noisy_snr:.4f} dB

【去噪后指标（K-SVD vs Clean, 整图, data_range=2）】
PSNR：{den_psnr:.4f} dB
SSIM：{den_ssim:.4f}
RMSE：{den_rmse:.4f}
SNR ：{den_snr:.4f} dB
=====================
"""

    print(final_log)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(final_log)

    txt_path = os.path.join(save_dir, f"ksvd_metrics_sigma{args.sigma}.txt")
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(final_log)

    print(f"\n所有结果已保存到：{save_dir}")