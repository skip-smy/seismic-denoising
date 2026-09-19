# -*- coding: utf-8 -*-
import argparse
import gc
import os
import time
import numpy as np
import segyio
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import orthogonal_mp

# ====================== 参数 ======================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', type=str, default='real')
    parser.add_argument('--noisy_segy_path', type=str, default='data/P.sgy', help='含噪SEGY文件路径')
    parser.add_argument('--clean_npy_path', type=str, default='data/real_clean.npy', help='干净NPY文件路径')
    parser.add_argument('--sigma', type=float, default=0.05, help='仅用于日志和文件名')
    parser.add_argument('--log_file', type=str, default='test_log.txt')
    parser.add_argument('--patch_h', type=int, default=8)
    parser.add_argument('--patch_w', type=int, default=8)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--dict_size', type=int, default=256)
    parser.add_argument('--n_nonzero', type=int, default=4)
    parser.add_argument('--max_iter', type=int, default=20)
    parser.add_argument('--train_patches_limit', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


# ====================== 归一化/反归一化 ======================
def normalize_seismic(x, pmin=None, pmax=None):
    x = x.astype(np.float32, copy=False)
    if pmin is None or pmax is None:
        pmin = np.percentile(x, 1)
        pmax = np.percentile(x, 99)

    x_clip = np.clip(x, pmin, pmax)
    x_norm = (x_clip - pmin) / (pmax - pmin + 1e-12) 
    x_norm = 2 * x_norm -1
    return x_norm.astype(np.float32), float(pmin), float(pmax)


def denormalize(x_norm, pmin, pmax):
    x_norm = (x_norm +1)/2
    x = x_norm * (pmax - pmin) + pmin
    return x.astype(np.float32)


# ====================== 保存SEGY ======================
def save_segy(data, ref_segy_path, save_path):
    data = data.astype(np.float32)

    with segyio.open(ref_segy_path, "r", ignore_geometry=True) as src:
        spec = segyio.spec()
        spec.sorting = src.sorting
        spec.format = src.format
        spec.samples = src.samples
        spec.tracecount = src.tracecount

        with segyio.create(save_path, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            for i in range(src.tracecount):
                dst.header[i] = src.header[i]
                dst.trace[i] = data[i]


# ====================== patch生成器 ======================
def patch_generator_2d(data, patch_h, patch_w, stride, chunk_size=500):
    H, W = data.shape
    patches = []
    positions = []

    for i in range(0, H - patch_h + 1, stride):
        for j in range(0, W - patch_w + 1, stride):
            patch = data[i:i + patch_h, j:j + patch_w].reshape(-1)
            patches.append(patch)
            positions.append((i, j))

            if len(patches) >= chunk_size:
                yield np.asarray(patches, dtype=np.float64), np.asarray(positions, dtype=np.int32)
                patches = []
                positions = []

    if len(patches) > 0:
        yield np.asarray(patches, dtype=np.float64), np.asarray(positions, dtype=np.int32)


def count_total_patches(data_shape, patch_h, patch_w, stride):
    H, W = data_shape
    n1 = (H - patch_h) // stride + 1
    n2 = (W - patch_w) // stride + 1
    return n1 * n2


# ====================== 评价指标 ======================
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
    H, W = noisy_norm.shape
    total_patches = count_total_patches(noisy_norm.shape, patch_h, patch_w, stride)
    print(f"总patch数: {total_patches}")

    rng = np.random.default_rng(random_state)

    # 第一遍：抽样训练字典
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
                j = rng.integers(0, seen)
                if j < train_patches_limit:
                    sampled[j] = patch.copy()

        del patches_chunk
        gc.collect()

    sampled = np.asarray(sampled, dtype=np.float64)
    means = sampled.mean(axis=1, keepdims=True)
    sampled_centered = sampled - means

    print(f"实际用于训练字典的patch数: {sampled_centered.shape[0]}")

    dico = DictionaryLearning(
        n_components=dict_size,
        transform_algorithm='omp',
        transform_n_nonzero_coefs=n_nonzero,
        max_iter=max_iter,
        random_state=random_state,
        fit_algorithm='cd'
    )
    code = dico.fit_transform(sampled_centered)
    dictionary = dico.components_.astype(np.float64)

    del sampled, means, sampled_centered, code, dico
    gc.collect()

    # 第二遍：分块编码与重建
    recon = np.zeros((H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    processed = 0
    for patches_chunk, positions_chunk in patch_generator_2d(
        noisy_norm, patch_h, patch_w, stride, chunk_size=chunk_size
    ):
        means_chunk = patches_chunk.mean(axis=1, keepdims=True)
        centered_chunk = patches_chunk - means_chunk

        code_chunk = orthogonal_mp(
            dictionary.T,
            centered_chunk.T,
            n_nonzero_coefs=n_nonzero
        ).T.astype(np.float64)

        recon_centered_chunk = np.dot(code_chunk, dictionary).astype(np.float64)
        recon_patches_chunk = recon_centered_chunk + means_chunk

        for k in range(len(recon_patches_chunk)):
            i, j = positions_chunk[k]
            patch = recon_patches_chunk[k].reshape(patch_h, patch_w)
            recon[i:i + patch_h, j:j + patch_w] += patch
            weight[i:i + patch_h, j:j + patch_w] += 1.0

        processed += len(patches_chunk)
        print(f"编码进度: {processed}/{total_patches}")

        del patches_chunk, positions_chunk, means_chunk
        del centered_chunk, code_chunk, recon_centered_chunk, recon_patches_chunk
        gc.collect()

    denoised_norm = recon / (weight + 1e-12)
    denoised_norm = np.clip(denoised_norm, -1, 1).astype(np.float64)

    del recon, weight
    gc.collect()

    return denoised_norm, dictionary


# ====================== 主流程 ======================
if __name__ == "__main__":
    args = parse_args()

    print("===== 当前运行参数 =====")
    print(f"Noisy SEGY路径: {args.noisy_segy_path}")
    print(f"Clean NPY路径: {args.clean_npy_path}")
    print(f"sigma: {args.sigma}")
    print(f"log文件: {args.log_file}")
    print(f"patch: ({args.patch_h}, {args.patch_w})")
    print(f"stride: {args.stride}")
    print(f"dict_size: {args.dict_size}")
    print(f"n_nonzero: {args.n_nonzero}")
    print(f"max_iter: {args.max_iter}")
    print("=======================\n")

    save_dir = os.path.join("real_results", f"{args.set_dir}_sigma{args.sigma}")
    os.makedirs("real_results", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, args.log_file)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"K-SVD自适应字典地震去噪测试日志 | sigma={args.sigma}\n")
        f.write(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=====================\n")

    # 1. 读取 clean.npy
    print("====读取干净NPY数据====")
    seismic_clean = np.load(args.clean_npy_path).astype(np.float32)
    print(f"clean shape: {seismic_clean.shape}")

    # 2. 读取 noisy.segy
    print("====读取含噪SEGY数据====")
    with segyio.open(args.noisy_segy_path, "r", ignore_geometry=True) as f:
        seismic_noisy = f.trace.raw[:].astype(np.float32)
    print(f"noisy shape: {seismic_noisy.shape}")

    # 3. 先把 noisy.segy 转成 noisy.npy
    noisy_npy_path = os.path.join(save_dir, "noisy_from_segy.npy")
    np.save(noisy_npy_path, seismic_noisy)
    print(f"已保存 noisy npy: {noisy_npy_path}")

    # 4. 检查尺寸
    if seismic_clean.shape != seismic_noisy.shape:
        raise ValueError(f"clean和noisy尺寸不一致: {seismic_clean.shape} vs {seismic_noisy.shape}")

    # 5. 同一种方式归一化：统一用 clean 的 pmin/pmax
    print("====统一归一化到[-1,1]====")
    seismic_clean_norm, pmin, pmax = normalize_seismic(seismic_clean)
    seismic_noisy_norm, _, _ = normalize_seismic(seismic_noisy, pmin=pmin, pmax=pmax)

    # 6. K-SVD自适应字典去噪
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

    # 7. 反归一化
    seismic_denoised = denormalize(seismic_denoised_norm, pmin, pmax)

    # 8. 保存结果
    print("====保存结果====")

    # 保存 npy

    np.save(os.path.join(save_dir, f"clean_norm_sigma{args.sigma}.npy"), seismic_clean_norm)
    np.save(os.path.join(save_dir, f"noisy_norm_sigma{args.sigma}.npy"), seismic_noisy_norm)
    np.save(os.path.join(save_dir, f"ksvd_denoised_norm_sigma{args.sigma}.npy"), seismic_denoised_norm)
    np.save(os.path.join(save_dir, f"ksvd_dictionary_sigma{args.sigma}.npy"), dictionary)

    # 9. 计算评价指标
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