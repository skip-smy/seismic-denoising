# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import segyio
from bm3d import bm3d


def parse_args():
    parser = argparse.ArgumentParser(description='Chunked BM3D for seismic -> npy only')
    parser.add_argument('--segy_path', type=str, required=True, help='input segy path')
    parser.add_argument('--save_dir', type=str, required=True, help='output folder')
    parser.add_argument('--sigma', type=float, default=0.05, help='noise sigma in normalized domain')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--chunk_traces', type=int, default=2000, help='number of traces per chunk')
    return parser.parse_args()


def normalize_seismic(seismic):
    pmin, pmax = np.percentile(seismic, 1), np.percentile(seismic, 99)
    seismic_norm = 2 * (seismic - pmin) / (pmax - pmin + 1e-12) - 1
    seismic_norm = np.clip(seismic_norm, -1, 1).astype(np.float32)
    return seismic_norm, pmin, pmax


def read_segy(segy_path):
    with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
        data = f.trace.raw[:].astype(np.float32)
    return data


def chunked_bm3d(noisy_norm, sigma, chunk_traces=2000):
    n_traces, n_samples = noisy_norm.shape
    out = np.zeros_like(noisy_norm, dtype=np.float32)

    n_chunks = (n_traces + chunk_traces - 1) // chunk_traces
    total_start = time.time()

    for k in range(n_chunks):
        s = k * chunk_traces
        e = min((k + 1) * chunk_traces, n_traces)

        block = noisy_norm[s:e, :]
        print(f'[{k+1}/{n_chunks}] BM3D block: traces {s}:{e}, shape={block.shape}')

        t0 = time.time()
        block_denoised = bm3d(block, sigma_psd=sigma)
        block_denoised = np.clip(block_denoised, -1, 1).astype(np.float32)
        out[s:e, :] = block_denoised
        print(f'[{k+1}/{n_chunks}] done, time = {time.time() - t0:.2f} s')

    print(f'All chunks done, total time = {time.time() - total_start:.2f} s')
    return out


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print('====读取原始地震数据====')
    seismic_original = read_segy(args.segy_path)
    print(f'原始数据形状: {seismic_original.shape}')

    print('====归一化====')
    seismic_clean_norm, pmin, pmax = normalize_seismic(seismic_original)
    print(f'p1={pmin:.6f}, p99={pmax:.6f}')
    print(f'clean_norm range = [{seismic_clean_norm.min():.6f}, {seismic_clean_norm.max():.6f}]')

    print('====添加高斯噪声====')
    np.random.seed(args.seed)
    seismic_noisy_norm = seismic_clean_norm + np.random.normal(
        0, args.sigma, seismic_clean_norm.shape
    ).astype(np.float32)
    seismic_noisy_norm = np.clip(seismic_noisy_norm, -1, 1).astype(np.float32)
    print(f'noisy_norm range = [{seismic_noisy_norm.min():.6f}, {seismic_noisy_norm.max():.6f}]')

    print('====分块BM3D去噪====')
    seismic_denoised_norm = chunked_bm3d(
        seismic_noisy_norm,
        sigma=args.sigma,
        chunk_traces=args.chunk_traces
    )

    print('====保存npy====')
    np.save(os.path.join(args.save_dir, f'clean_norm_sigma{args.sigma}.npy'), seismic_clean_norm)
    np.save(os.path.join(args.save_dir, f'noisy_norm_sigma{args.sigma}.npy'), seismic_noisy_norm)
    np.save(os.path.join(args.save_dir, f'bm3d_denoised_norm_sigma{args.sigma}.npy'), seismic_denoised_norm)

    np.savez(
        os.path.join(args.save_dir, f'norm_stats_sigma{args.sigma}.npz'),
        pmin=np.float32(pmin),
        pmax=np.float32(pmax)
    )

    print('保存完成：')
    print(os.path.join(args.save_dir, f'clean_norm_sigma{args.sigma}.npy'))
    print(os.path.join(args.save_dir, f'noisy_norm_sigma{args.sigma}.npy'))
    print(os.path.join(args.save_dir, f'bm3d_denoised_norm_sigma{args.sigma}.npy'))


if __name__ == '__main__':
    main()