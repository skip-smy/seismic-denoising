import argparse
import datetime
import os
import time
from typing import List, Tuple

import numpy as np
import segyio
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from models.model import Model
from utils import utils_option as option


def infer_in_channels_from_head(head_path: str) -> int:
    state = torch.load(head_path, map_location='cpu')
    if 'head_x.0.weight' not in state:
        raise KeyError(f'Cannot find head_x.0.weight in {head_path}')
    return int(state['head_x.0.weight'].shape[1] - 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt',
        type=str,
        default='options/test_denoising.json',
        help='Path to option JSON file.'
    )
    parser.add_argument(
        '--noisy_segy_path',
        type=str,
        default='seismic_data/P.sgy',
        help='Path to real noisy SEG-Y file.'
    )
    parser.add_argument(
        '--clean_npy_path',
        type=str,
        default='seismic_data/clean.npy',
        help='Path to pseudo-clean npy file.'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        nargs='+',
        default=[25,26,30],
        help='Sigma list fed to model.'
    )
    parser.add_argument(
        '--sigma_mode',
        type=str,
        default='pixel',
        choices=['pixel', 'norm'],
        help='"pixel": sigma/255 for model input; "norm": use sigma directly.'
    )
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument(
        '--result_dir',
        type=str,
        default='real_results_5_8',
        help='Directory to save outputs/metrics.'
    )

    return parser.parse_args()


def log(*args):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:'), *args)


def calc_snr(x_clean: np.ndarray, x_result: np.ndarray) -> float:
    noise = x_clean - x_result
    ps = np.sum(x_clean ** 2)
    pn = np.sum(noise ** 2)
    return float(10 * np.log10(ps / (pn + 1e-12)))


def load_segy_as_numpy(segy_path: str) -> np.ndarray:
    with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
        data = f.trace.raw[:].astype(np.float32)
    return data


def normalize_01_with_given_range(data: np.ndarray,
                                  vmin: float,
                                  vmax: float,
                                  eps: float = 1e-12) -> np.ndarray:
    data_norm = (data - vmin) / (vmax - vmin + eps)
    data_norm = np.clip(data_norm, 0.0, 1.0).astype(np.float32)
    return data_norm


def to_minus_one_one_from_01(x_01: np.ndarray) -> np.ndarray:
    x_11 = 2.0 * x_01 - 1.0
    return np.clip(x_11, -1.0, 1.0).astype(np.float32)


def extract_patches_2d(img: np.ndarray,
                       patch_size: int,
                       stride: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    h, w = img.shape
    h_starts = list(range(0, max(h - patch_size + 1, 1), stride))
    w_starts = list(range(0, max(w - patch_size + 1, 1), stride))

    if len(h_starts) == 0 or h_starts[-1] != h - patch_size:
        h_starts.append(max(h - patch_size, 0))
    if len(w_starts) == 0 or w_starts[-1] != w - patch_size:
        w_starts.append(max(w - patch_size, 0))

    h_starts = sorted(set(h_starts))
    w_starts = sorted(set(w_starts))

    patches, positions = [], []
    for i in h_starts:
        for j in w_starts:
            patch = img[i:i + patch_size, j:j + patch_size]
            if patch.shape != (patch_size, patch_size):
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='edge')
            patches.append(patch.astype(np.float32))
            positions.append((i, j))

    return np.array(patches, dtype=np.float32), positions


def reconstruct_from_patches(patches: np.ndarray,
                             positions: List[Tuple[int, int]],
                             out_shape: Tuple[int, int],
                             patch_size: int) -> np.ndarray:
    h, w = out_shape
    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    for patch, (i, j) in zip(patches, positions):
        ih = min(i + patch_size, h)
        jw = min(j + patch_size, w)
        ph = ih - i
        pw = jw - j
        output[i:ih, j:jw] += patch[:ph, :pw]
        weight[i:ih, j:jw] += 1.0

    weight[weight == 0] = 1.0
    return output / weight


def patch_inference(model: Model,
                    x_noisy: np.ndarray,
                    sigma: float,
                    patch_size: int,
                    stride: int,
                    batch_size: int) -> np.ndarray:
    patches, positions = extract_patches_2d(x_noisy, patch_size, stride)
    device = model.device
    out_patches = []

    model.net.eval()
    for start in range(0, len(patches), batch_size):
        end = min(start + batch_size, len(patches))
        batch = patches[start:end]
        batch_t = torch.from_numpy(batch).unsqueeze(1).to(device)
        sigma_t = torch.full(
            (batch_t.size(0), 1, 1, 1),
            sigma,
            device=device,
            dtype=torch.float32
        )
        with torch.no_grad():
            pred, _ = model.net(batch_t, sigma_t)
        out_patches.append(pred.squeeze(1).detach().cpu().numpy())

    denoised = np.concatenate(out_patches, axis=0)
    denoised = reconstruct_from_patches(denoised, positions, x_noisy.shape, patch_size)
    denoised = np.clip(denoised, 0.0, 1.0).astype(np.float32)
    return denoised


def main():
    args = parse_args()

    opt = option.parse(args.opt, is_train=False)

    head_path = os.path.join(opt['path']['pretrained_netG'], 'head.pth')
    if not os.path.exists(head_path):
        raise FileNotFoundError(f'Checkpoint not found: {head_path}')

    in_nc = infer_in_channels_from_head(head_path)
    opt['data']['n_channels'] = in_nc
    opt['netG']['in_nc'] = in_nc
    opt['netG']['out_nc'] = in_nc
    log(f'Infer checkpoint input channels: {in_nc}')

    model = Model(opt)
    model.init()

    if not os.path.exists(args.noisy_segy_path):
        raise FileNotFoundError(f'Noisy SEGY not found: {args.noisy_segy_path}')
    if not os.path.exists(args.clean_npy_path):
        raise FileNotFoundError(f'Clean NPY not found: {args.clean_npy_path}')

    log('Loading noisy.sgy ...')
    x_noisy_raw = load_segy_as_numpy(args.noisy_segy_path)

    log('Loading clean.npy ...')
    x_clean_raw = np.load(args.clean_npy_path).astype(np.float32)

    log(f'noisy raw shape: {x_noisy_raw.shape}')
    log(f'clean raw shape: {x_clean_raw.shape}')

    if x_noisy_raw.shape != x_clean_raw.shape:
        raise ValueError(f'shape mismatch: noisy={x_noisy_raw.shape}, clean={x_clean_raw.shape}')

    pmin, pmax = np.percentile(x_clean_raw, 1), np.percentile(x_clean_raw, 99)
    log(f'0-1 normalize with clean p1/p99: vmin={pmin:.6f}, vmax={pmax:.6f}')

    x_clean_01 = normalize_01_with_given_range(x_clean_raw, pmin, pmax)
    x_noisy_01 = normalize_01_with_given_range(x_noisy_raw, pmin, pmax)

    stem = os.path.splitext(os.path.basename(args.noisy_segy_path))[0]

    for sigma in args.sigma:
        save_dir = os.path.join(args.result_dir, f"real_dcdicl_sigma_{sigma:g}")
        os.makedirs(save_dir, exist_ok=True)

        sigma_model = sigma / 255.0 if args.sigma_mode == 'pixel' else sigma
        log(f'Using sigma_model={sigma_model:.6f}, sigma_arg={sigma}, sigma_mode={args.sigma_mode}')

        start = time.time()
        x_denoised_01 = patch_inference(
            model,
            x_noisy_01,
            sigma_model,
            args.patch_size,
            args.stride,
            args.batch_size
        )
        elapsed = time.time() - start

        x_clean_11 = to_minus_one_one_from_01(x_clean_01)
        x_noisy_11 = to_minus_one_one_from_01(x_noisy_01)
        x_denoised_11 = to_minus_one_one_from_01(x_denoised_01)

        base_psnr = compare_psnr(x_clean_11, x_noisy_11, data_range=2.0)
        base_ssim = compare_ssim(x_clean_11, x_noisy_11, data_range=2.0, channel_axis=None)
        base_snr = calc_snr(x_clean_11, x_noisy_11)
        base_rmse = float(np.sqrt(np.mean((x_clean_11 - x_noisy_11) ** 2)))

        den_psnr = compare_psnr(x_clean_11, x_denoised_11, data_range=2.0)
        den_ssim = compare_ssim(x_clean_11, x_denoised_11, data_range=2.0, channel_axis=None)
        den_snr = calc_snr(x_clean_11, x_denoised_11)
        den_rmse = float(np.sqrt(np.mean((x_clean_11 - x_denoised_11) ** 2)))

        log(f'[{sigma}] Baseline(noisy): PSNR={base_psnr:.4f}, SSIM={base_ssim:.4f}, SNR={base_snr:.4f}, RMSE={base_rmse:.6f}')
        log(f'[{sigma}] Denoised      : PSNR={den_psnr:.4f}, SSIM={den_ssim:.4f}, SNR={den_snr:.4f}, RMSE={den_rmse:.6f}, elapsed={elapsed:.2f}s')

        np.save(os.path.join(save_dir, f'{stem}_clean.npy'), x_clean_11)
        np.save(os.path.join(save_dir, f'{stem}_noisy.npy'), x_noisy_11)
        np.save(os.path.join(save_dir, f'{stem}_denoised.npy'), x_denoised_11)

        with open(os.path.join(save_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
            f.write(f"DCDicL日志 | sigma={sigma}\n")
            f.write(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=====================\n")
            f.write(f'noisy_segy_path={args.noisy_segy_path}\n')
            f.write(f'clean_npy_path={args.clean_npy_path}\n')
            f.write(f'sigma_arg={sigma}\n')
            f.write(f'sigma_model={sigma_model:.8f}\n')
            f.write('process_norm=zero_one\n')
            f.write('save_and_metric_norm=minus_one_one_by_2x_minus_1_with_clip\n')
            f.write(f'baseline_psnr={base_psnr:.6f}\n')
            f.write(f'baseline_ssim={base_ssim:.6f}\n')
            f.write(f'baseline_snr={base_snr:.6f}\n')
            f.write(f'baseline_rmse={base_rmse:.6f}\n')
            f.write(f'denoised_psnr={den_psnr:.6f}\n')
            f.write(f'denoised_ssim={den_ssim:.6f}\n')
            f.write(f'denoised_snr={den_snr:.6f}\n')
            f.write(f'denoised_rmse={den_rmse:.6f}\n')
            f.write(f'elapsed_sec={elapsed:.6f}\n')

        log(f'All results saved to: {save_dir}')


if __name__ == '__main__':
    main()