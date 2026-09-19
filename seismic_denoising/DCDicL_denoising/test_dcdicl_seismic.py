import argparse
import datetime
import os
import time
from typing import List, Tuple

import numpy as np
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
        '--clean_npy_path',
        type=str,
        default=os.path.join('seismic_data', '合成数据',
                             'clean_norm_sigma0.05.npy'),
        help='Path to clean npy. Expected value range: [-1, 1].'
    )
    parser.add_argument(
        '--noisy_npy_path',
        type=str,
        default=os.path.join('seismic_data', '合成数据',
                             'noisy_norm_sigma0.05.npy'),
        help='Path to noisy npy. Expected value range: [-1, 1].'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        nargs='+',
        default=[255.0 * 0.05 * 0.5],
        help='Sigma list fed to model. Default maps sigma=0.05 in [-1,1] to [0,1].'
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
        default='debug/pretrain_5_9_npy_results',
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


def load_norm_npy(path: str, name: str, tol: float = 1e-4) -> np.ndarray:
    data = np.load(path).astype(np.float32)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(
            f'{name} must be a 2D array after squeeze, got shape {data.shape}'
        )

    data_min = float(np.min(data))
    data_max = float(np.max(data))
    if data_min < -1.0 - tol or data_max > 1.0 + tol:
        raise ValueError(
            f'{name} must be in [-1, 1], got min={data_min:.8f}, max={data_max:.8f}'
        )
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def to_zero_one_from_minus_one_one(x_11: np.ndarray) -> np.ndarray:
    return np.clip((x_11 + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


def to_minus_one_one_from_zero_one(x_01: np.ndarray) -> np.ndarray:
    x_11 = 2.0 * x_01 - 1.0
    return np.clip(x_11, -1.0, 1.0).astype(np.float32)


def get_patch_starts(size: int, patch_size: int, stride: int) -> List[int]:
    starts = list(range(0, max(size - patch_size + 1, 1), stride))
    if len(starts) == 0 or starts[-1] != size - patch_size:
        starts.append(max(size - patch_size, 0))
    return sorted(set(starts))


def extract_patches_2d(img: np.ndarray,
                       patch_size: int,
                       stride: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    h, w = img.shape
    h_starts = get_patch_starts(h, patch_size, stride)
    w_starts = get_patch_starts(w, patch_size, stride)

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
                    x_noisy_01: np.ndarray,
                    sigma: float,
                    patch_size: int,
                    stride: int,
                    batch_size: int) -> np.ndarray:
    patches, positions = extract_patches_2d(x_noisy_01, patch_size, stride)
    device = model.device
    out_patches = []

    log(
        f'Patch inference: patch_size={patch_size}, stride={stride}, total_patches={len(patches)}, batch_size={batch_size}'
    )

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

    denoised_01 = np.concatenate(out_patches, axis=0)
    denoised_01 = reconstruct_from_patches(denoised_01, positions,
                                            x_noisy_01.shape, patch_size)
    return np.clip(denoised_01, 0.0, 1.0).astype(np.float32)


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

    if not os.path.exists(args.clean_npy_path):
        raise FileNotFoundError(f'Clean NPY not found: {args.clean_npy_path}')
    if not os.path.exists(args.noisy_npy_path):
        raise FileNotFoundError(f'Noisy NPY not found: {args.noisy_npy_path}')

    x_clean_11 = load_norm_npy(args.clean_npy_path, 'clean')
    x_noisy_11 = load_norm_npy(args.noisy_npy_path, 'noisy')

    log(
        f'clean shape: {x_clean_11.shape}, range: [{x_clean_11.min():.6f}, {x_clean_11.max():.6f}]'
    )
    log(
        f'noisy shape: {x_noisy_11.shape}, range: [{x_noisy_11.min():.6f}, {x_noisy_11.max():.6f}]'
    )

    if x_clean_11.shape != x_noisy_11.shape:
        raise ValueError(
            f'shape mismatch: clean={x_clean_11.shape}, noisy={x_noisy_11.shape}'
        )

    x_noisy_01 = to_zero_one_from_minus_one_one(x_noisy_11)
    clean_stem = os.path.splitext(os.path.basename(args.clean_npy_path))[0]
    noisy_stem = os.path.splitext(os.path.basename(args.noisy_npy_path))[0]
    stem = f'{clean_stem}__{noisy_stem}'

    for sigma_arg in [float(v) for v in args.sigma]:
        sigma_tag = str(sigma_arg).replace('.', 'p')
        save_dir = os.path.join(args.result_dir, f'sigma_{sigma_tag}')
        os.makedirs(save_dir, exist_ok=True)

        sigma_model = sigma_arg / 255.0 if args.sigma_mode == 'pixel' else sigma_arg
        log(
            f'Using sigma_model={sigma_model:.6f}, sigma_arg={sigma_arg}, sigma_mode={args.sigma_mode}'
        )

        base_psnr = compare_psnr(x_clean_11, x_noisy_11, data_range=2.0)
        base_ssim = compare_ssim(x_clean_11,
                                 x_noisy_11,
                                 data_range=2.0,
                                 channel_axis=None)
        base_snr = calc_snr(x_clean_11, x_noisy_11)
        base_rmse = float(np.sqrt(np.mean((x_clean_11 - x_noisy_11) ** 2)))

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

        x_denoised_11 = to_minus_one_one_from_zero_one(x_denoised_01)

        den_psnr = compare_psnr(x_clean_11, x_denoised_11, data_range=2.0)
        den_ssim = compare_ssim(x_clean_11,
                                x_denoised_11,
                                data_range=2.0,
                                channel_axis=None)
        den_snr = calc_snr(x_clean_11, x_denoised_11)
        den_rmse = float(np.sqrt(np.mean((x_clean_11 - x_denoised_11) ** 2)))

        log(f'[{sigma_arg}] Baseline(noisy): PSNR={base_psnr:.4f}, SSIM={base_ssim:.4f}, SNR={base_snr:.4f}, RMSE={base_rmse:.6f}')
        log(f'[{sigma_arg}] Denoised      : PSNR={den_psnr:.4f}, SSIM={den_ssim:.4f}, SNR={den_snr:.4f}, RMSE={den_rmse:.6f}, elapsed={elapsed:.2f}s')

        np.save(os.path.join(save_dir, f'{stem}_clean.npy'), x_clean_11)
        np.save(os.path.join(save_dir, f'{stem}_noisy.npy'), x_noisy_11)
        np.save(os.path.join(save_dir, f'{stem}_denoised.npy'), x_denoised_11)

        with open(os.path.join(save_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
            f.write(f'DCDicL synthetic npy test | sigma={sigma_arg}\n')
            f.write(f"test_time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=====================\n")
            f.write(f'clean_npy_path={args.clean_npy_path}\n')
            f.write(f'noisy_npy_path={args.noisy_npy_path}\n')
            f.write(f'sigma_arg={sigma_arg}\n')
            f.write(f'sigma_model={sigma_model:.8f}\n')
            f.write('input_norm=minus_one_one\n')
            f.write('model_input_norm=zero_one_by_x_plus_1_over_2\n')
            f.write('metric_norm=minus_one_one_original_npy_domain\n')
            f.write('save_npy_norm=minus_one_one\n')
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
