import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import segyio
except ImportError as exc:
    raise ImportError('Please install segyio to train on SEGY data.') from exc


def load_segy_as_numpy(segy_path: str) -> np.ndarray:
    with segyio.open(segy_path, 'r', ignore_geometry=True) as segy_f:
        return segy_f.trace.raw[:].astype(np.float32)


def normalize_to_minus_one_one(data: np.ndarray,
                               opt: Dict[str, Any]) -> Tuple[np.ndarray, float, float]:
    mode = opt.get('norm_mode', 'percentile')

    if mode == 'already_minus_one_one':
        return np.clip(data, -1.0, 1.0).astype(np.float32), -1.0, 1.0

    if mode == 'already_zero_one':
        data_01 = np.clip(data, 0.0, 1.0)
        return (2.0 * data_01 - 1.0).astype(np.float32), 0.0, 1.0

    if 'norm_vmin' in opt and 'norm_vmax' in opt:
        vmin = float(opt['norm_vmin'])
        vmax = float(opt['norm_vmax'])
    elif mode == 'percentile':
        p_low = float(opt.get('p_low', 1.0))
        p_high = float(opt.get('p_high', 99.0))
        vmin = float(np.percentile(data, p_low))
        vmax = float(np.percentile(data, p_high))
    elif mode == 'minmax':
        vmin = float(np.min(data))
        vmax = float(np.max(data))
    else:
        raise ValueError(f'Unsupported norm_mode: {mode}')

    data_01 = (data - vmin) / (vmax - vmin + 1e-12)
    data_01 = np.clip(data_01, 0.0, 1.0)
    return (2.0 * data_01 - 1.0).astype(np.float32), vmin, vmax


def to_zero_one_from_minus_one_one(data_11: np.ndarray) -> np.ndarray:
    return ((data_11 + 1.0) * 0.5).astype(np.float32)


def sigma_to_model_domain(sigma_arg: float, sigma_mode: str) -> float:
    if sigma_mode == 'minus_one_one':
        return float(sigma_arg) * 0.5
    if sigma_mode == 'zero_one':
        return float(sigma_arg)
    if sigma_mode == 'pixel':
        return float(sigma_arg) / 255.0
    raise ValueError(f'Unsupported sigma_mode: {sigma_mode}')


def get_patch_starts(size: int, patch_size: int, stride: int) -> List[int]:
    starts = list(range(0, max(size - patch_size + 1, 1), stride))
    if len(starts) == 0 or starts[-1] != size - patch_size:
        starts.append(max(size - patch_size, 0))
    return sorted(set(starts))


class DatasetSeismicSegy(Dataset):
    def __init__(self, opt_dataset: Dict[str, Any], phase: str):
        super().__init__()
        self.opt = opt_dataset
        self.phase = phase
        self.segy_path = opt_dataset['segy_path']
        self.patch_size = int(opt_dataset.get('H_size',
                                              opt_dataset.get('patch_size', 128)))
        self.stride = int(opt_dataset.get('stride', self.patch_size))
        self.num_patches_per_epoch = int(opt_dataset.get(
            'num_patches_per_epoch', 20000))
        self.sigma = opt_dataset.get('sigma', [0.05, 0.05])
        self.sigma_mode = opt_dataset.get('sigma_mode', 'minus_one_one')
        self.seed = int(opt_dataset.get('seed', 0))

        clean_raw = load_segy_as_numpy(self.segy_path)
        self.clean_11, self.norm_vmin, self.norm_vmax = normalize_to_minus_one_one(
            clean_raw, opt_dataset)
        self.name = os.path.splitext(os.path.basename(self.segy_path))[0]

        h, w = self.clean_11.shape
        self.positions: List[Tuple[int, int]] = []
        if phase != 'train':
            h_starts = get_patch_starts(h, self.patch_size, self.stride)
            w_starts = get_patch_starts(w, self.patch_size, self.stride)
            self.positions = [(i, j) for i in h_starts for j in w_starts]
            max_val_patches = opt_dataset.get('max_val_patches')
            if max_val_patches is not None:
                max_val_patches = int(max_val_patches)
                if max_val_patches > 0 and len(self.positions) > max_val_patches:
                    pick = np.linspace(0,
                                       len(self.positions) - 1,
                                       max_val_patches,
                                       dtype=int)
                    self.positions = [self.positions[i] for i in pick]

    def __len__(self):
        if self.phase == 'train':
            return self.num_patches_per_epoch
        return len(self.positions)

    def _sample_sigma(self, rng: Any) -> Tuple[float, float]:
        sigma = self.sigma
        if isinstance(sigma, (int, float)):
            sigma_arg = float(sigma)
        elif len(sigma) == 1:
            sigma_arg = float(sigma[0])
        else:
            sigma_arg = float(rng.uniform(float(sigma[0]), float(sigma[1])))
        return sigma_arg, sigma_to_model_domain(sigma_arg, self.sigma_mode)

    def _crop(self, top: int, left: int) -> np.ndarray:
        patch = self.clean_11[top:top + self.patch_size,
                              left:left + self.patch_size]
        if patch.shape != (self.patch_size, self.patch_size):
            pad_h = self.patch_size - patch.shape[0]
            pad_w = self.patch_size - patch.shape[1]
            patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='edge')
        return patch.astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        h, w = self.clean_11.shape

        if self.phase == 'train':
            rng = np.random
            top = int(rng.randint(0, max(h - self.patch_size, 0) + 1))
            left = int(rng.randint(0, max(w - self.patch_size, 0) + 1))
        else:
            rng = np.random.default_rng(self.seed + index)
            top, left = self.positions[index]

        clean_11 = self._crop(top, left)
        clean_01 = to_zero_one_from_minus_one_one(clean_11)
        sigma_arg, sigma_model = self._sample_sigma(rng)
        noise = rng.normal(0.0, sigma_model, clean_01.shape).astype(np.float32)
        noisy_01 = clean_01 + noise

        return {
            'y': torch.from_numpy(noisy_01).unsqueeze(0).float(),
            'y_gt': torch.from_numpy(clean_01).unsqueeze(0).float(),
            'sigma': torch.FloatTensor([sigma_model]).unsqueeze(1).unsqueeze(1),
            'path': f'{self.segy_path}:{top}:{left}',
            'sigma_arg': torch.FloatTensor([sigma_arg]),
        }
