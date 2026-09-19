import argparse
import logging
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_seismic_segy import DatasetSeismicSegy
from models.model import Model
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def calc_snr(x_clean: np.ndarray, x_result: np.ndarray) -> float:
    noise = x_clean - x_result
    ps = np.sum(x_clean ** 2)
    pn = np.sum(noise ** 2)
    return float(10 * np.log10(ps / (pn + 1e-12)))


def to_minus_one_one_from_zero_one(x_01: np.ndarray) -> np.ndarray:
    return 2.0 * x_01 - 1.0


def evaluate(model: Model, val_loader: DataLoader) -> Dict[str, float]:
    net = model.net
    was_training = net.training
    net.eval()

    metric_sums = {
        'baseline_psnr': 0.0,
        'baseline_ssim': 0.0,
        'baseline_snr': 0.0,
        'baseline_rmse': 0.0,
        'denoised_psnr': 0.0,
        'denoised_ssim': 0.0,
        'denoised_snr': 0.0,
        'denoised_rmse': 0.0,
    }
    count = 0

    with torch.no_grad():
        for data in tqdm(val_loader, desc='validate', leave=False):
            y = data['y'].to(model.device)
            y_gt = data['y_gt'].to(model.device)
            sigma = data['sigma'].to(model.device)
            pred, _ = net(y, sigma)
            pred = pred.clamp(0.0, 1.0)

            noisy_np = y.detach().cpu().numpy()
            clean_np = y_gt.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()

            for b in range(clean_np.shape[0]):
                clean_11 = to_minus_one_one_from_zero_one(clean_np[b, 0])
                noisy_11 = to_minus_one_one_from_zero_one(noisy_np[b, 0])
                pred_11 = to_minus_one_one_from_zero_one(pred_np[b, 0])

                metric_sums['baseline_psnr'] += compare_psnr(
                    clean_11, noisy_11, data_range=2.0)
                metric_sums['baseline_ssim'] += compare_ssim(
                    clean_11, noisy_11, data_range=2.0, channel_axis=None)
                metric_sums['baseline_snr'] += calc_snr(clean_11, noisy_11)
                metric_sums['baseline_rmse'] += float(
                    np.sqrt(np.mean((clean_11 - noisy_11) ** 2)))

                metric_sums['denoised_psnr'] += compare_psnr(
                    clean_11, pred_11, data_range=2.0)
                metric_sums['denoised_ssim'] += compare_ssim(
                    clean_11, pred_11, data_range=2.0, channel_axis=None)
                metric_sums['denoised_snr'] += calc_snr(clean_11, pred_11)
                metric_sums['denoised_rmse'] += float(
                    np.sqrt(np.mean((clean_11 - pred_11) ** 2)))
                count += 1

    if was_training:
        net.train()

    return {key: value / max(count, 1) for key, value in metric_sums.items()}


def main(json_path: str = 'options/train_seismic_finetune.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',
                        type=str,
                        default=json_path,
                        help='Path to option JSON file.')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    util.makedirs(
        [path for key, path in opt['path'].items() if 'pretrained' not in key])
    option.save(opt)

    logger_name = 'train_seismic'
    utils_logger.logger_info(
        logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    seed = int(opt['train'].get('manual_seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    opt_data_train: Dict[str, Any] = opt['data']['train']
    train_set = DatasetSeismicSegy(opt_data_train, 'train')
    logger.info(
        f'Train SEGY={train_set.segy_path}, shape={train_set.clean_11.shape}, '
        f'norm_vmin={train_set.norm_vmin:.8f}, norm_vmax={train_set.norm_vmax:.8f}'
    )

    opt_data_val: Dict[str, Any] = opt['data']['test']
    if opt_data_val.get('norm_mode', opt_data_train.get('norm_mode')) in [
            'percentile', 'minmax'
    ]:
        opt_data_val['norm_vmin'] = train_set.norm_vmin
        opt_data_val['norm_vmax'] = train_set.norm_vmax
    val_set = DatasetSeismicSegy(opt_data_val, 'val')
    logger.info(
        f'Val SEGY={val_set.segy_path}, shape={val_set.clean_11.shape}, '
        f'num_val_patches={len(val_set)}')

    train_loader = DataLoader(
        train_set,
        batch_size=opt_data_train['batch_size'],
        shuffle=True,
        num_workers=opt_data_train.get('num_workers', 0),
        drop_last=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_set,
        batch_size=opt_data_val.get('batch_size', 1),
        shuffle=False,
        num_workers=opt_data_val.get('num_workers', 0),
        drop_last=False,
        pin_memory=True)

    model = Model(opt)
    model.init()

    current_step = 0
    total_iters = int(opt['train'].get('total_iters', 20000))
    best_psnr = -float('inf')
    start_time = time.time()

    while current_step < total_iters:
        for train_data in tqdm(train_loader, desc='train'):
            current_step += 1
            model.feed_data(train_data)
            model.train()
            model.update_learning_rate(current_step)

            if current_step % opt['train']['checkpoint_log'] == 0:
                model.log_train(current_step, 0, logger)

            if current_step % opt['train']['checkpoint_test'] == 0:
                metrics = evaluate(model, val_loader)
                logger.info(
                    'Validation step:{:8,d}, baseline PSNR:{:.4f}, SSIM:{:.4f}, '
                    'SNR:{:.4f}, RMSE:{:.6f}'.format(
                        current_step, metrics['baseline_psnr'],
                        metrics['baseline_ssim'], metrics['baseline_snr'],
                        metrics['baseline_rmse']))
                logger.info(
                    'Validation step:{:8,d}, denoised PSNR:{:.4f}, SSIM:{:.4f}, '
                    'SNR:{:.4f}, RMSE:{:.6f}, elapsed:{:.2f}s'.format(
                        current_step, metrics['denoised_psnr'],
                        metrics['denoised_ssim'], metrics['denoised_snr'],
                        metrics['denoised_rmse'],
                        time.time() - start_time))
                start_time = time.time()

                if metrics['denoised_psnr'] > best_psnr:
                    best_psnr = metrics['denoised_psnr']
                    logger.info(
                        f'New best denoised PSNR={best_psnr:.4f}; saving model.')
                    model.save(logger)

            if current_step % opt['train']['checkpoint_savemodel'] == 0:
                model.save(logger)

            if current_step >= total_iters:
                break

    logger.info('Training finished.')
    model.save(logger)


if __name__ == '__main__':
    main()
