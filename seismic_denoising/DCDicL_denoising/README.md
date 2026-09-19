# Deep Convolutional Dictionary Learning for Image Denoising
Hongyi Zheng*, Hongwei Yong*, Lei Zhang, "Deep Convolutional Dictionary Learning for Image Denoising," in CVPR 2021. (* Equal contribution)

[[paper]](https://www4.comp.polyu.edu.hk/~cslzhang/paper/DCDicL-cvpr21-final.pdf) [[supp]](https://www4.comp.polyu.edu.hk/~cslzhang/paper/DCDicL-cvpr21-supp.pdf)

The implementation of DCDicL is based on the awesome Image Restoration Toolbox [[KAIR]](https://github.com/cszn/KAIR).

## Requirement
- PyTorch 1.6+
- prettytable
- tqdm

## Testing
**Step 1**

- Download pretrained models from [[OneDrive]](https://1drv.ms/u/c/1bb582448f973d92/EZTo0uKvydxIocrcLiic94sBB-3E1uBUoAa3jCXkgeRBMA?e=SpQGc1) or [[BaiduPan]](https://pan.baidu.com/share/init?surl=vIqN2XiZ9UH8vcUpZPbXnw) (password: flfw) or [[Google Drive]](https://drive.google.com/file/d/1MKs5E8bt1H6xnCzf_Wa8Yd-FKupN7C3-/view?usp=sharing).
- Unzip downloaded file and put the folders into ```./release/denoising```

**Step 2**

Configure ```options/test_denoising.json```. Important settings:
- task: task name.
- path/root: path to save the tasks.
- path/pretrained_netG: path to the folder containing the pretrained models.
- data/n_channels: 1 for greyscale and 3 for color.
- test/visualize: true for saving the noisy input/predicted dictionaries.
- train/load_strict: optional, defaults to true. Set to false if you intentionally load a partially matched checkpoint (for example, when you changed network depth/channels for adaptation experiments).

**Step 3**
```bash
python test_dcdicl.py
```

For synthetic-noise seismic evaluation directly from SEGY, use:
```bash
python test_dcdicl_seismic.py -opt options/test_denoising.json --src_segy_path /path/to/clean.segy --norm_mode zero_one --sigma_mode pixel --sigma 18.87 --patch_size 256 --stride 128 --batch_size 8 --metrics_norm_mode minus_one_one --save_npy_norm_mode minus_one_one
```
You can pass multiple sigma values in one run, e.g. `--sigma 18 18.87 19`. The script will create per-sigma subfolders under `result_dir`.
If `--sigma` is omitted, `test_dcdicl_seismic.py` falls back to `data.test.sigma` from the provided `-opt` JSON.
Use `--sigma_mode pixel` (default) to follow the original denoising convention (`sigma/255`), and prefer `--norm_mode zero_one` to match the repo's image normalization behavior.
`--metrics_norm_mode minus_one_one` computes baseline/denoised metrics in `[-1,1]` (via linear transform `x->2x-1`) even when processing is done in `[0,1]`.
`--save_npy_norm_mode minus_one_one` saves `*_clean.npy`, `*_noisy.npy`, and `*_denoised.npy` in `[-1,1]`.
For strict control-variable comparison across models, generate one noise realization and reuse it via `--noise_npy` (or provide fixed noisy data via `--noisy_npy`) so baseline noisy metrics are identical. Reused `noise_npy` must use the same `norm_mode`.
Example option templates for multi-model comparison are provided at `options/test_A.json`, `options/test_B.json`, and `options/test_C.json` (only `path/pretrained_netG` differs by default).

## Notes for seismic data
- Large seismic slices such as `17850 x 1332` can be tested, but they are memory-heavy. This repo crops image size to multiples of 8 during testing, so the effective inference area is `(17848, 1328)` for an input of `(17850, 1332)`.
- If your seismic data is single-channel, set `data/n_channels` to `1` in the testing options.
- If GPU memory is insufficient, run inference patch-by-patch and merge the output (overlap + weighted averaging), or run on CPU.
- If loading pretrained weights fails due to key/shape mismatch after your architecture edits, set `train/load_strict` to `false` first to debug incompatibilities, then decide whether you need to align the model definition or retrain.
- Testing in this repo expects image files in `data/test/...` folders (not SEGY directly). If your source is SEGY, first convert each slice/section to image format and keep paired clean/noisy evaluation protocol consistent.
- For fair metric comparison across different denoising models, keep the same clean input and exactly the same noisy input. In this repo test pipeline, Gaussian noise uses a fixed seed (`np.random.seed(0)`), so the baseline noisy PSNR/SSIM should be identical when the same test data and sigma are used.
- `test_dcdicl_seismic.py` supports direct SEGY input for synthetic-noise testing and reports baseline (noisy) vs denoised SNR/PSNR/SSIM/RMSE.



## Training

If you want to achieve the best performance:
- you have to first train a 1-stage model, then train a multi-stage (2~6) model based on the pretrained model. (Please refer the paper for more details.)
- you have to include [[Waterloo Exploration Database]](https://ece.uwaterloo.ca/~k29ma/exploration/) in the training sets.

**Step 1**

Prepare training/testing data. The folder structure should be similar to:

```
+-- data
|   +-- train
|       +-- training_dataset_1
|       +-- training_dataset_2
|   +-- test
|       +-- testing_dataset_1
|       +-- testing_dataset_2
```

**Step 2**

Configure ```options/train_denoising.json```. Important settings:
- task: task name.
- path/root: path to save the tasks.
- data/n_channels: 1 for greyscale and 3 for color.
- data/train/sigma: range of noise levels.
- netG/d_size: dictionary size.
- netG/n_iter: number of iterations.
- netG/nc_x: number of channels in NetX.
- netG/nb: number of blocks in NetX.
- test/visualize: true for saving the noisy input/predicted dictionaries.

If you want to reload a pretrained model, pay attention to following settings:
- path/pretrained_netG: path to the folder containing the pretrained models.
- train/reload_broadcast: if you want to load a pretrained 1-stage model into multi-stage model, please set this item to **true**.


**Step 3**
```bash
python train_dcdicl.py
```

**FAQ**
- Keep receiving ''WARNING batched routines are designed for mall sizes. It might be ...''.

This is the limitation of the backend linear algebra GPU accelerated libraries of PyTorch. The only way to get rid of it is to reduce the number of channels or spatial size of the dictionaries.

## Citation
```
@InProceedings{Zheng_2021_CVPR,
    author    = {Zheng, Hongyi and Yong, Hongwei and Zhang, Lei},
    title     = {Deep Convolutional Dictionary Learning for Image Denoising},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {630-641}
}
```
