# Common Run Commands

Adjust data paths before running. The commands below are templates.

## DCDicL

```bash
cd DCDicL_denoising

python test_dcdicl_seismic.py \
  -opt options/test_denoising.json \
  --src_segy_path seismic_data/test.segy \
  --norm_mode zero_one \
  --sigma_mode pixel \
  --sigma 18.87 \
  --patch_size 256 \
  --stride 128 \
  --batch_size 8
```

Real-data inference:

```bash
python real_test_dcdicl_seismic.py \
  -opt options/test_denoising.json \
  --noisy_segy_path seismic_data/P.sgy \
  --clean_npy_path seismic_data/clean.npy \
  --sigma 27.8 \
  --patch_size 256 \
  --stride 128 \
  --batch_size 8
```

Fine-tuning:

```bash
python train_dcdicl_seismic.py -opt options/train_seismic_finetune_fast.json
```

## BM3D

```bash
cd BM3D
python BM3D.py \
  --segy_path data/P.sgy \
  --sigma 0.05 \
  --set_dir bm3d_sigma0.05
```

Real-data inference:

```bash
python real_data_BM3D.py \
  --noisy_segy_path data/P.sgy \
  --clean_npy_path data/clean.npy \
  --sigma 0.074
```

## KSVD

```bash
cd ksvd/ksvd-master
python main_train_ksvd.py \
  --segy_path data/P.sgy \
  --sigma 0.05 \
  --patch_h 8 \
  --patch_w 8 \
  --stride 4 \
  --dict_size 256 \
  --n_nonzero 8 \
  --max_iter 20
```

## DnCNN

```bash
cd 3.8DNCNN
python main_test.py \
  --src_segy_path data/P.sgy \
  --sigma 0.05 \
  --model_1_dir models/DnCNN_sigma0.05 \
  --model_name model_047.pth \
  --result_dir results \
  --batch_size 64
```

Real-data inference:

```bash
python real_data_main_test.py \
  --noisy_npy_path data/real_noisy.npy \
  --clean_npy_path data/clean.npy \
  --sigma 0.05 \
  --model_1_dir models/DnCNN_sigma0.05 \
  --model_name model_047.pth
```

## Plotting

```bash
cd plotting
python plt3_ksvd_bm3d.py
```

Plotting scripts usually contain hardcoded paths. Update them to point to downloaded arrays.
