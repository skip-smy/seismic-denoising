# Seismic Denoising Code

Code and configs for seismic denoising experiments (DCDicL + baselines: DnCNN / BM3D / K-SVD).

## Layout

- `seismic_denoising/` 鈥?main code package
  - `DCDicL_denoising/` 鈥?proposed method
  - `3.8DNCNN/`, `BM3D/`, `ksvd/` 鈥?baselines
  - `plotting/` 鈥?figure scripts
  - `RUN_COMMANDS.md`, `DATA_MANIFEST.md`, `WHAT_TO_DOWNLOAD.md`
- `鍥剧墖/` 鈥?paper figures (optional)

## Not in this repo

Large datasets (`.segy` / `.npy`), full result tensors, and model weights are **not** uploaded (GitHub size limits).
See `seismic_denoising/WHAT_TO_DOWNLOAD.md` and `DATA_MANIFEST.md` for how to obtain them.

## Quick start

```bash
cd seismic_denoising
pip install -r requirements_minimal.txt
# then follow RUN_COMMANDS.md
```
