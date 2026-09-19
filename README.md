# Seismic Denoising Code

Code and configs for seismic denoising experiments (DCDicL + baselines: DnCNN / BM3D / K-SVD).

## Layout

- `seismic_denoising/` — main code package
  - `DCDicL_denoising/` — proposed method
  - `3.8DNCNN/`, `BM3D/`, `ksvd/` — baselines
  - `plotting/` — figure scripts
  - `RUN_COMMANDS.md`, `DATA_MANIFEST.md`, `WHAT_TO_DOWNLOAD.md`
- `图片/` — paper figures (optional)
- `sample_data/` — ~1MB clean/noisy patches for smoke tests

## Not in this repo

Full datasets (`.segy` / large `.npy`), result tensors, and model weights are **not** uploaded (GitHub size limits).
See `seismic_denoising/WHAT_TO_DOWNLOAD.md` and `DATA_MANIFEST.md` for how to obtain them.

## Quick start

```bash
cd seismic_denoising
pip install -r requirements_minimal.txt
# then follow RUN_COMMANDS.md
```

Smoke-test patches:

```bash
python -c "import numpy as np; print(np.load('sample_data/clean_patch.npy').shape)"
```
