# Seismic Denoising Handoff

This is a lightweight handoff package for the seismic denoising thesis project.
The original AutoDL workspace is about 42 GB under `/root/autodl-tmp`.
This package keeps code, configs, small required checkpoints, and handoff notes.
Large raw data, generated results, caches, and Git history are intentionally not included.

## Included Modules

- `DCDicL_denoising/`: main method, DCDicL adapted for seismic denoising.
- `BM3D/`: BM3D baseline.
- `ksvd/ksvd-master/`: KSVD baseline.
- `3.8DNCNN/`: DnCNN baseline. Only the latest checkpoint for each sigma is kept.
- `plotting/`: plotting scripts.
- `requirements_minimal.txt`: minimal Python dependencies.
- `RUN_COMMANDS.md`: common commands.
- `DATA_MANIFEST.md`: original large data paths on AutoDL.
- `WHAT_TO_DOWNLOAD.md`: what extra data/results should be downloaded.
- `ENVIRONMENT.txt`: AutoDL environment snapshot.
- `PROJECT_TREE.txt`: file tree of this handoff package.

## Not Included

- Raw seismic data: `*.sgy`, `*.segy`, `*.npy`, `*.npz`.
- Generated outputs: `results/`, `real_results/`, `debug/`, `fig_check/`.
- Git/cache folders: `.git/`, `.Trash-0/`, `.autodl/`, `__pycache__/`.
- Full DnCNN training checkpoints. Only final checkpoints are included.

## Quick Start

```bash
cd /path/to/seismic_denoising_handoff
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
pip install -r requirements_minimal.txt
```

Recommended onboarding order:

1. Read `WHAT_TO_DOWNLOAD.md` and download the required data/results.
2. Read `RUN_COMMANDS.md`.
3. Run DCDicL inference first to verify paths and checkpoints.
4. Run BM3D, KSVD, and DnCNN on the same data split for fair comparison.
5. Use `plotting/` to reproduce figures and metric comparison.
6. If continuing training, start from `DCDicL_denoising/options/train_seismic_finetune*.json`.

## Important Path Note

Some historical scripts may still contain absolute paths such as `/root/autodl-tmp/...`.
If this package is moved to a new machine, update paths in configs or pass paths through command line arguments.
The copied DCDicL test config was adjusted to use a relative pretrained model path when possible.
