# What To Download From AutoDL

Use this file to decide what extra data/results to download in addition to the lightweight code package.

## My Recommendation

Download these two groups:

1. Core raw data from `DCDicL_denoising/seismic_data`
2. Final plotting/result arrays from the plotting folder's `data` and `fig_check`

This is enough for the next person to continue the project, reproduce most figures, and run all methods after fixing paths.
Do not download every method's `data/`, `results/`, and `real_results/` folders unless you need a full historical backup.

## A. Must Download: Core Data

Recommended source folder:

```text
/root/autodl-tmp/DCDicL_denoising/seismic_data
```

Important files inside:

```text
train.segy                         about 795 MB  training data
val.segy                           about 99 MB   validation data
test.segy                          about 99 MB   synthetic/test data source
P.sgy                              about 429 MB  real noisy seismic data
clean.npy                          about 426 MB  clean/reference data
synthetic clean/noisy npy files    about 190 MB  prepared synthetic arrays
```

Approx size: 1.6 GB.

Why only this folder:

- The same `P.sgy` and `clean.npy` are copied in BM3D, DnCNN, and KSVD folders.
- Downloading each baseline `data/` folder would duplicate data.
- After download, copy or symlink these files into each method's expected `data/` folder.

## B. Strongly Recommended: Final Plotting Data And Figures

The plotting folder has a Chinese name on AutoDL. To find it:

```bash
find /root/autodl-tmp -maxdepth 2 -name plt1.py -printf '%h\n' | head -1
```

Recommended folders inside it:

```text
data        about 4.2 GB
fig_check   about 34 MB
```

Why download these:

- The plotting `data` folder already collects final arrays for multiple methods.
- It contains synthetic comparison arrays:
  - `bm3d_denoised_norm_sigma0.05.npy`
  - `dcdicl_denoised_norm_sigma0.05.npy`
  - `ksvd_denoised_norm_sigma0.05.npy`
  - `test_dncnn_denoised.npy`
  - `noisy_norm_sigma0.05.npy`
  - `test_clean.npy`
- It also contains real-data comparison arrays under `data/real`.
- `fig_check` contains existing paper/report figures and visual checks.

This is the best result folder to download if the goal is thesis handoff.

## C. Optional: Method-Specific Result Folders

Only download these if the next person needs to inspect each method's historical outputs.

```text
/root/autodl-tmp/DCDicL_denoising/real_results                 about 4.4 GB
/root/autodl-tmp/DCDicL_denoising/debug                        about 2.3 GB
/root/autodl-tmp/BM3D/results                                  about 1.3 GB
/root/autodl-tmp/BM3D/real_results                             about 1.2 GB
/root/autodl-tmp/ksvd/ksvd-master/results                      about 1.2 GB
/root/autodl-tmp/ksvd/ksvd-master/real_results                 about 2.0 GB
/root/autodl-tmp/3.8DNCNN/results                              about 939 MB
/root/autodl-tmp/3.8DNCNN/real_results                         about 2.4 GB
```

My practical choice:

- For thesis continuation: download only the plotting `data` and `fig_check` folders.
- For full audit/history: download all method-specific results too.

## D. Single Usable DCDicL Real Result

The absolute best metric folder is `real_dcdicl_sigma_27.8_yong`, but that folder only contains clean/noisy arrays and metrics.
For a usable handoff result with denoised output included, use:

```text
/root/autodl-tmp/DCDicL_denoising/real_results/real_dcdicl_sigma_27.7
```

Metrics recorded:

```text
baseline_psnr=21.957482
baseline_ssim=0.577431
baseline_snr=1.306899
baseline_rmse=0.159645
denoised_psnr=23.304150
denoised_ssim=0.629117
denoised_snr=2.653568
denoised_rmse=0.136717
```

If storage is tight and you only want one DCDicL real result, download this folder.

## E. Suggested Download Commands

From local Windows PowerShell:

```powershell
mkdir D:\autodl_handoff\seismic_extra

scp -r seetacloud-autodl:/root/autodl-tmp/DCDicL_denoising/seismic_data D:\autodl_handoff\seismic_extra\

$plotDir = ssh seetacloud-autodl "find /root/autodl-tmp -maxdepth 2 -name plt1.py -printf '%h\n' | head -1"
scp -r "seetacloud-autodl:$plotDir/data" D:\autodl_handoff\seismic_extra\plotting_data
scp -r "seetacloud-autodl:$plotDir/fig_check" D:\autodl_handoff\seismic_extra\fig_check
```

If you only want the best DCDicL real result:

```powershell
scp -r seetacloud-autodl:/root/autodl-tmp/DCDicL_denoising/real_results/real_dcdicl_sigma_27.7 D:\autodl_handoff\seismic_extra\
```

## F. Do Not Download Unless Necessary

Skip these for normal handoff:

```text
/root/autodl-tmp/.git
/root/autodl-tmp/.Trash-0
/root/autodl-tmp/.autodl
all .ipynb_checkpoints
all __pycache__
duplicate baseline data folders
```
