# Data Manifest

The handoff package does not include large raw data or generated arrays.
The original AutoDL paths are listed below.

## Original Data Paths On AutoDL

- DCDicL data: `/root/autodl-tmp/DCDicL_denoising/seismic_data`
- BM3D data: `/root/autodl-tmp/BM3D/data`
- KSVD data: `/root/autodl-tmp/ksvd/ksvd-master/data`
- DnCNN data: `/root/autodl-tmp/3.8DNCNN/data`
- Plotting data: the Chinese-named plotting folder under `/root/autodl-tmp`.
  Find it with:

```bash
find /root/autodl-tmp -maxdepth 2 -name plt1.py -printf '%h\n' | head -1
```

## Original Result Paths On AutoDL

- DCDicL synthetic/debug results: `/root/autodl-tmp/DCDicL_denoising/debug`
- DCDicL real results: `/root/autodl-tmp/DCDicL_denoising/real_results`
- BM3D synthetic results: `/root/autodl-tmp/BM3D/results`
- BM3D real results: `/root/autodl-tmp/BM3D/real_results`
- KSVD synthetic results: `/root/autodl-tmp/ksvd/ksvd-master/results`
- KSVD real results: `/root/autodl-tmp/ksvd/ksvd-master/real_results`
- DnCNN synthetic results: `/root/autodl-tmp/3.8DNCNN/results`
- DnCNN real results: `/root/autodl-tmp/3.8DNCNN/real_results`
- Final figures: `fig_check` under the plotting folder found above.

## Deduplication Note

Many files are duplicated across method folders.
For example, `P.sgy` appears under DCDicL, BM3D, KSVD, DnCNN, and the plotting folder.
Do not download every `data/` folder unless you want a full historical backup.
Use `WHAT_TO_DOWNLOAD.md` for the recommended minimal download set.
