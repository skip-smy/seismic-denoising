import os
import numpy as np
import segyio

def load_segy_as_numpy(segy_path):
    """
    读取 SEG-Y 文件，返回 [n_traces, n_samples]
    """
    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        data = segyio.tools.collect(f.trace[:])
    return np.asarray(data, dtype=np.float32)

def normalize_with_given_range(data, vmin, vmax, eps=1e-12):
    """
    按给定范围归一化到 [-1, 1]，并截断
    """
    data_norm = 2.0 * (data - vmin) / (vmax - vmin + eps) - 1.0
    data_norm = np.clip(data_norm, -1.0, 1.0)
    return data_norm

def sigma_std(noisy, clean):
    noise = noisy - clean
    return float(np.std(noise))

def sigma_mad(noisy, clean):
    noise = noisy - clean
    med = np.median(noise)
    mad = np.median(np.abs(noise - med))
    return float(1.4826 * mad)

# ====== 改这里 ======
noisy_path = "data/P.sgy"   # 或 .npy
clean_path = "data/clean.npy"
txt_path   = "real_results/sigma_results.txt"
# ====================

# 读取 noisy
if noisy_path.lower().endswith((".sgy", ".segy")):
    noisy = load_segy_as_numpy(noisy_path)
elif noisy_path.lower().endswith(".npy"):
    noisy = np.load(noisy_path).astype(np.float32)
else:
    raise ValueError(f"不支持的 noisy 文件格式: {noisy_path}")

# 读取 clean
if clean_path.lower().endswith((".sgy", ".segy")):
    clean = load_segy_as_numpy(clean_path)
elif clean_path.lower().endswith(".npy"):
    clean = np.load(clean_path).astype(np.float32)
else:
    raise ValueError(f"不支持的 clean 文件格式: {clean_path}")

print("noisy shape:", noisy.shape)
print("clean shape:", clean.shape)

if noisy.shape != clean.shape:
    raise ValueError(f"shape不一致: noisy{noisy.shape} vs clean{clean.shape}")

# 1) 原始尺度 sigma
sigma_raw_std = sigma_std(noisy, clean)
sigma_raw_mad = sigma_mad(noisy, clean)

# 2) 按 BM3D 方式统一归一化到 [-1,1]
#    用 clean 的 1% 和 99% 分位数作为归一化范围
pmin = np.percentile(clean, 1)
pmax = np.percentile(clean, 99)

noisy_11 = normalize_with_given_range(noisy, pmin, pmax)
clean_11 = normalize_with_given_range(clean, pmin, pmax)

sigma_11_std = sigma_std(noisy_11, clean_11)
sigma_11_mad = sigma_mad(noisy_11, clean_11)

lines = [
    "==== Sigma 结果（BM3D缩放方式）====\n",
    f"noisy_path: {noisy_path}\n",
    f"clean_path: {clean_path}\n",
    f"shape: {noisy.shape}\n",
    f"p1(clean): {pmin:.8f}\n",
    f"p99(clean): {pmax:.8f}\n",
    "\n",
    f"原始尺度 sigma (std): {sigma_raw_std:.8f}\n",
    f"原始尺度 sigma (mad): {sigma_raw_mad:.8f}\n",
    f"[-1,1] sigma (std): {sigma_11_std:.8f}\n",
    f"[-1,1] sigma (mad): {sigma_11_mad:.8f}\n",
]

for line in lines:
    print(line, end="")

out_dir = os.path.dirname(txt_path)
if out_dir != "":
    os.makedirs(out_dir, exist_ok=True)

with open(txt_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"\n结果已保存到: {os.path.abspath(txt_path)}")