import os
import numpy as np
import segyio
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter

# ===================== 参数 =====================
segy_path = "data/real/P.sgy"
save_path = "data/real/clean.npy"

dt = 0.001          # 采样间隔，单位秒。你之前很多数据是 6 ms
lowcut = 5          # 带通低截止频率 Hz
highcut = 80        # 带通高截止频率 Hz
filter_order = 4

use_smooth = True   # 是否做轻微平滑
sigma = (0.6, 0.6)  # 二维高斯平滑强度，别太大

# ===================== 读取segy =====================
with segyio.open(segy_path, "r", ignore_geometry=True) as f:
    data = segyio.tools.collect(f.trace[:])

data = np.array(data, dtype=np.float32)   # shape: (道数, 采样点数)
print("原始数据 shape:", data.shape)

# ===================== 带通滤波函数 =====================
def bandpass_filter_2d(seis, dt, lowcut, highcut, order=4):
    fs = 1.0 / dt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")

    out = np.zeros_like(seis, dtype=np.float32)
    for i in range(seis.shape[0]):   # 对每一道沿时间方向滤波
        out[i, :] = filtfilt(b, a, seis[i, :]).astype(np.float32)

    return out

# ===================== 第一步：带通滤波 =====================
pseudo_clean = bandpass_filter_2d(data, dt, lowcut, highcut, order=filter_order)

# ===================== 第二步：轻微平滑（可选） =====================
if use_smooth:
    pseudo_clean = gaussian_filter(pseudo_clean, sigma=sigma)

pseudo_clean = pseudo_clean.astype(np.float32)

# ===================== 保存 =====================
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.save(save_path, pseudo_clean)

print("伪干净数据已保存到:", save_path)
print("pseudo_clean shape:", pseudo_clean.shape)
print("min/max:", pseudo_clean.min(), pseudo_clean.max())