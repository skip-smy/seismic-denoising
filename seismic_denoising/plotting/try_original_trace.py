import os
import numpy as np
import matplotlib.pyplot as plt
from obspy import read

plt.rcParams["axes.unicode_minus"] = False

# ===================== 1. 基础配置 =====================
segy_file_path = "data/real/P.sgy"
start_trace = 0
end_trace = 100
fig_size = (12, 5)

# ===================== 2. 读取+截取数据 =====================
stream = read(segy_file_path)
trace_slice = stream[start_trace:end_trace]

n_samples = trace_slice[0].stats.npts
dt = trace_slice[0].stats.delta
time_axis = np.arange(n_samples) * dt

# ===================== 3. 提取二维剖面 =====================
section = np.array([tr.data for tr in trace_slice], dtype=np.float32).T

print("section shape:", section.shape)
print("min/max:", section.min(), section.max())

# ===================== 4. 显示范围 =====================
clip = np.percentile(np.abs(section), 99)   # 比99再稍微强一点
vmin, vmax = -clip, clip

# ===================== 5. 绘图 =====================
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=fig_size)
plt.imshow(
    np.abs(section),       # 取绝对值，只保留“亮度”
    aspect='auto',
    cmap='seismic',           # 白->红
    vmin=-clip,
    vmax=clip,
    extent=[start_trace, end_trace, time_axis[-1], 0]
)

plt.xlabel("Trace", fontsize=12)
plt.ylabel("Time (s)", fontsize=12)

# 不要 title，论文图一般标题放 caption 里
# plt.title("Seismic Section")

plt.tight_layout()
plt.savefig("fig_check/real_data_look/1_seismic.png", dpi=300, bbox_inches="tight")
plt.show()