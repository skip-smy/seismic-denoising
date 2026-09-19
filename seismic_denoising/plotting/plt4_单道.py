import os
import numpy as np
import matplotlib.pyplot as plt


def load_section_from_npy(npy_path):
    """
    读取 npy，并转置成真正的地震剖面格式:
    [samples, traces]
    """
    data = np.load(npy_path).astype(np.float32)
    return data.T


def plot_trace_amplitude_only(
    clean_path,
    noisy_path,
    bm3d_path,
    ksvd_path,
    dcdicl_path,
    trace_idx=50,
    dt_ms=6,
    save_path=None
):
    # 1. 读取并转置
    clean = load_section_from_npy(clean_path)
    noisy = load_section_from_npy(noisy_path)
    bm3d  = load_section_from_npy(bm3d_path)
    ksvd  = load_section_from_npy(ksvd_path)
    dncnn = load_section_from_npy(dncnn_path)
    dcdicl = load_section_from_npy(dcdicl_path)

    # 2. 检查尺寸
    if not (clean.shape == noisy.shape == bm3d.shape == ksvd.shape):
        raise ValueError(
            f"四个数据尺寸不一致:\n"
            f"clean={clean.shape}, noisy={noisy.shape}, bm3d={bm3d.shape}, ksvd={ksvd.shape}"
        )

    n_samples, n_traces = clean.shape
    if not (0 <= trace_idx < n_traces):
        raise ValueError(f"trace_idx={trace_idx} 超出范围，应在 [0, {n_traces-1}] 内")

    # 3. 时间轴（ms）
    time_axis = np.arange(n_samples) * dt_ms

    # 4. 取同一道
    clean_trace = clean[:, trace_idx]
    noisy_trace = noisy[:, trace_idx]
    # bm3d_trace  = bm3d[:, trace_idx]
    # ksvd_trace  = ksvd[:, trace_idx]
    # dncnn_trace  = dncnn[:, trace_idx]
    dcdicl_trace = dcdicl[:, trace_idx]

    # 5. 画图
    plt.figure(figsize=(6,2.8))
    plt.plot(time_axis, clean_trace, label="Clean", linewidth=1)
    plt.plot(time_axis, noisy_trace, label="Noisy", linewidth=1)
    # plt.plot(time_axis, bm3d_trace,  label="BM3D", linewidth=1)
    # plt.plot(time_axis, ksvd_trace,  label="K-SVD", linewidth=1)
    # plt.plot(time_axis, dncnn_trace,  label="DnCNN", linewidth=1)
    plt.plot(time_axis, dcdicl_trace,  label="DCDicl", linewidth=1)

    plt.xlim(3000, 3400)
    plt.xticks(np.arange(3000,3401,50))
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)
    plt.yticks(np.arange(-1,1,0.5))
    plt.legend(
)
    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



if __name__ == "__main__":
    clean_path = "data/test_clean.npy"
    noisy_path = "data/noisy_norm_sigma0.05.npy"
    bm3d_path  = "data/bm3d_denoised_norm_sigma0.05.npy"
    ksvd_path  = "data/ksvd_denoised_norm_sigma0.05.npy"
    dncnn_path = "data/test_dncnn_denoised.npy"
    dcdicl_path = "data/dcdicl_denoised_norm_sigma0.05.npy"

    plot_trace_amplitude_only(
        clean_path=clean_path,
        noisy_path=noisy_path,
        bm3d_path=bm3d_path,
        ksvd_path=ksvd_path,
        dcdicl_path=dcdicl_path,
        trace_idx=50,                 # 改成你想看的道号
        dt_ms=6,                      # 如果知道采样间隔，比如 0.002 或 0.006，就填进去
        save_path="fig_check/single_channel/trace_50_dcdicl.png"
    )