import torch.nn as nn
import torch.nn.init as init
import segyio
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# ========== 1. 配置区域（请根据你的实际情况修改！） ==========
# 文件路径
# ORIGINAL_SEGY = "./data/Train/train.segy"  # 你的原始SEGY文件
# MODEL_WEIGHT_PATH = "./models/model_056.pth"  # 训练好的模型权重
# OUTPUT_FOLDER = "./results/single_channel_comparison"  # 结果保存文件夹

ORIGINAL_SEGY = "/root/autodl-tmp/3.8DNCNN/data/Train/train.segy"
MODEL_WEIGHT_PATH = "/root/autodl-tmp/3.8DNCNN/models/DnCNN_sigma0.2/model_040.pth"
OUTPUT_FOLDER = "/root/autodl-tmp/3.8DNCNN/results/single_channel_comparison"

#
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文显示兼容
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']#linux系统兼容
plt.rcParams["axes.unicode_minus"] = False    # 负号显示兼容
# 道号和参数
TRACE_IDX = 100000  # 你要画的道号（从0开始计数！）
SIGMA = 0.2  # 训练时的噪声水平（和train.py里的sigma一致！）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========== 2. 辅助函数：严格复现训练时的逻辑 ==========
# 注意：这里的归一化、加噪必须和你 train.py/datagenerator.py 里的完全一致！

def load_single_trace(segy_path, trace_idx):
    """从SEGY提取指定道的原始数据"""
    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        total_traces = f.tracecount
        if trace_idx < 0 or trace_idx >= total_traces:
            raise ValueError(f"道号{trace_idx}不合法！总道数范围：0 ~ {total_traces - 1}")
        trace_data = f.trace[trace_idx].astype(np.float32)  # 原始地震道
    return trace_data, total_traces


def normalize_trace(trace_data, pmin=None, pmax=None):
    """
    归一化：和datagenerator里的逻辑一致（用1%和99%百分位数）
    如果没有传入pmin/pmax，就用当前道的统计量
    """
    if pmin is None or pmax is None:
        pmin = np.percentile(trace_data, 1)
        pmax = np.percentile(trace_data, 99)
    normalized = 2*(trace_data - pmin) / (pmax - pmin) - 1  # 归一到[0,1]
    return normalized, pmin, pmax





def add_noise(trace_tensor, sigma):
    """加噪：和DenoisingDataset里的逻辑完全一致"""
    noise = torch.randn_like(trace_tensor) * sigma # 注意：如果训练时sigma是除以255的，这里要一致！
    # （如果训练时没除以255，就直接用 sigma）
    noisy_tensor = trace_tensor + noise
    return noisy_tensor, noise


# ========== 3. 导入你的模型定义（必须和训练时完全一致！） ==========
# 示例：假设你的模型在 model.py 里，类名叫 DnCNN
# from model import DnCNN  # 取消注释，替换为你的实际模型导入
class HardSwish(nn.Module):
    def forward(self,x):
        return x*F.relu6(x+3.0)/6.0
# 这里为了演示，放一个简单的DnCNN示例（你要替换成自己的真实模型！）
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        self.hard_swish = HardSwish()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(self.hard_swish)
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(self.hard_swish)
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)



# ========== 4. 核心流程：提取 → 加噪 → 去噪 → 画图 ==========
def main():
    # 自动创建输出文件夹
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ---------- 4.1 加载原始地震道 ----------
    print(f"正在加载SEGY文件，提取第{TRACE_IDX}道...")
    trace_orig, total_traces = load_single_trace(ORIGINAL_SEGY, TRACE_IDX)
    print(f"成功提取！总道数：{total_traces}，该道采样点数：{len(trace_orig)}")

    # ---------- 4.2 归一化（和训练时一致） ----------
    trace_norm, pmin, pmax = normalize_trace(trace_orig)
    print(f"归一化完成：pmin={pmin:.4f}, pmax={pmax:.4f}")

    # ---------- 4.3 转Tensor，调整维度（适配模型输入） ----------
    # 注意：你的模型是处理2D patch的，单道是1D，我们把它转成 [1, 1, 1, n_samples]（batch, channel, height, width）
    # 或者如果你的模型接受1D输入，调整为 [1, 1, n_samples]
    trace_tensor = torch.FloatTensor(trace_norm).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,n_samples]
    trace_tensor = trace_tensor.to(DEVICE)

    # ---------- 4.4 加噪（和训练时一致） ----------
    print(f"正在加噪，sigma={SIGMA}...")
    noisy_tensor, noise_tensor = add_noise(trace_tensor, SIGMA)

    # ---------- 4.5 加载模型并推理 ----------
    print(f"正在加载模型权重：{MODEL_WEIGHT_PATH}...")
    # 初始化模型（替换成你的真实模型！）
    model = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    # 推理去噪（不计算梯度）
    print("正在模型推理去噪...")
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)

    # ---------- 4.6 转numpy，反归一化 ----------
    # 去掉batch/channel维度，转numpy
    trace_norm_np = trace_tensor.squeeze().cpu().numpy()
    noisy_norm_np = noisy_tensor.squeeze().cpu().numpy()
    denoised_norm_np = denoised_tensor.squeeze().cpu().numpy()



    # ---------- 4.7 保存三类数据（可选） ----------
    np.save(os.path.join(OUTPUT_FOLDER, f"trace_{TRACE_IDX}_original.npy"), trace_norm_np)
    np.save(os.path.join(OUTPUT_FOLDER, f"trace_{TRACE_IDX}_noisy.npy"), noisy_norm_np)
    np.save(os.path.join(OUTPUT_FOLDER, f"trace_{TRACE_IDX}_denoised.npy"), denoised_norm_np)
    print(f"三类数据已保存到：{OUTPUT_FOLDER}")

    # ---------- 4.8 画对比图（修改版：三曲线同图 + 局部放大600-650） ----------
    print("正在生成对比图...")

    # 定义要显示的范围（600-650，包含两端）
    start_idx =120
    end_idx =350  # 切片左闭右开，所以取到651

    # 只取这个范围内的数据
    x_axis = np.arange(start_idx, end_idx)
    orig_slice = trace_norm_np[start_idx:end_idx]
    noisy_slice = noisy_norm_np[start_idx:end_idx]
    denoised_slice = denoised_norm_np[start_idx:end_idx]

    # 创建单图
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    # 画三条曲线，加上标签
    ax.plot(x_axis, orig_slice, color='red', linewidth=1.5, label='Original (原始数据)')
    ax.plot(x_axis, noisy_slice, color='green', linewidth=1.0, alpha=0.7, label=f'Noisy (加噪, σ={SIGMA})')
    ax.plot(x_axis, denoised_slice, color='blue', linewidth=1.5, label='Denoised (HS-DnCNN去噪)')

    # 设置标题和标签
    ax.set_title(f"地震道去噪效果对比 (Trace {TRACE_IDX}, Sample {start_idx}-{end_idx})", fontsize=14,
                 fontweight='bold')
    ax.set_xlabel("Sample Index (采样点)", fontsize=12)
    ax.set_ylabel("Amplitude (振幅)", fontsize=12)

    # 设置x轴范围，确保正好是600-650
    ax.set_xlim(start_idx, end_idx - 1)

    # 显示网格和图例
    ax.grid(alpha=0.4, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')

    # 调整布局，保存图片
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, f"trace_{TRACE_IDX}_HS-DnCNN_comparison_{start_idx}-{end_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"HS-DnCNN对比图已保存到：{save_path}")

    # 显示图片
    plt.show()

if __name__ == "__main__":
    main()