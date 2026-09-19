import os
import numpy as np
import segyio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ========= clean =========
npy_path = 'data/test_clean.npy'   # 改成你的npy文件路径
save_dir = 'fig_check'
sigma=0.05

os.makedirs(save_dir, exist_ok=True)  
data = np.load(npy_path)
section = data[:1300,:].T
patch1 = section[820:980, 300:470]


#=============直接读取noisy==================
npy_path2 = 'data/noisy_norm_sigma0.05.npy'
data = np.load(npy_path2)
section = data[:1300,:].T
patch2 = section[820:980, 300:470]

#=============dncnn==========================
npy_path3 = 'data/test_dncnn_denoised.npy'
data = np.load(npy_path3)
section = data[:1300,:].T
patch3 = section[820:980, 300:470]



#=============ksvd==========================
npy_path4 = 'data/ksvd_denoised_norm_sigma0.05.npy'
data = np.load(npy_path4)
section = data[:1300,:].T
patch4 = section[820:980, 300:470]


#=============bm3d==========================
npy_path5 = 'data/bm3d_denoised_norm_sigma0.05.npy'
data = np.load(npy_path5)
section = data[:1300,:].T
patch5 = section[820:980, 300:470]


#===============dcdicl===================
npy_path6 = 'data/dcdicl_denoised_norm_sigma0.05.npy'
data = np.load(npy_path6)
section = data[:1300,:].T
patch6 = section[820:980, 300:470]



patch_list = [patch1, patch2, patch3, patch4, patch5,patch6]
title_list = ["patch1", "noisy_patch2", "patch3", "patch4", "patch5", "patch6"]
save_list = [
    "patch1_with_redbox.png",
    "noisy_patch2_with_redbox.png",
    "patch3_with_redbox.png",
    "patch4_with_redbox.png",
    "patch5_with_redbox.png",
    "patch6_with_redbox.png"
]

x, y, w, h = 15, 18, 27, 30
os.makedirs(os.path.join(save_dir, "denoised_patch") , exist_ok=True) 
for patch, title, save_name in zip(patch_list, title_list, save_list):
    clip = np.percentile(np.abs(patch), 99)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(patch, cmap="gray", aspect="auto", vmin=-clip, vmax=clip)

    rect = Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

   
    ax.set_xlabel("Trace")
    ax.set_ylabel("Sample")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "denoised_patch",save_name), dpi=300, bbox_inches="tight")
    plt.close()

denoised_list = [patch1, patch3, patch4, patch5, patch6]
name_list = ["clean", "dncnn", "ksvd", "bm3d","dcdicl"]
os.makedirs(os.path.join(save_dir, "denoising error") , exist_ok=True) 
for denoised_patch, name in zip(denoised_list, name_list):
    denoising_error = denoised_patch - patch1
    clip = np.percentile(np.abs(denoising_error), 99)

    plt.figure(figsize=(5, 4))
    plt.imshow(denoising_error, cmap="gray", aspect="auto", vmin=-clip, vmax=clip)

    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"denoising error", f"denoising_error_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

print("图片已保存到:", save_dir)