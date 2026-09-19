import os
import numpy as np
import segyio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


npy_path = 'data/real/noisy_norm.npy' 
data = np.load(npy_path)
section = data[:100,:].T
save_dir = 'fig_check/real_data_look'

clip = np.percentile(np.abs(section), 99)
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(section, cmap="seismic", aspect="auto", vmin=0, vmax=clip)

ax.set_xlabel("Trace")
ax.set_ylabel("Sample")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "original_50.png"), dpi=300, bbox_inches="tight")
plt.close()