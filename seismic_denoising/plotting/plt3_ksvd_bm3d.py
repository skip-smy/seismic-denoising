import math
import numpy as np
import matplotlib.pyplot as plt

def plot_dictionary_atoms_with_grid(D, atom_h=None, atom_w=None, n_cols=None,normalize_each=True, save_path=None, dpi=300):
    D = np.asarray(D)

    if D.shape[0] < D.shape[1]:
        n_atoms, patch_dim = D.shape
        atoms = D
    else:
        patch_dim, n_atoms = D.shape
        atoms = D.T

    if atom_h is None or atom_w is None:
        side = int(round(math.sqrt(atoms.shape[1])))
        if side * side != atoms.shape[1]:
            raise ValueError("patch_dim 不是完全平方数，请手动指定 atom_h 和 atom_w")
        atom_h, atom_w = side, side

    if n_cols is None:
        n_cols = int(math.ceil(math.sqrt(n_atoms)))
    n_rows = int(math.ceil(n_atoms / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.9, n_rows * 0.9))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("blue")

        if i < n_atoms:
            atom = atoms[i].reshape(atom_h, atom_w)

            if normalize_each:
                vmax = np.max(np.abs(atom))
                if vmax > 0:
                    atom = atom / vmax

            ax.imshow(atom, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
        else:
            ax.axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    plt.show()


if __name__ == "__main__":
    D = np.load("data/ksvd_dictionary_sigma0.05.npy")
    plot_dictionary_atoms_with_grid(D, save_path="fig_check/dict.png")