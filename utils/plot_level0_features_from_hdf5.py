#!/usr/bin/env python3
"""
Plot the coarse (level_0) features from a single AMR HDF5 plotfile.

This script:
  - Reads level_0/boxes and level_0/data:datatype=*
  - Reconstructs a (num_components, H, W) array on the coarse grid
  - Plots each component as a heatmap in a row of subplots

Usage:
  python plot_level0_features_from_hdf5.py \
      --infile DMR.plot.000052.2d.hdf5 \
      --out-png level0_snapshot.png

Requirements:
  pip install h5py matplotlib numpy
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def decode_attr(val, fallback):
    """Decode a bytes/np.bytes_ attribute to str, with fallback."""
    if val is None:
        return fallback
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("ascii", errors="ignore")
    return str(val)


def find_dataset_with_prefix(group, prefix):
    """Find the first dataset in group whose name starts with prefix."""
    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset) and name.startswith(prefix):
            return obj
    raise KeyError(f"No dataset starting with '{prefix}' found in group '{group.name}'")


def read_level0_features(h5_path, level=0):
    """
    Reconstruct coarse-grid features from level_<level>.

    Returns:
      feats: np.ndarray, shape (num_components, H, W)
      comp_names: list[str] of length num_components
      H, W: ints, coarse grid height and width
      time: float, simulation time from file attribute
    """
    with h5py.File(h5_path, "r") as f:
        # --- global metadata ---
        ncomp = int(f.attrs["num_components"])
        time = float(f.attrs.get("time", 0.0))
        comp_names = []
        for i in range(ncomp):
            name = decode_attr(f.attrs.get(f"component_{i}", None), f"comp_{i}")
            comp_names.append(name)

        # --- level group ---
        lev_group = f[f"level_{level}"]

        # boxes: (N_boxes,) with fields lo_i, lo_j, hi_i, hi_j
        boxes = lev_group["boxes"][()]

        # data + offsets: find datasets named like "data:datatype=0" and "data:offsets=0"
        data_ds = find_dataset_with_prefix(lev_group, "data:datatype=")
        offs_ds = find_dataset_with_prefix(lev_group, "data:offsets=")
        data = data_ds[()]      # 1D array of length n_cells * ncomp
        offsets = offs_ds[()]   # length N_boxes + 1

        # --- infer coarse grid H,W from level_0 boxes ---
        lo_i = boxes["lo_i"]
        hi_i = boxes["hi_i"]
        lo_j = boxes["lo_j"]
        hi_j = boxes["hi_j"]

        W = int(hi_i.max() - lo_i.min() + 1)
        H = int(hi_j.max() - lo_j.min() + 1)

        feats = np.zeros((ncomp, H, W), dtype=np.float64)

        # --- reconstruct per-box, fill into global grid ---
        for b_idx, box in enumerate(boxes):
            bi_lo_i = int(box["lo_i"])
            bi_lo_j = int(box["lo_j"])
            bi_hi_i = int(box["hi_i"])
            bi_hi_j = int(box["hi_j"])

            nx = bi_hi_i - bi_lo_i + 1
            ny = bi_hi_j - bi_lo_j + 1

            start = int(offsets[b_idx])
            end = int(offsets[b_idx + 1])
            block = data[start:end]

            # Each box stores ncomp * (ny * nx) values
            arr = block.reshape(ncomp, ny, nx)

            # place into the global coarse grid
            feats[:, bi_lo_j : bi_hi_j + 1, bi_lo_i : bi_hi_i + 1] = arr

    return feats, comp_names, H, W, time


def plot_features(feats, comp_names, H, W, time, out_png=None, show=False):
    """
    Plot (num_components, H, W) heatmaps in a row.
    """
    ncomp = feats.shape[0]
    fig, axes = plt.subplots(
        1, ncomp, figsize=(4 * ncomp, 4), squeeze=False
    )
    axes = axes[0]

    vmins = feats.reshape(ncomp, -1).min(axis=1)
    vmaxs = feats.reshape(ncomp, -1).max(axis=1)

    for i in range(ncomp):
        ax = axes[i]
        im = ax.imshow(
            feats[i],
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            vmin=vmins[i],
            vmax=vmaxs[i],
            cmap="viridis",
        )
        ax.set_title(comp_names[i])
        ax.set_xlabel("i")
        if i == 0:
            ax.set_ylabel("j")
        else:
            ax.set_yticks([])

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"{Path(out_png).name if out_png else ''}\n"
        f"level_0 snapshot, time = {time:.6e}, H={H}, W={W}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])

    if out_png is not None:
        fig.savefig(out_png, dpi=200)
        print(f"[INFO] Saved figure to {out_png}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infile", type=str, required=True, help="Path to DMR.plot.*.2d.hdf5"
    )
    ap.add_argument(
        "--out-png",
        type=str,
        default="level0_snapshot.png",
        help="Output PNG path",
    )
    ap.add_argument(
        "--level",
        type=int,
        default=0,
        help="AMR level to plot (default: 0 = coarsest)",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively",
    )
    args = ap.parse_args()

    feats, names, H, W, time = read_level0_features(args.infile, level=args.level)
    print(
        f"[INFO] Read {args.infile}: level={args.level}, "
        f"num_components={feats.shape[0]}, H={H}, W={W}, time={time}"
    )
    print("[INFO] Components:", ", ".join(names))

    plot_features(feats, names, H, W, time, out_png=args.out_png, show=args.show)


if __name__ == "__main__":
    main()
