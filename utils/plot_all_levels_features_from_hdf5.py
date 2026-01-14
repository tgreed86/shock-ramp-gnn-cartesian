#!/usr/bin/env python3
"""
Plot composite AMR features from a single AMReX HDF5 plotfile.

For each level_L group in the file, we:
  - Read level_L/boxes and level_L/data:datatype=*
  - Reconstruct component fields on that level's index grid
  - Upsample every level to the finest resolution grid (Lmax) and overlay:
        * coarser levels filled first
        * finer levels overwrite coarser

Result:
  - One composite array per feature on the finest grid
  - Single figure with one subplot per feature (density, x-momentum, etc.)

Usage:
  python plot_composite_features_from_hdf5.py \
      --infile DMR.plot.000052.2d.hdf5 \
      --out-png DMR_000052_composite.png

Requirements:
  pip install h5py matplotlib numpy
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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
    raise KeyError(
        f"No dataset starting with '{prefix}' found in group '{group.name}'"
    )


# ---------------------------------------------------------------------
# Core reading / composite logic
# ---------------------------------------------------------------------
def build_composite(h5_path):
    """
    Build a composite AMR solution on the finest grid.

    Steps:
      - Read all level_* groups.
      - Use level_L/attrs['prob_domain'] to get domain extents at each level:
            prob_domain = (lo_i, lo_j, hi_i, hi_j)
      - Finest grid size:
            W_fine = hi_i_Lmax - lo_i_Lmax + 1
            H_fine = hi_j_Lmax - lo_j_Lmax + 1
      - For each level L:
            * Compute refinement factor s_L = W_fine / W_L = H_fine / H_L
            * Reconstruct the level array on its own grid
            * Upsample by factor s_L in i,j using nearest neighbor
            * Paste into composite; finer levels overwrite coarser.

    Returns:
      composite: np.ndarray, shape (ncomp, H_fine, W_fine)
      comp_names: list[str]
      time: float
      Lmax: int
      H_fine, W_fine: ints
    """
    with h5py.File(h5_path, "r") as f:
        # ---- global metadata ----
        ncomp = int(f.attrs["num_components"])
        time = float(f.attrs.get("time", 0.0))
        comp_names = []
        for i in range(ncomp):
            name = decode_attr(f.attrs.get(f"component_{i}", None), f"comp_{i}")
            comp_names.append(name)

        # ---- find level groups ----
        level_groups = []
        for name, obj in f.items():
            if isinstance(obj, h5py.Group) and name.startswith("level_"):
                try:
                    L = int(name.split("_")[1])
                except ValueError:
                    continue
                level_groups.append((L, obj))

        if not level_groups:
            raise RuntimeError("No 'level_*' groups found in file")

        level_groups.sort(key=lambda x: x[0])
        levels = [L for (L, _) in level_groups]
        Lmax = max(levels)

        # ---- prob_domain per level ----
        prob_domains = {}
        for L, g in level_groups:
            prob = g.attrs["prob_domain"]  # (lo_i, lo_j, hi_i, hi_j)
            prob_domains[L] = prob
            print(f"[INFO] level_{L} prob_domain={prob}")

        # Finest grid size from level Lmax
        lo_i_f, lo_j_f, hi_i_f, hi_j_f = prob_domains[Lmax]
        W_fine = hi_i_f - lo_i_f + 1
        H_fine = hi_j_f - lo_j_f + 1
        print(f"[INFO] Finest grid: H_fine={H_fine}, W_fine={W_fine}")
        print(f"[INFO] Levels present: {levels}, Lmax={Lmax}")

        # Composite array, start filled with NaNs
        composite = np.full((ncomp, H_fine, W_fine), np.nan, dtype=np.float64)

        # ---- compute refinement factor s_L for each level ----
        factors = {}
        for L in levels:
            lo_i, lo_j, hi_i, hi_j = prob_domains[L]
            W_L = hi_i - lo_i + 1
            H_L = hi_j - lo_j + 1
            fi = W_fine // W_L
            fj = H_fine // H_L
            if fi != fj:
                raise RuntimeError(
                    f"Non-isotropic refinement between level {L} and Lmax: fi={fi}, fj={fj}"
                )
            s = fi
            factors[L] = s
            print(f"[INFO] level_{L}: H={H_L}, W={W_L}, refinement factor s={s}")

        # ---- process levels in ascending order, so finer overwrites ----
        for L, g in level_groups:
            s = factors[L]
            print(f"[INFO] Processing level_{L} (upsample factor s={s})")

            boxes = g["boxes"][()]
            if boxes.size == 0:
                print(f"  [WARN] level_{L} has no boxes; skipping")
                continue

            data_ds = find_dataset_with_prefix(g, "data:datatype=")
            offs_ds = find_dataset_with_prefix(g, "data:offsets=")
            data = data_ds[()]      # 1D array of floats
            offsets = offs_ds[()]   # shape (n_boxes+1,)

            lo_iL, lo_jL, hi_iL, hi_jL = prob_domains[L]

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

                if block.size != ncomp * ny * nx:
                    raise RuntimeError(
                        f"Unexpected block size at level {L}, box {b_idx}: "
                        f"got {block.size}, expected {ncomp * ny * nx}"
                    )

                arr = block.reshape(ncomp, ny, nx)

                # indices relative to this level's domain
                ci0 = bi_lo_i - lo_iL
                cj0 = bi_lo_j - lo_jL

                # map to finest-grid indices
                fi0 = ci0 * s
                fj0 = cj0 * s
                fi1 = (ci0 + nx) * s
                fj1 = (cj0 + ny) * s

                # upsample block -> (ncomp, ny*s, nx*s)
                block_up = np.repeat(
                    np.repeat(arr, s, axis=1), s, axis=2
                )

                # sanity checks
                if block_up.shape[1] != (fj1 - fj0) or block_up.shape[2] != (fi1 - fi0):
                    raise RuntimeError(
                        f"Upsampled block shape mismatch at level {L}, box {b_idx}: "
                        f"block_up={block_up.shape}, "
                        f"slice=({fj1 - fj0}, {fi1 - fi0})"
                    )

                composite[:, fj0:fj1, fi0:fi1] = block_up

        return composite, comp_names, time, Lmax, H_fine, W_fine


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_composite(composite, comp_names, time, out_png=None, show=False):
    """
    Plot one composite field per feature (one subplot per component).
    """
    ncomp, H, W = composite.shape

    # global vmin/vmax per component (ignore NaNs)
    vmins = np.nanmin(composite.reshape(ncomp, -1), axis=1)
    vmaxs = np.nanmax(composite.reshape(ncomp, -1), axis=1)

    fig, axes = plt.subplots(1, ncomp, figsize=(4 * ncomp, 4), squeeze=False)
    axes = axes[0]

    for i in range(ncomp):
        ax = axes[i]
        img = composite[i]

        # mask NaNs (if any)
        img_masked = np.ma.masked_invalid(img)

        im = ax.imshow(
            img_masked,
            origin="lower",
            interpolation="nearest",
            aspect="equal",  # keep physical aspect ratio
            vmin=vmins[i],
            vmax=vmaxs[i],
            cmap="viridis",
        )

        ax.set_title(comp_names[i])
        ax.set_xlabel("fine i")
        if i == 0:
            ax.set_ylabel("fine j")
        else:
            ax.set_yticks([])

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Composite AMR solution on finest grid\n"
        f"time = {time:.6e}, H={H}, W={W}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])

    if out_png is not None:
        fig.savefig(out_png, dpi=200)
        print(f"[INFO] Saved figure to {out_png}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infile", type=str, required=True, help="Path to DMR.plot.*.2d.hdf5"
    )
    ap.add_argument(
        "--out-png",
        type=str,
        default="composite_snapshot.png",
        help="Output PNG path",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively",
    )
    args = ap.parse_args()

    composite, comp_names, time, Lmax, H_fine, W_fine = build_composite(args.infile)

    print(
        f"[INFO] File {args.infile}: "
        f"Lmax={Lmax}, components={', '.join(comp_names)}, "
        f"composite shape={composite.shape}"
    )

    plot_composite(composite, comp_names, time, out_png=args.out_png, show=args.show)


if __name__ == "__main__":
    main()
