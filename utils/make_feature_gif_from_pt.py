#!/usr/bin/env python3
"""
Convert a .pt file produced by dynamic_mesh_reader_cells_v2.py
into a GIF showing the time evolution of all features.

- Each time step -> one frame in the GIF
- Each frame has one subplot per feature channel

Requirements:
    pip install torch imageio matplotlib

Usage:
    python make_feature_gif_from_pt.py \
        --pt shock_ramp_amr.pt \
        --out-gif shock_ramp_features.gif
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt


def extract_fields(obj):
    """
    Handle both dict snapshots and torch_geometric.data.Data snapshots.
    Returns a simple dict with torch tensors.
    """
    # PyG Data-like (has attributes)
    if hasattr(obj, "__dict__") and hasattr(obj, "x"):
        x = obj.x
        pos = getattr(obj, "pos")
        level = getattr(obj, "level")
        t = int(getattr(obj, "t", -1))
        H = int(getattr(obj, "H", 0))
        W = int(getattr(obj, "W", 0))

    # Plain dict
    else:
        x = obj.get("x", obj.get("features"))
        pos = obj["pos"]
        level = obj["level"]
        t = int(obj.get("t", -1))
        H = int(obj.get("H", 0))
        W = int(obj.get("W", 0))

    x = torch.as_tensor(x, dtype=torch.float32)
    pos = torch.as_tensor(pos, dtype=torch.float32)
    level = torch.as_tensor(level, dtype=torch.long)

    return {"x": x, "pos": pos, "level": level, "t": t, "H": H, "W": W}


def compute_global_clims(snapshots):
    """
    Compute global (vmin, vmax) per feature channel across all snapshots.
    Returns two numpy arrays of shape [F].
    """
    # Infer number of features from first snapshot
    F = snapshots[0]["x"].shape[1]
    mins = torch.full((F,), float("inf"), dtype=torch.float32)
    maxs = torch.full((F,), float("-inf"), dtype=torch.float32)

    for snap in snapshots:
        x = snap["x"]  # [N,F]
        mins = torch.minimum(mins, x.amin(dim=0))
        maxs = torch.maximum(maxs, x.amax(dim=0))

    return mins.numpy(), maxs.numpy()


def make_gif(pt_path, out_gif, fps=5, feature_names=None, cmap="viridis"):
    # ------------------------------------------------------------------
    # 1) Load snapshots and normalize structure
    # ------------------------------------------------------------------
    print(f"[INFO] Loading snapshots from {pt_path}")
    raw_list = torch.load(pt_path, map_location="cpu")
    snapshots = [extract_fields(obj) for obj in raw_list]

    # Sort by time, just in case
    snapshots.sort(key=lambda s: s["t"])

    T = len(snapshots)
    N, F = snapshots[0]["x"].shape
    print(f"[INFO] Loaded {T} time steps, N={N} cells, F={F} features")

    # Feature names
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(F)]
    else:
        if len(feature_names) != F:
            raise ValueError(
                f"Expected {F} feature names, got {len(feature_names)}"
            )

    # ------------------------------------------------------------------
    # 2) Global color limits per feature
    # ------------------------------------------------------------------
    print("[INFO] Computing global color limits per feature...")
    vmins, vmaxs = compute_global_clims(snapshots)

    # ------------------------------------------------------------------
    # 3) Prepare GIF writer
    # ------------------------------------------------------------------
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing GIF to {out_gif} (fps={fps})")

    with imageio.get_writer(out_gif, mode="I", duration=1.0 / fps) as writer:
        # ------------------------------------------------------------------
        # 4) Generate frames
        # ------------------------------------------------------------------
        for idx, snap in enumerate(snapshots):
            x = snap["x"].numpy()          # [N,F]
            pos = snap["pos"].numpy()      # [N,2], normalized to [0,1]
            level = snap["level"].numpy()  # [N]
            t_val = snap["t"]
            H = snap["H"]
            W = snap["W"]

            # per-cell marker size: smaller for finer levels
            base_size = 10.0
            sizes = base_size / (4.0 ** level)

            fig, axes = plt.subplots(
                1, F, figsize=(4 * F, 4), sharex=True, sharey=True
            )
            if F == 1:
                axes = [axes]

            for f in range(F):
                
                ax = axes[f]
                vals = x[:, f]

                sc = ax.scatter(
                    pos[:, 0],
                    pos[:, 1],
                    c=vals,
                    s=sizes,
                    cmap=cmap,
                    vmin=vmins[f],
                    vmax=vmaxs[f],
                    marker="s",
                    linewidths=0,
                )
                ax.set_title(feature_names[f])
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                #ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])

                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                '''
                ax = axes[f]
                vals = x[:, f]

                # --------------------------------------------------------
                # Convert AMR sample into a regular H×W raster
                # --------------------------------------------------------
                grid = np.zeros((H, W), dtype=float)
                count = np.zeros((H, W), dtype=int)

                # Convert normalized pos → raster indices
                ii = np.clip((pos[:, 0] * W).astype(int), 0, W - 1)
                jj = np.clip((pos[:, 1] * H).astype(int), 0, H - 1)

                # Accumulate values (AMR may map multiple fine cells to same coarse pixel)
                np.add.at(grid, (jj, ii), vals)
                np.add.at(count, (jj, ii), 1)

                # Avoid division by zero
                mask = count > 0
                grid[mask] /= count[mask]

                # --------------------------------------------------------
                # Render grid as an image
                # --------------------------------------------------------
                im = ax.imshow(
                    grid,
                    origin="lower",
                    vmin=vmins[f], vmax=vmaxs[f],
                    cmap=cmap,
                    interpolation="nearest",  # <-- removes all white seam artifacts
                    aspect="equal"
                )

                ax.set_title(feature_names[f])
                ax.set_xticks([])
                ax.set_yticks([])

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                '''

            fig.suptitle(f"t = {t_val}   (H={H}, W={W})", fontsize=12)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Backend-agnostic: convert canvas to RGB array (handles Retina / high-DPI)
            fig.canvas.draw()
            buf = fig.canvas.tostring_argb()
            w, h = fig.canvas.get_width_height()

            buf = np.frombuffer(buf, dtype=np.uint8)
            n_pixels = buf.size // 4

            # Detect Retina / high-DPI scaling
            logical_pixels = w * h
            if logical_pixels <= 0:
                raise RuntimeError(f"Invalid canvas size: w={w}, h={h}")

            scale = int(round((n_pixels / logical_pixels) ** 0.5))
            if scale < 1:
                scale = 1

            W = w * scale
            H = h * scale

            argb = buf.reshape((H, W, 4))
            rgb = argb[:, :, 1:]  # drop alpha → RGB
            img = rgb.copy()

            #img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(img)
            plt.close(fig)

            print(f"[INFO] Frame {idx+1}/{T} written (t={t_val})")

    print("[INFO] Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to the .pt file")
    ap.add_argument("--out-gif", required=True, help="Output GIF path")
    ap.add_argument(
        "--fps", type=int, default=5, help="Frames per second for the GIF"
    )
    ap.add_argument(
        "--feature-names",
        nargs="*",
        help="Optional names for feature channels (default: feat_0, feat_1, ...)",
    )
    ap.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name (default: viridis)",
    )
    args = ap.parse_args()

    make_gif(
        pt_path=args.pt,
        out_gif=args.out_gif,
        fps=args.fps,
        feature_names=args.feature_names,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
