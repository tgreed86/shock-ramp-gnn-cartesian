#!/usr/bin/env python3
"""
Plot AMR features from a single Chombo/AMReX plotfile using its companion
map file to recover physical (x,y) coordinates.

For a given time step (e.g. DMR.plot.000052.2d.hdf5 and
DMR.plot.000052.2d.map.hdf5), this script:

  * Reads all AMR levels from the plot file (cell-centered data).
  * Reads the matching NodeFArrayBox data from the map file (nodal X,Y).
  * For every cell on every level, computes the physical cell-center
    coordinates by averaging the four surrounding map nodes.
  * Produces a single figure with one subplot per feature
    (density, x-momentum, y-momentum, energy-density, ...).
  * All AMR levels are overlaid in physical space; finer levels are drawn
    with smaller markers and on top of coarser levels, so refinement
    overlays the background coarse solution.

Usage example
-------------
    python plot_composite_from_plot_and_map.py \
        --plot-file DMR.plot.000052.2d.hdf5 \
        --map-file  DMR.plot.000052.2d.map.hdf5 \
        --out-png   DMR_000052_composite_phys.png

"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------


def _decode_attr(val, fallback):
    """Decode a bytes/np.bytes_ attribute to str, with a fallback name."""
    if val is None:
        return fallback
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("ascii", errors="ignore")
    return str(val)


def _find_dataset_with_prefix(group, prefix):
    """Find the first dataset in an HDF5 group whose name starts with prefix."""
    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset) and name.startswith(prefix):
            return obj
    raise KeyError(
        f"No dataset starting with '{prefix}' found in group '{group.name}'"
    )


# ---------------------------------------------------------------------
# Core: read plot+map and build per-cell physical coordinates
# ---------------------------------------------------------------------


def read_cells_with_physical_coords(plot_path, map_path):
    """
    Read a single plotfile + mapfile pair and return:

      xs, ys          : 1D arrays of length Ncells with physical x,y
      levels          : 1D int array of level index for each cell
      feats           : 2D array, shape (ncomp, Ncells) with feature values
      component_names : list[str] of length ncomp
      time            : simulation time (from global attrs)
    """
    with h5py.File(plot_path, "r") as fp, h5py.File(map_path, "r") as fm:
        # ----- global metadata -----
        ncomp = int(fp.attrs["num_components"])
        time = float(fp.attrs.get("time", 0.0))

        component_names = []
        for i in range(ncomp):
            raw = fp.attrs.get(f"component_{i}", None)
            name = _decode_attr(raw, f"comp_{i}")
            component_names.append(name)

        # ----- discover levels -----
        level_ids = []
        for name, obj in fp.items():
            if isinstance(obj, h5py.Group) and name.startswith("level_"):
                try:
                    L = int(name.split("_")[1])
                except ValueError:
                    continue
                level_ids.append(L)
        level_ids = sorted(level_ids)
        if not level_ids:
            raise RuntimeError("No 'level_*' groups found in plot file")

        print(f"[INFO] Components: {component_names}")
        print(f"[INFO] Levels present: {level_ids}")

        xs_chunks = []
        ys_chunks = []
        level_chunks = []
        feat_chunks = [[] for _ in range(ncomp)]

        for L in level_ids:
            gP = fp[f"level_{L}"]
            gM = fm[f"level_{L}"]

            boxes = gP["boxes"][()]
            boxes_map = gM["boxes"][()]
            if not (boxes.shape == boxes_map.shape and np.array_equal(boxes, boxes_map)):
                raise RuntimeError(f"Plot/map boxes mismatch at level {L}")

            data = gP["data:datatype=0"][()]
            offs = gP["data:offsets=0"][()]
            data_m = gM["data:datatype=0"][()]
            offs_m = gM["data:offsets=0"][()]

            comps_plot = int(gP["data_attributes"].attrs["comps"])
            comps_map = int(gM["data_attributes"].attrs["comps"])

            if comps_plot != ncomp:
                raise RuntimeError(
                    f"Level {L}: plot comps={comps_plot} != global ncomp={ncomp}"
                )
            if comps_map != 2:
                raise RuntimeError(
                    f"Level {L}: map comps={comps_map} (expected 2 for (x,y) nodes)"
                )

            n_boxes = boxes.shape[0]
            print(
                f"[INFO] Level {L}: boxes={n_boxes}, "
                f"plot_data_len={len(data)}, map_data_len={len(data_m)}"
            )

            for b_idx, box in enumerate(boxes):
                lo_i = int(box["lo_i"])
                lo_j = int(box["lo_j"])
                hi_i = int(box["hi_i"])
                hi_j = int(box["hi_j"])

                nx = hi_i - lo_i + 1  # number of cells in i
                ny = hi_j - lo_j + 1  # number of cells in j

                # ----- cell-centered solution values -----
                start = int(offs[b_idx])
                end = int(offs[b_idx + 1])
                block = data[start:end]
                expected_cells = ncomp * ny * nx
                if block.size != expected_cells:
                    raise RuntimeError(
                        f"Level {L}, box {b_idx}: plot block size {block.size} "
                        f"!= {expected_cells} = ncomp*ny*nx"
                    )
                vals = block.reshape(ncomp, ny, nx)  # (ncomp, ny, nx)

                # ----- nodal X,Y from map (NodeFArrayBox) -----
                start_m = int(offs_m[b_idx])
                end_m = int(offs_m[b_idx + 1])
                block_m = data_m[start_m:end_m]

                nx_nodes = nx + 1
                ny_nodes = ny + 1
                expected_nodes = 2 * ny_nodes * nx_nodes
                if block_m.size != expected_nodes:
                    raise RuntimeError(
                        f"Level {L}, box {b_idx}: map block size {block_m.size} "
                        f"!= {expected_nodes} = 2*(ny+1)*(nx+1)"
                    )

                nodes = block_m.reshape(2, ny_nodes, nx_nodes)  # (2, ny+1, nx+1)
                X = nodes[0]
                Y = nodes[1]

                # ----- cell centers from 4 surrounding nodes -----
                # shape: (ny, nx)
                x_center = 0.25 * (
                    X[0:ny, 0:nx]
                    + X[0:ny, 1 : nx + 1]
                    + X[1 : ny + 1, 0:nx]
                    + X[1 : ny + 1, 1 : nx + 1]
                )
                y_center = 0.25 * (
                    Y[0:ny, 0:nx]
                    + Y[0:ny, 1 : nx + 1]
                    + Y[1 : ny + 1, 0:nx]
                    + Y[1 : ny + 1, 1 : nx + 1]
                )

                x_flat = x_center.ravel()
                y_flat = y_center.ravel()

                xs_chunks.append(x_flat)
                ys_chunks.append(y_flat)
                level_chunks.append(np.full_like(x_flat, L, dtype=np.int32))

                for k in range(ncomp):
                    feat_chunks[k].append(vals[k].ravel())

        xs = np.concatenate(xs_chunks)
        ys = np.concatenate(ys_chunks)
        levels = np.concatenate(level_chunks)
        feats = np.vstack([np.concatenate(ch) for ch in feat_chunks])

        print(f"[INFO] Total cells across all levels: {xs.shape[0]}")
        return xs, ys, levels, feats, component_names, time


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_composite_scatter(
    xs,
    ys,
    levels,
    feats,
    component_names,
    time,
    out_png=None,
    show=False,
):
    """
    Make a single figure with one subplot per feature.

    All levels are drawn in physical (x,y) space. Coarser levels use larger
    square markers, finer levels use smaller markers and are drawn on top,
    so refinement visually overlays the coarse background.
    """
    ncomp, N = feats.shape
    uniq_levels = np.unique(levels)

    # global vmin/vmax per component
    vmins = feats.min(axis=1)
    vmaxs = feats.max(axis=1)

    fig, axes = plt.subplots(1, ncomp, figsize=(4 * ncomp, 4), squeeze=False)
    axes = axes[0]

    # marker sizes per level (points^2): level 0 largest, then shrink
    base_size = 6.0
    size_per_level = {L: base_size / (2 ** L) for L in uniq_levels}

    for k in range(ncomp):
        ax = axes[k]

        # draw levels in ascending order so finer levels naturally overwrite
        for L in sorted(uniq_levels):
            mask = levels == L
            if not np.any(mask):
                continue

            s = size_per_level[L]
            sc = ax.scatter(
                xs[mask],
                ys[mask],
                c=feats[k, mask],
                s=s,
                cmap="viridis",
                vmin=vmins[k],
                vmax=vmaxs[k],
                linewidths=0,
                marker="s",
            )

        ax.set_title(component_names[k])
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x (physical)")
        if k == 0:
            ax.set_ylabel("y (physical)")
        else:
            ax.set_yticks([])

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Composite AMR solution in physical space\n"
        f"time = {time:.6e}",
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
        "--plot-file",
        type=str,
        required=True,
        help="Path to DMR.plot.XXXX.2d.hdf5 (cell-centered data)",
    )
    ap.add_argument(
        "--map-file",
        type=str,
        required=True,
        help="Path to DMR.plot.XXXX.2d.map.hdf5 (nodal X,Y map)",
    )
    ap.add_argument(
        "--out-png",
        type=str,
        default="composite_phys.png",
        help="Output PNG file for the composite plot.",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively.",
    )
    args = ap.parse_args()

    xs, ys, levels, feats, comp_names, time = read_cells_with_physical_coords(
        args.plot_file, args.map_file
    )

    print(
        f"[INFO] Read {args.plot_file} + {args.map_file}: "
        f"{len(comp_names)} components, {xs.shape[0]} cells."
    )

    plot_composite_scatter(
        xs,
        ys,
        levels,
        feats,
        comp_names,
        time,
        out_png=args.out_png,
        show=args.show,
    )


if __name__ == "__main__":
    main()
