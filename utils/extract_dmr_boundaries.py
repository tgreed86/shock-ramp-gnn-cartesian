#!/usr/bin/env python3
"""
extract_dmr_boundaries.py

Extract physical-space boundary curves from a Chombo
DMR.plot.XXXXX.2d.map.hdf5 file.

The map file stores a NodeFArrayBox with 2 components (x,y) on each AMR level.
This script reconstructs the global node coordinate arrays for a given level
and then pulls out:

  - bottom boundary (j = j_min, left -> right)
  - top boundary    (j = j_max, left -> right)
  - left boundary   (i = i_min, bottom -> top)
  - right boundary  (i = i_max, bottom -> top)
  - a closed CCW polygon around the outer domain

Output is written to a .npz file that you can reuse for all time steps.
"""

import argparse, os
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import torch



def save_wedge_mesh_spec_pt(
    out_path: str,
    boundaries: dict,
    n0: int = 64,
    max_level: int = 3,
):
    """
    Save all information needed to rebuild the AMR mesh in another project.

    We save:
      - outer polygon (and optionally the separate edges) as float32 tensors
      - n0 (base grid size) and max_level as ints

    In another project you can:
      1) load this .pt file,
      2) reconstruct the boundaries dict,
      3) call build_amr_mesh_for_wedge(boundaries, n0, max_level).
    """
    payload = {
        "outer_polygon_ccw": torch.from_numpy(boundaries["outer_polygon_ccw"]).float(),
        "bottom": torch.from_numpy(boundaries["bottom"]).float(),
        "top": torch.from_numpy(boundaries["top"]).float(),
        "left": torch.from_numpy(boundaries["left"]).float(),
        "right": torch.from_numpy(boundaries["right"]).float(),
        "n0": int(n0),
        "max_level": int(max_level),
    }
    torch.save(payload, out_path)
    print(f"[INFO] Saved wedge mesh spec to '{out_path}'.")


def plot_boundaries(
    boundaries: dict,
    show_polygon: bool = True,
    show_axes: bool = True,
    equal_aspect: bool = True,
    save_path: str | None = None,
):
    """
    Plot physical-space boundaries.

    Parameters
    ----------
    boundaries : dict
        Output of `extract_boundaries`, expected keys:
        'bottom', 'top', 'left', 'right', and optionally 'outer_polygon_ccw'.
    show_polygon : bool
        If True, also draw the outer CCW polygon as a single closed curve.
    show_axes : bool
        If False, hides axes for a cleaner boundary-only plot.
    equal_aspect : bool
        If True, sets axis aspect to 'equal' so geometry isn't distorted.
    save_path : str or None
        If provided, saves the figure to this path instead of (or in addition to)
        just showing it.
    """
    bottom = boundaries.get("bottom")
    top = boundaries.get("top")
    left = boundaries.get("left")
    right = boundaries.get("right")
    poly = boundaries.get("outer_polygon_ccw", None)

    fig, ax = plt.subplots()

    if bottom is not None:
        ax.plot(bottom[:, 0], bottom[:, 1], label="bottom")
    if top is not None:
        ax.plot(top[:, 0], top[:, 1], label="top")
    if left is not None:
        ax.plot(left[:, 0], left[:, 1], label="left")
    if right is not None:
        ax.plot(right[:, 0], right[:, 1], label="right")

    if show_polygon and poly is not None:
        # Close polygon if not already closed
        if not np.allclose(poly[0], poly[-1]):
            poly = np.vstack([poly, poly[0]])
        ax.plot(poly[:, 0], poly[:, 1], linestyle="--", linewidth=1.0, label="outer_polygon")

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x (physical)")
    ax.set_ylabel("y (physical)")
    ax.legend()

    if not show_axes:
        ax.axis("off")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()


def build_level_node_coords(map_path: str, level: str = "level_0"):
    """
    Build global node (x,y) arrays for a given AMR level using the same
    conventions as dynamic_mesh_reader_physical_space.py.

    Parameters
    ----------
    map_path : str
        Path to DMR.plot.XXXXX.2d.map.hdf5.
    level : str
        Level group name, e.g. 'level_0'.

    Returns
    -------
    X, Y : 2D np.ndarray
        Node coordinates with shape (Nj_nodes, Ni_nodes) and indexing [j, i].
    meta : dict
        {'i_min','i_max','j_min','j_max'} in *cell* index space for this level.
    """
    with h5py.File(map_path, "r") as fm:
        if level not in fm:
            raise KeyError(f"Level group '{level}' not found in '{map_path}'.")

        gM = fm[level]

        # Global domain extents in index space (same thing you use in the reader)
        if "prob_domain" not in gM.attrs:
            raise RuntimeError(f"{level} missing 'prob_domain' attribute")
        lo_iL, lo_jL, hi_iL, hi_jL = [int(v) for v in gM.attrs["prob_domain"]]
        W_L = hi_iL - lo_iL + 1  # cells in i
        H_L = hi_jL - lo_jL + 1  # cells in j

        # Global node grid is (cells + 1) in each direction
        ni_nodes = W_L + 1
        nj_nodes = H_L + 1

        X = np.full((nj_nodes, ni_nodes), np.nan, dtype=float)
        Y = np.full((nj_nodes, ni_nodes), np.nan, dtype=float)

        boxes = gM["boxes"][()]
        data_m = gM["data:datatype=0"][()]
        offs_m = gM["data:offsets=0"][()]

        comps_map = int(gM["data_attributes"].attrs["comps"])
        if comps_map != 2:
            raise RuntimeError(
                f"{level}: map comps={comps_map} (expected 2 for (x,y) nodes)"
            )

        n_boxes = boxes.shape[0]
        if offs_m.shape[0] != n_boxes + 1:
            raise RuntimeError(
                f"{level}: offsets length {offs_m.shape[0]} != boxes+1 ({n_boxes+1})"
            )

        # Stitch each NodeFArrayBox into the global node arrays
        for b_idx, box in enumerate(boxes):
            lo_i = int(box["lo_i"])
            lo_j = int(box["lo_j"])
            hi_i = int(box["hi_i"])
            hi_j = int(box["hi_j"])

            nx = hi_i - lo_i + 1          # cells in i for this box
            ny = hi_j - lo_j + 1          # cells in j for this box
            nx_nodes = nx + 1             # nodes in i
            ny_nodes = ny + 1             # nodes in j

            start_m = int(offs_m[b_idx])
            end_m = int(offs_m[b_idx + 1])
            block_m = data_m[start_m:end_m]

            expected_nodes = 2 * ny_nodes * nx_nodes
            if block_m.size != expected_nodes:
                raise RuntimeError(
                    f"{level} box {b_idx}: map block size {block_m.size} "
                    f"!= {expected_nodes} = 2*(ny+1)*(nx+1)"
                )

            # NodeFArrayBox layout: (comp, j, i) = (x,y) as in your reader
            nodes = block_m.reshape(2, ny_nodes, nx_nodes)
            X_local = nodes[0]
            Y_local = nodes[1]

            # Map from local box indices to global node indices.
            gi0 = lo_i - lo_iL
            gj0 = lo_j - lo_jL
            gi1 = gi0 + nx_nodes
            gj1 = gj0 + ny_nodes

            X[gj0:gj1, gi0:gi1] = X_local
            Y[gj0:gj1, gi0:gi1] = Y_local

    meta = {"i_min": lo_iL, "i_max": hi_iL, "j_min": lo_jL, "j_max": hi_jL}
    return X, Y, meta


def build_domain_polygon(boundaries: dict) -> Polygon:
    """
    Build a shapely Polygon for the wedge domain, from the outer CCW boundary.

    Parameters
    ----------
    boundaries : dict
        Output of `extract_boundaries`, must contain 'outer_polygon_ccw'.

    Returns
    -------
    P : shapely.geometry.Polygon
        Polygon of the domain in physical space.
    """
    outer = np.asarray(boundaries["outer_polygon_ccw"])
    if outer.shape[1] != 2:
        raise ValueError("outer_polygon_ccw must be an (N,2) array of (x,y) points.")
    return Polygon(outer)

def build_amr_mesh_for_wedge(
    boundaries: dict,
    n0: int = 64,
    max_level: int = 3,
):
    """
    Build an AMR-style quadtree covering of the wedge domain.

    Level-0: n0 x n0 cells on the bounding box of the wedge.
    Levels 1..max_level: each cell splits into 4 children (quadtree).
    
    Rules:
      * Cells completely outside the wedge are discarded.
      * Cells completely inside the wedge are kept at their current level
        (no further refinement), giving as many coarse (level-0) cells as possible.
      * Cells intersecting the wedge boundary are recursively refined until
        max_level, so you get a layer of fine cells hugging the boundary.
      * At the finest level, each cell is clipped to the wedge so no cell
        geometry extends outside the boundary.

    Parameters
    ----------
    boundaries : dict
        Output of `extract_boundaries`.
    n0 : int
        Number of level-0 cells in each direction (n0 x n0).
    max_level : int
        Maximum AMR level (0 = coarsest).

    Returns
    -------
    cells : list of dict
        Each element has keys:
          'level' : int
          'poly'  : shapely Polygon (cell geometry in physical space)
          'i'     : int, logical i-index at this level
          'j'     : int, logical j-index at this level
    """
    P = build_domain_polygon(boundaries)
    xmin, ymin, xmax, ymax = P.bounds

    dx0 = (xmax - xmin) / n0
    dy0 = (ymax - ymin) / n0

    cells: list[dict] = []

    def subdivide(level: int, i: int, j: int, x0: float, y0: float, dx: float, dy: float):
        """
        Recursively subdivide a cell.

        (i, j) are logical indices at the current level (not global across all levels).
        """
        cell_box = box(x0, y0, x0 + dx, y0 + dy)

        # Completely outside: discard
        if not P.intersects(cell_box):
            return

        # Completely inside: keep this cell, no further refinement
        if P.covers(cell_box) and level <= max_level:
            cells.append({"level": level, "i": i, "j": j, "poly": cell_box})
            return

        # Partially covered
        if level == max_level:
            # Finest level: clip to the wedge to avoid geometry outside
            inter = P.intersection(cell_box)
            if not inter.is_empty:
                # inter can be a Polygon or MultiPolygon – handle both
                if inter.geom_type == "Polygon":
                    cells.append({"level": level, "i": i, "j": j, "poly": inter})
                else:
                    # Split MultiPolygon into separate cells at same level
                    for geom in inter.geoms:
                        cells.append({"level": level, "i": i, "j": j, "poly": geom})
            return

            # If we get here we had some coverage but it vanished after clipping,
            # which is effectively "outside".
        else:
            # Refine this boundary-touching cell into 4 children
            next_level = level + 1
            dx2 = dx / 2.0
            dy2 = dy / 2.0

            # Child 0: lower-left
            subdivide(next_level, 2 * i, 2 * j, x0, y0, dx2, dy2)
            # Child 1: lower-right
            subdivide(next_level, 2 * i + 1, 2 * j, x0 + dx2, y0, dx2, dy2)
            # Child 2: upper-left
            subdivide(next_level, 2 * i, 2 * j + 1, x0, y0 + dy2, dx2, dy2)
            # Child 3: upper-right
            subdivide(next_level, 2 * i + 1, 2 * j + 1, x0 + dx2, y0 + dy2, dx2, dy2)

    # Loop over level-0 grid and kick off recursion
    for j0 in range(n0):
        y0 = ymin + j0 * dy0
        for i0 in range(n0):
            x0 = xmin + i0 * dx0
            subdivide(level=0, i=i0, j=j0, x0=x0, y0=y0, dx=dx0, dy=dy0)

    return cells

def plot_amr_mesh(cells, boundaries=None, ax=None):
    """
    Plot the AMR mesh (cells) in physical space.

    Parameters
    ----------
    cells : list of dict
        Output of build_amr_mesh_for_wedge.
    boundaries : dict or None
        If provided, draws the domain boundary as well.
    ax : matplotlib Axes or None
        Existing axes to draw into; if None, a new figure is created.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Draw cells colored by level
    levels = sorted({c["level"] for c in cells})
    level_to_color = {lvl: idx for idx, lvl in enumerate(levels)}

    for cell in cells:
        poly = cell["poly"]
        lvl = cell["level"]
        x, y = poly.exterior.xy
        ax.plot(x, y, linewidth=0.5, alpha=0.8, label=None)

    # Draw the wedge boundary on top
    if boundaries is not None and "outer_polygon_ccw" in boundaries:
        outer = boundaries["outer_polygon_ccw"]
        ax.plot(outer[:, 0], outer[:, 1], linewidth=1.5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (physical)")
    ax.set_ylabel("y (physical)")
    return ax


def extract_boundaries(X: np.ndarray, Y: np.ndarray):
    """
    Extract physical-space boundary polylines from node coordinate arrays.

    Parameters
    ----------
    X, Y : 2D np.ndarray
        Arrays with shape (Nj_nodes, Ni_nodes) and the same shape.

    Returns
    -------
    boundaries : dict
        Keys:
            'bottom' : (N_bottom, 2) array, bottom boundary (left -> right).
            'top'    : (N_top, 2) array, top boundary (left -> right).
            'left'   : (N_left, 2) array, left boundary (bottom -> top).
            'right'  : (N_right, 2) array, right boundary (bottom -> top).
            'outer_polygon_ccw' : (M, 2) array, closed CCW polygon around domain.
    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")

    nj, ni = X.shape
    mask = np.isfinite(X) & np.isfinite(Y)

    # Bottom edge: j = 0, natural order left -> right in index space
    mask_bottom = mask[0, :]
    bottom = np.column_stack([X[0, mask_bottom], Y[0, mask_bottom]])

    # Top edge: j = nj - 1, natural order left -> right
    mask_top = mask[-1, :]
    top = np.column_stack([X[-1, mask_top], Y[-1, mask_top]])

    # Left edge: i = 0, natural order bottom -> top
    mask_left = mask[:, 0]
    left = np.column_stack([X[mask_left, 0], Y[mask_left, 0]])

    # Right edge: i = ni - 1, natural order bottom -> top
    mask_right = mask[:, -1]
    right = np.column_stack([X[mask_right, -1], Y[mask_right, -1]])

    # Build a closed CCW polygon:
    #   bottom: left -> right
    #   right:  bottom -> top (skip first point)
    #   top:    right -> left (reverse, skip first point)
    #   left:   top -> bottom (reverse, skip first point)
    right_ccw = right[1:]
    top_ccw = top[::-1][1:]
    left_ccw = left[::-1][1:]
    outer_polygon = np.vstack([bottom, right_ccw, top_ccw, left_ccw])

    return {
        "bottom": bottom,
        "top": top,
        "left": left,
        "right": right,
        "outer_polygon_ccw": outer_polygon,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract physical-space boundary curves from a Chombo .map.hdf5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--map_file",
        help="Path to DMR.plot.XXXXX.2d.map.hdf5 (only the map file is needed).",
    )
    parser.add_argument(
        "-l",
        "--level",
        default="level_0",
        help="AMR level to use when building the global node map.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dmr_boundaries_level0.npz",
        help="Output .npz file where boundary arrays will be stored.",
    )
    args = parser.parse_args(argv)

    map_path = str(args.map_file)
    level = args.level

    n0 = 64
    max_level = 3

    print(f"[INFO] Reading node mapping from '{map_path}' at level '{level}'...")
    X, Y, meta = build_level_node_coords(map_path, level=level)
    print(
        f"[INFO] Node grid shape: X, Y = {X.shape}, "
        f"i in [{meta['i_min']}, {meta['i_max']}], "
        f"j in [{meta['j_min']}, {meta['j_max']}]"
    )

    boundaries = extract_boundaries(X, Y)

    plot_boundaries(boundaries, show_polygon=True, save_path=None)
        
    # Build AMR mesh: 64x64 base grid, up to level 3
    amr_cells = build_amr_mesh_for_wedge(boundaries, n0=n0, max_level=max_level)

    print(f"[INFO] Built AMR mesh with {len(amr_cells)} cells total.")
    # For example, count cells at each level:
    
    level_counts = Counter(c["level"] for c in amr_cells)
    print("[INFO] Cells per level:", dict(level_counts))

    # Optional: visualize
    plot_amr_mesh(amr_cells, boundaries)
    plt.show()

    # Save spec to reuse in PyTorch project
    mesh_spec_path = os.path.join(os.path.dirname(args.output), "wedge_mesh_spec.pt")
    save_wedge_mesh_spec_pt(mesh_spec_path, boundaries, n0=n0, max_level=max_level)

    out_path = str(args.output)
    np.savez(
        out_path,
        X=X,
        Y=Y,
        bottom=boundaries["bottom"],
        top=boundaries["top"],
        left=boundaries["left"],
        right=boundaries["right"],
        outer_polygon_ccw=boundaries["outer_polygon_ccw"],
        **meta,
    )
    print(f"[INFO] Saved boundaries to '{out_path}'.")
    print("[INFO] Arrays stored in the .npz file:")
    with np.load(out_path) as data:
        for key in ["X", "Y", "bottom", "right", "top", "left", "outer_polygon_ccw"]:
            arr = data[key]
            print(f"  {key:16s}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
