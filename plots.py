# plots.py
# -----------------------------------------------------------------------------
# Plot helpers for multi-level AMR:
#   - rasterize nodes (level, ij, values) to a fine 2**L grid
#   - draw AMR mesh overlays for any subset of levels
#   - create triptych pages: GT(t), Pred(t+1), GT(t+1) across feature channels
#   - projection utility (x-axis projection per feature) if needed later
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, PolyCollection
import os, copy
from utils_geom import build_idw_map, apply_idw_map, _targeted_map_to_pred


# ------------------------------ PDF wrapper --------------------------------- #

class PdfWriter:
    def __init__(self, path: str):
        self._pp = PdfPages(path)

    def savefig(self, fig):
        self._pp.savefig(fig)
        plt.close(fig)

    def close(self):
        self._pp.close()


# ------------------------------ rasterization ------------------------------- #

def _draw_amr_cells(
    ax,
    centers: torch.Tensor,          # (N,2)
    levels: torch.Tensor | None,    # (N,) int, None -> all zeros
    values: torch.Tensor,           # (N,F) or (N,)
    f_idx: int | None,              # which feature/column to draw (None if values is 1D)
    H: int, W: int,
    bbox: tuple[float,float,float,float],  # (xmin,xmax,ymin,ymax)
    vmin: float, vmax: float,
    cmap: str | None = None,
    edgecolor: str = "none",
    linewidth: float = 0.0,
):
    # normalize inputs
    cxy = centers.detach().cpu().numpy()
    vals = values[:, f_idx].detach().cpu().numpy() if values.ndim == 2 else values.detach().cpu().numpy()
    lv  = levels.detach().cpu().numpy().astype(np.int32) if (levels is not None) else np.zeros(len(cxy), dtype=np.int32)

    xmin, xmax, ymin, ymax = bbox
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    patches = []
    for (x, y), L in zip(cxy, lv):
        scale = 0.5 ** int(L)          # L=0 → 1*dx0, L=1 → 0.5*dx0, ...
        w = dx0 * scale
        h = dy0 * scale
        patches.append(Rectangle((x - 0.5*w, y - 0.5*h), w, h))

    pc = PatchCollection(patches, cmap=cmap or "viridis", edgecolor="none", linewidth=0.0)
    pc.set_edgecolor("none")
    pc.set_array(vals)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    return pc  # so caller can attach a colorbar

def _build_amr_verts_np(centers_np, levels_np, H, W, bbox):
    xmin, xmax, ymin, ymax = bbox
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    if levels_np is None:
        levels_np = np.zeros((centers_np.shape[0],), dtype=np.int32)
    else:
        levels_np = levels_np.astype(np.int32)

    # vectorized widths/heights
    scale = np.power(0.5, levels_np)  # (N,)
    w = dx0 * scale
    h = dy0 * scale

    x = centers_np[:, 0]
    y = centers_np[:, 1]
    x0 = x - 0.5 * w
    x1 = x + 0.5 * w
    y0 = y - 0.5 * h
    y1 = y + 0.5 * h

    # verts: (N, 4, 2)
    verts = np.stack([
        np.stack([x0, y0], axis=1),
        np.stack([x1, y0], axis=1),
        np.stack([x1, y1], axis=1),
        np.stack([x0, y1], axis=1),
    ], axis=1).astype(np.float32)

    return verts

def _draw_amr_cells_fast(ax, verts, vals_1d, bbox, vmin, vmax, cmap="viridis"):
    xmin, xmax, ymin, ymax = bbox
    pc = PolyCollection(
        verts,
        cmap=cmap,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
    )
    pc.set_rasterized(True)

    pc.set_array(vals_1d)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    return pc

def _subplot_field(ax, field, H, W, title, vmin=None, vmax=None):
    img = field.view(H, W).cpu().numpy()
    im = ax.imshow(img, origin='lower', extent=[0,1,0,1], vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return im

def _empty_panel(ax, label: str):
    ax.set_axis_off()
    ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes)

def _subplot_delta_field(ax, Z, H, W, title="", vmin=None, vmax=None, cmap="RdBu_r"):
    Z2d = Z.view(H, W).detach().cpu().numpy()
    if vmin is None or vmax is None:
        vmax_auto = np.nanmax(np.abs(Z2d))
        vmax = vmax_auto if vmax_auto > 0 else 1.0
        vmin = -vmax
    im = ax.imshow(
        Z2d,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal"
    )
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return im

def rasterize_piecewise_constant(
    x: torch.Tensor,
    level: torch.Tensor,
    ij: torch.Tensor,
    H: int,
    W: int,
    L_out: int,
) -> np.ndarray:
    """
    Fill a fine raster (H*2**L_out, W*2**L_out, F) by painting each cell's
    constant value over its covering block. Assumes cell-centered values that
    represent the cell average for visualization.

    x:     [N, F]
    level: [N] in {0..L_max}
    ij:    [N, 2] indices within each level's raster
    """
    x = torch.as_tensor(x, dtype=torch.float32, device="cpu")
    level = torch.as_tensor(level, dtype=torch.long, device="cpu")
    ij = torch.as_tensor(ij, dtype=torch.long, device="cpu")

    F = int(x.shape[1])
    Hf, Wf = H * (2 ** L_out), W * (2 ** L_out)
    out = np.zeros((Hf, Wf, F), dtype=np.float32)

    # We paint coarse->fine so fine cells overwrite their parents where they exist
    order = torch.argsort(level)  # 0,1,2,...
    for idx in order.tolist():
        l = int(level[idx].item())
        i_l = int(ij[idx, 0].item())
        j_l = int(ij[idx, 1].item())
        block = 2 ** (L_out - l)  # size of cell in fine-grid pixels per side

        i0 = i_l * block
        j0 = j_l * block
        i1 = i0 + block
        j1 = j0 + block

        # Broadcast value across the block
        val = x[idx].cpu().numpy().reshape(1, 1, F)
        out[i0:i1, j0:j1, :] = val

    return out  # shape (Hf, Wf, F)


# ------------------------------- mesh overlay -------------------------------- #

def _draw_mesh_from_centers_and_levels(
    ax,
    centers: torch.Tensor,      # (N, 2)  x,y
    levels:  torch.Tensor,      # (N,)
    H: int,
    W: int,
    bbox: Tuple[float, float, float, float],
    *,
    lw0: float = 0.6,
    color0: str = "0.3",
    lw1: float = 0.4,
    color1: str = "k",
    lw2: float = 0.3,
    color2: str = "b",
    lw3: float = 0.05,
    color3: str = "r",
) -> None:
    """
    Draw each cell exactly as encoded in (centers, levels).
    level=0 → full L0 cell
    level=1 → 1/2 cell
    level=2 → 1/4 cell
    ...
    This matches the way evaluate_mesh_first() actually stores things.
    """
    xmin, xmax, ymin, ymax = bbox
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    centers = torch.as_tensor(centers).to(torch.float32)
    levels  = torch.as_tensor(levels).to(torch.long)

    for (cx, cy), L in zip(centers.tolist(), levels.tolist()):
        scale = 2 ** L
        w = dx0 / scale
        h = dy0 / scale
        x0 = cx - 0.5 * w
        y0 = cy - 0.5 * h

        if L == 0:
            lw, c = lw0, color0
        elif L == 1:
            lw, c = lw1, color1
        elif L == 2:
            lw, c = lw2, color2
        else:
            lw, c = lw3, color3

        rect = Rectangle((x0, y0), w, h, fill=False, linewidth=lw, edgecolor=c, alpha=0.95)
        ax.add_patch(rect)

def _draw_amr_mesh(
    ax: plt.Axes,
    masks_by_level: Dict[int, torch.Tensor] | torch.Tensor,
    H: int,
    W: int,
    levels: Optional[Iterable[int]] = None,
    lw: float = 0.3,
    alpha: float = 0.35,
    color: str = "k",
    *,
    bbox: tuple[float,float,float,float] | None = None,  # optional (xmin,xmax,ymin,ymax)
) -> None:
    """
    Draw cell borders for True cells in masks_by_level[l] as thin rectangles.
    Renders in a pixel-like grid whose finest resolution is 2^Lmax * (H,W).
    If `bbox` is provided, we map this pixel grid linearly to bbox.
    """

    # Allow passing a single mask instead of dict; treat it as level 1
    if not isinstance(masks_by_level, dict):
        masks_by_level = {1: torch.as_tensor(masks_by_level)}

    # Determine which levels to draw
    if levels is None:
        levels = sorted([l for l in masks_by_level.keys() if l >= 1])
    else:
        levels = sorted([l for l in levels if l >= 1])
    if len(levels) == 0:
        # Nothing to draw; just clear axes nicely
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        return

    Lmax = max(levels)

    # Finest-grid span (in "pixels" before optional bbox mapping)
    WW = W * (1 << Lmax)
    HH = H * (1 << Lmax)

    # If bbox is provided, build a linear mapping from pixel grid -> bbox units
    if bbox is not None:
        xmin, xmax, ymin, ymax = bbox
        sx = (xmax - xmin) / float(WW)
        sy = (ymax - ymin) / float(HH)
    else:
        xmin = ymin = 0.0
        sx = sy = 1.0

    # Make sure the axes show the correct extent and aspect.
    # Use "origin lower": y increases upward. Our row index y=0 is the *top* row,
    # so we flip vertically when positioning rectangles.
    ax.set_xlim(xmin, xmin + sx * WW)
    ax.set_ylim(ymin, ymin + sy * HH)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

    # Plot finer levels last (so they show on top)
    for l in levels:
        mask_l = masks_by_level.get(l, None)
        if mask_l is None:
            continue
        m = torch.as_tensor(mask_l, dtype=torch.bool, device='cpu').squeeze()
        if m.dim() == 1:
            if m.numel() != H * W:
                # If you ever store per-level masks at native resolution (H*2^l, W*2^l), handle here.
                raise ValueError(f"Level-{l} mask has 1D {m.numel()} elems; expected {H*W} for coarse layout.")
            m = m.view(H, W)
        elif m.dim() == 2:
            if m.shape != (H, W):
                # If someone passed (H*2^l, W*2^l), we can downsample to coarse by OR-ing 2^l blocks.
                # For now, be strict so we notice bad inputs.
                raise ValueError(f"Level-{l} mask has shape {tuple(m.shape)}; expected {(H, W)}.")
        else:
            raise ValueError(f"Level-{l} mask has dim={m.dim()}; expected 1D or 2D.")

        # Size (in finest pixels) of a coarse cell at this level
        # (We draw one 'pixel block' per coarse cell flagged at level l.)
        block = 1 << (Lmax - l)  # = 2^(Lmax-l)

        # Draw rectangles. Convert coarse (row=y, col=x) to lower-left anchored bbox coords.
        ys, xs = torch.nonzero(m, as_tuple=True)
        for y, x in zip(ys.tolist(), xs.tolist()):
            # Pixel-space top-left corner of the level-l coarse cell
            j0_pix = x * block
            i0_pix_top = y * block
            # Flip vertical to place with origin lower
            i0_pix = HH - (i0_pix_top + block)

            # Map to output units (either pixel units or bbox)
            x0 = xmin + sx * j0_pix
            y0 = ymin + sy * i0_pix
            w0 = sx * block
            h0 = sy * block

            rect = Rectangle((x0, y0), width=w0, height=h0,
                             fill=False, linewidth=lw, edgecolor=color, alpha=alpha)
            ax.add_patch(rect)


def _rasterize_selected_mesh_to_grid(
    centers: torch.Tensor,          # (N,2) x,y in [xmin,xmax]×[ymin,ymax]
    levels: torch.Tensor,           # (N,) integer levels: 0..Lmax
    values: torch.Tensor,           # (N,C) feature values for selected-mesh cells
    H: int, W: int,                 # coarse grid shape
    bbox: tuple[float,float,float,float] = (0.0, 1.0, 0.0, 1.0)  # (xmin,xmax,ymin,ymax)
) -> tuple[torch.Tensor, int, int]:
    """
    Return (img_flat, HH, WW) where img_flat is (HH*WW, C) laid out row-major
    on the *finest* grid implied by levels. Supports arbitrary Lmax >= 0.

    Strategy:
      - Finest resolution: HH = H * 2^Lmax, WW = W * 2^Lmax
      - Each level-l cell occupies a block of size B = 2^(Lmax - l) pixels in each axis.
      - We place each cell's value into its block (replication for coarse),
        with higher levels overwriting lower levels.
    """

    # Normalize inputs
    if isinstance(centers, np.ndarray): centers = torch.from_numpy(centers)
    if isinstance(levels,  np.ndarray): levels  = torch.from_numpy(levels)
    if isinstance(values,  np.ndarray): values  = torch.from_numpy(values)

    centers = centers.detach().to(torch.float32)
    levels  = levels.detach().to(torch.int64).view(-1)
    values  = values.detach()

    N = centers.shape[0]
    if levels.numel() != N:
        raise ValueError(f"levels length {levels.numel()} != centers length {N}")
    if values.shape[0] != N:
        raise ValueError(f"values first dim {values.shape[0]} != centers length {N}")
    if centers.dim() != 2 or centers.shape[1] != 2:
        raise ValueError(f"centers shape must be (N,2), got {tuple(centers.shape)}")

    xmin, xmax, ymin, ymax = bbox
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"Invalid bbox={bbox}")

    # Coarse cell size
    dx = (xmax - xmin) / float(W)
    dy = (ymax - ymin) / float(H)

    # Find overall finest resolution
    if levels.numel() == 0:
        Lmax = 0
    else:
        Lmax = int(levels.max().item())
        if Lmax < 0:
            raise ValueError(f"levels has negative entries; got Lmax={Lmax}")

    scale = 1 << Lmax  # 2^Lmax
    HH, WW = H * scale, W * scale
    C = int(values.shape[1]) if values.dim() == 2 else 1
    vals = values if values.dim() == 2 else values.view(N, 1)

    img = torch.full((HH, WW, C), torch.nan, device=vals.device, dtype=vals.dtype)

    # Map centers to finest-grid integer pixel coords
    # Convert to coarse-cell units, then scale by 2^Lmax and floor.
    # Clamp to valid range to avoid boundary spillover.
    x_units = (centers[:, 0] - xmin) / dx   # in [0, W)
    y_units = (centers[:, 1] - ymin) / dy   # in [0, H)

    col_fine = torch.floor(x_units * scale).to(torch.int64)
    row_fine = torch.floor(y_units * scale).to(torch.int64)

    col_fine = torch.clamp(col_fine, 0, WW - 1)
    row_fine = torch.clamp(row_fine, 0, HH - 1)

    # Order: coarse (big blocks) first, refined (small blocks) last to overwrite
    order = torch.argsort(levels)  # 0,1,2,...,Lmax
    row_fine = row_fine[order]
    col_fine = col_fine[order]
    levels_o = levels[order]
    vals_o   = vals[order]

    # For each cell, align to its level's block and fill that block with its value
    for i in range(N):
        l = int(levels_o[i].item())
        # Block size at this level on finest grid
        block = 1 << (Lmax - l)  # 2^(Lmax - l); equals 1 for finest cells
        r = int(row_fine[i].item())
        c = int(col_fine[i].item())
        # Align to upper-left of the block
        r0 = (r // block) * block
        c0 = (c // block) * block
        r1 = min(r0 + block, HH)
        c1 = min(c0 + block, WW)
        img[r0:r1, c0:c1, :] = vals_o[i].view(1, 1, C)

    # Flatten row-major as expected by your plotting code
    img_flat = img.view(HH * WW, C)
    return img_flat, HH, WW

# ------------------------------ triptych page -------------------------------- #

def _vmin_vmax_per_channel(arrs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel [vmin, vmax] across a list of (H,W,F) arrays.
    """
    F = arrs[0].shape[-1]
    vmin = np.full((F,), np.inf, dtype=np.float32)
    vmax = np.full((F,), -np.inf, dtype=np.float32)
    for A in arrs:
        vmin = np.minimum(vmin, A.reshape(-1, F).min(axis=0))
        vmax = np.maximum(vmax, A.reshape(-1, F).max(axis=0))
    # Avoid degenerate ranges
    tol = 1e-8
    tight = vmax - vmin
    tight[tight < tol] = 1.0
    return vmin, vmax

# ------------------------------ projections (optional) ---------------------- #

def make_projection_page_xaxis(
    fields: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    H: int,
    W: int,
    L_out: int,
    feature_names: Optional[Sequence[str]] = None,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """
    For each provided label -> (x, level, ij), rasterize to fine grid, then
    project along x-axis (sum over rows) to produce a 1D profile per feature.
    Plots all labels overlaid for comparison.

    Example:
        page = make_projection_page_xaxis(
            fields={
                "GT(t)": (x_t, level_t, ij_t),
                "Pred(t+1)": (x_pred, level_pred, ij_pred),
                "GT(t+1)": (x_tp1, level_tp1, ij_tp1),
            },
            H=H, W=W, L_out=fine_L, feature_names=["ρ","u","E"]
        )
    """
    # Rasterize all first
    fine = {name: rasterize_piecewise_constant(x, lvl, ij, H, W, L_out)
            for name, (x, lvl, ij) in fields.items()}

    any_grid = next(iter(fine.values()))
    Hf, Wf, F = any_grid.shape
    names = list(feature_names) if feature_names is not None and len(feature_names) == F else [f"feat[{i}]" for i in range(F)]

    fig, axes = plt.subplots(nrows=F, ncols=1, figsize=(6.8, max(1, F) * 2.4), constrained_layout=True)
    if F == 1:
        axes = np.array([axes])

    xs = np.arange(Wf)
    for r in range(F):
        ax = axes[r]
        for label, grid in fine.items():
            prof = grid[:, :, r].sum(axis=0)  # sum over rows => project onto x
            ax.plot(xs, prof, label=label, linewidth=1.3)
        ax.set_xlim(0, Wf - 1)
        ax.set_ylabel(names[r])
        ax.grid(True, alpha=0.25)
        if r == 0:
            ax.legend(loc="best", fontsize=9)
        if r == F - 1:
            ax.set_xlabel("x (fine index)")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    return fig

def plot_features_and_mask_pdf(
    out_pdf_path: str,
    pred_coarse_feat: torch.Tensor,  # (H*W, F)
    gt_coarse_feat: torch.Tensor,    # (H*W, F)
    pred_mask: torch.Tensor,         # (H*W,) bool
    gt_mask: torch.Tensor | None,    # (H*W,) bool or None
    s_ref: torch.Tensor | None,      # (H*W,) or None
    H: int,
    W: int,
    feature_names: list[str] | None = None
):
    """(Legacy) 2x2 per-feature page: GT(t+1), Pred(t+1), indicator, pred mask (+GT contour)."""
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    F = pred_coarse_feat.size(1)
    feature_names = feature_names or [f"x_{i}" for i in range(F)]
    with PdfPages(out_pdf_path) as pdf:
        for f in range(F):
            fig, axs = plt.subplots(2, 2, figsize=(10,8), constrained_layout=True)
            vmin = min(pred_coarse_feat[:,f].min().item(), gt_coarse_feat[:,f].min().item())
            vmax = max(pred_coarse_feat[:,f].max().item(), gt_coarse_feat[:,f].max().item())
            im0 = _subplot_field(axs[0,0], gt_coarse_feat[:,f], H, W, f"GT {feature_names[f]} | t+1", vmin=vmin, vmax=vmax)
            _ = _subplot_field(axs[0,1], pred_coarse_feat[:,f], H, W, f"Pred {feature_names[f]} | t+1", vmin=vmin, vmax=vmax)
            fig.colorbar(im0, ax=axs[0,0], shrink=0.8)
            fig.colorbar(im0, ax=axs[0,1], shrink=0.8)

            if s_ref is not None:
                im1 = _subplot_field(axs[1,0], s_ref, H, W, "Refine indicator (coarse)")
                fig.colorbar(im1, ax=axs[1,0], shrink=0.8)
            else:
                _empty_panel(axs[1,0], "indicator: (n/a)")

            m_pred = pred_mask.view(H,W).float()
            axs[1,1].imshow(m_pred.cpu().numpy(), origin='lower', extent=[0,1,0,1], cmap='gray_r', vmin=0, vmax=1)
            axs[1,1].set_title("Pred refine mask (t+1)")
            axs[1,1].set_xlabel("x"); axs[1,1].set_ylabel("y")
            if gt_mask is not None:
                gt = gt_mask.view(H,W).cpu().numpy().astype(float)
                axs[1,1].contour(np.linspace(0,1,W), np.linspace(0,1,H), gt, levels=[0.5], colors='red', linewidths=1.0)

            pdf.savefig(fig); plt.close(fig)

def plot_loss_curves(out_path: str, epochs, train_losses, val_losses):
    # --- Skip first 10 epochs if total epochs > 20 ---
    if len(epochs) > 20:
        start_epoch = 10
        epochs = epochs[start_epoch:]
        train_losses = train_losses[start_epoch:]
        val_losses = val_losses[start_epoch:]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(epochs, train_losses, label="train")
    #ax.plot(epochs, val_losses, label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Loss vs epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_predictions_cell_refine_pdf(
    out_pdf_path: str,
    gt_mask_t_list,       # list[Tensor(H*W,)]
    pred_mask_tp1_list,   # list[Tensor(H*W,)]
    gt_mask_tp1_list,     # list[Tensor(H*W,)]
    H: int, W: int,
    titles: list[str] | None = None
):
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    titles = titles or [f"sample {i}" for i in range(len(gt_mask_t_list))]
    with PdfPages(out_pdf_path) as pdf:
        for i, (m_t, m_p, m_g) in enumerate(zip(gt_mask_t_list, pred_mask_tp1_list, gt_mask_tp1_list)):
            fig = plt.figure(figsize=(12,8), constrained_layout=True)
            gs = fig.add_gridspec(2, 3, height_ratios=[3,1])
            axs_img = [fig.add_subplot(gs[0, j]) for j in range(3)]
            axs_bar = [fig.add_subplot(gs[1, j]) for j in range(3)]

            # --- NEW: draw AMR mesh instead of gray mask ---
            def show_mesh(ax, mask, title):
                ax.cla()
                if isinstance(mask, dict):
                    masks_by_level = {l: torch.as_tensor(m).view(H, W).to(torch.bool) for l, m in mask.items()}
                else:
                    m = torch.as_tensor(mask).view(-1).to(torch.bool).view(H, W)
                    masks_by_level = {1: m}

                # If your data bbox is not (0,1,0,1), pass it here so the grid matches your domain.
                bbox = (0.0, 1.0, 0.0, 1.0)  # or cfg["data"]["bbox"] if available in scope
                _draw_amr_mesh(ax, masks_by_level, H, W, bbox=bbox)
                ax.set_title(title)

            #print(f"[PLOT DEBUG] sums: t={int(torch.as_tensor(m_t).sum())} "
            #    f"pred={int(torch.as_tensor(m_p).sum())} "
            #    f"gt={int(torch.as_tensor(m_g).sum())}")

            show_mesh(axs_img[0], m_t,  f"t (GT mesh) — {titles[i]}")
            show_mesh(axs_img[1], m_p,  "t+1 (Pred mesh)")
            show_mesh(axs_img[2], m_g,  "t+1 (GT mesh)")

            # Histograms of levels (unchanged)
            def bar_levels(ax, mask, title):
                vals = mask.view(-1).detach().cpu().numpy().astype(bool)
                counts = [np.sum(~vals), np.sum(vals)]  # L0, L1
                ax.bar(["L0","L1"], counts)
                ax.set_title(title)
                ax.set_ylabel("# cells")

            bar_levels(axs_bar[0], m_t,  "levels @ t (GT)")
            bar_levels(axs_bar[1], m_p,  "levels @ t+1 (Pred)")
            bar_levels(axs_bar[2], m_g,  "levels @ t+1 (GT)")

            fig.suptitle("AMR meshes: GT(t), Pred(t+1), GT(t+1)  +  level histograms")
            pdf.savefig(fig); plt.close(fig)

def plot_predictions_from_examples_pdf(
    out_pdf_path: str,
    examples: List[dict],
    H: int,
    W: int,
    *,
    bbox: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    titles: Optional[List[str]] = None,
):
    """
    3-panel PDF, but instead of trying to rebuild L2 from masks,
    it uses the explicit centers/levels that evaluate_mesh_first() saved.
      panel 0: GT at t  -> (centers_t,  level_t)
      panel 1: Pred     -> (centers_selected, levels_selected)
      panel 2: GT at t+1 -> (centers_tp1, level_tp1)  (if present)
    This should match the 'actual' mesh plots.
    """
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    n = len(examples)
    titles = titles or [f"sample {i}" for i in range(n)]

    print(f"[PLOT DEBUG] Writing predictions mesh PDF to: {out_pdf_path}")
    with PdfPages(out_pdf_path) as pdf:
        for i, ex in enumerate(examples):
            fig = plt.figure(figsize=(12, 4.9), constrained_layout=True)
            gs = fig.add_gridspec(1, 3)
            axs = [fig.add_subplot(gs[0, j]) for j in range(3)]

            # 0) GT at t
            ctr_t   = ex.get("centers_t", None)
            lev_t   = ex.get("level_t", None)
            # 1) Predicted mesh (the one used for eval)
            #ctr_sel = ex.get("centers_selected", None)
            #lev_sel = ex.get("levels_selected", None)
            ctr_sel = ex.get("pred_centers", None)
            lev_sel = ex.get("pred_levels", None)
            # 2) GT at t+1 (may be None)
            ctr_tp1 = ex.get("centers_tp1", None)
            lev_tp1 = ex.get("level_tp1", None)

            # panel 0
            if ctr_t is not None and lev_t is not None:
                _draw_mesh_from_centers_and_levels(
                    axs[0], ctr_t, lev_t, H, W, bbox,
                    lw0=0.6, color0="0.3",
                    lw1=0.4, color1="k",
                    lw2=0.3, color2="b",
                )
            axs[0].set_title(f"t (GT) — {titles[i]}")
            axs[0].set_aspect("equal")
            axs[0].set_xlabel("x"); axs[0].set_ylabel("y")

            # panel 1
            if ctr_sel is not None and lev_sel is not None:
                _draw_mesh_from_centers_and_levels(
                    axs[1], ctr_sel, lev_sel, H, W, bbox,
                    lw0=0.6, color0="0.3",
                    lw1=0.4, color1="k",
                    lw2=0.3, color2="b",
                )
            axs[1].set_title("t+1 (Pred mesh)")
            axs[1].set_aspect("equal")
            axs[1].set_xlabel("x"); axs[1].set_ylabel("y")

            # panel 2
            if ctr_tp1 is not None and lev_tp1 is not None:
                _draw_mesh_from_centers_and_levels(
                    axs[2], ctr_tp1, lev_tp1, H, W, bbox,
                    lw0=0.6, color0="0.3",
                    lw1=0.4, color1="k",
                    lw2=0.3, color2="b",
                )
            axs[2].set_title("t+1 (GT)")
            axs[2].set_aspect("equal")
            axs[2].set_xlabel("x"); axs[2].set_ylabel("y")

            fig.suptitle(f"H={H} W={W}")
            pdf.savefig(fig); plt.close(fig)


def plot_qual_2x3_pdf(
    out_pdf_path: str,
    gt_coarse_feat_t: torch.Tensor,     # (H*W, F)  [unused if selected-mesh is provided]
    gt_coarse_feat_tp1: torch.Tensor,   # (H*W, F)  [unused if selected-mesh is provided]
    pred_coarse_feat_tp1: torch.Tensor, # (H*W, F)  [unused if selected-mesh is provided]
    gt_mask_t_list,
    H: int, W: int,
    feature_names: list[str] | None = None,
    var_titles: list[str] | None = None,
    *,
    # ---- legacy fine-grid rasters (original behavior) ----
    fine_gt_t: torch.Tensor | None = None,      # (Hf*Wf, F)
    fine_gt_tp1: torch.Tensor | None = None,    # (Hf*Wf, F)
    fine_pred_tp1: torch.Tensor | None = None,  # (Hf*Wf, F)
    Hf: int | None = None,
    Wf: int | None = None,
    title: str = "",
    # ---- selected-mesh branch (preferred) ----
    centers: torch.Tensor | None = None,       # (N,2) selected centers (x,y) in domain units
    levels:  torch.Tensor | None = None,       # (N,) level per cell (0..Lmax)
    sel_gt_t: torch.Tensor | None = None,      # (N,F) GT(t) sampled on selected centers (ABSOLUTE UNITS)
    sel_gt_tp1: torch.Tensor | None = None,    # (N,F) GT(t+1) on selected centers (ABSOLUTE UNITS)
    sel_pred_tp1: torch.Tensor | None = None,  # (N,F) Pred(t+1) on selected centers (ABSOLUTE UNITS)
    mesh_mode: str | None = None,
    # ---- rendering options for the selected-mesh branch ----
    selected_raster: str = "block",            # "block" (AMR replication) or "idw"
    selected_bins: int = 256,                  # grid size for "idw" mode (selected_bins x selected_bins)
    selected_k: int = 8,                       # k-NN for "idw"
    selected_bbox: tuple[float,float,float,float] | None = None,  # (xmin,xmax,ymin,ymax); infer if None
):
    """
    Layout (2 rows x 3 cols) per-feature page:
      Top (GT):   [GT(t),   GT(t+1),        GT(t+1) - GT(t)]
      Bottom:     [Pred-GT, Pred(t+1),      Pred(t+1) - GT(t)]

    Behavior:
      - If `centers/levels/sel_*` are provided, renders **on the selected mesh**.
        Use `selected_raster="idw"` to visually match the PT 'actual' path.
      - Else, falls back to the original coarse/fine raster behavior unchanged.

    NOTE: Pass *absolute* (denormalized) tensors for sel_gt_t / sel_gt_tp1 / sel_pred_tp1.
    """

    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)

    # ---------- small helpers ----------
    def _imshow_field(ax, flat, HH, WW, label, vmin=None, vmax=None):
        arr = flat.view(HH, WW).detach().cpu().numpy()
        cmap = copy.copy(plt.get_cmap("viridis"))
        cmap.set_bad(color="white")
        im = ax.imshow(np.ma.masked_invalid(arr), origin="lower",
                       vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
        ax.set_title(label)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        return im

    def _imshow_delta(ax, flat, HH, WW, label, vmin=None, vmax=None):
        arr = flat.view(HH, WW).detach().cpu().numpy()
        cmap = copy.copy(plt.get_cmap("viridis"))
        cmap.set_bad(color="white")
        im = ax.imshow(np.ma.masked_invalid(arr), origin="lower",
                       vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
        ax.set_title(label)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        return im

    # fallback: AMR block replication (you already have this; imported from your module)
    def _rasterize_selected_mesh_to_grid_block(centers_, levels_, values_, Hc, Wc, bbox=(0.0,1.0,0.0,1.0)):
        # identical to your existing _rasterize_selected_mesh_to_grid, but ensure NaN init
        if isinstance(centers_, np.ndarray): centers_ = torch.from_numpy(centers_)
        if isinstance(levels_,  np.ndarray): levels_  = torch.from_numpy(levels_)
        if isinstance(values_,  np.ndarray): values_  = torch.from_numpy(values_)
        centers_ = centers_.to(torch.float32)
        levels_  = levels_.to(torch.int64).view(-1)
        values_  = values_
        N = centers_.shape[0]
        xmin, xmax, ymin, ymax = bbox
        dx = (xmax - xmin) / float(Wc)
        dy = (ymax - ymin) / float(Hc)
        Lmax = int(levels_.max().item()) if levels_.numel() > 0 else 0
        scale = 1 << Lmax
        HH, WW = Hc * scale, Wc * scale
        C = int(values_.shape[1]) if values_.dim() == 2 else 1
        vals = values_ if values_.dim() == 2 else values_.view(N, 1)
        device = vals.device
        img = torch.full((HH, WW, C), torch.nan, device=device, dtype=vals.dtype)  # NaN init

        x_units = (centers_[:, 0] - xmin) / dx
        y_units = (centers_[:, 1] - ymin) / dy
        col_fine = torch.clamp(torch.floor(x_units * scale).to(torch.int64), 0, WW - 1)
        row_fine = torch.clamp(torch.floor(y_units * scale).to(torch.int64), 0, HH - 1)

        order = torch.argsort(levels_)  # 0..Lmax
        row_fine = row_fine[order]; col_fine = col_fine[order]
        levels_o = levels_[order];  vals_o = vals[order]

        for i in range(N):
            l = int(levels_o[i].item())
            block = 1 << (Lmax - l)  # 2^(Lmax - l)
            r = int(row_fine[i].item()); c = int(col_fine[i].item())
            r0 = (r // block) * block; c0 = (c // block) * block
            r1 = min(r0 + block, HH);  c1 = min(c0 + block, WW)
            img[r0:r1, c0:c1, :] = vals_o[i].view(1, 1, C)
        return img.view(HH * WW, C), HH, WW

    # IDW resampling to a uniform grid (to mimic PT path)
    def _rasterize_selected_mesh_to_grid_idw(centers_, values_, bins=256, k=8, bbox=None, chunk=32768):
        """
        centers_: (N,2), values_: (N,F)
        Returns img_flat (bins*bins, F), bins, bins
        """
        if isinstance(centers_, np.ndarray): centers_ = torch.from_numpy(centers_)
        centers_ = centers_.to(values_.device).to(torch.float32)
        values_  = values_.to(torch.float32)
        if bbox is None:
            xmin = float(centers_[:,0].min().item()); xmax = float(centers_[:,0].max().item())
            ymin = float(centers_[:,1].min().item()); ymax = float(centers_[:,1].max().item())
            bbox = (xmin, xmax, ymin, ymax)
        xmin, xmax, ymin, ymax = bbox

        xs = torch.linspace(xmin, xmax, bins, device=centers_.device)
        ys = torch.linspace(ymin, ymax, bins, device=centers_.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (M,2), M=bins*bins

        M = grid.size(0); F = values_.size(1)
        out = torch.full((M, F), float('nan'), device=values_.device)

        # chunk to keep memory sane
        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            q = grid[s:e]                            # (m,2)
            # distances to all centers (m, N)
            d = torch.cdist(q, centers_)             # (m, N)
            d, idx = torch.topk(d, k=min(k, centers_.size(0)), dim=1, largest=False)
            w = 1.0 / (d + 1e-8)                     # inverse distance
            w = w / w.sum(dim=1, keepdim=True)       # normalize
            vals = values_.index_select(0, idx.reshape(-1)).view(idx.size(0), idx.size(1), F)  # (m,k,F)
            out[s:e] = (w.unsqueeze(-1) * vals).sum(dim=1)
        return out, bins, bins

    # ---------- selected vs fallback branch ----------
    use_selected_mesh = (
        centers is not None and levels is not None and
        sel_gt_t is not None and sel_gt_tp1 is not None and sel_pred_tp1 is not None
    )

    if use_selected_mesh:
        F = sel_gt_tp1.size(1)
        feature_names = feature_names or [f"x_{i}" for i in range(F)]
        var_titles    = var_titles or feature_names

        # bbox for rasterization
        if selected_bbox is None:
            xmin = float(centers[:,0].min().item()); xmax = float(centers[:,0].max().item())
            ymin = float(centers[:,1].min().item()); ymax = float(centers[:,1].max().item())
            bbox = (xmin, xmax, ymin, ymax)
        else:
            bbox = selected_bbox

        # Build rasters in the chosen mode
        if selected_raster.lower() == "idw":
            A_img, HH, WW = _rasterize_selected_mesh_to_grid_idw(
                centers, sel_gt_t, bins=selected_bins, k=selected_k, bbox=bbox
            )
            B_img, _,  _  = _rasterize_selected_mesh_to_grid_idw(
                centers, sel_gt_tp1, bins=selected_bins, k=selected_k, bbox=bbox
            )
            P_img, _,  _  = _rasterize_selected_mesh_to_grid_idw(
                centers, sel_pred_tp1, bins=selected_bins, k=selected_k, bbox=bbox
            )
        else:  # "block"
            A_img, HH, WW = _rasterize_selected_mesh_to_grid_block(centers, levels, sel_gt_t,   H, W, bbox=bbox)
            B_img, _,  _  = _rasterize_selected_mesh_to_grid_block(centers, levels, sel_gt_tp1, H, W, bbox=bbox)
            P_img, _,  _  = _rasterize_selected_mesh_to_grid_block(centers, levels, sel_pred_tp1, H, W, bbox=bbox)

        with PdfPages(out_pdf_path) as pdf:
            for f in range(F):
                fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
                A = A_img[:, f].view(-1); B = B_img[:, f].view(-1); P = P_img[:, f].view(-1)

                vmin = float(torch.min(torch.stack([torch.nan_to_num(A, nan=0.).nan_to_num(), 
                                                    torch.nan_to_num(B, nan=0.).nan_to_num(), 
                                                    torch.nan_to_num(P, nan=0.).nan_to_num()])).min())
                vmax = float(torch.max(torch.stack([torch.nan_to_num(A, nan=0.), 
                                                    torch.nan_to_num(B, nan=0.), 
                                                    torch.nan_to_num(P, nan=0.)])).max())

                Dgt = (B - A)
                Dpg = (P - B)
                Dpt = (P - A)

                m_gt = float(torch.nan_to_num(Dgt.abs()).max())
                m_pg = float(torch.nan_to_num(Dpg.abs()).max())
                m_pt = float(torch.nan_to_num(Dpt.abs()).max())

                im0 = _imshow_field(axs[0,0], A, HH, WW, f"GT(t): {feature_names[f]}", vmin=vmin, vmax=vmax)
                im1 = _imshow_field(axs[0,1], B, HH, WW, f"GT(t+1): {feature_names[f]}", vmin=vmin, vmax=vmax)
                im2 = _imshow_delta(axs[0,2], Dgt, HH, WW, "GT(t+1) − GT(t)", vmin=-m_gt, vmax=+m_gt)
                fig.colorbar(im0, ax=axs[0,0], shrink=0.7, location='right')
                fig.colorbar(im1, ax=axs[0,1], shrink=0.7, location='right')
                fig.colorbar(im2, ax=axs[0,2], shrink=0.7, location='right')

                impd0 = _imshow_delta(axs[1,0], Dpg, HH, WW, "Pred(t+1) − GT(t+1)", vmin=-m_pg, vmax=+m_pg)
                imp   = _imshow_field(axs[1,1], P,   HH, WW, f"Pred(t+1): {feature_names[f]}", vmin=vmin, vmax=vmax)
                impd2 = _imshow_delta(axs[1,2], Dpt, HH, WW, "Pred(t+1) − GT(t)", vmin=-m_pt, vmax=+m_pt)
                fig.colorbar(impd0, ax=axs[1,0], shrink=0.7, location='right')
                fig.colorbar(imp,   ax=axs[1,1], shrink=0.7, location='right')
                fig.colorbar(impd2, ax=axs[1,2], shrink=0.7, location='right')

                mesh_label = f"({mesh_mode or 'selected'}; {selected_raster}) @ {HH}×{WW}"
                fig.suptitle(var_titles[f] + '; ' + title + ' ' + mesh_label, fontsize=14)
                pdf.savefig(fig); plt.close(fig)
        return

    # ----------------- fallback: ORIGINAL raster behavior --------------------
    F = gt_coarse_feat_t.size(1)
    feature_names = feature_names or [f"x_{i}" for i in range(F)]
    var_titles    = var_titles or feature_names

    use_fine = (
        fine_gt_t is not None and fine_gt_tp1 is not None and fine_pred_tp1 is not None
        and (Hf is not None) and (Wf is not None)
    )

    with PdfPages(out_pdf_path) as pdf:
        for f in range(F):
            fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
            if use_fine:
                A = fine_gt_t[:, f].view(-1);  B = fine_gt_tp1[:, f].view(-1);  P = fine_pred_tp1[:, f].view(-1)
                HH, WW = Hf, Wf
            else:
                A = gt_coarse_feat_t[:, f].view(-1); B = gt_coarse_feat_tp1[:, f].view(-1); P = pred_coarse_feat_tp1[:, f].view(-1)
                HH, WW = H, W

            vmin = float(torch.min(torch.stack([A.min(), B.min(), P.min()])))
            vmax = float(torch.max(torch.stack([A.max(), B.max(), P.max()])))

            Dgt = (B - A); Dpg = (P - B); Dpt = (P - A)
            m_gt = float(Dgt.abs().max()); m_pg = float(Dpg.abs().max()); m_pt = float(Dpt.abs().max())

            im0 = _imshow_field(axs[0,0], A, HH, WW, f"GT(t): {feature_names[f]}", vmin=vmin, vmax=vmax)
            im1 = _imshow_field(axs[0,1], B, HH, WW, f"GT(t+1): {feature_names[f]}", vmin=vmin, vmax=vmax)
            im2 = _imshow_delta(axs[0,2], Dgt, HH, WW, "GT(t+1) − GT(t)", vmin=-m_gt, vmax=+m_gt)
            fig.colorbar(im0, ax=axs[0,0], shrink=0.7, location='right')
            fig.colorbar(im1, ax=axs[0,1], shrink=0.7, location='right')
            fig.colorbar(im2, ax=axs[0,2], shrink=0.7, location='right')

            impd0 = _imshow_delta(axs[1,0], Dpg, HH, WW, "Pred(t+1) − GT(t+1)", vmin=-m_pg, vmax=+m_pg)
            imp   = _imshow_field(axs[1,1], P,   HH, WW, f"Pred(t+1): {feature_names[f]}", vmin=vmin, vmax=vmax)
            impd2 = _imshow_delta(axs[1,2], Dpt, HH, WW, "Pred(t+1) − GT(t)", vmin=-m_pt, vmax=+m_pt)
            fig.colorbar(impd0, ax=axs[1,0], shrink=0.7, location='right')
            fig.colorbar(imp,   ax=axs[1,1], shrink=0.7, location='right')
            fig.colorbar(impd2, ax=axs[1,2], shrink=0.7, location='right')

            fig.suptitle(var_titles[f] + '; ' + title, fontsize=14)
            pdf.savefig(fig); plt.close(fig)


def _pt_to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

'''
def amr_composite_to_finest_grid(
    centers: torch.Tensor,  # (N,2)
    levels:  torch.Tensor,  # (N,)
    values:  torch.Tensor,  # (N,F)
    H: int, W: int,
    bbox=(0.0,1.0,0.0,1.0)
):
    """
    Composite an AMR field onto the *finest* implied grid (H*2^Lmax, W*2^Lmax),
    with 'fine overwriting coarse'. Returns:
      img_flat: (HH*WW, F) row-major
      level_flat: (HH*WW,) the level of the value written at each pixel
      HH, WW
    """
    if isinstance(centers, np.ndarray): centers = torch.from_numpy(centers)
    if isinstance(levels,  np.ndarray): levels  = torch.from_numpy(levels)
    if isinstance(values,  np.ndarray): values  = torch.from_numpy(values)

    centers = centers.to(torch.float32)
    levels  = levels.to(torch.int64).view(-1)
    vals    = values if values.dim()==2 else values.view(-1,1)

    xmin, xmax, ymin, ymax = bbox
    dx = (xmax - xmin) / float(W)
    dy = (ymax - ymin) / float(H)

    Lmax = int(levels.max().item()) if levels.numel() > 0 else 0
    scale = 1 << Lmax
    HH, WW = H*scale, W*scale
    N = centers.size(0)
    F = vals.size(1)

    # map centers to finest pixel coords
    x_units = (centers[:,0] - xmin) / dx   # in [0, W)
    y_units = (centers[:,1] - ymin) / dy   # in [0, H)
    colf = torch.floor(x_units * scale).clamp(0, WW-1).to(torch.int64)
    rowf = torch.floor(y_units * scale).clamp(0, HH-1).to(torch.int64)

    # order by increasing level so higher levels overwrite last
    order = torch.argsort(levels)
    rowf = rowf[order]; colf = colf[order]
    levO = levels[order]; valO = vals[order]

    img   = torch.zeros((HH, WW, F), dtype=vals.dtype, device=vals.device)
    lmap  = torch.full((HH, WW), -1, dtype=torch.int16, device=vals.device)

    for i in range(N):
        l = int(levO[i])
        block = 1 << (Lmax - l)           # side pixels for this level on finest grid
        r = int(rowf[i]); c = int(colf[i])
        r0 = (r // block) * block; c0 = (c // block) * block
        r1 = min(r0 + block, HH); c1 = min(c0 + block, WW)
        img[r0:r1, c0:c1, :] = valO[i].view(1,1,F)
        lmap[r0:r1, c0:c1]   = l

    return img.view(HH*WW, F), lmap.view(HH*WW), HH, WW
'''
def amr_composite_to_finest_grid(
    centers: torch.Tensor,  # (N,2)
    levels:  torch.Tensor,  # (N,)
    values:  torch.Tensor,  # (N,F)
    H: int, W: int,
    bbox=(0.0,1.0,0.0,1.0),
    refine_ratio: int = 2,
):
    """
    Composite an AMR field onto the *finest* implied grid
    (H*refine_ratio^Lmax, W*refine_ratio^Lmax),
    with 'fine overwriting coarse'.

    Returns:
      img_flat:   (HH*WW, F)   values on finest grid
      valid_flat: (HH*WW,)     bool mask: True where at least one AMR cell wrote
      HH, WW
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    if isinstance(centers, np.ndarray): centers = torch.from_numpy(centers)
    if isinstance(levels,  np.ndarray): levels  = torch.from_numpy(levels)
    if isinstance(values,  np.ndarray): values  = torch.from_numpy(values)

    centers = centers.to(torch.float32)
    levels  = levels.to(torch.int64).view(-1)
    vals    = values if values.dim() == 2 else values.view(-1, 1)

    xmin, xmax, ymin, ymax = bbox
    dx = (xmax - xmin) / float(W)
    dy = (ymax - ymin) / float(H)

    Lmax = int(levels.max().item()) if levels.numel() > 0 else 0
    scale = rr ** Lmax
    HH, WW = H * scale, W * scale
    N = centers.size(0)
    F = vals.size(1)

    # map centers to finest pixel coords
    x_units = (centers[:, 0] - xmin) / dx   # [0, W)
    y_units = (centers[:, 1] - ymin) / dy   # [0, H)
    colf = torch.floor(x_units * scale).clamp(0, WW - 1).to(torch.int64)
    rowf = torch.floor(y_units * scale).clamp(0, HH - 1).to(torch.int64)

    # order by increasing level so higher levels overwrite last
    order = torch.argsort(levels)
    rowf = rowf[order]
    colf = colf[order]
    levO = levels[order]
    valO = vals[order]

    img   = torch.zeros((HH, WW, F), dtype=vals.dtype, device=vals.device)
    valid = torch.zeros((HH, WW), dtype=torch.bool, device=vals.device)

    for i in range(N):
        l = int(levO[i])
        block = rr ** (Lmax - l)
        r = int(rowf[i])
        c = int(colf[i])
        r0 = (r // block) * block
        c0 = (c // block) * block
        r1 = min(r0 + block, HH)
        c1 = min(c0 + block, WW)

        img[r0:r1, c0:c1, :] = valO[i].view(1, 1, F)
        valid[r0:r1, c0:c1]  = True

    return (
        img.view(HH * WW, F),
        valid.view(HH * WW),
        HH,
        WW,
    )

def _as_np(x):
    if x is None: return None
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)

def _as_level(levels, N):
    if levels is None:  # allow None: treat as all level-0
        return np.zeros((N,), dtype=np.int64)
    L = _as_np(levels).astype(np.int64).reshape(-1)
    if L.size != N:
        raise ValueError(f"levels size {L.size} != N {N}")
    return L


def _gradmag_from_image(img: np.ndarray, bbox, HH: int, WW: int):
    """
    img: (HH,WW) array – already composited to the finest grid
    bbox: (xmin, xmax, ymin, ymax)
    Returns: (HH,WW) gradient magnitude with one-sided boundaries.
    """
    xmin, xmax, ymin, ymax = bbox
    dx = (xmax - xmin) / float(max(WW, 1))
    dy = (ymax - ymin) / float(max(HH, 1))
    # finite differences
    fx = np.zeros_like(img, dtype=np.float64)
    fy = np.zeros_like(img, dtype=np.float64)

    if WW > 1:
        fx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / (2.0 * dx)
        fx[:, 0]    = (img[:, 1] - img[:, 0])   / dx
        fx[:, -1]   = (img[:, -1] - img[:, -2]) / dx
    if HH > 1:
        fy[1:-1, :] = (img[2:, :] - img[:-2, :]) / (2.0 * dy)
        fy[0,    :] = (img[1,  :] - img[0,   :]) / dy
        fy[-1,   :] = (img[-1, :] - img[-2,  :]) / dy

    return np.sqrt(fx*fx + fy*fy)


# helper near the top of plots.py (optional, just for brevity)
def _fidx_or_none(t: torch.Tensor, f: int | None):
    return f if (t.ndim == 2 and f is not None) else None

def _unwrap_delta(entry, fallback_centers, fallback_levels):
    """
    Accepts either:
      - dict: {"vals": (N,F or N), "centers": (N,2), "levels": (N,)}
      - tensor: (N,F or N)
    Returns (vals, centers, levels).
    """
    if isinstance(entry, dict):
        vals    = entry["vals"]
        centers = entry.get("centers", fallback_centers)
        levels  = entry.get("levels", fallback_levels)
        return vals, centers, levels
    # legacy path: plain tensor
    return entry, fallback_centers, fallback_levels

@torch.no_grad()
def compute_plot_deltas(
    gt_t_centers: torch.Tensor, gt_t_feats: torch.Tensor,
    gt_tp1_centers: torch.Tensor, gt_tp1_feats: torch.Tensor,
    pred_centers: torch.Tensor, pred_levels: torch.Tensor,
    pred_parents: torch.Tensor, mask_pred: torch.Tensor,
    pred_feats: torch.Tensor,
    H: int, W: int, knn_k: int = 8, chunk: int = 8192,
):
    """
    Returns dict with:
      - delta_gt      (N_tp1, F)  = GT(t+1) - map[GT(t) -> GT(t+1)]          on GT(t+1) mesh
      - delta_pred_gt (N_pred, F) = Pred(t+1) - map[GT(t+1) -> Pred(t+1)]    on Pred mesh
      - delta_pred_t  (N_pred, F) = Pred(t+1) - map[GT(t)   -> Pred(t+1)]    on Pred mesh
      - gt_t_on_gt_tp1, gt_tp1_on_pred, gt_t_on_pred (for optional inspection)
    """
    # --- GT(t) -> GT(t+1)
    idx, w = build_idw_map(gt_tp1_centers, gt_t_centers, k=knn_k, chunk=chunk)
    gt_t_on_gt_tp1 = apply_idw_map(idx, w, gt_t_feats)
    delta_gt = gt_tp1_feats - gt_t_on_gt_tp1

    # --- GT(t+1) -> Pred(t+1)
    gt_tp1_on_pred, _ = _targeted_map_to_pred(
        pred_centers=pred_centers,
        pred_levels=pred_levels,
        pred_parents=pred_parents,
        mask_pred=mask_pred,
        src_centers=gt_tp1_centers,
        src_feats=gt_tp1_feats,
        H=H, W=W,
        mask_src_parent=None, src_parent_feats=None,
    )
    delta_pred_gt = pred_feats - gt_tp1_on_pred

    # --- GT(t) -> Pred(t+1)
    gt_t_on_pred, _ = _targeted_map_to_pred(
        pred_centers=pred_centers,
        pred_levels=pred_levels,
        pred_parents=pred_parents,
        mask_pred=mask_pred,
        src_centers=gt_t_centers,
        src_feats=gt_t_feats,
        H=H, W=W,
        mask_src_parent=None, src_parent_feats=None,
    )
    delta_pred_t = pred_feats - gt_t_on_pred

    return {
        "gt_t_on_gt_tp1": gt_t_on_gt_tp1,
        "gt_tp1_on_pred": gt_tp1_on_pred,
        "gt_t_on_pred": gt_t_on_pred,
        "delta_gt": delta_gt,
        "delta_pred_gt": delta_pred_gt,
        "delta_pred_t": delta_pred_t,
    }


@torch.no_grad()
def plot_qual_2x3_pdf_with_cells(
    examples: list[dict],
    cfg: dict,
    out_pdf_path: str,
    feature_names: list[str] | None = None,
    unify_clims: bool = False,  # if True, uses global color limits across all pages; else per-page
):
    """
    Builds one combined PDF over all examples.
    Each page: top row = [GT(t), Pred(t+1), GT(t+1)],
               bottom = [GT(t+1)-GT(t) on GT(t+1),
                         Pred(t+1)-GT(t+1) on Pred,
                         Pred(t+1)-GT(t)   on Pred]
    """
    speed = cfg.get("speed", {})
    knn_k = int(cfg["loss"].get("interp_k", speed.get("knn_k", 8)))
    chunk = int(speed.get("interp_chunk", 8192))

    # Default names if not provided
    if feature_names is None and len(examples) > 0:
        F = int(examples[0]["gt_t"].shape[1])
        feature_names = [f"Feat {i}" for i in range(F)]

    # Optionally pre-compute global color limits (top row and deltas) across all pages
    global_top_min = None
    global_top_max = None
    global_d_abs_99 = None  # symmetric limits around 0 for deltas

    if unify_clims and len(examples) > 0:
        tops = []
        dels = []
        for ex in examples:
            gt_t_feats   = ex["gt_t"].to(torch.float32)
            pred_feats   = ex["pred_tp1"].to(torch.float32)
            gt_tp1_feats = ex["gt_tp1"].to(torch.float32)
            tops.append(torch.cat([gt_t_feats.reshape(-1), pred_feats.reshape(-1), gt_tp1_feats.reshape(-1)]))

            # Need deltas too → compute once with cheap configs per page
            H, W = int(ex["H"]), int(ex["W"])
            mask_pred_parent = _recover_parent_mask(ex, H, W)
            deltas = compute_plot_deltas(
                gt_t_centers=ex["centers_t"].to(torch.float32),
                gt_t_feats=gt_t_feats,
                gt_tp1_centers=ex["centers_tp1"].to(torch.float32),
                gt_tp1_feats=gt_tp1_feats,
                pred_centers=ex["pred_centers"].to(torch.float32),
                pred_levels=ex["pred_levels"].to(torch.int64),
                pred_parents=ex["pred_parents"].to(torch.int64),
                mask_pred=mask_pred_parent,
                pred_feats=pred_feats,
                H=H, W=W, knn_k=knn_k, chunk=chunk,
            )
            dels.append(torch.cat([
                deltas["delta_gt"].abs().reshape(-1),
                deltas["delta_pred_gt"].abs().reshape(-1),
                deltas["delta_pred_t"].abs().reshape(-1),
            ]))

        tops_cat = torch.cat(tops)
        global_top_min = float(torch.nanquantile(tops_cat, 0.01))
        global_top_max = float(torch.nanquantile(tops_cat, 0.99))

        dels_cat = torch.cat(dels)
        global_d_abs_99 = float(torch.nanquantile(dels_cat, 0.99))

    with PdfPages(out_pdf_path) as pdf:
        for ex in examples:
            H, W = int(ex["H"]), int(ex["W"])
            #print("t: ", int(ex["t"]))
            t = int(ex.get("t", 0))

            print(f"Plotting example at t={t}")

            # Pull tensors (already CPU from your evaluate)
            gt_t_centers    = ex["centers_t"].to(torch.float32)
            gt_t_feats      = ex["gt_t"].to(torch.float32)
            gt_tp1_centers  = ex["centers_tp1"].to(torch.float32)
            gt_tp1_feats    = ex["gt_tp1"].to(torch.float32)
            pred_centers    = ex["pred_centers"].to(torch.float32)
            pred_levels     = ex["pred_levels"].to(torch.int64)
            pred_parents    = ex["pred_parents"].to(torch.int64)
            pred_feats      = ex["pred_tp1"].to(torch.float32)

            mask_pred_parent = _recover_parent_mask(ex, H, W)

            # Compute deltas inside (keeps main clean)
            deltas = compute_plot_deltas(
                gt_t_centers=gt_t_centers, gt_t_feats=gt_t_feats,
                gt_tp1_centers=gt_tp1_centers, gt_tp1_feats=gt_tp1_feats,
                pred_centers=pred_centers, pred_levels=pred_levels,
                pred_parents=pred_parents, mask_pred=mask_pred_parent,
                pred_feats=pred_feats,
                H=H, W=W, knn_k=knn_k, chunk=chunk,
            )

            # ---- figure ----
            F = gt_t_feats.shape[1]

            for f in range(F):
                fig, ax = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
                fig.suptitle(f"t={t} — {feature_names[f]}", fontsize=12)

                # Top row color limits (per-page or global)
                if unify_clims and global_top_min is not None:
                    tmin, tmax = global_top_min, global_top_max
                else:
                    top_vals = torch.cat([
                        gt_t_feats[:, f].reshape(-1),
                        pred_feats[:, f].reshape(-1),
                        gt_tp1_feats[:, f].reshape(-1),
                    ])
                    tmin = float(torch.nanquantile(top_vals, 0.01))
                    tmax = float(torch.nanquantile(top_vals, 0.99))

                def sym_lims(x):
                    if unify_clims and global_d_abs_99 is not None:
                        a = global_d_abs_99
                    else:
                        a = float(torch.nanquantile(x.abs().reshape(-1), 0.99))
                    return (-a, a)

                d1min, d1max = sym_lims(deltas["delta_gt"][:, f])
                d2min, d2max = sym_lims(deltas["delta_pred_gt"][:, f])
                d3min, d3max = sym_lims(deltas["delta_pred_t"][:, f])

                # --- Top row
                # GT(t)
                pc = _draw_amr_cells(
                    ax[0,0],
                    centers=gt_t_centers,
                    levels=ex.get("level_t", None),
                    values=gt_t_feats,           # (N_t, F)
                    f_idx=f,
                    H=H, W=W, bbox=ex["bbox"],
                    vmin=tmin, vmax=tmax,
                    cmap="viridis",
                    edgecolor="k", linewidth=0.15,
                )
                fig.colorbar(pc, ax=ax[0,0])
                ax[0,0].set_title("GT(t)")

                # Pred(t+1) | M_pred
                pc = _draw_amr_cells(
                    ax[0,1],
                    centers=pred_centers,
                    levels=pred_levels,
                    values=pred_feats,           # (N_pred, F)
                    f_idx=f,
                    H=H, W=W, bbox=ex["bbox"],
                    vmin=tmin, vmax=tmax,
                    cmap="viridis",
                    edgecolor="k", linewidth=0.15,
                )
                fig.colorbar(pc, ax=ax[0,1])
                ax[0,1].set_title("Pred(t+1) | M_pred")

                # GT(t+1)
                pc = _draw_amr_cells(
                    ax[0,2],
                    centers=gt_tp1_centers,
                    levels=ex.get("level_tp1", None),
                    values=gt_tp1_feats,         # (N_tp1, F)
                    f_idx=f,
                    H=H, W=W, bbox=ex["bbox"],
                    vmin=tmin, vmax=tmax,
                    cmap="viridis",
                    edgecolor="k", linewidth=0.15,
                )
                fig.colorbar(pc, ax=ax[0,2])
                ax[0,2].set_title("GT(t+1)")

                vals_gt,    centers_gt,    levels_gt    = _unwrap_delta(
                    deltas.get("delta_gt"),
                    fallback_centers=gt_tp1_centers,
                    fallback_levels=ex.get("level_tp1", None),
                )
                vals_pgt,   centers_pgt,   levels_pgt   = _unwrap_delta(
                    deltas.get("delta_pred_gt"),
                    fallback_centers=pred_centers,
                    fallback_levels=pred_levels,
                )
                vals_pt,    centers_pt,    levels_pt    = _unwrap_delta(
                    deltas.get("delta_pred_t"),
                    fallback_centers=pred_centers,
                    fallback_levels=pred_levels,
                )

                # Δ₁ = GT(t+1) − GT(t)  (on GT(t+1))
                pc = _draw_amr_cells(
                    ax[1,0],
                    centers=centers_gt,
                    levels=levels_gt,
                    values=vals_gt,
                    f_idx=_fidx_or_none(vals_gt, f),
                    H=H, W=W, bbox=ex["bbox"],
                    vmin=d1min, vmax=d1max,
                    cmap="coolwarm",
                    edgecolor="k", linewidth=0.15,
                )
                fig.colorbar(pc, ax=ax[1,0])
                ax[1,0].set_title("Δ₁ = GT(t+1) − GT(t)  (on GT(t+1))")

                # Δ₂ = Pred(t+1) − GT(t+1)  (on Pred)
                pc = _draw_amr_cells(
                    ax[1,1],
                    centers=centers_pgt,
                    levels=levels_pgt,
                    values=vals_pgt,
                    f_idx=_fidx_or_none(vals_pgt, f),
                    H=H, W=W, bbox=ex["bbox"],
                    vmin=d1min, vmax=d1max,
                    cmap="coolwarm",
                    edgecolor="k", linewidth=0.15,
                )
                fig.colorbar(pc, ax=ax[1,1])
                ax[1,1].set_title("Δ₂ = Pred(t+1) − GT(t+1)  (on Pred)")

                # Δ₃ = Pred(t+1) − GT(t)  (on Pred)
                pc = _draw_amr_cells(
                    ax[1,2],
                    centers=centers_pt,
                    levels=levels_pt,
                    values=vals_pt,
                    f_idx=_fidx_or_none(vals_pt, f),
                    H=H, W=W, bbox=ex["bbox"],
                    vmin=d1min, vmax=d1max,
                    cmap="coolwarm",
                    edgecolor="k", linewidth=0.15,
                )
                fig.colorbar(pc, ax=ax[1,2])
                ax[1,2].set_title("Δ₃ = Pred(t+1) − GT(t)  (on Pred)")

                pdf.savefig(fig)
                plt.close(fig)

@torch.no_grad()
def plot_qual_pdf(
    examples: list[dict],
    cfg: dict,
    out_pdf_path: str,
    feature_names: list[str] | None = None,
    unify_clims: bool = False,
    dpi: int = 150,
    rasterize: bool = True,
    colorbars: str = "row",  # "row" (2 total), "none", or "each" (slow)
):
    """
    Fast qualitative PDF generator intended for routine use.

    Each page (per example, per feature):
      top row    = [GT(t), Pred(t+1), GT(t+1)]
      bottom row = [Δ₁=GT(t+1)-GT(t) on GT(t+1),
                    Δ₂=Pred(t+1)-GT(t+1) on Pred,
                    Δ₃=Pred(t+1)-GT(t)   on Pred]

    Key speedups vs plot_qual_2x3_pdf_with_deltas:
      - Precompute cell polygons once per mesh per example (via _build_amr_verts_np).
      - Use PolyCollection-based renderer (via _draw_amr_cells_fast) rather than per-cell Rectangle objects.
      - Optionally rasterize the heavy artists before writing to PDF.
      - Default to 2 colorbars per page (one for top row, one for bottom row).

    Assumes you have already added helper functions:
      - _build_amr_verts_np(centers_np, levels_np, H, W, bbox) -> verts (N,4,2)
      - _draw_amr_cells_fast(ax, verts, vals_1d, bbox, vmin, vmax, cmap=...)
      - compute_plot_deltas(...)
      - _recover_parent_mask(ex, H, W)
      - _unwrap_delta(entry, fallback_centers, fallback_levels)
      - _fidx_or_none(vals, f)   (optional; used if your deltas may return 1D arrays)
    """

    if not examples:
        raise ValueError("examples is empty; nothing to plot.")

    speed = cfg.get("speed", {})
    knn_k = int(cfg.get("loss", {}).get("interp_k", speed.get("knn_k", 8)))
    chunk = int(speed.get("interp_chunk", 8192))

    # Feature naming
    if feature_names is None:
        F0 = int(examples[0]["gt_t"].shape[1])
        feature_names = [f"Feat {i}" for i in range(F0)]

    # Optional global color limits
    global_top_min = global_top_max = None
    global_d_abs_99 = None

    if unify_clims:
        tops = []
        dels = []
        for ex in examples:
            gt_t_feats   = ex["gt_t"].to(torch.float32)
            pred_feats   = ex["pred_tp1"].to(torch.float32)
            gt_tp1_feats = ex["gt_tp1"].to(torch.float32)

            tops.append(torch.cat([
                gt_t_feats.reshape(-1),
                pred_feats.reshape(-1),
                gt_tp1_feats.reshape(-1),
            ]))

            H, W = int(ex["H"]), int(ex["W"])
            mask_pred_parent = _recover_parent_mask(ex, H, W)

            deltas = compute_plot_deltas(
                gt_t_centers=ex["centers_t"].to(torch.float32),
                gt_t_feats=gt_t_feats,
                gt_tp1_centers=ex["centers_tp1"].to(torch.float32),
                gt_tp1_feats=gt_tp1_feats,
                pred_centers=ex["pred_centers"].to(torch.float32),
                pred_levels=ex["pred_levels"].to(torch.int64),
                pred_parents=ex["pred_parents"].to(torch.int64),
                mask_pred=mask_pred_parent,
                pred_feats=pred_feats,
                H=H, W=W, knn_k=knn_k, chunk=chunk,
            )

            # Collect abs deltas for symmetric limits
            dels.append(torch.cat([
                deltas["delta_gt"].abs().reshape(-1),
                deltas["delta_pred_gt"].abs().reshape(-1),
                deltas["delta_pred_t"].abs().reshape(-1),
            ]))

        tops_cat = torch.cat(tops)
        global_top_min = float(torch.nanquantile(tops_cat, 0.01))
        global_top_max = float(torch.nanquantile(tops_cat, 0.99))

        dels_cat = torch.cat(dels)
        global_d_abs_99 = float(torch.nanquantile(dels_cat, 0.99))

    def _sym_lims_from_abs99(abs99: float) -> tuple[float, float]:
        a = float(abs99)
        if not np.isfinite(a) or a <= 0.0:
            a = 1.0
        return (-a, a)

    with PdfPages(out_pdf_path) as pdf:
        for ex in examples:
            H, W = int(ex["H"]), int(ex["W"])
            t = int(ex.get("t", 0))
            bbox = ex["bbox"]

            print(f"[PLOT] example t={t}")

            # --- Pull tensors once (CPU/numpy conversion once) ---
            gt_t_centers_t   = ex["centers_t"].to(torch.float32)
            gt_t_feats_t     = ex["gt_t"].to(torch.float32)
            gt_tp1_centers_t = ex["centers_tp1"].to(torch.float32)
            gt_tp1_feats_t   = ex["gt_tp1"].to(torch.float32)

            pred_centers_t   = ex["pred_centers"].to(torch.float32)
            pred_levels_t    = ex["pred_levels"].to(torch.int64)
            pred_parents_t   = ex["pred_parents"].to(torch.int64)
            pred_feats_t     = ex["pred_tp1"].to(torch.float32)

            # Levels for GT meshes may or may not exist in ex
            level_t_t = ex.get("level_t", None)
            if torch.is_tensor(level_t_t):
                level_t_np = level_t_t.to(torch.int64).cpu().numpy()
            else:
                level_t_np = None

            level_tp1_t = ex.get("level_tp1", None)
            if torch.is_tensor(level_tp1_t):
                level_tp1_np = level_tp1_t.to(torch.int64).cpu().numpy()
            else:
                level_tp1_np = None

            # Convert centers/levels to numpy once
            gt_t_xy_np   = gt_t_centers_t.cpu().numpy()
            gt_tp1_xy_np = gt_tp1_centers_t.cpu().numpy()
            pred_xy_np   = pred_centers_t.cpu().numpy()
            pred_lv_np   = pred_levels_t.cpu().numpy()

            # Build verts once per mesh (this is the big win)
            print("  Building AMR verts...")
            gt_t_verts   = _build_amr_verts_np(gt_t_xy_np, level_t_np,   H, W, bbox)
            gt_tp1_verts = _build_amr_verts_np(gt_tp1_xy_np, level_tp1_np, H, W, bbox)
            pred_verts   = _build_amr_verts_np(pred_xy_np, pred_lv_np,   H, W, bbox)

            # Recover predicted parent mask
            print("  Recovering predicted parent mask...")
            mask_pred_parent = _recover_parent_mask(ex, H, W)

            # Compute deltas once per example
            print("  Computing deltas...")
            deltas = compute_plot_deltas(
                gt_t_centers=gt_t_centers_t,
                gt_t_feats=gt_t_feats_t,
                gt_tp1_centers=gt_tp1_centers_t,
                gt_tp1_feats=gt_tp1_feats_t,
                pred_centers=pred_centers_t,
                pred_levels=pred_levels_t,
                pred_parents=pred_parents_t,
                mask_pred=mask_pred_parent,
                pred_feats=pred_feats_t,
                H=H, W=W, knn_k=knn_k, chunk=chunk,
            )

            # Unwrap (supports legacy dict/tensor style if you use it elsewhere)
            print("  Unwrapping deltas...")
            vals_gt,  _, _ = _unwrap_delta(deltas.get("delta_gt"),      gt_tp1_centers_t, level_tp1_t)
            vals_pgt, _, _ = _unwrap_delta(deltas.get("delta_pred_gt"), pred_centers_t,   pred_levels_t)
            vals_pt,  _, _ = _unwrap_delta(deltas.get("delta_pred_t"),  pred_centers_t,   pred_levels_t)

            # Convert feature arrays to numpy once
            gt_t_feats_np   = gt_t_feats_t.cpu().numpy()
            pred_feats_np   = pred_feats_t.cpu().numpy()
            gt_tp1_feats_np = gt_tp1_feats_t.cpu().numpy()

            # Note: delta arrays may be (N,F) or (N,)
            vals_gt_np  = vals_gt.cpu().numpy()
            vals_pgt_np = vals_pgt.cpu().numpy()
            vals_pt_np  = vals_pt.cpu().numpy()

            F = int(gt_t_feats_np.shape[1])

            print("  Plotting features...")
            for f in range(F):
                fig, ax = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
                fig.suptitle(f"t={t} — {feature_names[f] if f < len(feature_names) else f'Feat {f}'}", fontsize=12)

                # --- Top row color limits ---
                if unify_clims and global_top_min is not None:
                    tmin, tmax = global_top_min, global_top_max
                else:
                    top_vals = np.concatenate([
                        gt_t_feats_np[:, f].reshape(-1),
                        pred_feats_np[:, f].reshape(-1),
                        gt_tp1_feats_np[:, f].reshape(-1),
                    ])
                    # robust quantiles
                    tmin = float(np.nanquantile(top_vals, 0.01))
                    tmax = float(np.nanquantile(top_vals, 0.99))
                    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmin == tmax:
                        tmin, tmax = float(np.nanmin(top_vals)), float(np.nanmax(top_vals) + 1e-12)

                # --- Delta symmetric limits ---
                if unify_clims and global_d_abs_99 is not None:
                    dmin, dmax = _sym_lims_from_abs99(global_d_abs_99)
                else:
                    # Use Δ₁ scale (common for all three deltas on this page/feature)
                    if vals_gt_np.ndim == 2:
                        base = vals_gt_np[:, f]
                    else:
                        base = vals_gt_np
                    a = float(np.nanquantile(np.abs(base).reshape(-1), 0.99))
                    dmin, dmax = _sym_lims_from_abs99(a)

                # --- Draw top row ---
                pc00 = _draw_amr_cells_fast(ax[0, 0], gt_t_verts,   gt_t_feats_np[:, f],   bbox, tmin, tmax, cmap="viridis")
                pc01 = _draw_amr_cells_fast(ax[0, 1], pred_verts,   pred_feats_np[:, f],   bbox, tmin, tmax, cmap="viridis")
                pc02 = _draw_amr_cells_fast(ax[0, 2], gt_tp1_verts, gt_tp1_feats_np[:, f], bbox, tmin, tmax, cmap="viridis")

                ax[0, 0].set_title("GT(t)")
                ax[0, 1].set_title("Pred(t+1) | M_pred")
                ax[0, 2].set_title("GT(t+1)")

                # --- Draw bottom row ---
                if vals_gt_np.ndim == 2:
                    d1 = vals_gt_np[:, f]
                    d2 = vals_pgt_np[:, f] if vals_pgt_np.ndim == 2 else vals_pgt_np
                    d3 = vals_pt_np[:, f]  if vals_pt_np.ndim == 2  else vals_pt_np
                else:
                    # 1D (rare)
                    d1, d2, d3 = vals_gt_np, vals_pgt_np, vals_pt_np

                pc10 = _draw_amr_cells_fast(ax[1, 0], gt_tp1_verts, d1, bbox, dmin, dmax, cmap="coolwarm")
                pc11 = _draw_amr_cells_fast(ax[1, 1], pred_verts,   d2, bbox, dmin, dmax, cmap="coolwarm")
                pc12 = _draw_amr_cells_fast(ax[1, 2], pred_verts,   d3, bbox, dmin, dmax, cmap="coolwarm")

                ax[1, 0].set_title("Δ₁ = GT(t+1) − GT(t)  (on GT(t+1))")
                ax[1, 1].set_title("Δ₂ = Pred(t+1) − GT(t+1)  (on Pred)")
                ax[1, 2].set_title("Δ₃ = Pred(t+1) − GT(t)  (on Pred)")

                # Rasterize heavy artists before PDF write (major speedup)
                if rasterize:
                    for pc in (pc00, pc01, pc02, pc10, pc11, pc12):
                        pc.set_rasterized(True)

                # Colorbars (default: 2 total per page)
                if colorbars == "each":
                    fig.colorbar(pc00, ax=ax[0, 0])
                    fig.colorbar(pc01, ax=ax[0, 1])
                    fig.colorbar(pc02, ax=ax[0, 2])
                    fig.colorbar(pc10, ax=ax[1, 0])
                    fig.colorbar(pc11, ax=ax[1, 1])
                    fig.colorbar(pc12, ax=ax[1, 2])
                elif colorbars == "row":
                    # One for top row, one for bottom row
                    fig.colorbar(pc02, ax=ax[0, :], shrink=0.95, pad=0.01)
                    fig.colorbar(pc12, ax=ax[1, :], shrink=0.95, pad=0.01)
                elif colorbars == "none":
                    pass
                else:
                    raise ValueError("colorbars must be one of: 'row', 'none', 'each'")

                pdf.savefig(fig, dpi=dpi)
                plt.close(fig)


def _recover_parent_mask(ex: dict, H: int, W: int) -> torch.Tensor:
    """
    Retrieve a parent-level (H,W) bool mask for the predicted mesh from an example dict.
    Falls back to all-True if not present in a convenient form.
    """
    mp = ex.get("mask_pred", None)
    if isinstance(mp, dict):
        m = mp.get(1, None)
        if m is None:
            return torch.ones((H, W), dtype=torch.bool)
        return m.to(torch.bool).view(H, W)
    if torch.is_tensor(mp):
        return mp.to(torch.bool).view(H, W)
    return torch.ones((H, W), dtype=torch.bool)
