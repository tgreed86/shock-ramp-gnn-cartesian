# amr_policy.py
# -----------------------------------------------------------------------------
# Deterministic (non-ML) AMR policy utilities for multi-level meshes.
#
# Exposed API (used by train_mesh_first.py / rollout_eval.py):
#   - build_multilevel_masks_from_gradients(x_t, masks_t, H, W, dx, dy, L_max, cfg)
#   - dynamic_cells_from_masks(masks_by_level, H, W, dx, dy)
#   - map_features_to_mesh(src, dst, H, W, mode="standard", k=8)
#   - map_pred_to_gt_mesh(pred, gt, mode="standard", k=8)
#
# Notes:
# * "masks_by_level" is a dict {l: bool[H*2**l, W*2**l]} for l=0..L_max.
#   We assume mask_0 is the full coarse grid (all True). If missing, we add it.
# * "leaf cells" = cells that have no refined children. The graph is built on
#   the leaf set. Edges are face-adjacency across same level and cross-level
#   seams (computed via a finest-grid occupancy sweep).
# * Gradients are computed on downsampled representations derived from a finest
#   rasterization of x_t (piecewise-constant per leaf cell), then mean-pooled
#   to each level. Refinement is top-down using thresholds per level.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------- small helpers / config -------------------------- #

def _get(cfg: dict, path: str, default=None):
    node = cfg
    for p in path.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def _get_refine_ratio(cfg: dict) -> int:
    pol = cfg.get("policy", {}) or {}
    mesh = cfg.get("mesh", {}) or {}
    rr = pol.get("refine_ratio", mesh.get("refine_ratio", 2))
    rr = int(rr)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {rr}")
    return rr


def _as_list_or_repeat(x, L_max: int):
    if isinstance(x, (list, tuple)):
        if len(x) == L_max:
            return list(x)
        raise ValueError(f"Expected list length {L_max}, got {len(x)}")
    # scalar -> repeat
    return [float(x)] * L_max


def _ensure_mask0(masks_by_level: Dict[int, torch.Tensor], H: int, W: int, device: torch.device):
    if 0 not in masks_by_level or masks_by_level[0] is None:
        masks_by_level[0] = torch.ones((H, W), dtype=torch.bool, device=device)
    return masks_by_level


def _device_of_masks(masks_by_level: Dict[int, torch.Tensor]) -> torch.device:
    for v in masks_by_level.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")

def _get_bbox(cfg: Dict[str,Any]) -> Tuple[float,float,float,float]:
    if cfg.get("data", {}).get("bbox"):
        xmin, xmax, ymin, ymax = cfg["data"]["bbox"]
    else:
        dom = cfg.get("domain", {})
        xmin = float(dom.get("xmin", 0.0)); xmax = float(dom.get("xmax", 1.0))
        ymin = float(dom.get("ymin", 0.0)); ymax = float(dom.get("ymax", 1.0))
    return xmin, xmax, ymin, ymax

def _resolve_channel_indices(cfg: dict, Fdim: int) -> torch.Tensor:
    """
    Decide which feature channels to use when combining gradients.

    Looks for cfg["policy"]["grad_channels"], which should be either:
      - None / missing  -> use all channels [0..Fdim-1]
      - a single int    -> use [that] (clipped to valid range)
      - a list of ints  -> use those, filtered to [0..Fdim-1]

    Returns a 1D LongTensor of indices.
    """
    pol = cfg.get("policy", {})
    ch = pol.get("grad_channels", None)

    if ch is None:
        # default: all channels
        return torch.arange(Fdim, dtype=torch.long)

    # normalise to list of ints
    if isinstance(ch, (int, np.integer)):
        ch_list = [int(ch)]
    else:
        ch_list = [int(c) for c in ch]

    idx = torch.as_tensor(ch_list, dtype=torch.long)
    # keep only valid indices
    mask = (idx >= 0) & (idx < Fdim)
    idx = idx[mask]

    if idx.numel() == 0:
        # safety fallback: all channels
        return torch.arange(Fdim, dtype=torch.long)

    return idx

def _parents_from_level_ij(levels: torch.Tensor,
                           ij: torch.Tensor,
                           H: int, W: int,
                           refine_ratio: int = 2) -> torch.Tensor:
    """
    Map each level-l cell with indices (i,j) at its own resolution
    (H*refine_ratio^l, W*refine_ratio^l)
    to its coarse parent index in a flattened (H*W) coarse grid.

    Works for arbitrary l >= 0. Assumes ij are integer cell indices at the cell's level.
    """
    if levels.dim() != 1:
        levels = levels.view(-1)
    i = ij[:, 0].long()
    j = ij[:, 1].long()

    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    # scale = refine_ratio^level, elementwise
    scale = torch.pow(torch.tensor(rr, device=levels.device, dtype=torch.long), levels)
    # integer floor division to get coarse parent row/col
    row = torch.div(i, scale, rounding_mode='floor').clamp_(0, H - 1)
    col = torch.div(j, scale, rounding_mode='floor').clamp_(0, W - 1)

    parents = row * W + col
    return parents.long()

def _sample_id_from_batch(batch: Dict[str, torch.Tensor]) -> int | None:
    for k in ["pid", "uid", "seq", "t"]:
        if k in batch and torch.is_tensor(batch[k]):
            try:
                return int(batch[k].item())
            except Exception:
                pass
    return None

def _normalize_level_ij(level: torch.Tensor, ij: torch.Tensor,
                        H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure level:[N] (long) and ij:[N,2] (long). Accepts inputs with shapes:
      - level: [N], [1,N], [N,1], [1], scalar
      - ij: [N,2], [1,N,2], [2,N], [1,2,N]
    If level has the wrong length, reconstruct per-cell levels from ij and (H,W).
    """
    level = torch.as_tensor(level)
    ij = torch.as_tensor(ij)

    # Squeeze leading singleton batch dims
    if level.ndim >= 2 and level.shape[0] == 1:
        level = level.squeeze(0)
    if ij.ndim >= 3 and ij.shape[0] == 1:
        ij = ij.squeeze(0)

    # Fix ij axes if transposed
    if ij.ndim == 2 and ij.shape[0] == 2 and ij.shape[1] != 2:
        ij = ij.transpose(0, 1)  # (2,N) -> (N,2)
    elif ij.ndim == 3 and ij.shape[0] == 2 and ij.shape[1] != 2 and ij.shape[2] == 1:
        ij = ij.squeeze(-1).transpose(0, 1)

    # If ij still not [...,2], but ends with 2, flatten all but last
    if ij.ndim > 2 and ij.shape[-1] == 2:
        ij = ij.reshape(-1, 2)

    if ij.ndim != 2 or ij.shape[1] != 2:
        raise ValueError(f"ij must be [N,2] after normalization, got {tuple(ij.shape)}")

    N = ij.shape[0]

    # Flatten level and fix length
    level = level.view(-1)
    if level.numel() != N:
        # Reconstruct per-cell level from ij and (H,W)
        i = ij[:, 0].to(torch.float32)
        j = ij[:, 1].to(torch.float32)
        Li = torch.ceil(torch.log2(torch.clamp((i + 1.0) / float(W), min=1.0))).to(torch.long)
        Lj = torch.ceil(torch.log2(torch.clamp((j + 1.0) / float(H), min=1.0))).to(torch.long)
        level = torch.maximum(Li, Lj)

    return level.to(torch.long), ij.to(torch.long)


def _ensure_leaf_inputs(x: torch.Tensor,
                        level: torch.Tensor,
                        ij: torch.Tensor,
                        H: int, W: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize x:[N,F], level:[N], ij:[N,2].
    Accepts inputs with a leading singleton batch dim: [1,N,*] and flattens it.
    If level isn't [N], reconstruct per-cell levels from ij and (H,W).
    """
    x = torch.as_tensor(x)
    level = torch.as_tensor(level)
    ij = torch.as_tensor(ij)

    # Strip a leading singleton batch dim consistently
    if x.ndim >= 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if level.ndim >= 2 and level.shape[0] == 1:
        level = level.squeeze(0)
    if ij.ndim >= 3 and ij.shape[0] == 1:
        ij = ij.squeeze(0)

    # Ensure x is [N,F]
    if x.ndim == 1:
        x = x.unsqueeze(1)                     # [N] -> [N,1]
    elif x.ndim > 2:
        F = x.shape[-1]
        x = x.reshape(-1, F)                   # flatten all but last

    # Ensure ij is [N,2]
    if ij.ndim != 2 or ij.shape[-1] != 2:
        # if ij is [...,2], flatten all but last
        if ij.ndim > 2 and ij.shape[-1] == 2:
            ij = ij.reshape(-1, 2)
        else:
            raise ValueError(f"ij must end with size 2; got {tuple(ij.shape)}")

    N = x.shape[0]
    if ij.shape[0] != N:
        raise ValueError(f"x and ij length mismatch: {N} vs {ij.shape[0]}")

    # Ensure level is [N]; if not, reconstruct from ij and (H,W)
    level = level.view(-1)
    if level.numel() != N:
        i = ij[:, 0].to(torch.float32)
        j = ij[:, 1].to(torch.float32)
        Li = torch.ceil(torch.log2(torch.clamp((i + 1.0) / float(W), min=1.0))).to(torch.long)
        Lj = torch.ceil(torch.log2(torch.clamp((j + 1.0) / float(H), min=1.0))).to(torch.long)
        level = torch.maximum(Li, Lj)

    # Dtypes
    x = x.to(torch.float32)
    level = level.to(torch.long)
    ij = ij.to(torch.long)
    return x, level, ij


def _to_tensor_2d(x, H, W, device=None, dtype=torch.float32):
    """
    Accepts: torch.Tensor or np.ndarray (flat or 2D) and returns a 2D torch tensor (H,W).
    """
    if isinstance(x, torch.Tensor):
        t = x.detach()
        # if flat, reshape
        if t.numel() == H * W and t.dim() != 2:
            t = t.view(H, W)
        elif t.dim() == 2:
            if t.shape != (H, W):
                t = t.reshape(H, W)
        else:
            t = t.reshape(H, W)
        return t.to(device=device, dtype=dtype, copy=False)

    # numpy path
    arr = np.asarray(x)
    if arr.size == H * W and arr.ndim != 2:
        arr = arr.reshape(H, W)
    elif arr.ndim == 2 and arr.shape != (H, W):
        arr = arr.reshape(H, W)
    else:
        arr = arr.reshape(H, W)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _combine_level(
    per_lvl: torch.Tensor,
    use_cols: torch.Tensor,
    combine: str = "l2",
    w: torch.Tensor | None = None,
) -> torch.Tensor:
        """
        per_lvl: (h, w, F) tensor of per-feature |∇f_k| (CPU or device); may contain NaNs.
        returns: (h, w) combined map (same device as input).
        """
        dev_local = per_lvl.device
        sel = per_lvl.index_select(-1, use_cols.to(dev_local))
        sel = torch.nan_to_num(sel, nan=0.0)  # be robust to NaNs
        if combine == "max":
            return sel.max(dim=2).values
        elif combine in ("sum", "weighted_sum"):
            ww = (w.to(dev_local) if w is not None else torch.ones(sel.shape[-1], device=dev_local, dtype=sel.dtype))
            return (sel * ww.view(1, 1, -1)).sum(dim=2)
        else:  # "l2" or "weighted_l2"
            ww = (w.to(dev_local) if w is not None else torch.ones(sel.shape[-1], device=dev_local, dtype=sel.dtype))
            return torch.sqrt(((sel * ww.view(1, 1, -1)) ** 2).sum(dim=2))


@torch.no_grad()
def compute_hierarchical_gradients_from_gt_raster(
    batch: dict,
    cfg: dict,
    H: int, W: int,
    dx0: float, dy0: float,
    *,
    feature_idx=None, feature_names=None
):
    """
    Fast vectorized gradient backend:
      1) Compose selected channels to finest grid Hf×Wf (Lmax).
      2) ∇ via conv2d (central-diff) on GPU if available.
      3) L2 combine per-channel to (Hf,Wf); then max-pool down to each level.

    Returns:
      {
        "grad_by_level":     {L: (h,w) },
        "grad_feat_by_level":{L: (h,w,C)}
      }
    """
    from plots import amr_composite_to_finest_grid 

    dev = next((v.device for v in batch.values() if torch.is_tensor(v)), torch.device("cpu"))

    pol    = cfg.get("policy", {})
    Lmax   = int(pol.get("max_level", 2))
    rr = _get_refine_ratio(cfg)
    xmin, xmax, ymin, ymax = cfg["data"]["bbox"]

    # Resolve channels
    """
    F_total = batch["dyn_feat_t"].shape[-1]
    if feature_idx is not None:
        use_cols = torch.as_tensor(list(feature_idx), dtype=torch.long)
    elif feature_names is not None:
        chmap = cfg.get("channels", {})
        use_cols = torch.as_tensor([chmap[nm] for nm in feature_names], dtype=torch.long)
    else:
        use_cols = torch.arange(F_total, dtype=torch.long)
    use_cols = use_cols[(use_cols >= 0) & (use_cols < F_total)]
    C = int(use_cols.numel())
    if C == 0: raise ValueError("No valid feature channels.")
    """
    # resolve channels (use same selection as policy)
    F_total = batch["dyn_feat_t"].shape[-1]
    use_cols = _resolve_channel_indices(cfg, F_total).detach().cpu()
    C = int(use_cols.numel())
    if C == 0:
        raise ValueError("No valid feature channels.")

    # Compose each channel to finest
    imgs = []
    valid2d = None   # geometry mask, same for all channels
    for c in use_cols.tolist():
        img_flat, valid_flat, Hc, Wc = amr_composite_to_finest_grid(
            batch["centers_t"].detach().cpu(),
            (None if "level_t" not in batch else batch["level_t"].detach().cpu()),
            batch["center_feat_t"][:, [c]].detach().cpu(),
            H, W, (xmin, xmax, ymin, ymax),
            refine_ratio=rr,
        )
        img2d = _to_tensor_2d(
            img_flat, Hc, Wc,
            device=dev,
            dtype=torch.float32,
        )
        imgs.append(img2d)   # (Hf, Wf)
        # Capture validity mask ONCE (geometry-only)
        if valid2d is None:
            valid2d = _to_tensor_2d(
                valid_flat, Hc, Wc,
                device=dev,
                dtype=torch.bool,
            )

    X = torch.stack(imgs, dim=0).to(dev, dtype=torch.float32)  # (C,Hf,Wf)

    # Central-diff conv kernels (replicate padding), scaled by Δ
    dx_f = dx0 / (float(rr) ** Lmax)
    dy_f = dy0 / (float(rr) ** Lmax)
    kx = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], dtype=torch.float32, device=dev).view(1,1,3,3) / dx_f
    ky = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], dtype=torch.float32, device=dev).view(1,1,3,3) / dy_f

    #Xp = F.pad(X.unsqueeze(0), (1,1,1,1), mode="replicate")  # (1,C,Hf+2,Wf+2)
    # X is (C,Hf,Wf). Pad in 3D, then add batch dim -> (1,C,Hf+2,Wf+2)
    try:
        Xp = F.pad(X, (1,1,1,1), mode="replicate").unsqueeze(0)
    except NotImplementedError:
        # Some MPS builds don’t support replicate here; reflect is a close fallback
        Xp = F.pad(X, (1,1,1,1), mode="reflect").unsqueeze(0)

    gx = F.conv2d(Xp, kx.expand(C,1,3,3), groups=C)          # (1,C,Hf,Wf)
    gy = F.conv2d(Xp, ky.expand(C,1,3,3), groups=C)

    stencil_ok = (
        valid2d
        & torch.roll(valid2d,  1, dims=0)
        & torch.roll(valid2d, -1, dims=0)
        & torch.roll(valid2d,  1, dims=1)
        & torch.roll(valid2d, -1, dims=1)
    )
    gx = gx * stencil_ok[None, None]
    gy = gy * stencil_ok[None, None]

    gfeat_fine = torch.sqrt(torch.clamp(gx*gx + gy*gy, min=0.0))[0]  # (C,Hf,Wf)

    # Return both per-feature and combined per level
    grad_feat_by_level, grad_by_level = {}, {}
    combine = str(cfg.get("policy", {}).get("combine", "l2")).lower()
    raw_w = cfg.get("policy", {}).get("refine_weights", None)
    if raw_w is not None:
        w = torch.as_tensor(raw_w, device=dev, dtype=torch.float32)
        if w.numel() != C:
            w = (torch.cat([w, w.new_ones(C - w.numel())]) if w.numel() < C else w[:C])
    else:
        w = None

    def _combine(stack):  # stack: (C,h,w)
        if combine == "max":
            return stack.max(dim=0).values
        if combine in ("sum", "weighted_sum"):
            return (stack if w is None else (w[:,None,None]*stack)).sum(dim=0)
        # l2 / weighted_l2
        if w is None:
            return torch.sqrt((stack**2).sum(dim=0))
        return torch.sqrt(((w[:,None,None]*stack)**2).sum(dim=0))

    # Finest level = Lmax
    grad_feat_by_level[Lmax] = gfeat_fine.permute(1,2,0).contiguous()  # (Hf,Wf,C)
    grad_by_level[Lmax]      = _combine(gfeat_fine)

    # Down-pool to parents
    for L in range(Lmax-1, -1, -1):
        scale = rr ** (Lmax - L)
        # max-pool per channel to preserve sharpest gradients
        pooled = F.max_pool2d(gfeat_fine, kernel_size=scale, stride=scale)  # (C,h,w)
        grad_feat_by_level[L] = pooled.permute(1,2,0).contiguous()
        grad_by_level[L]      = _combine(pooled)

    return {"grad_by_level": grad_by_level, "grad_feat_by_level": grad_feat_by_level}

@torch.no_grad()
def predict_masks_hierarchical_from_gt_gradients(
    batch: dict,
    cfg: dict,
    H: int,
    W: int,
    dx_unused: float,
    dy_unused: float,
    *,
    device=None,
    debug_out: dict | None = None,
    timing_out: dict | None = None,
) -> dict[int, torch.Tensor]:
    t_all_t0 = time.perf_counter()
    timing: Dict[str, float] = {
        "predict_grad_raster_s": 0.0,
        "predict_combine_levels_s": 0.0,
        "predict_pool_up_s": 0.0,
        "predict_prev_masks_s": 0.0,
        "predict_threshold_hysteresis_s": 0.0,
        "predict_dilation_s": 0.0,
        "predict_normalize_masks_s": 0.0,
        "predict_debug_out_s": 0.0,
        "predict_legacy_policy_s": 0.0,
    }
    dev = device or next((v.device for v in batch.values() if torch.is_tensor(v)), torch.device("cpu"))

    pol          = cfg.get("policy", {})
    rr           = _get_refine_ratio(cfg)
    tau_by_level = pol.get("tau_by_level", None)
    tau_low_def  = float(pol.get("tau_low",  0.02))
    tau_high_def = float(pol.get("tau_high", 0.03))

    mode = str(pol.get("hysteresis_mode", "absolute")).lower()  # "absolute" or "percentile"
    pct_low_default  = float(pol.get("percentile_low",  75.0))
    pct_high_default = float(pol.get("percentile_high", 90.0))
    pct_by_level     = pol.get("percentiles_by_level", {})
    pct_mode = "auto"
    if isinstance(pct_by_level, dict):
        pct_mode = str(pct_by_level.get("selection", "auto")).strip().lower()
    if pct_mode not in ("auto", "global", "per_level", "per-level", "level", "levels"):
        raise ValueError(
            "policy.percentiles_by_level.selection must be one of: "
            "auto, global, per_level"
        )
    use_global_pct = (pct_mode == "global")
    use_level_pct = (pct_mode in ("per_level", "per-level", "level", "levels"))
    dbg_thr = bool(pol.get("debug_thresholds", False))

    def _pct_level_entry(L: int):
        if not isinstance(pct_by_level, dict):
            return None
        if L in pct_by_level:
            return pct_by_level[L]
        s = str(L)
        if s in pct_by_level:
            return pct_by_level[s]
        return None

    def _taus(L: int) -> tuple[float, float]:
        if isinstance(tau_by_level, dict) and L in tau_by_level:
            tl = tau_by_level[L]
            if isinstance(tl, dict):
                return float(tl.get("low", tau_low_def)), float(tl.get("high", tau_high_def))
            else:
                return (tau_low_def, float(tl))
        return (tau_low_def, tau_high_def)

    xmin, xmax, ymin, ymax = cfg["data"]["bbox"]
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    # --- compute grads (on the same path used by the policy) ---
    t_grad_t0 = time.perf_counter()
    grads = compute_hierarchical_gradients_from_gt_raster(
        batch, cfg, H, W, dx0, dy0,
        feature_idx=None,
        feature_names=None,
    )
    timing["predict_grad_raster_s"] = float(time.perf_counter() - t_grad_t0)

    # --- per-feature combine -> single (h,w) per level ---
    t_combine_t0 = time.perf_counter()
    combine = str(cfg.get("policy", {}).get("combine", "l2")).lower()
    _anyL = next(iter(grads["grad_feat_by_level"].keys()))
    Fdim = grads["grad_feat_by_level"][_anyL].shape[-1]
    use_cols = _resolve_channel_indices(cfg, Fdim)

    raw_w = cfg.get("policy", {}).get("refine_weights", None)
    if raw_w is not None:
        w = torch.as_tensor(raw_w, dtype=torch.float32)
        if w.numel() != use_cols.numel():
            if w.numel() < use_cols.numel():
                w = torch.cat([w, w.new_ones(use_cols.numel() - w.numel())])
            else:
                w = w[:use_cols.numel()]
    else:
        w = None

    # G[L] : (h_L, w_L) combined gradient magnitude per level
    G = {
        L: _combine_level(
            torch.nan_to_num(grads["grad_feat_by_level"][L], nan=0.0),
            use_cols=use_cols,
            combine=combine,
            w=w,
        )
        for L in grads["grad_feat_by_level"]
    }
    timing["predict_combine_levels_s"] = float(time.perf_counter() - t_combine_t0)

    # --- pooled-up max propagation (level-agnostic) ---
    t_pool_t0 = time.perf_counter()
    pooled_up = {L: G[L].clone() for L in G}
    maxL = max(G.keys())
    for L in range(maxL - 1, -1, -1):
        for Lc in range(L + 1, maxL + 1):
            scale = rr ** (Lc - L)
            pooled = torch.nn.functional.max_pool2d(G[Lc][None, None], kernel_size=scale, stride=scale)[0, 0]
            pooled_up[L] = torch.maximum(pooled_up[L], pooled)
    timing["predict_pool_up_s"] = float(time.perf_counter() - t_pool_t0)

    # Reconstruct previous refine masks per level from current mesh state.
    # prev_refine_by_level[L] lives on the parent grid for level L (shape H*rr^(L-1), W*rr^(L-1)).
    t_prev_masks_t0 = time.perf_counter()
    prev_refine_by_level: Dict[int, torch.Tensor] = {}
    # Hysteresis previous-state source:
    # - default uses current mesh state keys (level_t/ij_t)
    # - optional override keys (prev_level_t/prev_ij_t) allow callers to
    #   stabilize refinement with a different previous mesh state while still
    #   composing gradients from GT(t).
    level_t_raw = batch.get("prev_level_t", batch.get("level_t", None))
    ij_t_raw = batch.get("prev_ij_t", batch.get("ij_t", None))
    if level_t_raw is not None and ij_t_raw is not None:
        try:
            lv_prev = torch.as_tensor(level_t_raw, device=dev).view(-1).long()
            ij_prev = torch.as_tensor(ij_t_raw, device=dev)

            if ij_prev.ndim >= 3 and ij_prev.shape[0] == 1:
                ij_prev = ij_prev.squeeze(0)
            if ij_prev.ndim == 2 and ij_prev.shape[0] == 2 and ij_prev.shape[1] != 2:
                ij_prev = ij_prev.transpose(0, 1)
            if ij_prev.ndim > 2 and ij_prev.shape[-1] == 2:
                ij_prev = ij_prev.reshape(-1, 2)

            if ij_prev.ndim == 2 and ij_prev.shape[1] == 2 and ij_prev.shape[0] == lv_prev.numel():
                ij_prev = ij_prev.to(torch.long)
                for L in range(1, maxL + 1):
                    h_p = int(H * (rr ** (L - 1)))
                    w_p = int(W * (rr ** (L - 1)))
                    prev_mask_L = torch.zeros((h_p, w_p), dtype=torch.bool, device=dev)

                    sel = (lv_prev >= int(L))
                    if bool(sel.any()):
                        rel_lv = (lv_prev[sel] - int(L - 1)).to(torch.long)
                        parent_flat = _parents_from_level_ij(
                            rel_lv,
                            ij_prev[sel],
                            h_p,
                            w_p,
                            refine_ratio=rr,
                        ).view(-1).long()
                        if parent_flat.numel() > 0:
                            parent_flat = parent_flat.clamp_(0, h_p * w_p - 1)
                            prev_mask_L.view(-1)[parent_flat] = True
                    prev_refine_by_level[L] = prev_mask_L
            elif dbg_thr:
                print("[THR] warning: level_t/ij_t shape mismatch; falling back to L1 mask_t hysteresis only.")
        except Exception as e:
            if dbg_thr:
                print(f"[THR] warning: failed to build per-level previous masks ({e}); using fallback.")
    timing["predict_prev_masks_s"] = float(time.perf_counter() - t_prev_masks_t0)

    # --- thresholding with absolute/percentile hysteresis (unchanged logic) ---
    t_thr_t0 = time.perf_counter()
    dilation_s = 0.0
    masks_by_level = {}
    for L in range(1, maxL + 1):
        parent = L - 1
        Gparent = pooled_up[parent].to(dev)

        if mode == "percentile":
            if use_global_pct:
                p_low, p_high = pct_low_default, pct_high_default
            else:
                pL = _pct_level_entry(L)
                if use_level_pct and (pL is None):
                    p_low, p_high = pct_low_default, pct_high_default
                elif isinstance(pL, dict):
                    p_low = float(pL.get("low", pct_low_default))
                    p_high = float(pL.get("high", pct_high_default))
                else:
                    p_low, p_high = pct_low_default, pct_high_default
            p_low  = max(0.0, min(100.0, p_low))
            p_high = max(0.0, min(100.0, p_high))
            if p_low > p_high: p_low, p_high = p_high, p_low

            finite = torch.isfinite(Gparent)
            if finite.any():
                thr_low  = torch.quantile(Gparent[finite], p_low  / 100.0)
                thr_high = torch.quantile(Gparent[finite], p_high / 100.0)
            else:
                thr_low  = torch.tensor(float("inf"), device=Gparent.device)
                thr_high = torch.tensor(float("inf"), device=Gparent.device)
        else:
            tau_low, tau_high = _taus(L)
            thr_low  = torch.as_tensor(tau_low,  device=Gparent.device, dtype=Gparent.dtype)
            thr_high = torch.as_tensor(tau_high, device=Gparent.device, dtype=Gparent.dtype)

        if dbg_thr:
            finite = torch.isfinite(Gparent)
            gmin = Gparent[finite].min().item() if finite.any() else float("nan")
            gmax = Gparent[finite].max().item() if finite.any() else float("nan")
            print(f"[THR] L{L} mode={mode} thr_low={float(thr_low):.3e} thr_high={float(thr_high):.3e} "
                  f"min={gmin:.3e} max={gmax:.3e}")

        h_p, w_p = int(Gparent.shape[0]), int(Gparent.shape[1])
        prev_mask = prev_refine_by_level.get(L, None)

        # Backward-compatible fallback for L1 when richer previous level state is unavailable.
        if prev_mask is None and L == 1:
            prev_flat_raw = batch.get("prev_mask_t", batch.get("mask_t", None))
            if prev_flat_raw is None:
                prev_flat = None
            else:
                prev_flat = torch.as_tensor(prev_flat_raw, device=dev)
        if prev_mask is None and L == 1 and prev_flat is not None:
            H0 = int(cfg["data"].get("H", H))
            W0 = int(cfg["data"].get("W", W))
            if H0 * W0 == prev_flat.numel():
                prev2d = prev_flat.view(H0, W0).float()
            else:
                side = int(round(prev_flat.numel() ** 0.5))
                if side * side == prev_flat.numel():
                    prev2d = prev_flat.view(side, side).float()
                else:
                    prev2d = torch.zeros((h_p, w_p), device=dev, dtype=torch.float32)
            prev_mask = prev2d.bool()

        if prev_mask is not None:
            if prev_mask.shape != (h_p, w_p):
                prev_mask = torch.nn.functional.interpolate(
                    prev_mask.float()[None, None], size=(h_p, w_p), mode="nearest"
                )[0, 0].bool()
            keep = prev_mask & (Gparent > thr_low)
            newr = (~prev_mask) & (Gparent > thr_high)
            M = keep | newr
        else:
            M = (Gparent > thr_high)

        if L > 1:
            allow = torch.nn.functional.interpolate(
                masks_by_level[L - 1].float()[None, None], scale_factor=float(rr), mode="nearest"
            )[0, 0].bool()
            h, w = Gparent.shape
            M = M & allow[:h, :w]

        # Optional: dilation halo around refine mask.
        # Preferred config is cell-based halo:
        #   policy.dilate_cells_L{L} = n_cells
        # Optional convenience for finest level:
        #   policy.dilate_cells_lmax = n_cells (applies only when L==maxL)
        # Backward compatible:
        #   policy.dilate_phys_L{L} = physical radius
        pol_cfg = cfg.get("policy", {}) or {}
        dxL_here = (xmax - xmin) / float(W * (rr ** (L - 1)))

        n_cells = None
        raw_cells = pol_cfg.get(f"dilate_cells_L{L}", None)
        if raw_cells is None and (L == maxL):
            raw_cells = pol_cfg.get("dilate_cells_lmax", None)
        if raw_cells is not None:
            try:
                n_cells = float(raw_cells)
            except Exception:
                n_cells = None

        if n_cells is not None:
            r_phys = float(max(0.0, n_cells) * dxL_here)
        else:
            r_phys = float(pol_cfg.get(f"dilate_phys_L{L}", 0.0))

        if r_phys > 0:
            t_dilate_t0 = time.perf_counter()
            r_cells = max(1, int(round(r_phys / dxL_here)))
            k = 2 * r_cells + 1
            M = torch.nn.functional.max_pool2d(
                M.float()[None, None],
                kernel_size=k,
                stride=1,
                padding=r_cells,
            )[0, 0].bool()
            dilation_s += float(time.perf_counter() - t_dilate_t0)

        masks_by_level[L] = M
    thr_total = float(time.perf_counter() - t_thr_t0)
    timing["predict_dilation_s"] = float(dilation_s)
    timing["predict_threshold_hysteresis_s"] = max(0.0, thr_total - float(dilation_s))

    # --- Normalize masks to parent grid resolution expected by geometry builder ---
    t_norm_t0 = time.perf_counter()
    norm_masks_by_level = {}
    for L, M in masks_by_level.items():
        M = M.to(torch.bool)
        expected_h = H * (rr ** (L - 1))
        expected_w = W * (rr ** (L - 1))
        h, w = int(M.shape[0]), int(M.shape[1])

        if (h, w) == (expected_h, expected_w):
            norm_masks_by_level[L] = M
            continue

        if (h, w) == (expected_h * rr, expected_w * rr):
            Mp = torch.nn.functional.max_pool2d(
                M.float()[None, None],
                kernel_size=rr,
                stride=rr,
            )[0, 0].to(torch.bool)
            norm_masks_by_level[L] = Mp
            continue

        Mp = torch.nn.functional.interpolate(
            M.float()[None, None],
            size=(expected_h, expected_w),
            mode="nearest",
        )[0, 0].to(torch.bool)
        norm_masks_by_level[L] = Mp

    masks_by_level = norm_masks_by_level
    timing["predict_normalize_masks_s"] = float(time.perf_counter() - t_norm_t0)

    # --- optional: export gradient maps for debugging / plotting ---
    t_dbg_out_t0 = time.perf_counter()
    if debug_out is not None:
        # store *copies* so policy can freely mutate its own tensors later
        debug_out["G"] = {L: v.detach().clone() for L, v in G.items()}
        debug_out["pooled_up"] = {L: v.detach().clone() for L, v in pooled_up.items()}
    timing["predict_debug_out_s"] = float(time.perf_counter() - t_dbg_out_t0)
    timing["predict_legacy_policy_s"] = float(time.perf_counter() - t_all_t0)
    if isinstance(timing_out, dict):
        timing_out.clear()
        timing_out.update(timing)

    return masks_by_level



def _per_channel_grad_mag(Xsel, dx, dy, p_pow: float = 1.0):
    C, HH, WW = Xsel.shape
    fx = torch.zeros_like(Xsel)
    fy = torch.zeros_like(Xsel)
    if HH > 1:  # x-direction (rows)
        fx[:, 1:-1, :] = (Xsel[:, 2:, :] - Xsel[:, :-2, :]) / (2.0 * dx)
        fx[:, 0,    :] = (Xsel[:, 1,  :] - Xsel[:, 0,   :]) / dx
        fx[:, -1,   :] = (Xsel[:, -1, :] - Xsel[:, -2,  :]) / dx
    if WW > 1:  # y-direction (cols)
        fy[:, :, 1:-1] = (Xsel[:, :, 2:] - Xsel[:, :, :-2]) / (2.0 * dy)
        fy[:, :, 0]    = (Xsel[:, :, 1]  - Xsel[:, :, 0])  / dy
        fy[:, :, -1]   = (Xsel[:, :, -1] - Xsel[:, :, -2]) / dy
    g = torch.sqrt(fx * fx + fy * fy)
    return g if p_pow == 1.0 else (g ** p_pow)


def _sorted_indices_by_level(level: torch.Tensor) -> torch.Tensor:
    """Coarse→fine order for painting."""
    return torch.argsort(level, dim=0)

def coarse_aggregate_from_dynamic(X: torch.Tensor, parents: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Aggregate node-wise features X (N,D) to the coarse parent grid (H*W,D)
    using 'parents' indices in [0..H*W-1].

    X:        [N, D]
    parents:  [N]  (long)
    returns:  [H*W, D]
    """
    assert X.dim() == 2, f"X must be [N,D], got {tuple(X.shape)}"
    N, D = X.shape
    parents = parents.view(-1).long()
    assert parents.numel() == N, f"parents length ({parents.numel()}) must equal N ({N})"

    out = X.new_zeros((H * W, D))
    cnt = X.new_zeros((H * W, 1))

    # Sum features into parents
    out.index_add_(0, parents, X)

    # Count how many children contributed to each parent
    ones = X.new_ones((N, 1))
    cnt.index_add_(0, parents, ones)

    # Avoid divide-by-zero; parents with no contributions remain zero
    cnt.clamp_min_(1.0)
    out = out / cnt

    return out

# ---------------------------- rasterization core ---------------------------- #

def _rasterize_leaf_to_fine(x: torch.Tensor,
                            level: torch.Tensor,
                            ij: torch.Tensor,
                            H: int, W: int, L_out: int) -> torch.Tensor:
    # Normalize/validate shapes; infer per-cell level when needed
    x, level, ij = _ensure_leaf_inputs(x, level, ij, H, W)
    N, F = x.shape

    Hf, Wf = H * (2 ** L_out), W * (2 ** L_out)
    out = x.new_zeros((Hf, Wf, F))

    order = _sorted_indices_by_level(level)  # coarse→fine
    for idx in order.tolist():
        l = int(level[idx].item())
        blk = 2 ** (L_out - l)
        # row=j, col=i
        row0 = int(ij[idx, 1].item()) * blk
        col0 = int(ij[idx, 0].item()) * blk
        rr2 = min(row0 + blk, Hf)
        cc2 = min(col0 + blk, Wf)
        out[row0:rr2, col0:cc2, :] = x[idx].view(1, 1, F)
    return out



def _mean_pool_blocks(A: np.ndarray, blk: int) -> np.ndarray:
    """
    Average-pool A over non-overlapping square blocks of size blk.
    A: (Hf, Wf, F) -> (Hf/blk, Wf/blk, F)
    """
    Hf, Wf, F = A.shape
    assert Hf % blk == 0 and Wf % blk == 0
    h = Hf // blk
    w = Wf // blk
    A_reshaped = A.reshape(h, blk, w, blk, F)
    # mean over block dims
    return A_reshaped.mean(axis=(1, 3))


def _maxpool_or(mask_child: torch.Tensor, ratio: int = 2) -> torch.Tensor:
    """
    Downsample a child-level boolean mask by ratio×ratio using OR pooling.
    Accepts shapes [Hc,Wc], [1,Hc,Wc], [Hc,Wc,1], or [B,Hc,Wc] (B=1).
    Returns [Hp,Wp] where Hp=ceil(Hc/ratio), Wp=ceil(Wc/ratio).
    """
    rr = int(ratio)
    if rr < 2:
        raise ValueError(f"ratio must be >=2, got {ratio}")

    m = torch.as_tensor(mask_child).to(torch.bool)

    # Strip a leading/trailing singleton or collapse a (B,H,W) with B>1 via any().
    if m.ndim == 3:
        if m.shape[0] == 1:      # [1,H,W] -> [H,W]
            m = m.squeeze(0)
        elif m.shape[-1] == 1:   # [H,W,1] -> [H,W]
            m = m.squeeze(-1)
        else:
            # Collapse any batch dimension via OR, leaving [H,W]
            m = m.any(dim=0)

    if m.ndim != 2:
        raise ValueError(f"_maxpool_or expects 2D after squeeze; got {tuple(m.shape)}")

    Hc, Wc = m.shape

    # Pad to multiples of rr so we can view into rr×rr tiles safely
    padH = (rr - (Hc % rr)) % rr
    padW = (rr - (Wc % rr)) % rr
    if padH or padW:
        m = F.pad(m, (0, padW, 0, padH), value=False)
        Hc, Wc = m.shape

    # OR over rr×rr tiles
    m = m.view(Hc // rr, rr, Wc // rr, rr).any(dim=(1, 3))
    return m  # dtype=bool, shape [Hc//rr, Wc//rr]


def _upsample2(mask_parent: torch.Tensor) -> torch.Tensor:
    """
    Parent -> child by 2x upsample (repeat each dimension by 2).
    """
    return mask_parent.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)


def _finite_diff_grad_mag(A: np.ndarray) -> np.ndarray:
    """
    Central-diff gradient magnitude per channel on a 2D image A[..., F].
    Returns shape (H, W, F).
    """
    # np.gradient handles edges with first-order differences.
    gx = np.gradient(A, axis=0)
    gy = np.gradient(A, axis=1)
    # gx, gy are lists when A has multiple dims; we need channel-wise grads
    # np.gradient on last axis not requested; so gx,gy are arrays (H,W,F).
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag


def _combine_channels(mag: np.ndarray, sel: Optional[list], mode: str) -> np.ndarray:
    """
    Combine per-channel magnitudes into a scalar field via l2/l1/max over
    selected channels. Returns (H,W) float32.
    """
    if sel is None:
        M = mag
    else:
        M = mag[..., sel]

    if mode.lower() in ("l2", "euclid", "euclidean"):
        out = np.sqrt((M ** 2).sum(axis=-1))
    elif mode.lower() in ("l1", "manhattan"):
        out = np.abs(M).sum(axis=-1)
    elif mode.lower() in ("max", "linf", "l_inf", "inf"):
        out = np.max(np.abs(M), axis=-1)
    else:
        # default: l2
        out = np.sqrt((M ** 2).sum(axis=-1))
    return out.astype(np.float32)


# ------------------------ leaf set & graph construction ---------------------- #

def _leaf_masks_from_hierarchy(
    masks_by_level: Dict[int, torch.Tensor],
    L_max: int,
    refine_ratio: int = 2,
) -> Dict[int, torch.Tensor]:
    """
    Given {l: mask_l}, compute leaf masks per level by subtracting parents that
    are refined at the next level.
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    out = {}
    for l in range(0, L_max + 1):
        mask_l = masks_by_level.get(l, None)
        if mask_l is None:
            Hl = masks_by_level[0].shape[0] * (rr ** l)
            Wl = masks_by_level[0].shape[1] * (rr ** l)
            mask_l = torch.zeros((Hl, Wl), dtype=torch.bool, device=_device_of_masks(masks_by_level))
        if l < L_max and masks_by_level.get(l + 1, None) is not None:
            refined_parent = _maxpool_or(masks_by_level[l + 1], ratio=rr)
            leaf = mask_l & (~refined_parent.to(mask_l.device))
        else:
            leaf = mask_l
        out[l] = leaf
    return out


def _centers_from_level_ij(level: torch.Tensor, ij: torch.Tensor,
                           H: int, W: int, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """
    Return normalized cell centers in [0,1]x[0,1] for each leaf cell.
    Accepts arbitrary shapes for level/ij; uses (H,W) and per-cell levels.
    """
    level, ij = _normalize_level_ij(level, ij, H, W)

    i = ij[:, 0].to(torch.float32)
    j = ij[:, 1].to(torch.float32)
    Lf = level.to(torch.float32)

    Wl = (W * (2.0 ** Lf))
    Hl = (H * (2.0 ** Lf))

    # normalized centers in [0,1]
    x = (i + 0.5) / Wl
    y = (j + 0.5) / Hl
    return torch.stack([x, y], dim=1)


def _centers_from_level_ij_known_dxdy(level: torch.Tensor, ij: torch.Tensor,
                                      H: int, W: int, dx: float, dy: float) -> torch.Tensor:
    """
    Same as above, signature kept for callers that pass dx,dy; result is still
    normalized [0,1] centers (dx,dy are not needed for normalization).
    """
    return _centers_from_level_ij(level, ij, H, W, dx, dy)


# -------------------------- feature mapping (IDW/NN) ------------------------ #


@torch.no_grad()
def _idw_map(src_x: torch.Tensor,
             src_centers: torch.Tensor,
             dst_centers: torch.Tensor,
             k: int = 8,
             p: float = 2.0) -> torch.Tensor:
    """
    Inverse Distance Weighted mapping from src points to dst points.

    Accepts src_x of shape [N,F], [1,N,F], [N], [N,1], or [N,F,1].
    Accepts centers as [N,2], [1,N,2], or [2,N] (transposed) and normalizes.
    Returns tensor of shape [M, F].
    """
    def _to_NF(x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x)
        if x.ndim >= 3 and x.shape[0] == 1:
            x = x.squeeze(0)              # [1,N,F] -> [N,F]
        if x.ndim == 1:
            x = x.unsqueeze(1)            # [N] -> [N,1]
        elif x.ndim > 2:
            F = x.shape[-1]
            x = x.reshape(-1, F)          # [...,F] -> [N,F]
        return x.contiguous()

    def _to_N2(x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x)
        if x.ndim >= 3 and x.shape[0] == 1:
            x = x.squeeze(0)              # [1,N,2] -> [N,2]
        if x.ndim == 2 and x.shape[0] == 2 and x.shape[1] != 2:
            x = x.transpose(0, 1)         # [2,N] -> [N,2]
        if x.ndim > 2 and x.shape[-1] == 2:
            x = x.reshape(-1, 2)          # [...,2] -> [N,2]
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"centers must be [N,2], got {tuple(x.shape)}")
        return x.contiguous()

    src_x = _to_NF(src_x).to(torch.float32)
    S = src_x.shape[0]
    if S == 0:
        # No sources → zero output of matching feature width
        return torch.zeros((dst_centers.shape[0], src_x.shape[1]), dtype=src_x.dtype, device=src_x.device)

    src_xy = _to_N2(src_centers).to(src_x.device, dtype=torch.float32)
    dst_xy = _to_N2(dst_centers).to(src_x.device, dtype=torch.float32)

    # Pairwise distances [M,N]
    d = torch.cdist(dst_xy, src_xy, p=2)
    if d.numel() == 0:
        return torch.zeros((dst_xy.shape[0], src_x.shape[1]), dtype=src_x.dtype, device=src_x.device)

    # Choose neighbors
    k_eff = int(max(1, min(k, d.shape[1])))
    topk_d, topk_idx = torch.topk(d, k=k_eff, largest=False)  # [M,k]

    # Weights
    eps = 1e-12
    w = 1.0 / torch.clamp(topk_d, min=eps) ** float(p)        # [M,k]
    wsum = w.sum(dim=1, keepdim=True).clamp_min(eps)

    # Gather src_x for neighbors
    F = src_x.shape[1]
    gather = src_x.index_select(0, topk_idx.reshape(-1))      # [M*k, F]
    gather = gather.view(topk_idx.shape[0], k_eff, F)         # [M,k,F]

    out = (w.unsqueeze(-1) * gather).sum(dim=1) / wsum        # [M,F]

    # Exact matches: enforce identity
    exact = (topk_d[:, 0] <= 1e-14)
    if exact.any():
        rows = torch.nonzero(exact, as_tuple=False).view(-1)
        out[rows] = src_x.index_select(0, topk_idx[rows, 0])

    return out
