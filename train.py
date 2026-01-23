# ===================== train_mesh_first.py (NEW) =====================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh‑first training pipeline for dynamic AMR feature prediction.

Workflow (per sample t → t+1):
  1) Predict the refine mask at t+1 *deterministically* from GT(t+1) features by
     computing coarse‑grid gradient magnitudes and applying hysteresis thresholds.
     (Supports multi‑feature refinement; see policy.combine/refine_channels.)
  2) Build the predicted dynamic mesh from that mask.
  3) Interpolate GT node features at t onto the predicted mesh centers → X_pred.
  4) Run the GNN to predict features at t+1 on the predicted mesh.
  5) Interpolate predictions back to the GT nodes at t+1 and compute the loss.

Also produces qualitative PDFs of feature fields per sample using
`plot_qual_2x3_pdf` from plots.py.
"""

from __future__ import annotations
print("[train.py] module import started", flush=True)
from typing import Dict, Any, List, Tuple
import os, io, json, time, zipfile, random, hashlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch import optim
from torch.amp import autocast
from contextlib import nullcontext, contextmanager
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import CellRefineTemporalDataset, CellRefineWindowDataset, \
                    preprocess_timesteps_once
from models import FeatureNet
from amr_policy import coarse_aggregate_from_dynamic, predict_masks_hierarchical_from_gt_gradients
            
from plots import plot_loss_curves, plot_qual_2x3_pdf, plot_predictions_from_examples_pdf, \
                compute_plot_deltas, plot_qual_2x3_pdf_with_cells, plot_qual_pdf
from utils_geom import build_coarse_n4, \
             build_idw_map, apply_idw_map, _targeted_map_to_pred, \
             knn_interpolate_cuda_cdist, knn_interpolate_matmul, \
             parents_from_pos, apply_precomputed_idw_map

from pretrain import make_collate_with_precompute, \
                precompute_pred_mesh_and_interps_for_rollout, CollateWithPrecompute

from utils.precomp_h5 import LazyPrecompH5
from utils.uniform_mesh_engine import UniformMeshEngine
import utils.dec_ops as dec


debug_once = False


@torch.no_grad()
def _compute_norm_stats_from_loader(loader, device):
    """
    Compute mean/std of physics features over the training loader.

    Supports:
      - old single-step batches with 'center_feat_t' (and optionally 'center_feat_tp1')
      - new multi-step batches with 'feat_list' (list of tensors per time step)
    """
    n_total = 0
    mean = None
    M2 = None  # for Welford's online variance

    for batch in loader:
        # -------- single-step case (old dataset) --------
        if "center_feat_t" in batch:
            x_t = batch["center_feat_t"].to(device).float()   # (N_t, F)
            xs = [x_t]

            # if you also want to include t+1 in stats (like before), keep this:
            if "center_feat_tp1" in batch:
                x_tp1 = batch["center_feat_tp1"].to(device).float()
                xs.append(x_tp1)

            x = torch.cat(xs, dim=0)  # (N_all, F)

        # -------- windowed case (CellRefineWindowDataset) --------
        elif "feat_list" in batch:
            feat_list = batch["feat_list"]  # list/tuple of length K, each (N_k, F)

            xs = []
            for f in feat_list:
                if f is None:
                    continue
                f = f.to(device).float()
                if f.numel() == 0:
                    continue
                xs.append(f)

            if not xs:
                continue  # nothing in this batch

            x = torch.cat(xs, dim=0)  # (sum_k N_k, F)

        else:
            # Nothing we recognize; skip this batch
            continue

        # -------- Welford update --------
        if x.numel() == 0:
            continue

        if mean is None:
            # first batch
            mean = x.mean(dim=0)
            # unbiased variance components
            diff = x - mean
            M2 = (diff * diff).sum(dim=0)
            n_total = x.size(0)
        else:
            n_batch = x.size(0)
            new_n_total = n_total + n_batch

            delta = x.mean(dim=0) - mean
            mean = mean + delta * (n_batch / new_n_total)

            diff = x - mean
            M2 = M2 + (diff * diff).sum(dim=0) + (delta * delta) * (n_total * n_batch / new_n_total)

            n_total = new_n_total

    if mean is None or n_total == 0:
        raise RuntimeError("Could not compute normalization stats: no feature tensors found in loader.")

    var = M2 / max(n_total - 1, 1)
    std = torch.sqrt(var + 1e-8)

    # Keep them on the same device; _maybe_norm moves them to x.device later.
    return mean, std


def _maybe_norm(x: torch.Tensor, mu, sigma):
    """
    Normalize x using mu, sigma (broadcastable), moving stats to x.device.
    mu/sigma can be tensors, lists, or numpy arrays.
    """
    #if mu is None or sigma is None:
    #    return x

    # Ensure mu, sigma are tensors on x.device with the right dtype
    if not torch.is_tensor(mu):
        mu_t = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
    else:
        mu_t = mu.to(device=x.device, dtype=x.dtype)

    if not torch.is_tensor(sigma):
        sigma_t = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    else:
        sigma_t = sigma.to(device=x.device, dtype=x.dtype)

    return (x - mu_t) / sigma_t


def _maybe_denorm(x: torch.Tensor, mu, sigma):
    """
    Inverse of _maybe_norm: x * sigma + mu, moving stats to x.device.
    """
    #if mu is None or sigma is None:
    #    return x

    if not torch.is_tensor(mu):
        mu_t = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
    else:
        mu_t = mu.to(device=x.device, dtype=x.dtype)

    if not torch.is_tensor(sigma):
        sigma_t = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    else:
        sigma_t = sigma.to(device=x.device, dtype=x.dtype)

    return x * sigma_t + mu_t


# ------------------------- Small utilities -------------------------


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


# Per‑edge Laplacian smoothness on cell‑graph predictions

def laplacian_smoothness(pred_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return pred_feat.new_zeros(())
    u, v = edge_index[0], edge_index[1]
    diff = pred_feat[u] - pred_feat[v]
    return (diff**2).mean()


def temporal_consistency(pred_feat: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_feat, feat_t)

def move_precomp_to_device(precomp, device):
    """
    Supports both:
      (A) old in-memory precomp: dict[str, list[Optional[Tensor]]]
      (B) streaming H5 handle:   {"type":"h5","path":..., "T":..., "H":..., "W":...}
    """
    device = torch.device(device)

    # --- streaming H5 handle path ---
    if isinstance(precomp, dict) and precomp.get("type") == "h5":
        return LazyPrecompH5(
            path=precomp["path"],
            T=int(precomp["T"]),
            H=int(precomp["H"]),
            W=int(precomp["W"]),
            device=device,
        )

    # --- existing in-memory path ---
    if not isinstance(precomp, dict):
        return precomp

    out = {}
    for key, seq in precomp.items():
        if not isinstance(seq, (list, tuple)):
            out[key] = seq
            continue

        new_seq = []
        for x in seq:
            if x is None:
                new_seq.append(None)
            elif torch.is_tensor(x):
                new_seq.append(x.to(device))
            else:
                new_seq.append(x)
        out[key] = new_seq

    return out


# -------------------- Fast kNN‑IDW interpolation (vectorized) --------------------
# This replaces expensive Python loops and can be cached per sample.

def _sync_device(device):
    # Make timings honest on accelerators
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try: torch.mps.synchronize()
        except Exception: pass


# -------------------- Mesh‑first (deterministic) mask --------------------

@torch.no_grad()
def _coarse_grad_mag(field_coarse: torch.Tensor, H: int, W: int, dx: float, dy: float, p: float = 1.0) -> torch.Tensor:
    """Centered‑difference gradient magnitude on the coarse H×W grid.
    field_coarse: (H*W,) → returns (H*W,) magnitudes (optionally raised to power p).
    """
    f = field_coarse.view(H, W)
    fx = torch.zeros_like(f)
    fy = torch.zeros_like(f)
    if W > 1:
        fx[:,1:-1] = (f[:,2:] - f[:,:-2]) / (2.0*dx)
        fx[:,0]    = (f[:,1] - f[:,0]) / dx
        fx[:,-1]   = (f[:,-1] - f[:,-2]) / dx
    if H > 1:
        fy[1:-1,:] = (f[2:,:] - f[:-2,:]) / (2.0*dy)
        fy[0,:]    = (f[1,:] - f[0,:]) / dy
        fy[-1,:]   = (f[-1,:] - f[-2,:]) / dy
    g = torch.sqrt(fx*fx + fy*fy).reshape(-1)
    if p != 1.0:
        g = g**p
    return g


def _resolve_channel_indices(cfg: Dict[str,Any]) -> List[int]:
    chmap = cfg.get("channels", {})
    names = cfg.get("policy", {}).get("refine_channels") or cfg.get("features", {}).get("supervised")
    idxs: List[int] = []
    if names:
        for nm in names:
            if isinstance(nm, int):
                idxs.append(int(nm))
            elif nm in chmap:
                idxs.append(int(chmap[nm]))
    if idxs:
        return idxs
    if cfg.get("features", {}).get("use_columns"):
        return list(cfg["features"]["use_columns"])
    if cfg.get("data", {}).get("feature_idx"):
        return list(cfg["data"]["feature_idx"])
    return [0,1,2]

def _t2f(x):
    """tensor/number -> python float"""
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x.detach().float().mean().cpu().item()) if x.numel() > 0 else 0.0
    return float(x)

def _fmt(x, w=11):
    if x is None:
        return " " * (w - 4) + "None"
    return f"{x:{w}.4e}"

def _loss_comp_print(
    *,
    tag: str,
    step_k: int,
    dt_hat,
    center_loss,
    lap_w,
    lap_loss,
    tmp_w,
    tmp_loss,
    dec_use,
    dec_blend_w,
    dec_resid_w,
    phy_loss,
    total_loss,
    y_nn=None,
    y_pred=None,
):
    dtv = _t2f(dt_hat)
    c   = _t2f(center_loss)
    ll  = _t2f(lap_loss)
    tl  = _t2f(tmp_loss)
    pl  = _t2f(phy_loss)
    tot = _t2f(total_loss)

    lap_term = (float(lap_w) * (ll if ll is not None else 0.0)) if lap_w else 0.0
    tmp_term = (float(tmp_w) * (tl if tl is not None else 0.0)) if tmp_w else 0.0
    phy_term = (float(dec_resid_w) * (pl if pl is not None else 0.0)) if dec_resid_w else 0.0

    yn = None if y_nn is None else _t2f(y_nn.abs().mean())
    yp = None if y_pred is None else _t2f(y_pred.abs().mean())

    print(
        f"[LOSS] {tag} k={step_k:02d} dt_hat={dtv:.4e} | "
        f"center={_fmt(c)} | "
        f"lap={_fmt(lap_term)} (w={lap_w:.2e}, raw={_fmt(ll)}) | "
        f"tmp={_fmt(tmp_term)} (w={tmp_w:.2e}, raw={_fmt(tl)}) | "
        f"dec_resid={_fmt(phy_term)} (w={dec_resid_w:.2e}, raw={_fmt(pl)}) | "
        f"TOTAL={_fmt(tot)} | "
        f"dec_use={int(bool(dec_use))} dec_blend_w={dec_blend_w:.2e} | "
        f"|y_nn|={_fmt(yn)} | |y_pred|={_fmt(yp)}"
    )


@torch.no_grad()
def predict_mask_from_gt_features(batch: Dict[str, torch.Tensor], cfg: Dict[str, Any], H: int, W: int, dx: float, dy: float):
    """Multi‑feature refinement support with hysteresis thresholds.
    Returns: mask_pred (H,W,bool), combined_score (H,W), per_feature_scores dict.
    """
    pol = cfg.get("policy", {})
    p_pow     = float(pol.get("p", 1.0))
    tau_low   = float(pol.get("tau_low", 0.02))
    tau_high  = float(pol.get("tau_high", 0.03))
    combine   = str(pol.get("combine", "l2")).lower()
    weights   = pol.get("refine_weights", None)

    idxs = _resolve_channel_indices(cfg)
    #print("idxs for refinement:", idxs)

    gt_dyn_tp1 = batch["dyn_feat_t"].to(torch.float32)
    parents    = batch["dyn_parents"]
    prev_mask  = batch["mask_t"]

    coarse_tp1 = coarse_aggregate_from_dynamic(gt_dyn_tp1, parents, H, W)  # (H*W, F)

    names = cfg.get("features", {}).get("dataset_order") or [f"ch{i}" for i in range(coarse_tp1.size(1))]
    score_list = []
    per_name_scores = {}
    for col in idxs:
        if 0 <= col < coarse_tp1.size(1):
            s = _coarse_grad_mag(coarse_tp1[:, col], H, W, dx, dy, p=p_pow)
            score_list.append(s)
            nm = names[col] if col < len(names) else f"ch{col}"
            per_name_scores[nm] = s
    if not score_list:
        score_list = [_coarse_grad_mag(coarse_tp1[:, 0], H, W, dx, dy, p=p_pow)]

    S = torch.stack(score_list, dim=0)  # (C, K)
    if weights is not None:
        w = torch.as_tensor(weights, device=S.device, dtype=S.dtype)
        if w.numel() != S.size(0):
            if w.numel() < S.size(0):
                w = torch.cat([w, w.new_ones(S.size(0)-w.numel())])
            else:
                w = w[:S.size(0)]
    else:
        w = torch.ones(S.size(0), device=S.device, dtype=S.dtype)

    if combine == "max":
        s_ref = S.max(dim=0).values
    elif combine in ("sum", "weighted_sum"):
        s_ref = (w.view(-1,1) * S).sum(dim=0)
    else:  # l2 or weighted_l2
        s_ref = torch.sqrt(((w.view(-1,1) * S)**2).sum(dim=0))

    prev = prev_mask.view(-1).bool()
    keep = prev & (s_ref > tau_low)
    newr = (~prev) & (s_ref > tau_high)
    mask_pred = (keep | newr).view(H, W)
    return mask_pred, s_ref.view(H, W), per_name_scores


# -------------------- Training / Evaluation (mesh‑first) --------------------

def _build_X(feat, centers, levels, cfg):
    """
    Build node features for FeatureNet:
      [ physics_feat ] (+ [x,y] if use_pos) (+ [level] if use_level)

    feat:    (N, F) on whatever device we want to run the model on (cpu/cuda/mps)
    centers: (N, 2) (typically from precompute; may be on cpu)
    levels:  (N,) or (N,1)
    """
    build_cfg = cfg.get("features", {}).get("build", {})
    use_pos   = build_cfg.get("use_pos", True)
    use_level = build_cfg.get("use_level", True)

    dev = feat.device
    Xs = [feat]  # already on dev

    if use_pos:
        Xs.append(centers.to(dev))

    if use_level:
        lvl = levels
        if lvl.dim() == 1:
            lvl = lvl.unsqueeze(-1)
        Xs.append(lvl.to(dev).to(feat.dtype))

    return torch.cat(Xs, dim=-1)

def _forward_main_head(model: FeatureNet, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    out = model(X, edge_index)
    if isinstance(out, (list, tuple)):
        return out[0]
    return out

def _forward_main_head_with_edge_attr(model, x_in, edge_index, edge_attr=None):
    """
    Backward compatible: if your model ignores edge_attr, it still works.
    """
    if edge_attr is not None and not hasattr(_forward_main_head_with_edge_attr, "_printed"):
        _forward_main_head_with_edge_attr._printed = True
        print("[DEC-CHK] Forward called with edge_attr present.")

    if edge_attr is None:
        return _forward_main_head(model, x_in, edge_index)  # your existing helper
    try:
        return _forward_main_head(model, x_in, edge_index, edge_attr=edge_attr)
    except TypeError:
        # model/_forward_main_head doesn’t accept edge_attr yet
        return _forward_main_head(model, x_in, edge_index)

def _get_bbox(cfg: Dict[str,Any]) -> Tuple[float,float,float,float]:
    if cfg.get("data", {}).get("bbox"):
        xmin, xmax, ymin, ymax = cfg["data"]["bbox"]
    else:
        dom = cfg.get("domain", {})
        xmin = float(dom.get("xmin", 0.0)); xmax = float(dom.get("xmax", 1.0))
        ymin = float(dom.get("ymin", 0.0)); ymax = float(dom.get("ymax", 1.0))
    return xmin, xmax, ymin, ymax


def _map_pred_to_next_pred(pred_centers_src,
                           feats_src,
                           levels_src,
                           parents_src,
                           pred_centers_dst,
                           levels_dst,
                           parents_dst,
                           mask_pred_dst,
                           H, W, knn_k=8, chunk=8192):
    """
    Map predicted features from mesh k -> k+1.

    - First: copy coarse cells where parent index is unchanged between src and dst.
    - Then: run IDW *only* for the remaining nodes.

    All tensors are moved to the device of feats_src.
    """
    dev = feats_src.device

    pred_centers_src = pred_centers_src.to(dev)
    pred_centers_dst = pred_centers_dst.to(dev)
    levels_src       = levels_src.to(dev)
    levels_dst       = levels_dst.to(dev)
    parents_src      = parents_src.to(dev)
    parents_dst      = parents_dst.to(dev)
    mask_pred_dst    = mask_pred_dst.to(dev)

    #feats_src = feats_src.to(dev)
    feats_src = feats_src.detach().to(dev)

    N_dst, F = pred_centers_dst.shape[0], feats_src.shape[1]
    out = feats_src.new_empty((N_dst, F), device=dev)

    # ---- 1) Coarse parent copy --------------------------------------------
    src_is_coarse = (levels_src == 0)
    dst_is_coarse = (levels_dst == 0)

    parent_src = parents_src.clamp(0, H * W - 1)
    parent_dst = parents_dst.clamp(0, H * W - 1)

    src_coarse_idx = torch.nonzero(src_is_coarse, as_tuple=False).view(-1)
    dst_coarse_idx = torch.nonzero(dst_is_coarse, as_tuple=False).view(-1)

    need_idw = torch.ones(N_dst, dtype=torch.bool, device=dev)

    copied = 0
    if src_coarse_idx.numel() > 0 and dst_coarse_idx.numel() > 0:
        # lookup[parent] = index of coarse src node whose parent is `parent`
        lookup = -torch.ones(H * W, dtype=torch.long, device=dev)
        lookup[parent_src[src_coarse_idx]] = src_coarse_idx

        # for each dst coarse node, find matching src coarse node
        src_match_for_dst = lookup[parent_dst[dst_coarse_idx]]
        has_match = src_match_for_dst >= 0
        if has_match.any():
            dst_keep = dst_coarse_idx[has_match]
            src_keep = src_match_for_dst[has_match]
            out[dst_keep] = feats_src[src_keep]
            need_idw[dst_keep] = False
            copied = int(has_match.sum().item())

    # ---- 2) IDW for the remaining nodes -----------------------------------
    q_idx = need_idw.nonzero(as_tuple=True)[0]
    if q_idx.numel() > 0:
        q_pts = pred_centers_dst[q_idx]  # (Q,2)
        idx_map, w_map = build_idw_map(q_pts, pred_centers_src,
                                        k=knn_k, chunk=chunk)
        vals = apply_idw_map(idx_map, w_map, feats_src)  # (Q,F)
        out[q_idx] = vals

    #if cfg.get("debug", {}).get("idw_stats", False):
    #print(f"[IDW stats] coarse_copied={copied}, "
    #        f"idw_points={int(q_idx.numel())}, N_dst={N_dst}")

    return out


def _require_list(batch, name):
    if name not in batch:
        raise RuntimeError(
            f"Batch is missing required key '{name}'. "
            f"This training loop is strict: precompute and include '{name}' in the collate."
        )
    return batch[name]

def _pred_mesh_for_step_strict(k_idx, *, pred_lists):
    """
    Return predicted geometry for the DESTINATION step (k_idx+1).
    pred_lists is a 5-tuple of lists pulled from the batch.
    Raises on any shape/index mismatch.
    """
    pc_list, pl_list, pp_list, pei_list, mp_list = pred_lists
    dst = k_idx + 1
    try:
        return (pc_list[dst], pl_list[dst], pp_list[dst], pei_list[dst], mp_list[dst])
    except Exception as e:
        raise RuntimeError(
            f"Invalid pred_*_list lengths or index when accessing step {dst}. "
            f"Ensure all pred lists are length-K and aligned."
        ) from e
    
def _tstats(name: str, t: torch.Tensor, max_elems: int = 0):
    if t is None:
        print(f"[NAN-DBG] {name}: None")
        return
    if not torch.is_tensor(t):
        print(f"[NAN-DBG] {name}: (non-tensor) {type(t)} = {t}")
        return

    tt = t.detach()
    finite = torch.isfinite(tt) if tt.dtype.is_floating_point or tt.dtype.is_complex else None

    n = tt.numel()
    msg = f"[NAN-DBG] {name}: shape={tuple(tt.shape)} dtype={tt.dtype} dev={tt.device} "

    # For float/complex: report finiteness and summary stats on finite values
    if tt.dtype.is_floating_point or tt.dtype.is_complex:
        nf = int(finite.sum().item())
        msg += f"finite={nf}/{n} "
        if nf > 0:
            v = tt[finite]
            msg += (
                f"min={v.min().item():.3e} max={v.max().item():.3e} "
                f"mean={v.mean().item():.3e} absmax={v.abs().max().item():.3e}"
            )
        else:
            msg += "ALL_NONFINITE"
        print(msg)

        if max_elems > 0 and nf < n:
            bad = torch.nonzero(~finite, as_tuple=False)
            bad = bad[:max_elems]
            print(f"[NAN-DBG] {name}: first nonfinite indices (up to {max_elems}): {bad.tolist()}")
        return

    # For non-float: no finiteness concept; report integer/bool stats
    # Always safe: min/max
    try:
        tmin = tt.min().item()
        tmax = tt.max().item()
        msg += f"min={tmin} max={tmax} "
    except Exception as e:
        msg += f"(min/max failed: {repr(e)}) "
        print(msg)
        return

    # If integer, report mean/absmax by casting to float FOR REPORTING ONLY
    if tt.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.long):
        v = tt.to(torch.float32)
        msg += f"mean={v.mean().item():.3e} absmax={v.abs().max().item():.3e}"
    elif tt.dtype == torch.bool:
        msg += f"true_count={int(tt.sum().item())}/{n}"
    else:
        # fallback
        pass

    print(msg)


def _assert_finite(name: str, t: torch.Tensor, crash: bool = True):
    if t is None or (not torch.is_tensor(t)):
        return True
    if not (t.dtype.is_floating_point or t.dtype.is_complex):
        return True  # finiteness not meaningful for integer/bool
    ok = bool(torch.isfinite(t).all().item())
    if not ok:
        _tstats(name, t, max_elems=10)
        if crash:
            raise RuntimeError(f"Non-finite detected in {name}")
    return ok

def train_one_epoch_multi_step(
    model,
    loader,
    opt,
    cfg,
    device,
    *,
    H: int,
    W: int,
    dx=None,
    dy=None,
    scaler=None,
    mu=None,
    sigma=None,
):
    """
    STRICT multi-step mesh-first training with:
      - Variant-B physics baseline + learned correction (rate mode)
      - Optional PARC operator inputs (advection/diffusion terms)
      - Optional physics residual loss (delta-form), as before

    Assumes model predicts RATE (cfg["model"]["predict_type"] == "rate").
    """

    model.train()
    total_loss_accum = 0.0
    mae_accum = 0.0
    n_steps = 0

    dbg = cfg.get("debug", {})
    nan_watch = bool(dbg.get("nan_watch", True))          # turn on/off
    nan_watch_first_only = bool(dbg.get("nan_first_only", True))

    if not hasattr(train_one_epoch_multi_step, "_nan_printed"):
        train_one_epoch_multi_step._nan_printed = False

    def _should_print():
        if not nan_watch:
            return False
        if nan_watch_first_only and train_one_epoch_multi_step._nan_printed:
            return False
        return True

    def _mark_printed():
        train_one_epoch_multi_step._nan_printed = True

    # ---- dx,dy ----
    if dx is None or dy is None:
        bbox = cfg.get("data", {}).get("bbox", None)
        if bbox is None:
            raise ValueError("dx/dy not provided and cfg['data']['bbox'] missing.")
        x0, x1, y0, y1 = map(float, bbox)
        dx = (x1 - x0) / float(W)
        dy = (y1 - y0) / float(H)

    speed = cfg.get("speed", {})
    use_amp = bool(speed.get("amp", True)) and device.type == "cuda"

    huber_delta = float(cfg["loss"].get("huber_delta", 0.05))
    lap_w  = float(cfg["loss"].get("laplacian_weight", 0.0))
    tmp_w  = float(cfg["loss"].get("temporal_weight", 0.0))
    use_huber = bool(cfg["loss"].get("interp_use_huber", True))

    predict_type = str(cfg.get("model", {}).get("predict_type", "rate")).lower()
    if predict_type != "rate":
        raise RuntimeError(f"PARC/Variant-B implementation below assumes predict_type='rate', got '{predict_type}'")

    # DEC / PARC controls
    loss_cfg = cfg.get("loss", {}) or {}
    dec_use = bool(loss_cfg.get("dec", False))
    parc_use = bool(loss_cfg.get("parc", False) or loss_cfg.get("parc_inputs", False))

    # Variant-B baseline strength (reuse your existing name)
    dec_blend_w = float(loss_cfg.get("blend_weight", 0.0))        # baseline weight (recommend 1.0 when parc_use)
    dec_resid_w = float(loss_cfg.get("residual_weight", 0.0))     # optional residual loss weight

    # Determine whether we must compute physics operators this step
    need_phy = parc_use or (dec_use and (dec_blend_w != 0.0 or dec_resid_w != 0.0))

    if scaler is None and use_amp and device.type == "cuda":
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    amp_ctx = (torch.cuda.amp.autocast if use_amp and device.type == "cuda"
               else (lambda **kw: torch.autocast("cpu", enabled=False)))

    train_one_epoch_multi_step._printed_loss_components = False

    for batch in loader:
        dt_list = batch.get("dt_list", None)
        if dt_list is None:
            raise RuntimeError("Missing dt_list in batch. Ensure CollateWithPrecompute attaches dt_list.")

        # Required lists (existing)
        centers_list  = _require_list(batch, "centers_list")
        feat_list     = _require_list(batch, "feat_list")
        level_list    = _require_list(batch, "level_list")
        ij_list       = _require_list(batch, "ij_list")
        ei_list       = _require_list(batch, "ei_list")
        parents_list  = _require_list(batch, "parents_list")
        mask_list     = _require_list(batch, "mask_list")

        pred_centers_list  = _require_list(batch, "pred_centers_list")
        pred_levels_list   = _require_list(batch, "pred_levels_list")
        pred_parents_list  = _require_list(batch, "pred_parents_list")
        pred_ei_list       = _require_list(batch, "pred_ei_list")
        mask_pred_list     = _require_list(batch, "mask_pred_list")

        # Optional DEC edge_attr list aligned with pred_ei_list
        pred_ea_list = batch.get("pred_ea_list", None) or batch.get("pred_edge_attr_list", None)

        # step-0 GT→pred
        feat_t_on_pred_list   = _require_list(batch, "feat_t_on_pred_list")
        feat_tp1_on_pred_list = _require_list(batch, "feat_tp1_on_pred_list")

        # pred→pred IDW maps
        pred2pred_idx_list = batch.get("pred2pred_idx_list", None)
        pred2pred_w_list   = batch.get("pred2pred_w_list", None)

        K = len(centers_list)
        if K < 2:
            raise RuntimeError("window_size must be ≥ 2")

        if need_phy and pred_ea_list is None:
            raise RuntimeError(
                "PARC/DEC enabled but batch is missing pred_ea_list / pred_edge_attr_list. "
                "Update H5 loader + CollateWithPrecompute to attach edge_attr per step."
            )

        # Pack pred lists
        pred_lists = (pred_centers_list, pred_levels_list, pred_parents_list, pred_ei_list, mask_pred_list)

        window_loss = 0.0
        window_mae = 0.0
        window_loss_graph = None

        opt.zero_grad(set_to_none=True)

        # ==========================================================
        # Helper closure: one step computation (k -> k+1 on pred mesh)
        # ==========================================================
        def _run_step(
            *,
            step_k: int,
            pred_centers,
            pred_levels,
            pred_parents,
            pred_ei,
            pred_ea,
            x_in_abs: torch.Tensor,     # [N,F] absolute state at time t (mapped onto pred mesh at t+1)
            x_tgt_abs: torch.Tensor,    # [N,F] absolute GT(t+1) mapped onto pred mesh at t+1
            dt_phys: torch.Tensor,      # scalar
            dt_ref_scalar,              # None or scalar (from batch)
        ):
            norm_in  = _maybe_norm(x_in_abs,  mu, sigma)
            norm_tgt = _maybe_norm(x_tgt_abs, mu, sigma)

            if _should_print():
                print(f"[NAN-DBG] step={step_k} ---- norm checks ----")
                _tstats("x_in_abs", x_in_abs)
                _tstats("x_tgt_abs", x_tgt_abs)
                _tstats("norm_in", norm_in)
                _tstats("norm_tgt", norm_tgt)
                if sigma is not None and torch.is_tensor(sigma):
                    _tstats("sigma", sigma)
                if mu is not None and torch.is_tensor(mu):
                    _tstats("mu", mu)
                _mark_printed()

            _assert_finite("norm_in", norm_in)
            _assert_finite("norm_tgt", norm_tgt)

            dt_ref_t = (torch.tensor(float(dt_ref_scalar), device=device, dtype=norm_in.dtype)
                        if dt_ref_scalar is not None else None)
            dt_hat = (dt_phys / dt_ref_t) if dt_ref_t is not None else dt_phys

            if _should_print():
                print(f"[NAN-DBG] step={step_k} ---- dt checks ----")
                _tstats("dt_phys", dt_phys)
                _tstats("dt_ref_t", dt_ref_t if dt_ref_t is not None else None)
                _tstats("dt_hat", dt_hat)
                _mark_printed()

            _assert_finite("dt_hat", dt_hat)

            # ---------------------------
            # NAN DEBUG: dt and targets
            # ---------------------------
            if _should_print():
                print(f"[NAN-DBG] step={step_k} ---- dt/target checks ----")
                _tstats("dt_phys", dt_phys)
                _tstats("dt_ref_t", dt_ref_t if dt_ref_t is not None else None)
                _tstats("dt_hat", dt_hat)
                print("[NAN-DBG] dt_hat <= 0:", bool((dt_hat <= 0).any().item()) if torch.is_tensor(dt_hat) else (dt_hat <= 0))
                _mark_printed()

            _assert_finite("dt_hat", dt_hat)

            if torch.is_tensor(dt_hat):
                if (dt_hat <= 0).any():
                    print("[NAN-DBG][WARN] dt_hat has non-positive values")

            with amp_ctx():
                pei = pred_ei.to(device) if torch.is_tensor(pred_ei) else pred_ei
                pea = pred_ea.to(device) if (pred_ea is not None and torch.is_tensor(pred_ea)) else pred_ea

                # ----- physics operators (float32, no autocast) -----
                r_adv_abs = r_diff_abs = r_phy_abs = area = None
                ch_mask = None
                parc_extra = None

                # ---------------------------
                # NAN DEBUG: graph + geometry
                # ---------------------------
                if _should_print():
                    print(f"[NAN-DBG] step={step_k} ---- graph/edge_attr checks ----")

                    # edge_index sanity
                    if torch.is_tensor(pei):
                        _tstats("pei(edge_index)", pei)
                        try:
                            ei_min = int(pei.min().item())
                            ei_max = int(pei.max().item())
                            Nnodes = int(x_in_abs.size(0))
                            print(f"[NAN-DBG] edge_index min={ei_min} max={ei_max} Nnodes={Nnodes}")
                            if ei_min < 0 or ei_max >= Nnodes:
                                print("[NAN-DBG][WARN] edge_index is out of bounds for node count!")
                        except Exception as e:
                            print("[NAN-DBG][WARN] edge_index bounds check failed:", repr(e))

                    # edge_attr sanity (layout: [nx, ny, face_len, dual_len, tau])
                    if pea is None:
                        print("[NAN-DBG] pea(edge_attr): None")
                    else:
                        _tstats("pea(edge_attr)", pea)
                        if pea.ndim == 2 and pea.size(1) >= 5:
                            nx = pea[:, 0]; ny = pea[:, 1]
                            face_len = pea[:, 2]; dual_len = pea[:, 3]; tau = pea[:, 4]
                            _tstats("pea[nx]", nx)
                            _tstats("pea[ny]", ny)
                            _tstats("pea[face_len]", face_len)
                            _tstats("pea[dual_len]", dual_len)
                            _tstats("pea[tau]", tau)
                            print("[NAN-DBG] face_len<=0 count:", int((face_len <= 0).sum().item()))
                            print("[NAN-DBG] dual_len<=0 count:", int((dual_len <= 0).sum().item()))
                            print("[NAN-DBG] tau nonfinite count:", int((~torch.isfinite(tau)).sum().item()))
                        else:
                            print("[NAN-DBG][WARN] pea does not look like [E,>=5]. shape=", tuple(pea.shape))

                    _mark_printed()

                if need_phy:
                    with torch.autocast(device_type=device.type, enabled=False):
                        adv_w  = float(loss_cfg.get("adv_weight", 1.0))
                        diff_w = float(loss_cfg.get("diff_weight", 1.0))

                        # Only compute what you need (PARC may still want both; we handle that explicitly below)
                        need_adv  = (abs(adv_w) > 0.0) or bool(loss_cfg.get("parc_use_adv", True))
                        need_diff = (abs(diff_w) > 0.0) or bool(loss_cfg.get("parc_use_diff", True))

                        with torch.autocast(device_type=device.type, enabled=False):
                            r_adv_abs, r_diff_abs, area = dec.dec_advdiff_terms_abs(
                                x_abs=x_in_abs.float(),
                                edge_index=pei.long(),
                                pred_ea=pea.float(),
                                levels=pred_levels.long().to(device),
                                dx0=float(dx),
                                dy0=float(dy),
                                cfg=cfg,
                                compute_adv=need_adv,
                                compute_diff=need_diff,
                            )

                            # If not computed, dec_ops should return None; if it returns something else, normalize here:
                            if (r_adv_abs is None) and need_adv:
                                raise RuntimeError("dec_advdiff_terms_abs returned r_adv_abs=None but need_adv=True")
                            if (r_diff_abs is None) and need_diff:
                                raise RuntimeError("dec_advdiff_terms_abs returned r_diff_abs=None but need_diff=True")

                            # Build baseline in a way that NEVER does 0*inf
                            r_phy_abs = torch.zeros_like(x_in_abs.float())
                            if abs(diff_w) > 0.0:
                                r_phy_abs = r_phy_abs + diff_w * r_diff_abs
                            if abs(adv_w) > 0.0:
                                r_phy_abs = r_phy_abs + adv_w * r_adv_abs

                        # channel mask: apply baseline only to configured physics channels
                        ch_mask = dec.build_channel_mask_from_loss(
                            cfg, x_in_abs.size(1), device=device, dtype=torch.float32
                        )
                        if ch_mask is None:
                            # default to rho + E (your intent) if not provided
                            idx_map = dec.infer_feature_indices(cfg, x_in_abs.size(1))
                            ch_mask = torch.zeros((x_in_abs.size(1),), device=device, dtype=torch.float32)
                            ch_mask[idx_map["rho"]] = 1.0
                            ch_mask[idx_map["E"]] = 1.0

                    # ---------------------------
                    # NAN DEBUG: physics operator outputs
                    # ---------------------------
                    if _should_print():
                        print(f"[NAN-DBG] step={step_k} ---- operator outputs ----")
                        _tstats("r_adv_abs", r_adv_abs)
                        _tstats("r_diff_abs", r_diff_abs)
                        _tstats("area", area)
                        _tstats("r_phy_abs", r_phy_abs)
                        _tstats("ch_mask", ch_mask)
                        _mark_printed()

                    _assert_finite("r_adv_abs", r_adv_abs, crash=False)
                    _assert_finite("r_diff_abs", r_diff_abs, crash=False)
                    _assert_finite("area", area)
                    _assert_finite("r_phy_abs", r_phy_abs)

                # ----- build node input X (PARC appends operator inputs) -----
                x_in = _build_X(norm_in, pred_centers, pred_levels, cfg)

                if parc_use and r_adv_abs is not None and r_diff_abs is not None:
                    parc_extra = dec.parc_terms_to_node_inputs(
                        r_adv_abs.to(device=device, dtype=torch.float32),
                        r_diff_abs.to(device=device, dtype=torch.float32),
                        dt_phys=dt_phys.to(device=device, dtype=torch.float32),
                        dt_ref=(dt_ref_t.to(device=device, dtype=torch.float32) if dt_ref_t is not None else None),
                        sigma=(sigma.to(device, dtype=torch.float32) if (sigma is not None and torch.is_tensor(sigma)) else None),
                        predict_type=predict_type,
                        cfg=cfg,
                        dtype=x_in.dtype,
                        detach=True,
                    )
                    if parc_extra.numel() > 0:
                        x_in = torch.cat([x_in, parc_extra], dim=1)

                    # ---------------------------
                    # NAN DEBUG: PARC inputs
                    # ---------------------------
                    if _should_print():
                        print(f"[NAN-DBG] step={step_k} ---- PARC extra checks ----")
                        _tstats("x_in (before PARC maybe)", x_in)
                        if parc_extra is None:
                            print("[NAN-DBG] parc_extra: None")
                        else:
                            _tstats("parc_extra", parc_extra)
                        _mark_printed()

                    if parc_extra is not None:
                        _assert_finite("parc_extra", parc_extra)
                    _assert_finite("x_in (final)", x_in)

                # ----- network outputs correction rate in model units -----
                y_corr = _forward_main_head_with_edge_attr(model, x_in, pei, edge_attr=pea)

                # ---------------------------
                # NAN DEBUG: NN forward output
                # ---------------------------
                if _should_print():
                    print(f"[NAN-DBG] step={step_k} ---- y_corr checks ----")
                    _tstats("y_corr", y_corr)
                    _mark_printed()

                _assert_finite("y_corr", y_corr)

                # ----- add Variant-B baseline in model units -----
                y_pred = y_corr
                '''
                if need_phy and (dec_blend_w != 0.0) and (r_phy_abs is not None):
                    phy_units = dec.physics_to_model_units(
                        r_phy_abs.to(dtype=y_corr.dtype),
                        dt_phys=dt_phys,
                        dt_ref=dt_ref_t,
                        sigma=(sigma.to(device, dtype=y_corr.dtype) if (sigma is not None and torch.is_tensor(sigma)) else None),
                        predict_type=predict_type,
                    )
                    # mask baseline to selected channels
                    phy_units = phy_units * ch_mask.view(1, -1).to(dtype=y_corr.dtype)
                    y_pred = y_corr + dec_blend_w * phy_units
                '''
                if need_phy and (dec_blend_w != 0.0) and (r_phy_abs is not None):
                    # --- compute baseline in fp32 to avoid fp16 NaNs on CUDA ---
                    with autocast("cuda", enabled=False):
                        r_phy_f32 = r_phy_abs.to(dtype=torch.float32)
                        
                        sigma_f32 = None
                        if sigma is not None and torch.is_tensor(sigma):
                            sigma_f32 = sigma.to(device, dtype=torch.float32)
                            
                            phy_units_f32 = dec.physics_to_model_units(
                                r_phy_f32,
                                dt_phys=dt_phys,
                                dt_ref=dt_ref_t,
                                sigma=sigma_f32,
                                predict_type=predict_type,
                            )

                            # mask baseline to selected channels (still fp32)
                            phy_units_f32 = phy_units_f32 * ch_mask.view(1, -1).to(dtype=torch.float32)

                        # Cast back to model dtype for the model arithmetic
                        phy_units = phy_units_f32.to(dtype=y_corr.dtype)
                        y_pred = y_corr + dec_blend_w * phy_units

                    if need_phy and (dec_blend_w != 0.0) and (r_phy_abs is not None):
                        if _should_print():
                            print(f"[NAN-DBG] step={step_k} ---- baseline unit checks ----")
                            _tstats("phy_units", phy_units)
                            _tstats("y_pred", y_pred)
                            _mark_printed()

                        _assert_finite("phy_units", phy_units)
                        _assert_finite("y_pred", y_pred)
                    else:
                        if _should_print():
                            print(f"[NAN-DBG] step={step_k} ---- y_pred (no baseline) ----")
                            _tstats("y_pred", y_pred)
                            _mark_printed()
                        _assert_finite("y_pred", y_pred)

                # ----- supervision target -----
                delta_target = norm_tgt - norm_in
                rate_target = delta_target / dt_hat.clamp_min(1e-12)

                if _should_print():
                    print(f"[NAN-DBG] step={step_k} ---- rate_target checks ----")
                    _tstats("delta_target", delta_target)
                    _tstats("rate_target", rate_target)
                    _mark_printed()

                _assert_finite("rate_target", rate_target)

                center_loss = (F.huber_loss(y_pred, rate_target, delta=huber_delta)
                               if use_huber else F.l1_loss(y_pred, rate_target))

                y_pred_abs = _maybe_denorm(norm_in + y_pred * dt_hat, mu, sigma)

                # ----- optional existing regularizers -----
                lap_loss = (laplacian_smoothness(y_pred, pei) if lap_w > 0 else y_pred.new_zeros(()))
                tmp_loss = (temporal_consistency(x_in, norm_tgt) if tmp_w > 0 else y_pred.new_zeros(()))

                # ----- optional physics residual loss (delta-form) -----
                phy_loss = y_pred.new_zeros(())
                if need_phy and dec_resid_w > 0.0 and (r_phy_abs is not None):
                    with torch.autocast(device_type=device.type, enabled=False):
                        phy_loss = dec.physics_residual_loss_delta(
                            y_pred_abs=y_pred_abs.float(),
                            x_in_abs=x_in_abs.float(),
                            dt_phys=dt_phys.float(),          # <-- correct dt for this step
                            r_phy_abs=r_phy_abs.float(),
                            area=area.float(),
                            sigma=(sigma.to(device, dtype=torch.float32) if (sigma is not None and torch.is_tensor(sigma)) else None),
                            channel_mask=ch_mask,
                        ).to(dtype=y_pred.dtype)

                loss_step = center_loss + lap_w * lap_loss + tmp_w * tmp_loss + dec_resid_w * phy_loss

                if cfg.get("debug", {}).get("print_loss_components", False) and (not train_one_epoch_multi_step._printed_loss_components):
                    _loss_comp_print(
                        tag="train",
                        step_k=step_k,
                        dt_hat=dt_hat,
                        center_loss=center_loss,
                        lap_w=lap_w,
                        lap_loss=lap_loss,
                        tmp_w=tmp_w,
                        tmp_loss=tmp_loss,
                        dec_use=need_phy,
                        dec_blend_w=dec_blend_w,
                        dec_resid_w=dec_resid_w,
                        phy_loss=phy_loss,
                        total_loss=loss_step,
                        y_nn=y_corr,
                        y_pred=y_pred,
                    )
                    train_one_epoch_multi_step._printed_loss_components = True

            return loss_step, y_pred_abs, x_tgt_abs

        # ======================
        # STEP 0 (k=0)
        # ======================
        k = 0
        pred_centers_1, pred_levels_1, pred_parents_1, pred_ei_1, mask_pred_1 = _pred_mesh_for_step_strict(k, pred_lists=pred_lists)
        pred_ea_1 = pred_ea_list[k + 1] if (pred_ea_list is not None) else None

        x_in_abs0  = feat_t_on_pred_list[k + 1].to(device)     # absolute GT(t) on pred(k+1)
        x_tgt_abs0 = feat_tp1_on_pred_list[k + 1].to(device)   # absolute GT(t+1) on pred(k+1)

        dt0 = dt_list[0]
        dt0 = dt0.to(device=device, dtype=x_in_abs0.dtype) if torch.is_tensor(dt0) else torch.tensor(float(dt0), device=device, dtype=x_in_abs0.dtype)
        dt_ref_scalar = batch.get("dt_ref", None)

        loss0, y_pred_abs0, _x_tgt_abs0 = _run_step(
            step_k=0,
            pred_centers=pred_centers_1,
            pred_levels=pred_levels_1,
            pred_parents=pred_parents_1,
            pred_ei=pred_ei_1,
            pred_ea=pred_ea_1,
            x_in_abs=x_in_abs0,
            x_tgt_abs=x_tgt_abs0,
            dt_phys=dt0,
            dt_ref_scalar=dt_ref_scalar,
        )

        window_loss += float(loss0.detach().cpu())
        window_mae  += float(torch.mean(torch.abs(y_pred_abs0.detach() - x_tgt_abs0)).cpu())
        n_steps += 1
        window_loss_graph = loss0 if window_loss_graph is None else (window_loss_graph + loss0)

        # chain absolute prediction forward
        pred_feats_k = y_pred_abs0

        # ======================
        # STEPS 1..K-2
        # ======================
        for k in range(1, K - 1):
            pred_centers_next, pred_levels_next, pred_parents_next, pred_ei_next, mask_pred_next = _pred_mesh_for_step_strict(k, pred_lists=pred_lists)
            pred_ea_next = pred_ea_list[k + 1] if (pred_ea_list is not None) else None

            # map pred(k)->pred(k+1)
            idx_km1 = pred2pred_idx_list[k-1].to(device)
            w_km1   = pred2pred_w_list[k-1].to(device)
            x_in_abs = apply_precomputed_idw_map(idx_km1, w_km1, pred_feats_k).to(device)

            x_tgt_abs = feat_tp1_on_pred_list[k + 1].to(device)

            dtk = dt_list[k]
            dtk = dtk.to(device=device, dtype=x_in_abs.dtype) if torch.is_tensor(dtk) else torch.tensor(float(dtk), device=device, dtype=x_in_abs.dtype)

            loss_k, y_pred_abs_k, _ = _run_step(
                step_k=k,
                pred_centers=pred_centers_next,
                pred_levels=pred_levels_next,
                pred_parents=pred_parents_next,
                pred_ei=pred_ei_next,
                pred_ea=pred_ea_next,
                x_in_abs=x_in_abs,
                x_tgt_abs=x_tgt_abs,
                dt_phys=dtk,
                dt_ref_scalar=dt_ref_scalar,
            )

            window_loss += float(loss_k.detach().cpu())
            window_mae  += float(torch.mean(torch.abs(y_pred_abs_k.detach() - x_tgt_abs)).cpu())
            n_steps += 1
            window_loss_graph = window_loss_graph + loss_k

            pred_feats_k = y_pred_abs_k

        # ===== backward + step once per window =====
        if window_loss_graph is not None:
            if scaler is not None:
                scaler.scale(window_loss_graph).backward()
                scaler.step(opt)
                scaler.update()
            else:
                window_loss_graph.backward()
                opt.step()
        else:
            opt.step()

        total_loss_accum += window_loss
        mae_accum        += window_mae

    denom = max(n_steps, 1)
    return total_loss_accum / denom, mae_accum / denom, {"num_windows": len(loader), "num_steps": n_steps}


def evaluate_one_epoch_multi_step(
    model,
    loader,
    cfg,
    device,
    *,
    H: int,
    W: int,
    dx=None,
    dy=None,
    mu=None,
    sigma=None,
    collect_examples: bool = False,
):
    model.eval()

    # ---- dx,dy ----
    if dx is None or dy is None:
        bbox = cfg.get("data", {}).get("bbox", None)
        if bbox is None:
            raise ValueError("dx/dy not provided and cfg['data']['bbox'] missing; cannot compute area weights.")
        x0, x1, y0, y1 = map(float, bbox)
        dx = (x1 - x0) / float(W)
        dy = (y1 - y0) / float(H)

    speed = cfg.get("speed", {})
    use_amp = bool(speed.get("amp", True)) and device.type == "cuda"

    huber_delta = float(cfg["loss"].get("huber_delta", 0.05))
    lap_w  = float(cfg["loss"].get("laplacian_weight", 0.0))
    tmp_w  = float(cfg["loss"].get("temporal_weight", 0.0))
    use_huber = bool(cfg["loss"].get("interp_use_huber", True))

    predict_type = str(cfg.get("model", {}).get("predict_type", "rate")).lower()
    if predict_type != "rate":
        raise RuntimeError(f"PARC/Variant-B implementation below assumes predict_type='rate', got '{predict_type}'")

    loss_cfg = cfg.get("loss", {}) or {}
    dec_use = bool(loss_cfg.get("dec", False))   # <-- FIXED
    parc_use = bool(loss_cfg.get("parc", False) or loss_cfg.get("parc_inputs", False))

    dec_blend_w = float(loss_cfg.get("blend_weight", 0.0))
    dec_resid_w = float(loss_cfg.get("residual_weight", 0.0))

    need_phy = parc_use or (dec_use and (dec_blend_w != 0.0 or dec_resid_w != 0.0))

    amp_ctx = (torch.cuda.amp.autocast if use_amp and device.type == "cuda"
               else (lambda **kw: torch.autocast("cpu", enabled=False)))

    # ---- metric accumulators (weighted by cell area) ----
    step_wsum = []
    step_mae_num = []
    step_mse_num = []
    step_gt2_num = []
    by_t = {}

    total_loss_accum = 0.0
    n_steps_total = 0
    examples = [] if collect_examples else None

    def _ensure_step_capacity(k: int, Fdim: int):
        while len(step_wsum) <= k:
            step_wsum.append(0.0)
            step_mae_num.append(torch.zeros(Fdim, dtype=torch.float64))
            step_mse_num.append(torch.zeros(Fdim, dtype=torch.float64))
            step_gt2_num.append(torch.zeros(Fdim, dtype=torch.float64))

    def _accumulate_metrics(*, k: int, t_abs: int | None, pred_abs: torch.Tensor, gt_abs: torch.Tensor, pred_levels: torch.Tensor):
        if pred_abs.ndim == 1: pred_abs_ = pred_abs[:, None]
        else: pred_abs_ = pred_abs
        if gt_abs.ndim == 1: gt_abs_ = gt_abs[:, None]
        else: gt_abs_ = gt_abs
        if pred_abs_.shape != gt_abs_.shape:
            raise RuntimeError(f"Metric shape mismatch: pred {pred_abs_.shape} vs gt {gt_abs_.shape}")
        N, Fdim = pred_abs_.shape
        _ensure_step_capacity(k, Fdim)

        dtype = pred_abs_.dtype
        dev = pred_abs_.device
        w = dec.cell_area_from_levels(pred_levels, dx0=float(dx), dy0=float(dy), dtype=dtype, device=dev)  # [N]
        wsum_add = float(w.sum().detach().cpu())

        diff = (pred_abs_ - gt_abs_)
        mae_add = (w[:, None] * diff.abs()).sum(dim=0).detach().cpu().to(torch.float64)
        mse_add = (w[:, None] * diff.pow(2)).sum(dim=0).detach().cpu().to(torch.float64)
        gt2_add = (w[:, None] * gt_abs_.pow(2)).sum(dim=0).detach().cpu().to(torch.float64)

        step_wsum[k] += wsum_add
        step_mae_num[k] += mae_add
        step_mse_num[k] += mse_add
        step_gt2_num[k] += gt2_add

        if t_abs is not None:
            rec = by_t.get(int(t_abs), None)
            if rec is None:
                by_t[int(t_abs)] = {"wsum": wsum_add, "mae": mae_add.clone(), "mse": mse_add.clone(), "gt2": gt2_add.clone()}
            else:
                rec["wsum"] += wsum_add
                rec["mae"]  += mae_add
                rec["mse"]  += mse_add
                rec["gt2"]  += gt2_add

    def _append_example_step(
        *,
        step_idx: int,
        pred_centers,
        pred_levels,
        pred_parents,
        y_pred_step_abs,
        centers_list,
        feat_list,
        level_list,
        parents_list,
        batch,
    ):
        if not collect_examples:
            return
        idx_tp1 = step_idx + 1
        t_indices = batch.get("t_indices", None)
        t_idx = int(t_indices[idx_tp1].item()) if (t_indices is not None and torch.is_tensor(t_indices)) else (int(t_indices[idx_tp1]) if t_indices is not None else int(idx_tp1))
        bbox = tuple(cfg.get("data", {}).get("bbox", (0.0, 1.0, 0.0, 1.0)))
        examples.append({
            "pred_centers": pred_centers.detach().cpu(),
            "pred_levels":  pred_levels.detach().cpu(),
            "pred_parents": pred_parents.detach().cpu(),
            "gt_t":         feat_list[step_idx].detach().cpu(),
            "gt_tp1":       feat_list[idx_tp1].detach().cpu(),
            "pred_tp1":     y_pred_step_abs.detach().cpu(),
            "H": H, "W": W, "bbox": bbox,
            "t": int(t_idx),

            # True-mesh geometry for plotting
            "centers_t":    centers_list[step_idx].detach().cpu(),
            "level_t":      level_list[step_idx].detach().cpu(),
            "parents_t":    parents_list[step_idx].detach().cpu(),
            "feat_t":       feat_list[step_idx].detach().cpu(),

            "centers_tp1":  centers_list[idx_tp1].detach().cpu(),
            "level_tp1":    level_list[idx_tp1].detach().cpu(),
            "parents_tp1":  parents_list[idx_tp1].detach().cpu(),
            "feat_tp1":     feat_list[idx_tp1].detach().cpu(),
        })

    with torch.no_grad():
        for batch in loader:
            dt_list = batch.get("dt_list", None)
            if dt_list is None:
                raise RuntimeError("Missing dt_list in batch. Ensure CollateWithPrecompute attaches dt_list.")

            centers_list  = _require_list(batch, "centers_list")
            feat_list     = _require_list(batch, "feat_list")
            level_list    = _require_list(batch, "level_list")
            ij_list       = _require_list(batch, "ij_list")
            ei_list       = _require_list(batch, "ei_list")
            parents_list  = _require_list(batch, "parents_list")
            mask_list     = _require_list(batch, "mask_list")

            pred_centers_list  = _require_list(batch, "pred_centers_list")
            pred_levels_list   = _require_list(batch, "pred_levels_list")
            pred_parents_list  = _require_list(batch, "pred_parents_list")
            pred_ei_list       = _require_list(batch, "pred_ei_list")
            mask_pred_list     = _require_list(batch, "mask_pred_list")

            pred_ea_list = batch.get("pred_ea_list", None) or batch.get("pred_edge_attr_list", None)

            feat_t_on_pred_list   = _require_list(batch, "feat_t_on_pred_list")
            feat_tp1_on_pred_list = _require_list(batch, "feat_tp1_on_pred_list")

            pred2pred_idx_list = batch.get("pred2pred_idx_list", None)
            pred2pred_w_list   = batch.get("pred2pred_w_list", None)

            t_indices = batch.get("t_indices", None)

            K = len(centers_list)
            if K < 2:
                raise RuntimeError("window_size must be ≥ 2")

            if need_phy and pred_ea_list is None:
                raise RuntimeError(
                    "PARC/DEC enabled but batch is missing pred_ea_list / pred_edge_attr_list."
                )

            pred_lists = (pred_centers_list, pred_levels_list, pred_parents_list, pred_ei_list, mask_pred_list)
            dt_ref_scalar = batch.get("dt_ref", None)

            # helper closure for one step (mirrors train)
            def _run_step_eval(
                *,
                step_k: int,
                pred_centers,
                pred_levels,
                pred_parents,
                pred_ei,
                pred_ea,
                x_in_abs,
                x_tgt_abs,
                dt_phys,
            ):
                norm_in  = _maybe_norm(x_in_abs,  mu, sigma)
                norm_tgt = _maybe_norm(x_tgt_abs, mu, sigma)

                dt_ref_t = (torch.tensor(float(dt_ref_scalar), device=device, dtype=norm_in.dtype)
                            if dt_ref_scalar is not None else None)
                dt_hat = (dt_phys / dt_ref_t) if dt_ref_t is not None else dt_phys

                with amp_ctx():
                    pei = pred_ei.to(device) if torch.is_tensor(pred_ei) else pred_ei
                    pea = pred_ea.to(device) if (pred_ea is not None and torch.is_tensor(pred_ea)) else pred_ea

                    r_adv_abs = r_diff_abs = r_phy_abs = area = None
                    ch_mask = None

                    if need_phy:
                        with torch.autocast(device_type=device.type, enabled=False):
                            r_adv_abs, r_diff_abs, area = dec.dec_advdiff_terms_abs(
                                x_abs=x_in_abs.float(),
                                edge_index=pei.long(),
                                pred_ea=pea.float(),
                                levels=pred_levels.long().to(device),
                                dx0=float(dx),
                                dy0=float(dy),
                                cfg=cfg,
                                compute_adv=True,
                                compute_diff=True,
                            )
                            adv_w  = float(loss_cfg.get("adv_weight", 1.0))
                            diff_w = float(loss_cfg.get("diff_weight", 1.0))
                            r_phy_abs = adv_w * r_adv_abs + diff_w * r_diff_abs

                            ch_mask = dec.build_channel_mask_from_loss(
                                cfg, x_in_abs.size(1), device=device, dtype=torch.float32
                            )
                            if ch_mask is None:
                                idx_map = dec.infer_feature_indices(cfg, x_in_abs.size(1))
                                ch_mask = torch.zeros((x_in_abs.size(1),), device=device, dtype=torch.float32)
                                ch_mask[idx_map["rho"]] = 1.0
                                ch_mask[idx_map["E"]] = 1.0

                    x_in = _build_X(norm_in, pred_centers, pred_levels, cfg)

                    if parc_use and r_adv_abs is not None and r_diff_abs is not None:
                        parc_extra = dec.parc_terms_to_node_inputs(
                            r_adv_abs.to(device=device, dtype=torch.float32),
                            r_diff_abs.to(device=device, dtype=torch.float32),
                            dt_phys=dt_phys.to(device=device, dtype=torch.float32),
                            dt_ref=(dt_ref_t.to(device=device, dtype=torch.float32) if dt_ref_t is not None else None),
                            sigma=(sigma.to(device, dtype=torch.float32) if (sigma is not None and torch.is_tensor(sigma)) else None),
                            predict_type=predict_type,
                            cfg=cfg,
                            dtype=x_in.dtype,
                            detach=True,
                        )
                        if parc_extra.numel() > 0:
                            x_in = torch.cat([x_in, parc_extra], dim=1)

                    y_corr = _forward_main_head_with_edge_attr(model, x_in, pei, edge_attr=pea)
                    '''
                    y_pred = y_corr
                    if need_phy and (dec_blend_w != 0.0) and (r_phy_abs is not None):
                        phy_units = dec.physics_to_model_units(
                            r_phy_abs.to(dtype=y_corr.dtype),
                            dt_phys=dt_phys,
                            dt_ref=dt_ref_t,
                            sigma=(sigma.to(device, dtype=y_corr.dtype) if (sigma is not None and torch.is_tensor(sigma)) else None),
                            predict_type=predict_type,
                        )
                        phy_units = phy_units * ch_mask.view(1, -1).to(dtype=y_corr.dtype)
                        y_pred = y_corr + dec_blend_w * phy_units
                    '''
                    y_pred = y_corr
                    if need_phy and (dec_blend_w != 0.0) and (r_phy_abs is not None):
                        # ---- compute baseline in fp32 to avoid fp16 inf/NaN during eval rollouts ----
                        with torch.autocast(device_type=device.type, enabled=False):
                            phy_units_f32 = dec.physics_to_model_units(
                                r_phy_abs.to(dtype=torch.float32),
                                dt_phys=(dt_phys.to(dtype=torch.float32) if torch.is_tensor(dt_phys) else dt_phys),
                                dt_ref=(dt_ref_t.to(dtype=torch.float32) if (dt_ref_t is not None and torch.is_tensor(dt_ref_t)) else dt_ref_t),
                                sigma=(sigma.to(device, dtype=torch.float32) if (sigma is not None and torch.is_tensor(sigma)) else None),
                                predict_type=predict_type,
                            )
                            phy_units_f32 = phy_units_f32 * ch_mask.view(1, -1).to(dtype=torch.float32)

                        phy_units = phy_units_f32.to(dtype=y_corr.dtype)
                        y_pred = y_corr + dec_blend_w * phy_units
                    '''
                    delta_target = norm_tgt - norm_in
                    rate_target = delta_target / dt_hat.clamp_min(1e-12)

                    center_loss = (F.huber_loss(y_pred, rate_target, delta=huber_delta)
                                   if use_huber else F.l1_loss(y_pred, rate_target))

                    y_pred_abs = _maybe_denorm(norm_in + y_pred * dt_hat, mu, sigma)
                    '''
                    with torch.autocast(device_type=device.type, enabled=False):
                        delta_target_f32 = (norm_tgt - norm_in).to(dtype=torch.float32)
                        dt_hat_f32 = dt_hat.to(dtype=torch.float32)
                        rate_target_f32 = delta_target_f32 / dt_hat_f32.clamp_min(1e-12)
                        
                        # compare in fp32 (more stable), then cast loss back if you want
                        y_pred_f32 = y_pred.to(dtype=torch.float32)
                        center_loss = (F.huber_loss(y_pred_f32, rate_target_f32, delta=huber_delta)
                                       if use_huber else F.l1_loss(y_pred_f32, rate_target_f32))

                        y_pred_abs = _maybe_denorm(
                            norm_in.to(dtype=torch.float32) + y_pred_f32 * dt_hat_f32,
                            mu, sigma
                        )

                    lap_loss = (laplacian_smoothness(y_pred, pei) if lap_w > 0 else y_pred.new_zeros(()))
                    tmp_loss = (temporal_consistency(x_in, norm_tgt) if tmp_w > 0 else y_pred.new_zeros(()))

                    phy_loss = y_pred.new_zeros(())
                    if need_phy and dec_resid_w > 0.0 and (r_phy_abs is not None):
                        with torch.autocast(device_type=device.type, enabled=False):
                            phy_loss = dec.physics_residual_loss_delta(
                                y_pred_abs=y_pred_abs.float(),
                                x_in_abs=x_in_abs.float(),
                                dt_phys=dt_phys.float(),      # <-- correct dt for this step
                                r_phy_abs=r_phy_abs.float(),
                                area=area.float(),
                                sigma=(sigma.to(device, dtype=torch.float32) if (sigma is not None and torch.is_tensor(sigma)) else None),
                                channel_mask=ch_mask,
                            ).to(dtype=y_pred.dtype)

                    loss_step = center_loss + lap_w * lap_loss + tmp_w * tmp_loss + dec_resid_w * phy_loss

                return loss_step, y_pred_abs

            # ===== STEP 0 =====
            k = 0
            pred_centers_1, pred_levels_1, pred_parents_1, pred_ei_1, mask_pred_1 = _pred_mesh_for_step_strict(k, pred_lists=pred_lists)
            pred_ea_1 = pred_ea_list[k + 1] if pred_ea_list is not None else None

            x_in_abs0  = feat_t_on_pred_list[k + 1].to(device)
            x_tgt_abs0 = feat_tp1_on_pred_list[k + 1].to(device)

            dt0 = dt_list[0]
            dt0 = dt0.to(device=device, dtype=x_in_abs0.dtype) if torch.is_tensor(dt0) else torch.tensor(float(dt0), device=device, dtype=x_in_abs0.dtype)

            loss0, y_pred_abs0 = _run_step_eval(
                step_k=0,
                pred_centers=pred_centers_1,
                pred_levels=pred_levels_1,
                pred_parents=pred_parents_1,
                pred_ei=pred_ei_1,
                pred_ea=pred_ea_1,
                x_in_abs=x_in_abs0,
                x_tgt_abs=x_tgt_abs0,
                dt_phys=dt0,
            )

            total_loss_accum += float(loss0.detach().cpu())
            n_steps_total += 1

            t_abs0 = None
            if t_indices is not None:
                t_abs0 = int(t_indices[1].item()) if torch.is_tensor(t_indices) else int(t_indices[1])

            _accumulate_metrics(k=0, t_abs=t_abs0, pred_abs=y_pred_abs0, gt_abs=x_tgt_abs0, pred_levels=pred_levels_1)

            _append_example_step(
                step_idx=0,
                pred_centers=pred_centers_1,
                pred_levels=pred_levels_1,
                pred_parents=pred_parents_1,
                y_pred_step_abs=y_pred_abs0,
                centers_list=centers_list,
                feat_list=feat_list,
                level_list=level_list,
                parents_list=parents_list,
                batch=batch,
            )

            pred_feats_k = y_pred_abs0

            # ===== STEPS 1..K-2 =====
            for k in range(1, K - 1):
                pred_centers_next, pred_levels_next, pred_parents_next, pred_ei_next, mask_pred_next = _pred_mesh_for_step_strict(k, pred_lists=pred_lists)
                pred_ea_next = pred_ea_list[k + 1] if pred_ea_list is not None else None

                idx_km1 = pred2pred_idx_list[k-1].to(device)
                w_km1   = pred2pred_w_list[k-1].to(device)
                x_in_abs = apply_precomputed_idw_map(idx_km1, w_km1, pred_feats_k).to(device)

                x_tgt_abs = feat_tp1_on_pred_list[k + 1].to(device)

                dtk = dt_list[k]
                dtk = dtk.to(device=device, dtype=x_in_abs.dtype) if torch.is_tensor(dtk) else torch.tensor(float(dtk), device=device, dtype=x_in_abs.dtype)

                loss_k, y_pred_abs_k = _run_step_eval(
                    step_k=k,
                    pred_centers=pred_centers_next,
                    pred_levels=pred_levels_next,
                    pred_parents=pred_parents_next,
                    pred_ei=pred_ei_next,
                    pred_ea=pred_ea_next,
                    x_in_abs=x_in_abs,
                    x_tgt_abs=x_tgt_abs,
                    dt_phys=dtk,
                )

                total_loss_accum += float(loss_k.detach().cpu())
                n_steps_total += 1

                t_absk = None
                if t_indices is not None:
                    t_absk = int(t_indices[k + 1].item()) if torch.is_tensor(t_indices) else int(t_indices[k + 1])

                _accumulate_metrics(k=k, t_abs=t_absk, pred_abs=y_pred_abs_k, gt_abs=x_tgt_abs, pred_levels=pred_levels_next)

                _append_example_step(
                    step_idx=k,
                    pred_centers=pred_centers_next,
                    pred_levels=pred_levels_next,
                    pred_parents=pred_parents_next,
                    y_pred_step_abs=y_pred_abs_k,
                    centers_list=centers_list,
                    feat_list=feat_list,
                    level_list=level_list,
                    parents_list=parents_list,
                    batch=batch,
                )

                pred_feats_k = y_pred_abs_k

    # ---- finalize metrics ----
    eps = 1e-12
    S = len(step_wsum)
    if S == 0:
        raise RuntimeError("No steps accumulated; check loader/window_size.")

    Fdim = step_mae_num[0].numel()
    maew_feat_by_step = torch.zeros((S, Fdim), dtype=torch.float64)
    rell2w_feat_by_step = torch.zeros((S, Fdim), dtype=torch.float64)
    maew_by_step = []
    rell2w_by_step = []

    for k in range(S):
        wsum = step_wsum[k]
        if wsum <= 0:
            maew_feat = torch.full((Fdim,), float("nan"), dtype=torch.float64)
            rell2w_feat = torch.full((Fdim,), float("nan"), dtype=torch.float64)
        else:
            maew_feat = step_mae_num[k] / wsum
            rell2w_feat = torch.sqrt(step_mse_num[k] / (step_gt2_num[k] + eps))
        maew_feat_by_step[k] = maew_feat
        rell2w_feat_by_step[k] = rell2w_feat
        maew_by_step.append(float(maew_feat.mean().item()))
        rell2w_by_step.append(float(rell2w_feat.mean().item()))

    t_values = sorted(by_t.keys())
    if len(t_values) > 0:
        maew_feat_by_t = torch.zeros((len(t_values), Fdim), dtype=torch.float64)
        rell2w_feat_by_t = torch.zeros((len(t_values), Fdim), dtype=torch.float64)
        maew_by_t = []
        rell2w_by_t = []
        for i, t_abs in enumerate(t_values):
            rec = by_t[t_abs]
            wsum = rec["wsum"]
            if wsum <= 0:
                maew_feat = torch.full((Fdim,), float("nan"), dtype=torch.float64)
                rell2w_feat = torch.full((Fdim,), float("nan"), dtype=torch.float64)
            else:
                maew_feat = rec["mae"] / wsum
                rell2w_feat = torch.sqrt(rec["mse"] / (rec["gt2"] + eps))
            maew_feat_by_t[i] = maew_feat
            rell2w_feat_by_t[i] = rell2w_feat
            maew_by_t.append(float(maew_feat.mean().item()))
            rell2w_by_t.append(float(rell2w_feat.mean().item()))
    else:
        maew_feat_by_t = None
        rell2w_feat_by_t = None
        maew_by_t = None
        rell2w_by_t = None

    avg_loss = total_loss_accum / max(n_steps_total, 1)

    stats = {
        "num_windows": len(loader),
        "num_steps": n_steps_total,
        "maew_by_rollout_step": maew_by_step,
        "rell2w_by_rollout_step": rell2w_by_step,
        "maew_feat_by_rollout_step": maew_feat_by_step,
        "rell2w_feat_by_rollout_step": rell2w_feat_by_step,
        "t_values": t_values,
        "maew_by_t": maew_by_t,
        "rell2w_by_t": rell2w_by_t,
        "maew_feat_by_t": maew_feat_by_t,
        "rell2w_feat_by_t": rell2w_feat_by_t,
    }
    if collect_examples:
        stats["examples"] = examples
    return avg_loss, stats


def parent_mask_from_selected(pred_parents: torch.Tensor,
                              pred_levels: torch.Tensor,
                              H: int, W: int,
                              min_level: int = 1) -> torch.Tensor:
    """
    Returns an (H, W) bool mask over the coarse-parent grid.
    A parent cell is marked True if ANY selected node with level >= min_level
    belongs to that parent.
    """
    device = pred_parents.device
    m = torch.zeros(H * W, dtype=torch.bool, device=device)
    sel = (pred_levels >= min_level)
    if sel.any():
        p = pred_parents[sel].long().clamp_(0, H*W - 1)
        m[p] = True
    return m.view(H, W)


# --- put this at top-level in train_v2.py (outside any function) ---
def identity_collate(batch):
    # batch is a list of length == batch_size; we use batch_size=1
    return batch[0]


def build_model_from_cfg(cfg, device):

    # match how you did it in main():
    H = int(cfg["data"].get("H", 64)); W = int(cfg["data"].get("W", 64))
    # infer F the same way as in main (using the dataset field you used there)
    # If you want to avoid touching data here, you can instead use len(cfg["features"]["use_columns"])
    use_cols = cfg.get("features", {}).get("use_columns", [0,1,3])
    F = len(use_cols)

    b = cfg.get("features", {}).get("build", {})
    in_ch  = F + (2 if b.get("use_pos", True)   else 0) + (1 if b.get("use_level", True) else 0)
    if cfg.get("loss", {}).get("parc", False) or cfg.get("loss", {}).get("parc_inputs", False):
        # Fdim is your state feature count (4 for your main task)
        Fdim = int(cfg.get("features", {}).get("num_features", 4))  # or however you compute F
        extra = dec.parc_extra_in_channels(cfg, Fdim)
        in_ch += extra

    out_ch = F

    model = FeatureNet(
        in_channels=in_ch,
        out_channels=out_ch,
        hidden=int(cfg.get("model", {}).get("hidden", 128)),   # must match the training cfg
        layers=int(cfg.get("model", {}).get("layers", 3)),      # must match the training cfg
        dropout=float(cfg.get("model", {}).get("dropout", 0.1)),
        make_score_head=True,
    ).to(device)
    
    return model

# ------------------------------ Main ------------------------------

def main(config_path: str | None = None):
    # -------- load & normalize config --------
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config_feature_first.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    loss = cfg.setdefault("loss", {})

    # Do NOT force dec on by default unless you really mean it.
    loss.setdefault("dec", False)

    loss.setdefault("mode", "diffusion")           # "diffusion" | "advection" | "advdiff"
    loss.setdefault("blend_weight", 0.0)
    loss.setdefault("residual_weight", 0.0)

    loss.setdefault("nu", 0.0)
    loss.setdefault("advection_scheme", "upwind")
    loss.setdefault("rho_eps", 1e-8)

    loss.setdefault("parc", False)                 # enable PARC operator inputs + Variant-B baseline behavior
    loss.setdefault("parc_input_form", "rate")     # "rate" (dt_ref*r/sigma) or "delta" (dt*r/sigma)
    loss.setdefault("parc_include_adv", True)
    loss.setdefault("parc_include_diff", True)
    loss.setdefault("parc_input_weighted", False)  # usually False
    loss.setdefault("parc_detach_inputs", True)

    # If PARC is enabled, recommend:
    # - ensure DEC geometry exists
    # - apply baseline by default
    if bool(loss.get("parc", False)):
        loss["dec"] = True
        loss.setdefault("blend_weight", 1.0)       # Variant-B baseline strength (1.0 recommended)
        loss.setdefault("channels", ["rho", "E"])  # default physics channels

    mode = str(loss.get("mode", "diffusion")).lower()

    # Enforce mode semantics if weights aren't explicitly provided
    if "adv_weight" not in loss and "diff_weight" not in loss:
        if mode == "diffusion":
            loss["adv_weight"] = 0.0
            loss["diff_weight"] = 1.0
        elif mode == "advection":
            loss["adv_weight"] = 1.0
            loss["diff_weight"] = 0.0
        else:  # "advdiff"
            loss["adv_weight"] = 1.0
            loss["diff_weight"] = 1.0
    else:
        loss.setdefault("adv_weight", 0.0)
        loss.setdefault("diff_weight", 1.0)

    print("[DEC-CFG] dec=", loss.get("dec"),
      "mode=", loss.get("mode"),
      "adv_w=", loss.get("adv_weight"),
      "diff_w=", loss.get("diff_weight"),
      "nu=", loss.get("nu"),
      "resid_w=", loss.get("residual_weight"),
      "blend_w=", loss.get("blend_weight"),
      "channels=", loss.get("channels"))

    # Speed defaults (non‑breaking):
    cfg.setdefault("speed", {}).setdefault("amp", True)
    cfg.setdefault("speed", {}).setdefault("interp_chunk", 8192)
    cfg.setdefault("speed", {}).setdefault("knn_k", 8)
    cfg.setdefault("speed", {}).setdefault("cache_interps", True)

    # Ensure dataset sees the intended columns
    if cfg.get("data", {}).get("feature_idx") and not cfg.get("features", {}).get("use_columns"):
        cfg.setdefault("features", {}).setdefault("use_columns", cfg["data"]["feature_idx"])

    #device = pick_device(cfg.get("train", {}).get("device", "auto"))
    raw_dev = cfg.get("device", "cpu")
    device = torch.device(raw_dev)
    pre_device = torch.device("cpu")  # Pretraining steps are actually faster on CPU
    #print(f"[INFO] Using device: {device}")
    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    H = int(cfg["data"].get("H", 64)); W = int(cfg["data"].get("W", 64))
    xmin, xmax, ymin, ymax = _get_bbox(cfg)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H

    # -------- dataset & loaders --------
    pt_path = cfg["data"]["pt_path"]
    if pt_path.endswith(".zip"):
        with zipfile.ZipFile(pt_path, "r") as zf:
            member = next(m for m in zf.namelist() if m.endswith(".pt") or m.endswith(".pth"))
            with zf.open(member, "r") as fh:
                buf = io.BytesIO(fh.read())
        data_list = torch.load(buf)
    else:
        data_list = torch.load(pt_path)

    model = build_model_from_cfg(cfg, device)

    def _get_lr(optimizer):
            return optimizer.param_groups[0]["lr"]

    # 1) Build modules first
    opt_groups = [{"params": model.parameters(), "lr": float(cfg["train"]["lr"])}]

    if cfg.get("train", {}).get("interp_type", "standard") == "gnn":
        D      = int(cfg["features"].get("num_dynamic_feats", 3))
        phid   = int(cfg.get("interp_nn", {}).get("hidden", 256))
        pdepth = int(cfg.get("interp_nn", {}).get("depth", 2))

        prolong_head     = ProlongationHead(D, hidden=phid, depth=pdepth).to(device)
        restriction_head = RestrictionHead(D, hidden=phid, depth=pdepth).to(device)
        N4 = build_coarse_n4(H, W, device=device)

        base_lr = float(cfg.get("interp_nn", {}).get("lr", cfg.get("optim", {}).get("lr", 1e-3)))
        opt_groups.append({"params": prolong_head.parameters(), "lr": base_lr})
        opt_groups.append({"params": restriction_head.parameters(), "lr": base_lr})

    # 2) Create optimizer with *all* groups already present
    opt = optim.AdamW(opt_groups, weight_decay=float(cfg["train"].get("weight_decay", 0.0)))

    # 3) Now create the scheduler (safe: param_groups size is final)
    sch_cfg   = cfg.get("scheduler", {})
    use_sched = bool(sch_cfg.get("use", True))
    scheduler = None
    if use_sched:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=sch_cfg.get("mode", "min"),
            factor=float(sch_cfg.get("factor", 0.5)),
            patience=int(sch_cfg.get("patience", 5)),
            threshold=float(sch_cfg.get("threshold", 1e-4)),
            threshold_mode=sch_cfg.get("threshold_mode", "rel"),
            cooldown=int(sch_cfg.get("cooldown", 0)),
            min_lr=float(sch_cfg.get("min_lr", 1e-6)),
            #verbose=bool(sch_cfg.get("verbose", True)),
        )

    # Load the raw list[Data] and let the dataset preprocess ONCE in memory:
    #raw_series = torch.load(raw_path, map_location="cpu")  # list of Data (one per timestep)

    K      = int(cfg["data"].get("window_size", 2))
    stride = int(cfg["data"].get("stride", 1))

    min_T = None
    max_T = None
    #if max_T is not None:
    #    data_list = data_list[min_T:max_T]  
    if min_T is not None or max_T is not None:
        n_total = len(data_list)
        start = 0 if min_T is None else int(min_T)
        end   = n_total if max_T is None else int(max_T)

        # clamp to valid bounds
        start = max(0, min(start, n_total))
        end   = max(start, min(end, n_total))  # ensure end >= start

        data_list = data_list[start:end]

        if len(data_list) == 0:
            raise ValueError(
                f"Selected empty time range: start={start}, end={end}, "
                f"n_total={n_total}"
            )

    # -------------------------------
    # NEW: build dt transitions (t -> t+1) from the raw snapshots
    # -------------------------------
    # Each snapshot has snapshot["time"] saved by the reader. :contentReference[oaicite:1]{index=1}
    times = torch.stack([
        (snap["time"].detach().cpu().float() if torch.is_tensor(snap["time"]) else torch.tensor(float(snap["time"])))
        for snap in data_list
    ]).view(-1)  # shape (T,)

    # Sanity check: times should be nondecreasing
    if torch.any(times[1:] < times[:-1]):
        raise RuntimeError("Snapshot times are not monotonically increasing; cannot form dt reliably.")

    dt_transitions = (times[1:] - times[:-1]).contiguous()  # shape (T-1,)
    eps_dt = 1e-12
    if torch.any(dt_transitions <= 0):
        # You can decide whether to raise or clamp; clamping avoids divide-by-zero explosions.
        dt_transitions = dt_transitions.clamp_min(eps_dt)

    # Optional: choose a reference dt to keep magnitudes stable (recommended but not required)
    dt_ref = dt_transitions.median()  # scalar

    full_ds = CellRefineWindowDataset(
        series=data_list,       # <-- pass the list directly
        cfg=cfg,
        window_size=K,
        stride=stride,
        H=H, W=W, device=str(device)
        # is_processed_file can be left None; passing raw list triggers in-memory preprocess
    )

    N = len(full_ds)
    train_frac = float(cfg["split"].get("train", 0.8))
    val_frac = float(cfg["split"].get("val_frac", 0.1))
    test_frac = float(cfg["split"].get("test_frac", 0.1))
    nv = max(1, int(round(val_frac * N)))
    nt = max(1, int(round(test_frac * N)))
    ntr = max(1, N - nv - nt)


    T = len(full_ds)  # number of windows, not raw timesteps
    idxs = np.arange(T)

    # ---- RANDOMLY SHUFFLE IDXs BEFORE SPLIT ----
    seed = int(cfg.get("seed", 1337))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(T)
    idxs = idxs[perm]

    # now compute split sizes
    train_frac = cfg["split"].get("train", 0.8)
    val_frac   = cfg["split"].get("val", 0.1)
    test_frac   = cfg["split"].get("test", 0.1)
    #test_frac  = 1.0 - train_frac - val_frac

    n_train = int(round(train_frac * T))
    n_val   = int(round(val_frac   * T))
    # keep the rest for test
    n_test  = T - n_train - n_val

    train_idx = idxs[:n_train]
    val_idx   = idxs[n_train:n_train + n_val]
    test_idx  = idxs[n_train + n_val:]

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    test_ds  = Subset(full_ds, test_idx.tolist())

    cache_path = cfg["train"].get("precomp_cache_path", None)
    force_recompute = bool(cfg["train"].get("precomp_force_recompute", False))

    precomp = precompute_pred_mesh_and_interps_for_rollout(
        full_ds.steps,
        cfg,
        H,
        W,
        dx,
        dy,
        device=device,
        progress=True,
        cache_path=cache_path,
        force_recompute=force_recompute,
    )

    precomp = move_precomp_to_device(precomp, device)

    # --- DEBUG: verify DEC edge attributes exist in precomp ---
    if cfg.get("debug", {}).get("print_dec_checks", False):
        t = 1
        pea = precomp["pred_edge_attr"][t] if "pred_edge_attr" in precomp else None
        print("[DEC-CHK] loaded pred_edge_attr[t=1]:", None if pea is None else (tuple(pea.shape), pea.dtype, pea.device))

        print("[DEC-CHK] precomp keys:", sorted(list(precomp.keys())))

        pea_key = "pred_edge_attr" if "pred_edge_attr" in precomp else ("pred_ea" if "pred_ea" in precomp else None)
        if pea_key is None:
            print("[DEC-CHK] precomp has NO pred_edge_attr/pred_ea")
        else:
            pea = precomp[pea_key]
            pei = precomp["pred_ei"]
            print(f"[DEC-CHK] precomp[{pea_key}] length={len(pea)}; precomp[pred_ei] length={len(pei)}")
            # show one timestep shapes
            i0 = 0
            if torch.is_tensor(pea[i0]) and torch.is_tensor(pei[i0]):
                print(f"[DEC-CHK] t={i0}: pred_ei shape={tuple(pei[i0].shape)} dtype={pei[i0].dtype} dev={pei[i0].device}")
                print(f"[DEC-CHK] t={i0}: pred_ea shape={tuple(pea[i0].shape)} dtype={pea[i0].dtype} dev={pea[i0].device}")
                if pea[i0].ndim == 2 and pea[i0].size(1) >= 5:
                    cols = pea[i0].size(1)
                    print(f"[DEC-CHK] pred_ea columns={cols} (expect >=5: nx,ny,face_len,dual_len,tau)")

    dec_cfg = cfg.get("loss", {}) or {}
    
    if bool(cfg.get("loss", {}).get("dec", False)) and (
        float(dec_cfg.get("blend_weight", 0.0)) != 0.0 or float(dec_cfg.get("residual_weight", 0.0)) != 0.0
    ):
        # After move_precomp_to_device(precomp, device)
        # precomp must include a list aligned to pred_ei_list, e.g. key "pred_ea_list" or "pred_edge_attr_list"
        if isinstance(precomp, dict):
            if ("pred_edge_attr" not in precomp) and ("pred_ea" not in precomp):
                raise RuntimeError(
                    "DEC enabled but precomp is missing pred_ea_list/pred_edge_attr_list. "
                    "Update H5->precomp loader and CollateWithPrecompute to read/store pred_edge_attr."
                )

    collate = CollateWithPrecompute(precomp, dt_transitions=dt_transitions, dt_ref=dt_ref)

    train_loader = DataLoader(
        train_ds, batch_size=1, sampler=RandomSampler(train_ds),
        num_workers=0, pin_memory=False, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, sampler=RandomSampler(val_ds),
        num_workers=0, pin_memory=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, sampler=RandomSampler(test_ds),
        num_workers=0, pin_memory=False, collate_fn=collate
    )

     # ---- Normalization stats (μ/σ) ----
    feats_cfg = cfg.get("features", {})
    do_norm = bool(feats_cfg.get("normalize", True))
    #mu, sigma = _as_mu_sigma(cfg, device)
    mu = sigma = None

    if do_norm and (mu is None or sigma is None):
        # compute from training loader on CPU/GPU (device already set)
        mu, sigma = _compute_norm_stats_from_loader(train_loader, device)
        # persist into cfg so eval uses the same stats
        cfg.setdefault("features", {}).setdefault("norm_stats", {})
        cfg["features"]["norm_stats"]["mu"] = mu.tolist()
        cfg["features"]["norm_stats"]["sigma"] = sigma.tolist()
        # keep tensors on device for this process
        mu = mu.to(device)
        sigma = sigma.to(device)
    elif do_norm:
        # stats provided in config
        mu = mu.to(device)
        sigma = sigma.to(device)
    else:
        mu = sigma = None
    
    # -------- training loop --------
    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)
    log_csv = os.path.join(cfg["train"]["save_dir"], "train_log.csv")
    with open(log_csv, "w") as f:
        f.write("epoch,split,loss,mae\n")

    print("[INFO] Starting training...")
    best_val = float("inf")
    TR, VL = [], []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        t0 = time.time()
        batch = next(iter(train_loader))

        # unpack: loss, mae, stats
        tr_loss, tr_mae, tr_stats = train_one_epoch_multi_step(
            model, train_loader, opt, cfg, device,
            H=H, W=W, dx=dx, dy=dy, mu=mu, sigma=sigma
        )
        vl_loss, vl_stats = evaluate_one_epoch_multi_step(
            model, val_loader, cfg, device,
            H=H, W=W, dx=dx, dy=dy, mu=mu, sigma=sigma,
            collect_examples=False
        )

        TR.append(tr_loss)
        VL.append(vl_loss)
        with open(log_csv, "a") as f:
            # epoch, split, loss, mae
            f.write(f"{epoch},train,{tr_loss:.10f},{tr_mae:.10f}\n")
            f.write(f"{epoch},val,{vl_loss:.6f}\n")

        # track best model by validation loss
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(),
                    os.path.join(cfg["train"]["save_dir"], "best_model.pt"))

        dt = time.time() - t0

        # keep ReduceLROnPlateau param-groups safe
        if scheduler is not None and hasattr(scheduler, "min_lrs"):
            if len(scheduler.min_lrs) != len(opt.param_groups):
                base_min = scheduler.min_lrs[0] if scheduler.min_lrs else float(
                    cfg.get("scheduler", {}).get("min_lr", 1e-6)
                )
                scheduler.min_lrs = [base_min] * len(opt.param_groups)

        if scheduler is not None:
            scheduler.step(vl_loss)

        print(
            f"[INFO] Epoch {epoch:03d}: "
            f"train {tr_loss:.6f} (MAE {tr_mae:.6f}) | "
            f"val {vl_loss:.6f} | "
            f"{dt:.1f}s | lr={_get_lr(opt):.3e}"
        )

    def _precomp_to_cpu(precomp):
        out = {}
        for key, lst in precomp.items():
            if isinstance(lst, list):
                new_lst = []
                for v in lst:
                    if torch.is_tensor(v):
                        new_lst.append(v.detach().cpu())
                    elif isinstance(v, tuple):
                        # handle (idx, w) style entries if you ever store tuples
                        new_lst.append(
                            tuple(x.detach().cpu() if torch.is_tensor(x) else x
                                for x in v)
                        )
                    else:
                        new_lst.append(v)
                out[key] = new_lst
            else:
                out[key] = lst
        return out

    precomp_cpu = _precomp_to_cpu(precomp)

    # ---- save "last" bundle with norm stats & cfg for reproducibility ----
    save_dict = {
        "model": model.state_dict(),
        "norm_stats": {
            "mu": None if mu is None else mu.detach().cpu().tolist(),
            "sigma": None if sigma is None else sigma.detach().cpu().tolist(),
        },
        "cfg": cfg,
        "precomp": precomp_cpu
    }
    torch.save(save_dict, os.path.join(cfg["train"]["save_dir"], "last_model.pt"))

    plot_loss_curves(
        os.path.join(cfg["train"]["save_dir"], "loss_curves.png"),
        list(range(1, len(TR) + 1)),
        TR, VL
    )

    # -------- qualitative PDFs on test --------
    print("test_loader length:", len(test_loader))
    print("=== Running test evaluation ===")
    test_loss, test_stats = evaluate_one_epoch_multi_step(
        model,
        test_loader,
        cfg,
        device,
        H=H,
        W=W,
        dx=dx,
        dy=dy,
        mu=mu,
        sigma=sigma,
        collect_examples=True,
    )

    print(f"[TEST] loss={test_loss:.4e}")

    test_examples = test_stats["examples"] 
    num = int(cfg.get("eval", {}).get("num_examples", -1))
    if num < 0:
        num = len(test_examples)

    # AMR mesh triptychs (existing)
    titles = [f"sample {e['t']}" for e in test_examples[:num]]

    # Build the three mask lists in a way that respects mesh_mode
    mesh_mode = str(cfg.get("train", {}).get("mesh_mode", "predicted")).lower()

    first = test_examples[0]
    # Build coarse parent mask for predicted L1+ refinement
    M_pred_L1 = parent_mask_from_selected(first["pred_parents"], first["pred_levels"], H, W, min_level=1)

    """
    print("[INFO] Generating qualitative PDFs...")
    plot_predictions_from_examples_pdf(
        os.path.join(cfg["train"]["save_dir"], "predictions_from_examples.pdf"),
        test_examples[:num],
        H, W,
        bbox=tuple(cfg.get("data", {}).get("bbox", (0.0, 1.0, 0.0, 1.0))),
        titles=[f"sample {e['t']}" for e in test_examples[:num]],
    )
    """

    print("[INFO] Generating 2x3 feature PDFs...")
    # Per-sample 2×3 feature pages (GT vs Pred)
    # Prefer selected-mesh tensors if present; otherwise fall back to coarse/fine rasters
    feat_names = (cfg.get("features", {}) or {}).get("dataset_order")
    def _names_for(T, names):
        if not names: return None
        F = T.size(1) if (T is not None and hasattr(T, "size")) else len(names)
        return names[:F]

    for i, e in enumerate(test_examples[:num]):
        out_pdf = os.path.join(cfg["train"]["save_dir"], f"qual_2x3_sample_{i:03d}.pdf")
        title = titles[i]

        # Preferred: selected-mesh values provided by evaluate_mesh_first
        has_selected = all(k in e for k in (
            "centers_selected", "levels_selected",
            "gt_selected_t", "gt_selected_tp1", "pred_selected_tp1"
        ))

        # Optional: use domain bbox from cfg if you have it
        bbox = tuple(cfg["data"]["bbox"]) if "data" in cfg and "bbox" in cfg["data"] else Non

    out_pdf = os.path.join(cfg["train"]["save_dir"], "qual_with_deltas.pdf")
    feature_names = cfg.get("features", {}).get("names", ["U", "V", "E"])
    '''
    print(f"[INFO] Generating qualitative PDF with deltas: {out_pdf}")
    #for ex in test_examples:
    #    print("t: ", int(ex["t"]))

    plot_qual_pdf(
        examples=test_examples,
        cfg=cfg,
        out_pdf_path=out_pdf,
        feature_names=feature_names,
        unify_clims=False,
        dpi=150,
        rasterize=True,
        colorbars="row",
    )
    '''
    # This is very computionally heavy; use only if needed
    """
    plot_qual_2x3_pdf_with_cells(
        examples=test_examples,
        cfg=cfg,
        out_pdf_path=out_pdf,
        feature_names=feature_names,
        unify_clims=False,  # set True if you want global color limits across all pages
    )
    """
    """
    proj_titles = [f"sample {e['t']} [{mesh_mode}]" for e in test_examples[:num]]

    # use selected-mesh outputs so resolution matches the mesh (H×W or 2H×2W)
    centers_list = [e["centers_selected"]   for e in test_examples[:num]]
    levels_list  = [e["levels_selected"]    for e in test_examples[:num]]
    pred_list    = [e["pred_selected_tp1"]  for e in test_examples[:num]]

    feat_names = (cfg.get("features", {}) or {}).get("dataset_order")
    proj_pdf = os.path.join(cfg["train"]["save_dir"], "pred_x_projection.pdf")

    # bbox for labeling (falls back to [0,1]×[0,1] if not provided)
    bbox = tuple(cfg.get("data", {}).get("bbox", (0.0, 1.0, 0.0, 1.0)))
    
    plot_pred_projection_x_pdf(
        proj_pdf,
        centers_list, levels_list, pred_list,
        H, W,
        titles=proj_titles,
        feature_names=feat_names,
        bbox=bbox,
        reduction="sum",       # "mean" or "sum"
        max_features=3
    )
    """
'''
def main(config_path: str | None = None):
    import os, io, json, time, zipfile
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader, Subset, RandomSampler

    # -------- load & normalize config --------
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config_feature_first.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Speed defaults (non-breaking):
    cfg.setdefault("speed", {}).setdefault("amp", True)
    cfg.setdefault("speed", {}).setdefault("interp_chunk", 8192)
    cfg.setdefault("speed", {}).setdefault("knn_k", 8)
    cfg.setdefault("speed", {}).setdefault("cache_interps", True)

    # Default mesh mode is the current behavior
    #   "predicted" => EXACT existing precomputed non-uniform mesh pipeline
    #   "uniform"   => constant uniform mesh test
    mesh_mode = str(cfg.get("train", {}).get("mesh_mode", "predicted")).lower()

    # Uniform mesh dims (only used in uniform mode)
    uniform_H = int(cfg.get("train", {}).get("uniform_H", 256))
    uniform_W = int(cfg.get("train", {}).get("uniform_W", 256))
    uniform_teacher_forcing = bool(cfg.get("train", {}).get("teacher_forcing_uniform", False))
    uniform_diag_edges = bool(cfg.get("train", {}).get("uniform_diag_edges", False))

    # Ensure dataset sees the intended columns
    if cfg.get("data", {}).get("feature_idx") and not cfg.get("features", {}).get("use_columns"):
        cfg.setdefault("features", {}).setdefault("use_columns", cfg["data"]["feature_idx"])

    raw_dev = cfg.get("device", "cpu")
    device = torch.device(raw_dev)
    pre_device = torch.device("cpu")  # kept to match your current structure

    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    H = int(cfg["data"].get("H", 64))
    W = int(cfg["data"].get("W", 64))
    xmin, xmax, ymin, ymax = _get_bbox(cfg)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H

    # -------- dataset loading --------
    pt_path = cfg["data"]["pt_path"]
    if pt_path.endswith(".zip"):
        with zipfile.ZipFile(pt_path, "r") as zf:
            member = next(m for m in zf.namelist() if m.endswith(".pt") or m.endswith(".pth"))
            with zf.open(member, "r") as fh:
                buf = io.BytesIO(fh.read())
        data_list = torch.load(buf)
    else:
        data_list = torch.load(pt_path)

    model = build_model_from_cfg(cfg, device)

    def _get_lr(optimizer):
        return optimizer.param_groups[0]["lr"]

    # -------- optimizer / scheduler (UNCHANGED) --------
    opt_groups = [{"params": model.parameters(), "lr": float(cfg["train"]["lr"])}]

    if cfg.get("train", {}).get("interp_type", "standard") == "gnn":
        D      = int(cfg["features"].get("num_dynamic_feats", 3))
        phid   = int(cfg.get("interp_nn", {}).get("hidden", 256))
        pdepth = int(cfg.get("interp_nn", {}).get("depth", 2))

        prolong_head     = ProlongationHead(D, hidden=phid, depth=pdepth).to(device)
        restriction_head = RestrictionHead(D, hidden=phid, depth=pdepth).to(device)
        N4 = build_coarse_n4(H, W, device=device)

        base_lr = float(cfg.get("interp_nn", {}).get("lr", cfg.get("optim", {}).get("lr", 1e-3)))
        opt_groups.append({"params": prolong_head.parameters(), "lr": base_lr})
        opt_groups.append({"params": restriction_head.parameters(), "lr": base_lr})

    opt = optim.AdamW(opt_groups, weight_decay=float(cfg["train"].get("weight_decay", 0.0)))

    sch_cfg   = cfg.get("scheduler", {})
    use_sched = bool(sch_cfg.get("use", True))
    scheduler = None
    if use_sched:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=sch_cfg.get("mode", "min"),
            factor=float(sch_cfg.get("factor", 0.5)),
            patience=int(sch_cfg.get("patience", 5)),
            threshold=float(sch_cfg.get("threshold", 1e-4)),
            threshold_mode=sch_cfg.get("threshold_mode", "rel"),
            cooldown=int(sch_cfg.get("cooldown", 0)),
            min_lr=float(sch_cfg.get("min_lr", 1e-6)),
            verbose=bool(sch_cfg.get("verbose", True)),
        )

    # -------- windows dataset (UNCHANGED) --------
    K      = int(cfg["data"].get("window_size", 2))
    stride = int(cfg["data"].get("stride", 1))

    min_T = None
    max_T = None
    if min_T is not None or max_T is not None:
        n_total = len(data_list)
        start = 0 if min_T is None else int(min_T)
        end   = n_total if max_T is None else int(max_T)

        start = max(0, min(start, n_total))
        end   = max(start, min(end, n_total))
        data_list = data_list[start:end]

        if len(data_list) == 0:
            raise ValueError(
                f"Selected empty time range: start={start}, end={end}, n_total={n_total}"
            )

    # -------------------------------
    # Build dt transitions from raw snapshots (UNCHANGED)
    # -------------------------------
    times = torch.stack([
        (snap["time"].detach().cpu().float() if torch.is_tensor(snap["time"]) else torch.tensor(float(snap["time"])))
        for snap in data_list
    ]).view(-1)

    if torch.any(times[1:] < times[:-1]):
        raise RuntimeError("Snapshot times are not monotonically increasing; cannot form dt reliably.")

    dt_transitions = (times[1:] - times[:-1]).contiguous()  # (T-1,)
    eps_dt = 1e-12
    if torch.any(dt_transitions <= 0):
        dt_transitions = dt_transitions.clamp_min(eps_dt)

    dt_ref = dt_transitions.median()  # scalar

    full_ds = CellRefineWindowDataset(
        series=data_list,
        cfg=cfg,
        window_size=K,
        stride=stride,
        H=H, W=W, device=str(device),
    )

    N = len(full_ds)
    train_frac = float(cfg["split"].get("train", 0.8))
    val_frac = float(cfg["split"].get("val_frac", 0.1))
    test_frac = float(cfg["split"].get("test_frac", 0.1))
    nv = max(1, int(round(val_frac * N)))
    nt = max(1, int(round(test_frac * N)))
    ntr = max(1, N - nv - nt)

    T = len(full_ds)
    idxs = np.arange(T)

    seed = int(cfg.get("seed", 1337))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(T)
    idxs = idxs[perm]

    train_frac = cfg["split"].get("train", 0.8)
    val_frac   = cfg["split"].get("val", 0.1)
    test_frac  = cfg["split"].get("test", 0.1)

    n_train = int(round(train_frac * T))
    n_val   = int(round(val_frac   * T))
    n_test  = T - n_train - n_val

    train_idx = idxs[:n_train]
    val_idx   = idxs[n_train:n_train + n_val]
    test_idx  = idxs[n_train + n_val:]

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    test_ds  = Subset(full_ds, test_idx.tolist())

    # -------- collate + loaders (BRANCHED, but predicted path is identical) --------
    class CollateWithDTOnly:
        """
        Minimal collate for uniform-mesh testing:
          - preserves dataset keys
          - attaches dt_list + dt_ref exactly like CollateWithPrecompute does conceptually
        Expects batch_size=1 windows (same as your current loaders).
        """
        def __init__(self, dt_transitions: torch.Tensor, dt_ref: torch.Tensor | float | None):
            self.dt_transitions = dt_transitions.detach().cpu()
            self.dt_ref = float(dt_ref) if dt_ref is not None else None

        def __call__(self, batch_list):
            if len(batch_list) != 1:
                raise RuntimeError("This collate expects batch_size=1 (one window per batch).")
            batch = batch_list[0]

            t_indices = batch.get("t_indices", None)
            if t_indices is None:
                raise RuntimeError(
                    "Uniform-mesh mode requires batch['t_indices'] so dt_list can be formed."
                )

            if torch.is_tensor(t_indices):
                t_idx_list = t_indices.detach().cpu().long().tolist()
            else:
                t_idx_list = [int(x) for x in t_indices]

            dt_list = []
            for i in range(len(t_idx_list) - 1):
                t0 = int(t_idx_list[i])
                if t0 < 0 or t0 >= int(self.dt_transitions.numel()):
                    raise RuntimeError(
                        f"t_indices[{i}]={t0} out of range for dt_transitions (len={int(self.dt_transitions.numel())})."
                    )
                dt_list.append(self.dt_transitions[t0])

            batch["dt_list"] = dt_list
            if self.dt_ref is not None:
                batch["dt_ref"] = float(self.dt_ref)

            return batch

    if mesh_mode in ("predicted", "precomputed", "nonuniform", "amr"):
        # ----- EXACT existing precompute path -----
        cache_path = cfg["train"].get("precomp_cache_path", None)
        force_recompute = bool(cfg["train"].get("precomp_force_recompute", False))

        precomp = precompute_pred_mesh_and_interps_for_rollout(
            full_ds.steps,
            cfg,
            H,
            W,
            dx,
            dy,
            device=device,
            progress=True,
            cache_path=cache_path,
            force_recompute=force_recompute,
        )
        precomp = move_precomp_to_device(precomp, device)

        collate = CollateWithPrecompute(precomp, dt_transitions=dt_transitions, dt_ref=dt_ref)

        train_loader = DataLoader(
            train_ds, batch_size=1, sampler=RandomSampler(train_ds),
            num_workers=0, pin_memory=False, collate_fn=collate
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, sampler=RandomSampler(val_ds),
            num_workers=0, pin_memory=False, collate_fn=collate
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, sampler=RandomSampler(test_ds),
            num_workers=0, pin_memory=False, collate_fn=collate
        )

    elif mesh_mode in ("uniform", "uniform_mesh", "constant_uniform"):
        precomp = None  # explicitly absent

        collate = CollateWithDTOnly(dt_transitions=dt_transitions, dt_ref=dt_ref)

        train_loader = DataLoader(
            train_ds, batch_size=1, sampler=RandomSampler(train_ds),
            num_workers=0, pin_memory=False, collate_fn=collate
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, sampler=RandomSampler(val_ds),
            num_workers=0, pin_memory=False, collate_fn=collate
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, sampler=RandomSampler(test_ds),
            num_workers=0, pin_memory=False, collate_fn=collate
        )
    else:
        raise ValueError(f"Unknown mesh_mode='{mesh_mode}'. Use 'predicted' or 'uniform'.")

    # -------- Normalization stats (predicted path unchanged; uniform path computes consistent stats) --------
    feats_cfg = cfg.get("features", {})
    do_norm = bool(feats_cfg.get("normalize", True))
    mu = sigma = None

    # uniform helper for stats
    @torch.no_grad()
    def _compute_norm_stats_uniform_from_loader(engine, loader, device, max_batches: int | None = None):
        """
        MPS-safe norm stats computation.

        Strategy:
        1) Get uniform-mapped GT on engine.device (may be MPS).
        2) Move to CPU in float32 FIRST (MPS cannot do float64).
        3) Accumulate sums on CPU (float32 is fine; avoids MPS float64 entirely).
        """
        sum_x = None
        sum_x2 = None
        count = 0.0

        nb = 0
        for b in loader:
            gt_u, mask_u = engine.map_window_gt_to_uniform(b)  # list length K

            for k in range(len(gt_u)):
                x = gt_u[k]          # [Nu, F], on engine.device
                m = mask_u[k]        # [Nu] or None

                if m is not None:
                    # make boolean mask on same device as x
                    m_bool = m.to(device=x.device, dtype=torch.bool)
                    x_sel = x[m_bool]
                else:
                    x_sel = x

                if x_sel.numel() == 0:
                    continue

                # CRITICAL: move to CPU in float32 FIRST (MPS-safe), then accumulate
                x_cpu = x_sel.detach().to(device="cpu", dtype=torch.float32)

                if sum_x is None:
                    Fdim = int(x_cpu.shape[1])
                    sum_x = torch.zeros((Fdim,), dtype=torch.float32, device="cpu")
                    sum_x2 = torch.zeros((Fdim,), dtype=torch.float32, device="cpu")

                sum_x += x_cpu.sum(dim=0)
                sum_x2 += (x_cpu * x_cpu).sum(dim=0)
                count += float(x_cpu.shape[0])

            nb += 1
            if (max_batches is not None) and (nb >= int(max_batches)):
                break

        if sum_x is None or count <= 0:
            raise RuntimeError("Could not compute uniform-mesh norm stats (no samples).")

        mu_ = (sum_x / count).to(dtype=torch.float32)
        var = (sum_x2 / count) - (mu_ * mu_)
        var = torch.clamp(var, min=1e-12)
        sigma_ = torch.sqrt(var).to(dtype=torch.float32)

        return mu_.to(device), sigma_.to(device)


    # Build uniform engine early if needed (for norm stats + train/eval)
    uniform_engine = None
    if mesh_mode in ("uniform", "uniform_mesh", "constant_uniform"):
        from utils.uniform_mesh_engine import UniformMeshEngine  # helper file you created

        # Hooks come from your existing codebase (these names already exist in your file)
        #   _build_X, _forward_main_head, coarse_aggregate_from_dynamic, laplacian_smoothness, temporal_consistency
        lap_w = float(cfg["loss"].get("laplacian_weight", 0.0))
        tmp_w = float(cfg["loss"].get("temporal_weight", 0.0))

        uniform_engine = UniformMeshEngine(
            cfg=cfg,
            device=device,
            H0=H,
            W0=W,
            Hu=uniform_H,
            Wu=uniform_W,
            bbox=tuple(cfg.get("data", {}).get("bbox", (0.0, 1.0, 0.0, 1.0))),
            build_X_fn=_build_X,
            forward_fn=_forward_main_head,
            coarse_agg_fn=coarse_aggregate_from_dynamic,
            lap_fn=laplacian_smoothness if lap_w > 0 else None,
            tmp_fn=temporal_consistency if tmp_w > 0 else None,
            teacher_forcing=uniform_teacher_forcing,
            diag_edges=uniform_diag_edges,
        )

    if do_norm and (mu is None or sigma is None):
        if mesh_mode in ("predicted", "precomputed", "nonuniform", "amr"):
            # EXACT current behavior
            mu, sigma = _compute_norm_stats_from_loader(train_loader, device)
            cfg.setdefault("features", {}).setdefault("norm_stats", {})
            cfg["features"]["norm_stats"]["mu"] = mu.tolist()
            cfg["features"]["norm_stats"]["sigma"] = sigma.tolist()
            mu = mu.to(device)
            sigma = sigma.to(device)
        else:
            # uniform mode: compute stats on the actual uniform-mapped GT used by training
            max_batches = feats_cfg.get("norm_max_batches", None)  # optional; None => full loader
            mu, sigma = _compute_norm_stats_uniform_from_loader(uniform_engine, train_loader, device, max_batches=max_batches)
            cfg.setdefault("features", {}).setdefault("norm_stats", {})
            cfg["features"]["norm_stats"]["mu"] = mu.detach().cpu().tolist()
            cfg["features"]["norm_stats"]["sigma"] = sigma.detach().cpu().tolist()
    elif do_norm:
        mu = mu.to(device)
        sigma = sigma.to(device)
    else:
        mu = sigma = None

    # -------- training loop (branched; predicted path unchanged) --------
    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)
    log_csv = os.path.join(cfg["train"]["save_dir"], "train_log.csv")
    with open(log_csv, "w") as f:
        f.write("epoch,split,loss,mae\n")

    print(f"[INFO] Starting training (mesh_mode={mesh_mode})...")
    best_val = float("inf")
    TR, VL = [], []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        t0 = time.time()

        # (kept from your current code; harmless)
        _ = next(iter(train_loader))

        if mesh_mode in ("predicted", "precomputed", "nonuniform", "amr"):
            tr_loss, tr_mae, tr_stats = train_one_epoch_multi_step(
                model, train_loader, opt, cfg, device,
                H=H, W=W, dx=dx, dy=dy, mu=mu, sigma=sigma
            )
            vl_loss, vl_stats = evaluate_one_epoch_multi_step(
                model, val_loader, cfg, device,
                H=H, W=W, dx=dx, dy=dy, mu=mu, sigma=sigma,
                collect_examples=False
            )
        else:
            tr_loss, tr_mae, tr_stats = uniform_engine.train_one_epoch(
                model, train_loader, opt, mu=mu, sigma=sigma, scaler=None
            )
            vl_loss, vl_stats = uniform_engine.evaluate_one_epoch(
                model, val_loader, mu=mu, sigma=sigma, collect_examples=False
            )

        TR.append(tr_loss)
        VL.append(vl_loss)

        with open(log_csv, "a") as f:
            f.write(f"{epoch},train,{tr_loss:.10f},{tr_mae:.10f}\n")
            f.write(f"{epoch},val,{vl_loss:.6f}\n")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), os.path.join(cfg["train"]["save_dir"], "best_model.pt"))

        dt = time.time() - t0

        # keep ReduceLROnPlateau param-groups safe (UNCHANGED)
        if scheduler is not None and hasattr(scheduler, "min_lrs"):
            if len(scheduler.min_lrs) != len(opt.param_groups):
                base_min = scheduler.min_lrs[0] if scheduler.min_lrs else float(
                    cfg.get("scheduler", {}).get("min_lr", 1e-6)
                )
                scheduler.min_lrs = [base_min] * len(opt.param_groups)

        if scheduler is not None:
            scheduler.step(vl_loss)

        print(
            f"[INFO] Epoch {epoch:03d}: "
            f"train {tr_loss:.6f} (MAE {tr_mae:.6f}) | "
            f"val {vl_loss:.6f} | "
            f"{dt:.1f}s | lr={_get_lr(opt):.3e}"
        )

    # -------- save last bundle (predicted path includes precomp EXACTLY; uniform stores no precomp) --------
    def _precomp_to_cpu(precomp_obj):
        out = {}
        for key, lst in precomp_obj.items():
            if isinstance(lst, list):
                new_lst = []
                for v in lst:
                    if torch.is_tensor(v):
                        new_lst.append(v.detach().cpu())
                    elif isinstance(v, tuple):
                        new_lst.append(tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in v))
                    else:
                        new_lst.append(v)
                out[key] = new_lst
            else:
                out[key] = lst
        return out

    save_dict = {
        "model": model.state_dict(),
        "norm_stats": {
            "mu": None if mu is None else mu.detach().cpu().tolist(),
            "sigma": None if sigma is None else sigma.detach().cpu().tolist(),
        },
        "cfg": cfg,
    }

    if mesh_mode in ("predicted", "precomputed", "nonuniform", "amr"):
        precomp_cpu = _precomp_to_cpu(precomp)
        save_dict["precomp"] = precomp_cpu
    else:
        save_dict["precomp"] = None
        save_dict["uniform_mesh"] = {
            "Hu": int(uniform_H),
            "Wu": int(uniform_W),
            "teacher_forcing": bool(uniform_teacher_forcing),
            "diag_edges": bool(uniform_diag_edges),
            "bbox": tuple(cfg.get("data", {}).get("bbox", (0.0, 1.0, 0.0, 1.0))),
        }

    torch.save(save_dict, os.path.join(cfg["train"]["save_dir"], "last_model.pt"))

    plot_loss_curves(
        os.path.join(cfg["train"]["save_dir"], "loss_curves.png"),
        list(range(1, len(TR) + 1)),
        TR, VL
    )

    # -------- test evaluation --------
    print("test_loader length:", len(test_loader))
    print("=== Running test evaluation ===")

    if mesh_mode in ("predicted", "precomputed", "nonuniform", "amr"):
        test_loss, test_stats = evaluate_one_epoch_multi_step(
            model,
            test_loader,
            cfg,
            device,
            H=H,
            W=W,
            dx=dx,
            dy=dy,
            mu=mu,
            sigma=sigma,
            collect_examples=True,
        )

        print(f"[TEST] loss={test_loss:.4e}")

        # Preserve your existing qualitative pipeline exactly as before:
        # (If you have additional plotting blocks below in your file, keep them unchanged.)
        test_examples = test_stats["examples"]

        out_pdf = os.path.join(cfg["train"]["save_dir"], "qual_with_deltas.pdf")
        feature_names = cfg.get("features", {}).get("names", ["U", "V", "E"])
        print(f"[INFO] Generating qualitative PDF with deltas: {out_pdf}")

        plot_qual_pdf(
            examples=test_examples,
            cfg=cfg,
            out_pdf_path=out_pdf,
            feature_names=feature_names,
            unify_clims=False,
            dpi=150,
            rasterize=True,
            colorbars="row",
        )

    else:
        test_loss, test_stats = uniform_engine.evaluate_one_epoch(
            model,
            test_loader,
            mu=mu,
            sigma=sigma,
            collect_examples=True,
        )

        print(f"[UNIFORM TEST] loss={test_loss:.4e}")
        # Save test metrics/examples for later inspection (plotting differs from AMR pipeline)
        torch.save(
            {"loss": float(test_loss), "stats": test_stats},
            os.path.join(cfg["train"]["save_dir"], "uniform_test_stats.pt")
        )
        print("[INFO] Saved uniform test stats to uniform_test_stats.pt")

        test_examples = test_stats["examples"]

        out_pdf = os.path.join(cfg["train"]["save_dir"], "qual_with_deltas_uniform.pdf")
        feature_names = cfg.get("features", {}).get("names", ["U", "V", "E"])
        print(f"[INFO] Generating qualitative PDF with deltas: {out_pdf}")

        plot_qual_pdf(
            examples=test_examples,
            cfg=cfg,
            out_pdf_path=out_pdf,
            feature_names=feature_names,
            unify_clims=False,
            dpi=150,
            rasterize=True,
            colorbars="row",
        )
'''

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config")
    args = ap.parse_args()
    print("Running main...", flush=True)
    main(args.config)


