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
from typing import Dict, Any, List, Tuple, Optional, Sequence
import os, io, json, time, zipfile, random, sys, csv, datetime, hashlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch import optim
from torch_geometric.data import Data
from contextlib import nullcontext
import numpy as np

from dataset import CellRefineWindowDataset
from models import FeatureNet, ParcFeatureAdapter
from amr_policy import coarse_aggregate_from_dynamic, predict_masks_hierarchical_from_gt_gradients
            
from plots import plot_loss_curves
from utils_geom import build_idw_map, apply_idw_map, apply_precomputed_idw_map, dynamic_cells_from_parent_masks

from pretrain import (
    precompute_pred_mesh_and_interps_for_rollout,
    precompute_uniform_mesh_in_memory,
    CollateWithPrecompute,
    CollateWithDtOnly,
    _build_starting_mesh_from_spec,
    build_amr_face_adjacency_edges,
    build_amr_local_knn_edges,
    dec_edge_attr_for_dyadic_quads,
    _clip_pred_mesh_to_wedge,
    _load_wedge_path_from_spec,
    _get_wedge_clip_level_lookup,
    _lookup_level_masks_on_device,
)

from utils.precomp_h5 import LazyPrecompH5
import utils.dec_ops as dec
#from utils.mls import SolveGradientsLST, SolveWeightLST2d, apply_laplacian
import utils.mls as mls



debug_once = False

def _get_refine_ratio(cfg: dict) -> int:
    pol = cfg.get("policy", {}) or {}
    mesh = cfg.get("mesh", {}) or {}
    rr = pol.get("refine_ratio", mesh.get("refine_ratio", 2))
    rr = int(rr)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {rr}")
    return rr

def _resolve_momentum_sigma_mode(cfg: dict) -> str:
    """
    Normalization mode for momentum-channel sigma values.
    Supported values:
      - "independent": keep per-channel sigma as computed
      - "shared_rms": tie mx/my to sqrt((sigma_mx^2 + sigma_my^2)/2)
    """
    feats_cfg = cfg.get("features", {}) or {}
    norm_cfg = feats_cfg.get("normalization", {}) or {}

    raw_mode = norm_cfg.get("momentum_sigma_mode", None)
    if raw_mode is None:
        # Backward-compatible convenience bools.
        tie_flag = norm_cfg.get("tie_momentum_sigma", feats_cfg.get("tie_momentum_sigma", None))
        if tie_flag is not None:
            raw_mode = "shared_rms" if bool(tie_flag) else "independent"

    if raw_mode is None:
        raw_mode = "independent"

    mode = str(raw_mode).strip().lower()
    aliases = {
        "independent": "independent",
        "per_channel": "independent",
        "separate": "independent",
        "shared_rms": "shared_rms",
        "joint_rms": "shared_rms",
        "shared": "shared_rms",
        "tied": "shared_rms",
    }
    if mode not in aliases:
        raise ValueError(
            "features.normalization.momentum_sigma_mode must be one of "
            "{independent, shared_rms} (aliases: per_channel, separate, joint_rms, shared, tied). "
            f"Got: {raw_mode!r}"
        )
    return aliases[mode]

def _apply_momentum_sigma_mode(
    sigma: torch.Tensor,
    *,
    mode: str,
    momentum_x_idx: int,
    momentum_y_idx: int,
) -> torch.Tensor:
    if mode == "independent":
        return sigma
    if mode != "shared_rms":
        raise ValueError(f"Unsupported momentum sigma mode: {mode!r}")

    max_idx = max(int(momentum_x_idx), int(momentum_y_idx))
    if sigma.numel() <= max_idx:
        raise ValueError(
            "Cannot apply shared momentum sigma: feature dimension is too small for momentum indices "
            f"mx={momentum_x_idx}, my={momentum_y_idx}, numel={sigma.numel()}."
        )

    out = sigma.clone()
    sig_x = out[int(momentum_x_idx)]
    sig_y = out[int(momentum_y_idx)]
    # Shared momentum scale: RMS of per-axis stds so total momentum block scale stays comparable.
    sig_shared = torch.sqrt(0.5 * (sig_x * sig_x + sig_y * sig_y)).clamp_min(1e-12)
    out[int(momentum_x_idx)] = sig_shared
    out[int(momentum_y_idx)] = sig_shared
    return out

@torch.no_grad()
def _compute_norm_stats_from_loader(
    loader,
    device,
    *,
    momentum_sigma_mode: str = "independent",
    momentum_x_idx: int = 1,
    momentum_y_idx: int = 2,
):
    n_total = 0
    mean = None
    M2 = None

    for batch_idx, batch in enumerate(loader):
        # Prefer the distribution the model actually trains on (pred-mesh mapped)
        xs = []

        if "feat_t_on_pred_list" in batch and "feat_tp1_on_pred_list" in batch:
            ft = batch["feat_t_on_pred_list"]
            f1 = batch["feat_tp1_on_pred_list"]
            for f in list(ft)[1:] + list(f1)[1:]:
                if f is None or f.numel() == 0: 
                    continue
                xs.append(f.to(device).float())

        elif "feat_list" in batch:
            for f in batch["feat_list"]:
                if f is None or f.numel() == 0:
                    continue
                xs.append(f.to(device).float())

        elif "center_feat_t" in batch:
            xs.append(batch["center_feat_t"].to(device).float())
            if "center_feat_tp1" in batch:
                xs.append(batch["center_feat_tp1"].to(device).float())

        else:
            continue

        if not xs:
            continue

        x = torch.cat(xs, dim=0)  # (N, F)
        if x.numel() == 0:
            continue

        batch_n = x.size(0)
        batch_mean = x.mean(dim=0)
        batch_M2 = ((x - batch_mean) ** 2).sum(dim=0)

        if mean is None:
            mean = batch_mean
            M2 = batch_M2
            n_total = batch_n
        else:
            new_n = n_total + batch_n
            delta = batch_mean - mean
            mean = mean + delta * (batch_n / new_n)
            M2 = M2 + batch_M2 + (delta * delta) * (n_total * batch_n / new_n)
            n_total = new_n

    if mean is None or n_total < 2:
        raise RuntimeError("Could not compute normalization stats: no feature tensors found in loader.")

    var = M2 / (n_total - 1)
    std = torch.sqrt(var + 1e-8)
    std = _apply_momentum_sigma_mode(
        std,
        mode=momentum_sigma_mode,
        momentum_x_idx=momentum_x_idx,
        momentum_y_idx=momentum_y_idx,
    )
    return mean, std

def save_config(run_dir: str, cfg: dict, argv=None):
    argv = list(sys.argv) if argv is None else argv

    # 1) resolved config                                                                                                                                                      
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

def _maybe_norm(x: torch.Tensor, mu, sigma):
    """
    Normalize x using mu, sigma (broadcastable), moving stats to x.device.
    mu/sigma can be tensors, lists, or numpy arrays.
    """
    if mu is None or sigma is None:
        return x

    # Ensure mu, sigma are tensors on x.device with the right dtype
    if not torch.is_tensor(mu):
        mu_t = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
    else:
        mu_t = mu.to(device=x.device, dtype=x.dtype)

    if not torch.is_tensor(sigma):
        sigma_t = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    else:
        sigma_t = sigma.to(device=x.device, dtype=x.dtype)

    sigma_t = sigma_t.clamp_min(1e-12)

    return (x - mu_t) / sigma_t


def _maybe_denorm(x: torch.Tensor, mu, sigma):
    """
    Inverse of _maybe_norm: x * sigma + mu, moving stats to x.device.
    """
    if mu is None or sigma is None:
        return x

    if not torch.is_tensor(mu):
        mu_t = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
    else:
        mu_t = mu.to(device=x.device, dtype=x.dtype)

    if not torch.is_tensor(sigma):
        sigma_t = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    else:
        sigma_t = sigma.to(device=x.device, dtype=x.dtype)

    sigma_t = sigma_t.clamp_min(1e-12)

    return x * sigma_t + mu_t

_MLS_STATE = {
    "sig": None,
    "grad": None,
    "lapw": None,
    "adv": None,
    "diff": None,
}

def _mls_sig_from_cfg(cfg: dict) -> tuple:
    loss_cfg = cfg.get("loss", {}) or {}
    # only include knobs that affect MLS solver behavior
    return (
        bool(loss_cfg.get("mls_cache_by_geometry", False)),
        bool(loss_cfg.get("mls_use_2hop_extension", True)),
        bool(loss_cfg.get("mls_use_neighbor_damping", True)),
        float(loss_cfg.get("mls_damping_alpha", 0.5)),
        int(loss_cfg.get("mls_poly_order", 2)),
        int(loss_cfg.get("mls_min_neighbors", 6)),
    )

def _get_mls_ops(cfg: dict):
    """
    Lazily construct MLS solvers once (per process) using cfg knobs.
    IMPORTANT for dynamic AMR: default cache_by_geometry=False.
    """
    sig = _mls_sig_from_cfg(cfg)
    if _MLS_STATE["sig"] == sig and _MLS_STATE["adv"] is not None:
        return _MLS_STATE["adv"], _MLS_STATE["diff"]

    cache_by_geometry, use_2hop, use_damp, alpha, poly_order, min_nbrs = sig

    grad = mls.SolveGradientsLST(
        cache_by_geometry=cache_by_geometry,
        use_2hop_extension=use_2hop,
        use_neighbor_damping=use_damp,
        damping_alpha=alpha,
    )
    lapw = mls.SolveWeightLST2d(
        polynomial_order=poly_order,
        min_neighbors=min_nbrs,
        cache_by_geometry=cache_by_geometry,
        use_2hop_extension=use_2hop,
        use_neighbor_damping=use_damp,
        damping_alpha=alpha,
    )

    adv  = mls.AdvectionMLS(grad)
    diff = mls.DiffusionMLS(lapw)

    _MLS_STATE.update({"sig": sig, "grad": grad, "lapw": lapw, "adv": adv, "diff": diff})
    return adv, diff


@torch.no_grad()
def mls_advdiff_terms_abs_faceadj(
    *,
    x_abs: torch.Tensor,          # (N,F) absolute state on current pred mesh
    pos: torch.Tensor,            # (N,2) centers (same mesh as x_abs)
    edge_index: torch.Tensor,     # (2,E) FACE-ADJ edges from your precomp H5
    levels: torch.Tensor,         # (N,)
    dx0: float,
    dy0: float,
    cfg: dict,
    compute_adv: bool,
    compute_diff: bool,
):
    """
    Runs MLS advection/diffusion using the mls.py solvers, but on your face-adjacency edge_index.
    Still supports 2-hop and neighbor damping via solver options.
    """
    loss_cfg = cfg.get("loss", {}) or {}
    rho_floor = float(loss_cfg.get("rho_floor", 1e-6))
    u_clip    = float(loss_cfg.get("u_clip", 1000.0))
    nu        = float(loss_cfg.get("nu", 0.0))

    # Choose where MLS runs (CPU recommended on MPS; avoids scatter/index_add headaches)
    ops_dev = torch.device(loss_cfg.get("mls_ops_device", "cpu"))

    x_ops   = x_abs.to(device=ops_dev, dtype=torch.float32)
    pos_ops = pos.to(device=ops_dev, dtype=torch.float32)
    ei_ops  = edge_index.to(device=ops_dev, dtype=torch.long)

    data = Data(pos=pos_ops, edge_index=ei_ops)

    # velocity from conservative state (assumes [rho,mx,my,E])
    idx = dec.infer_feature_indices(cfg, x_ops.size(1))
    rho = x_ops[:, idx["rho"]].abs().clamp_min(rho_floor)
    mx  = x_ops[:, idx["mx"]]
    my  = x_ops[:, idx["my"]]
    vel = torch.stack([mx / rho, my / rho], dim=1).clamp(-u_clip, u_clip)  # (N,2)

    mls_adv, mls_diff = _get_mls_ops(cfg)

    r_adv = None
    if compute_adv:
        r_adv = mls_adv(x_ops, vel, data)  # (N,F) on ops_dev

    r_diff = None
    if compute_diff:
        r_diff = mls_diff(x_ops, data)     # (N,F) on ops_dev
        if nu != 0.0:
            r_diff = nu * r_diff

    # area (keep compatibility with your residual-loss codepath)
    area = dec.cell_area_from_levels(
        levels.long().to(device=x_abs.device),
        dx0=float(dx0),
        dy0=float(dy0),
        dtype=torch.float32,
        device=x_abs.device,
        refine_ratio=_get_refine_ratio(cfg),
    )

    # move outputs back to training device
    if r_adv is not None:
        r_adv = r_adv.to(device=x_abs.device, dtype=torch.float32)
    if r_diff is not None:
        r_diff = r_diff.to(device=x_abs.device, dtype=torch.float32)

    return r_adv, r_diff, area

# ------------------------- Small utilities -------------------------

import math

def _to_scalar_dt(dt, device, dtype):
    if torch.is_tensor(dt):
        return dt.to(device=device, dtype=dtype).view(())
    return torch.tensor(float(dt), device=device, dtype=dtype).view(())

@torch.no_grad()
def _pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    # a,b: (N,) float tensors
    a = a.float().flatten()
    b = b.float().flatten()
    am = a.mean()
    bm = b.mean()
    av = a - am
    bv = b - bm
    denom = (av.std(unbiased=False) * bv.std(unbiased=False)).clamp_min(eps)
    return float((av * bv).mean() / denom)

@torch.no_grad()
def _wrel_l2(err: torch.Tensor, ref: torch.Tensor, w: torch.Tensor, eps: float = 1e-12) -> float:
    # err/ref: (N,) ; w: (N,)
    err2 = (err.float() * err.float()) * w.float()
    ref2 = (ref.float() * ref.float()) * w.float()
    num = torch.sqrt(err2.sum().clamp_min(eps))
    den = torch.sqrt(ref2.sum().clamp_min(eps))
    return float((num / den).item())

@torch.no_grad()
def stepA_check_ops_vs_gt_delta(
    batch: dict,
    cfg: dict,
    device: torch.device,
    *,
    dx: float,
    dy: float,
    step_k: int = 0,
    u_max: float = 1e3,
    rho_floor: float = 1e-6,
    E_floor: float = 1e-6,
    use_area_weights: bool = True,
):
    """
    Step A: Compare GT delta on pred mesh vs physics delta from DEC ops on the same mesh, with NO NN.
    Uses:
      x_in_abs  = feat_t_on_pred_list[k+1]   (GT(t+k) mapped to pred mesh at t+k+1)
      x_tgt_abs = feat_tp1_on_pred_list[k+1] (GT(t+k+1) mapped to pred mesh at t+k+1)

    And ops are computed on pred mesh at (k+1) using pred_ei_list[k+1] and pred_ea_list[k+1].
    """

    # -------- required tensors/lists --------
    dt_list = batch.get("dt_list", None)
    if dt_list is None:
        raise RuntimeError("batch missing dt_list")

    feat_t_on_pred_list   = batch.get("feat_t_on_pred_list", None)
    feat_tp1_on_pred_list = batch.get("feat_tp1_on_pred_list", None)
    if feat_t_on_pred_list is None or feat_tp1_on_pred_list is None:
        raise RuntimeError("batch missing feat_t_on_pred_list / feat_tp1_on_pred_list")

    pred_levels_list  = batch.get("pred_levels_list", None)
    pred_ei_list      = batch.get("pred_ei_list", None)
    if pred_levels_list is None or pred_ei_list is None:
        raise RuntimeError("batch missing pred_levels_list / pred_ei_list")

    pred_ea_list = batch.get("pred_ea_list", None) or batch.get("pred_edge_attr_list", None)
    if pred_ea_list is None:
        raise RuntimeError("batch missing pred_ea_list or pred_edge_attr_list")

    K = len(feat_t_on_pred_list)
    if step_k < 0 or step_k > (K - 2):
        raise ValueError(f"step_k must be in [0, {K-2}], got {step_k}")

    # -------- pick the mesh/time index you use in training (k+1) --------
    j = step_k + 1

    x_in_abs  = feat_t_on_pred_list[j].to(device).float()    # (N,F)
    x_tgt_abs = feat_tp1_on_pred_list[j].to(device).float()  # (N,F)

    pred_levels = pred_levels_list[j].to(device)
    pei = pred_ei_list[j].to(device)
    pea = pred_ea_list[j].to(device)

    dt = _to_scalar_dt(dt_list[step_k], device=device, dtype=x_in_abs.dtype)

    # -------- sanity prints --------
    Fdim = x_in_abs.size(1)
    N = x_in_abs.size(0)
    print("\n[STEP-A] ================================================")
    print(f"[STEP-A] step_k={step_k} using index j=k+1={j}")
    print(f"[STEP-A] N={N} F={Fdim} dt={float(dt.item()):.6e}")
    print(f"[STEP-A] x_in_abs min/max: {float(x_in_abs.min()):.3e} / {float(x_in_abs.max()):.3e}")
    print(f"[STEP-A] x_tgt_abs min/max: {float(x_tgt_abs.min()):.3e} / {float(x_tgt_abs.max()):.3e}")

    # -------- build weights (cell area) --------
    if use_area_weights:
        area = dec.cell_area_from_levels(
            pred_levels.long(),
            dx0=float(dx),
            dy0=float(dy),
            dtype=torch.float32,
            device=device,
            refine_ratio=_get_refine_ratio(cfg),
        ).view(-1)
    else:
        area = torch.ones((N,), device=device, dtype=torch.float32)

    # -------- sanitize state for ops (IMPORTANT: ops should not see insane mx/rho) --------
    # Use your sanitizer if you prefer; this is a momentum-by-rho limiter that matches u_max.
    idx_map = dec.infer_feature_indices(cfg, Fdim)
    rho_i, mx_i, my_i, E_i = idx_map["rho"], idx_map["mx"], idx_map["my"], idx_map["E"]

    x_ops = x_in_abs.clone()
    rho = x_ops[:, rho_i].clamp_min(rho_floor)
    x_ops[:, rho_i] = rho
    x_ops[:, E_i]   = x_ops[:, E_i].clamp_min(E_floor)

    mmax = (u_max * rho).to(dtype=x_ops.dtype)
    x_ops[:, mx_i] = x_ops[:, mx_i].clamp(-mmax, mmax)
    x_ops[:, my_i] = x_ops[:, my_i].clamp(-mmax, mmax)

    # -------- compute DEC operator terms (absolute units) --------
    loss_cfg = cfg.get("loss", {}) or {}
    adv_w  = float(loss_cfg.get("adv_weight", 1.0))
    diff_w = float(loss_cfg.get("diff_weight", 1.0))
    inc_adv  = bool(loss_cfg.get("parc_include_adv", True))
    inc_diff = bool(loss_cfg.get("parc_include_diff", True))

    compute_adv  = inc_adv  and (adv_w != 0.0)
    compute_diff = inc_diff and (diff_w != 0.0)

    with torch.autocast(device_type=device.type, enabled=False):
        r_adv_abs, r_diff_abs, _area2 = dec.dec_advdiff_terms_abs(
            x_abs=x_ops.float(),
            edge_index=pei.long(),
            pred_ea=pea.float(),
            levels=pred_levels.long(),
            dx0=float(dx),
            dy0=float(dy),
            cfg=cfg,
            compute_adv=compute_adv,
            compute_diff=compute_diff,
        )

    # Normalize missing terms to zeros for combination:
    ref = x_in_abs.float()
    if r_adv_abs is None:
        r_adv_abs = torch.zeros_like(ref)
    if r_diff_abs is None:
        r_diff_abs = torch.zeros_like(ref)

    r_phy_abs = (adv_w * r_adv_abs) + (diff_w * r_diff_abs)   # (N,F)
    delta_phy = dt * r_phy_abs                                # (N,F)
    delta_gt  = (x_tgt_abs - x_in_abs)                         # (N,F)

    # -------- metrics --------
    print("[STEP-A] ---- per-channel delta comparison ----")
    for c in range(Fdim):
        gt_c  = delta_gt[:, c]
        phy_c = delta_phy[:, c]
        err_c = phy_c - gt_c

        corr = _pearson_corr(phy_c, gt_c)
        rel  = _wrel_l2(err_c, gt_c, area)

        # also compare rate (optional): r_phy vs delta_gt/dt
        gt_rate_c = (gt_c / dt.clamp_min(1e-12))
        corr_r = _pearson_corr(r_phy_abs[:, c], gt_rate_c)

        print(
            f"[STEP-A] ch={c:02d}  "
            f"corr(delta_phy,delta_gt)={corr:+.4f}  "
            f"wRelL2(delta_phy-delta_gt, delta_gt)={rel:.4e}  "
            f"corr(r_phy, delta_gt/dt)={corr_r:+.4f}  "
            f"delta_gt min/max={float(gt_c.min()):+.3e}/{float(gt_c.max()):+.3e}  "
            f"delta_phy min/max={float(phy_c.min()):+.3e}/{float(phy_c.max()):+.3e}"
        )

    # -------- overall metric (all channels stacked) --------
    gt_all  = delta_gt.reshape(-1)
    phy_all = delta_phy.reshape(-1)
    err_all = phy_all - gt_all
    area_all = area.repeat_interleave(Fdim)

    corr_all = _pearson_corr(phy_all, gt_all)
    rel_all  = _wrel_l2(err_all, gt_all, area_all)

    print("[STEP-A] ---- overall ----")
    print(f"[STEP-A] corr(delta_phy,delta_gt)={corr_all:+.4f}")
    print(f"[STEP-A] wRelL2(delta_phy-delta_gt, delta_gt)={rel_all:.4e}")
    print("[STEP-A] ================================================\n")

    return {
        "delta_gt": delta_gt.detach().cpu(),
        "delta_phy": delta_phy.detach().cpu(),
        "r_adv_abs": r_adv_abs.detach().cpu(),
        "r_diff_abs": r_diff_abs.detach().cpu(),
        "r_phy_abs": r_phy_abs.detach().cpu(),
        "dt": float(dt.item()),
    }

@torch.no_grad()
def stepA_check_given_state_vs_gt(
    *,
    tag: str,
    x_in_abs: torch.Tensor,     # (N,F) state you are *actually* using at this step
    x_tgt_abs: torch.Tensor,    # (N,F) GT(t+1) on same mesh
    pred_levels: torch.Tensor,  # (N,)
    pei: torch.Tensor,          # (2,E)
    pea: torch.Tensor,          # (E,5)
    dt: torch.Tensor,           # scalar
    cfg: dict,
    device: torch.device,
    dx: float,
    dy: float,
    u_max: float = 1e3,
    rho_floor: float = 1e-6,
    E_floor: float = 1e-6,
):
    x_in_abs = x_in_abs.to(device).float()
    x_tgt_abs = x_tgt_abs.to(device).float()
    pred_levels = pred_levels.to(device)
    pei = pei.to(device)
    pea = pea.to(device)
    dt = dt.to(device=device, dtype=torch.float32).view(())

    Fdim = x_in_abs.size(1)
    N = x_in_abs.size(0)

    area = dec.cell_area_from_levels(
        pred_levels.long(),
        dx0=float(dx),
        dy0=float(dy),
        dtype=torch.float32,
        device=device,
        refine_ratio=_get_refine_ratio(cfg),
    ).view(-1)

    # sanitize state for ops (same limiter style as before)
    idx_map = dec.infer_feature_indices(cfg, Fdim)
    rho_i, mx_i, my_i, E_i = idx_map["rho"], idx_map["mx"], idx_map["my"], idx_map["E"]

    x_ops = x_in_abs.clone()
    rho = x_ops[:, rho_i].clamp_min(rho_floor)
    x_ops[:, rho_i] = rho
    x_ops[:, E_i]   = x_ops[:, E_i].clamp_min(E_floor)

    mmax = (u_max * rho).to(dtype=x_ops.dtype)
    x_ops[:, mx_i] = x_ops[:, mx_i].clamp(-mmax, mmax)
    x_ops[:, my_i] = x_ops[:, my_i].clamp(-mmax, mmax)

    loss_cfg = cfg.get("loss", {}) or {}
    adv_w  = float(loss_cfg.get("adv_weight", 1.0))
    diff_w = float(loss_cfg.get("diff_weight", 1.0))
    inc_adv  = bool(loss_cfg.get("parc_include_adv", True))
    inc_diff = bool(loss_cfg.get("parc_include_diff", True))

    compute_adv  = inc_adv  and (adv_w != 0.0)
    compute_diff = inc_diff and (diff_w != 0.0)

    with torch.autocast(device_type=device.type, enabled=False):
        r_adv_abs, r_diff_abs, _ = dec.dec_advdiff_terms_abs(
            x_abs=x_ops.float(),
            edge_index=pei.long(),
            pred_ea=pea.float(),
            levels=pred_levels.long(),
            dx0=float(dx),
            dy0=float(dy),
            cfg=cfg,
            compute_adv=compute_adv,
            compute_diff=compute_diff,
        )

    ref = x_in_abs.float()
    if r_adv_abs is None:
        r_adv_abs = torch.zeros_like(ref)
    if r_diff_abs is None:
        r_diff_abs = torch.zeros_like(ref)

    r_phy_abs = adv_w * r_adv_abs + diff_w * r_diff_abs
    delta_phy = dt * r_phy_abs
    delta_gt  = x_tgt_abs - x_in_abs

    print(f"\n[STEP-A*] {tag}  N={N} F={Fdim} dt={float(dt.item()):.3e}")
    print(f"[STEP-A*] x_in_abs absmax={float(x_in_abs.abs().max()):.3e}  x_tgt_abs absmax={float(x_tgt_abs.abs().max()):.3e}")

    for c in range(Fdim):
        gt_c  = delta_gt[:, c]
        phy_c = delta_phy[:, c]
        err_c = phy_c - gt_c
        corr = _pearson_corr(phy_c, gt_c)
        rel  = _wrel_l2(err_c, gt_c, area)
        print(f"[STEP-A*] ch={c:02d} corr={corr:+.4f}  wRelL2={rel:.4e}  "
              f"gt[min/max]={float(gt_c.min()):+.3e}/{float(gt_c.max()):+.3e}  "
              f"phy[min/max]={float(phy_c.min()):+.3e}/{float(phy_c.max()):+.3e}")


@torch.no_grad()
def _print_chan_stats(name: str, x: torch.Tensor, idx: dict, max_q: bool = True):
    """
    x: [N,F]
    idx: {'rho':i,'mx':i,'my':i,'E':i}
    """
    if x is None or (not torch.is_tensor(x)):
        print(f"[ABS-CHECK] {name}: None")
        return
    if x.ndim != 2:
        print(f"[ABS-CHECK] {name}: expected [N,F], got {tuple(x.shape)}")
        return

    N, F = x.shape
    x_f = x.detach().to(torch.float32)

    # Per-channel mean/std/min/max
    mean = x_f.mean(dim=0)
    std  = x_f.std(dim=0, unbiased=False)
    xmin = x_f.min(dim=0).values
    xmax = x_f.max(dim=0).values

    def _fmt(v):  # short float
        return [float(f"{t:.4g}") for t in v.detach().cpu()]

    print(f"\n[ABS-CHECK] {name} shape={tuple(x.shape)} dtype={x.dtype} dev={x.device}")
    print("  mean:", _fmt(mean))
    print("  std :", _fmt(std))
    print("  min :", _fmt(xmin))
    print("  max :", _fmt(xmax))

    # rho/E negativity is a strong “is this really absolute?” signal
    rho = x_f[:, idx["rho"]]
    E   = x_f[:, idx["E"]]
    neg_rho = float((rho <= 0).float().mean().item())
    neg_E   = float((E   <= 0).float().mean().item())
    print(f"  neg_frac rho<=0: {neg_rho:.3e}   E<=0: {neg_E:.3e}")

    if max_q:
        # a couple robust quantiles for rho/E
        qs = torch.tensor([0.01, 0.5, 0.99], device=x_f.device)
        rho_q = torch.quantile(rho, qs).detach().cpu()
        E_q   = torch.quantile(E, qs).detach().cpu()
        print(f"  rho q01/q50/q99: {[float(f'{t:.4g}') for t in rho_q]}")
        print(f"  E   q01/q50/q99: {[float(f'{t:.4g}') for t in E_q]}")

@torch.no_grad()
def abs_means_abs_check(
    *,
    cfg: dict,
    batch: dict,
    mu,
    sigma,
    device: torch.device,
    tag: str = "ABS-CHECK",
):
    """
    Confirms whether tensors labeled *_abs are truly in absolute/physical units
    (i.e., not already standardized by the dataset).

    Prints:
      - ABS stats for several tensors
      - NORM stats using current mu/sigma
      - heuristics for “already normalized?”
    """
    def _safe_norm(x: torch.Tensor, mu, sigma):
        if (mu is None) or (sigma is None):
            return None
        mu_t = mu if torch.is_tensor(mu) else torch.as_tensor(mu, dtype=x.dtype, device=x.device)
        sg_t = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        mu_t = mu_t.to(device=x.device, dtype=x.dtype)
        sg_t = sg_t.to(device=x.device, dtype=x.dtype).clamp_min(1e-12)
        return (x - mu_t) / sg_t

    # Pull a few canonical tensors out of the batch
    # (If any key is missing, we just skip it.)
    cand = {}
    try:
        feat_list = batch.get("feat_list", None)
        if isinstance(feat_list, list) and len(feat_list) >= 2:
            cand["feat_list[t0]_abs (GT mesh)"]  = feat_list[0]
            cand["feat_list[t1]_abs (GT mesh)"]  = feat_list[1]
    except Exception:
        pass

    try:
        ft = batch.get("feat_t_on_pred_list", None)
        ftp1 = batch.get("feat_tp1_on_pred_list", None)
        if isinstance(ft, list) and len(ft) >= 2:
            cand["feat_t_on_pred_list[j=1]_abs"] = ft[1]
        if isinstance(ftp1, list) and len(ftp1) >= 2:
            cand["feat_tp1_on_pred_list[j=1]_abs"] = ftp1[1]
    except Exception:
        pass

    # If nothing found, bail early
    if len(cand) == 0:
        print(f"[{tag}] No candidate tensors found in batch to inspect.")
        return

    # Determine channel mapping once
    # Use tensor Fdim from first available candidate
    first = next(iter(cand.values()))
    first = first.to(device) if torch.is_tensor(first) else first
    Fdim = int(first.size(1))
    idx = dec.infer_feature_indices(cfg, Fdim)
    print(f"\n[{tag}] inferred idx map: {idx}")

    # Print mu/sigma summaries (these are what you *think* correspond to abs space)
    if mu is None or sigma is None:
        print(f"[{tag}] mu/sigma are None -> normalization disabled or not passed here.")
    else:
        mu_t = mu.detach().to("cpu") if torch.is_tensor(mu) else torch.as_tensor(mu).to("cpu")
        sg_t = sigma.detach().to("cpu") if torch.is_tensor(sigma) else torch.as_tensor(sigma).to("cpu")
        print(f"[{tag}] mu    :", [float(f"{v:.4g}") for v in mu_t])
        print(f"[{tag}] sigma :", [float(f"{v:.4g}") for v in sg_t])

    # Inspect each candidate in ABS and NORM space
    for name, x in cand.items():
        if not torch.is_tensor(x):
            continue
        x = x.to(device)

        _print_chan_stats(f"{name} [ABS]", x, idx)

        x_norm = _safe_norm(x, mu, sigma)
        if x_norm is not None:
            _print_chan_stats(f"{name} [NORM using mu/sigma]", x_norm, idx)

            # Heuristic: if rho/E in ABS look ~standard normal (mean~0,std~1, many negatives),
            # then ABS is probably already standardized (dataset normalized) -> DEC sees wrong units.
            rho_abs = x[:, idx["rho"]].to(torch.float32)
            E_abs   = x[:, idx["E"]].to(torch.float32)

            rho_m = float(rho_abs.mean().item()); rho_s = float(rho_abs.std(unbiased=False).item())
            E_m   = float(E_abs.mean().item());   E_s   = float(E_abs.std(unbiased=False).item())
            neg_rho = float((rho_abs <= 0).float().mean().item())
            neg_E   = float((E_abs   <= 0).float().mean().item())

            looks_std_rho = (abs(rho_m) < 0.5) and (0.5 < rho_s < 2.0) and (neg_rho > 0.05)
            looks_std_E   = (abs(E_m)   < 0.5) and (0.5 < E_s   < 2.0) and (neg_E   > 0.05)

            if looks_std_rho or looks_std_E:
                print(f"[{tag}][WARN] {name}: rho/E ABS look already standardized "
                      f"(mean~0 std~1 with nontrivial negatives). "
                      f"If true, DEC ops are being computed on normalized units, not physical units.")


def _as_feature_vec(x: torch.Tensor, F: int, device=None, dtype=None) -> torch.Tensor:
    """
    Make sure mu/sigma broadcast as (1,F) on the right device/dtype.
    Accepts shape (F,), (1,F), (F,1) etc.
    """
    if x is None:
        raise ValueError("mu/sigma is None but debug needs it.")
    t = x
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype)
    t = t.view(-1)
    if t.numel() != F:
        raise ValueError(f"Expected vector of length F={F}, got {t.numel()}")
    return t.view(1, F)

def norm_to_abs(x_norm: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # x_abs = mu + sigma * x_norm
    F = x_norm.size(-1)
    mu_ = _as_feature_vec(mu, F, device=x_norm.device, dtype=x_norm.dtype)
    sg_ = _as_feature_vec(sigma, F, device=x_norm.device, dtype=x_norm.dtype)
    return mu_ + sg_ * x_norm

def abs_to_norm(x_abs: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    F = x_abs.size(-1)
    mu_ = _as_feature_vec(mu, F, device=x_abs.device, dtype=x_abs.dtype)
    sg_ = _as_feature_vec(sigma, F, device=x_abs.device, dtype=x_abs.dtype)
    return (x_abs - mu_) / sg_

@torch.no_grad()
def quantile_cpu(x: torch.Tensor, q: float, dim: int = 0) -> torch.Tensor:
    """
    CPU-safe quantile via sort (works on MPS/CUDA by moving once).
    Returns tensor with the same shape as x with dim removed.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1]")
    # Move to CPU for sort/selection
    xc = x.detach().to("cpu")
    n = xc.size(dim)
    if n <= 0:
        raise ValueError("empty tensor in quantile_cpu")
    # nearest-rank index (clamped)
    k = int(math.floor(q * (n - 1)))
    k = max(0, min(n - 1, k))
    xs, _ = torch.sort(xc, dim=dim)
    out = xs.select(dim, k)
    return out.to(x.device)

@torch.no_grad()
def clamp_pos_inplace(
    x_abs: torch.Tensor,
    *,
    rho_idx: int = 0,
    E_idx: int = 3,
    rho_floor: float = 1e-6,
    E_floor: float = 1e-6,
) -> torch.Tensor:
    """
    Applies the same kind of positivity clamp you’re already doing.
    Returns x_abs (in-place).
    """
    if rho_idx is not None:
        x_abs[:, rho_idx].clamp_(min=float(rho_floor))
    if E_idx is not None:
        x_abs[:, E_idx].clamp_(min=float(E_floor))
    return x_abs

@torch.no_grad()
def oracle_debug_step(
    *,
    step_k: int,
    x_in_norm: torch.Tensor,     # (N,F) normalized/model-space state at time t
    x_tgt_norm: torch.Tensor,    # (N,F) normalized/model-space teacher state at time t+1
    pred_norm: torch.Tensor,     # (N,F) your *processed* model output (after any adapters/clips), same space as x_in_norm
    dt: torch.Tensor,            # (N,) or scalar
    x_roll_abs: torch.Tensor,    # (N,F) the rollout state you actually produced (after clamp)
    x_tgt_abs: torch.Tensor,     # (N,F) teacher in abs space (after your normal abs extraction)
    mu: torch.Tensor,
    sigma: torch.Tensor,
    centers: Optional[torch.Tensor] = None,  # (N,2) optional
    levels: Optional[torch.Tensor] = None,   # (N,) optional
    feature_names: Sequence[str] = ("rho", "mx", "my", "E"),
    rho_idx: int = 0,
    E_idx: int = 3,
    rho_floor: float = 1e-6,
    E_floor: float = 1e-6,
    topk: int = 10,
) -> None:
    """
    1) Infers whether your current integration is behaving like "delta add" or "rate*dt add"
       by comparing reconstructed x_next_abs to your produced x_roll_abs.
    2) Runs an ORACLE step (using GT delta/rate) under each mode and reports error vs teacher.
    3) Reports pre-clamp negativity rates for rho/E under each candidate.
    """
    device = x_in_norm.device
    N, F = x_in_norm.shape

    # dt -> (N,1)
    if dt.numel() == 1:
        dtv = dt.reshape(1).expand(N)
    else:
        dtv = dt.view(-1)
        if dtv.numel() != N:
            raise ValueError(f"dt has {dtv.numel()} elems but N={N}")
    dtv = dtv.to(device=device, dtype=x_in_norm.dtype)
    dt_col = dtv.view(N, 1)

    # Reconstruct two candidate "your code might be doing this"
    # Candidate A: treat pred_norm as delta in norm space
    xA_norm = x_in_norm + pred_norm
    xA_abs_pre = norm_to_abs(xA_norm, mu, sigma)
    xA_abs = xA_abs_pre.clone()
    clamp_pos_inplace(xA_abs, rho_idx=rho_idx, E_idx=E_idx, rho_floor=rho_floor, E_floor=E_floor)

    # Candidate B: treat pred_norm as rate in norm space
    xB_norm = x_in_norm + pred_norm * dt_col
    xB_abs_pre = norm_to_abs(xB_norm, mu, sigma)
    xB_abs = xB_abs_pre.clone()
    clamp_pos_inplace(xB_abs, rho_idx=rho_idx, E_idx=E_idx, rho_floor=rho_floor, E_floor=E_floor)

    # Which candidate matches the x_roll_abs you actually produced?
    # (Use mean absolute error vs your produced rollout.)
    errA = (xA_abs - x_roll_abs).abs().mean().item()
    errB = (xB_abs - x_roll_abs).abs().mean().item()
    mode = "DELTA_ADD" if errA <= errB else "RATE_TIMES_DT_ADD"

    # Pre-clamp negativity rates (this is what triggers your clamp)
    negE_A = (xA_abs_pre[:, E_idx] < E_floor).float().mean().item()
    negE_B = (xB_abs_pre[:, E_idx] < E_floor).float().mean().item()
    negR_A = (xA_abs_pre[:, rho_idx] < rho_floor).float().mean().item()
    negR_B = (xB_abs_pre[:, rho_idx] < rho_floor).float().mean().item()

    # ORACLE: build GT delta/rate in norm space
    gt_delta_norm = (x_tgt_norm - x_in_norm)
    gt_rate_norm = gt_delta_norm / dt_col

    # Oracle under delta-add
    xoA_norm = x_in_norm + gt_delta_norm
    xoA_abs_pre = norm_to_abs(xoA_norm, mu, sigma)
    xoA_abs = xoA_abs_pre.clone()
    clamp_pos_inplace(xoA_abs, rho_idx=rho_idx, E_idx=E_idx, rho_floor=rho_floor, E_floor=E_floor)

    # Oracle under rate*dt-add
    xoB_norm = x_in_norm + gt_rate_norm * dt_col
    xoB_abs_pre = norm_to_abs(xoB_norm, mu, sigma)
    xoB_abs = xoB_abs_pre.clone()
    clamp_pos_inplace(xoB_abs, rho_idx=rho_idx, E_idx=E_idx, rho_floor=rho_floor, E_floor=E_floor)

    # Oracle errors vs teacher (abs)
    oerrA = (xoA_abs - x_tgt_abs).abs().mean().item()
    oerrB = (xoB_abs - x_tgt_abs).abs().mean().item()

    # How much clamp would oracle have triggered? (should be ~0.0 if everything is consistent)
    onegE_A = (xoA_abs_pre[:, E_idx] < E_floor).float().mean().item()
    onegE_B = (xoB_abs_pre[:, E_idx] < E_floor).float().mean().item()

    print(f"[ORACLE] step_k={step_k} inferred_mode={mode}  match_err(delta)={errA:.3e} match_err(rate*dt)={errB:.3e}")
    print(f"[ORACLE] step_k={step_k} preclamp_neg_frac pred:  rho A={negR_A:.3e} B={negR_B:.3e} | E A={negE_A:.3e} B={negE_B:.3e}")
    print(f"[ORACLE] step_k={step_k} oracle_mae_vs_teacher:  A(delta-add)={oerrA:.3e}  B(rate*dt-add)={oerrB:.3e}")
    print(f"[ORACLE] step_k={step_k} oracle_preclamp_negE_frac: A={onegE_A:.3e} B={onegE_B:.3e}")

    # Optional: show top-k nodes by *your actual* L1 drift (abs) to correlate with your existing DRIFT-C print
    if topk > 0:
        drift_l1 = (x_roll_abs - x_tgt_abs).abs().sum(dim=1)  # (N,)
        k = min(int(topk), N)
        vals, idx = torch.topk(drift_l1.detach().to("cpu"), k=k, largest=True, sorted=True)
        print(f"[ORACLE] step_k={step_k} top-{k} nodes by L1 drift (abs):")
        for r in range(k):
            i = int(idx[r].item())
            score = float(vals[r].item())
            xy_str = ""
            lvl_str = ""
            if centers is not None:
                xy = centers[i].detach().to("cpu").tolist()
                xy_str = f" xy={xy}"
            if levels is not None:
                lvl = int(levels[i].detach().to("cpu").item())
                lvl_str = f" level={lvl}"
            # show roll/teach for quick sanity
            rollv = x_roll_abs[i].detach().to("cpu").tolist()
            teachv = x_tgt_abs[i].detach().to("cpu").tolist()
            fn = feature_names
            print(f"  node={i:6d} score={score:.3e}{xy_str}{lvl_str}")
            print(f"    roll [{','.join(fn)}]={rollv}")
            print(f"    teach[{','.join(fn)}]={teachv}")


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


def _chan_name_to_index_map():
    # accepts both your config names and your internal short names
    return {
        "density": 0, "rho": 0,
        "x_momentum": 1, "mx": 1,
        "y_momentum": 2, "my": 2,
        "energy": 3, "E": 3,
    }

def _channels_to_indices(ch_list):
    m = _chan_name_to_index_map()
    out = []
    for s in (ch_list or []):
        if s not in m:
            raise ValueError(f"Unknown channel name '{s}'. Expected one of: {sorted(m.keys())}")
        out.append(m[s])
    return out

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

def _budget_feature_indices(cfg: dict, Fdim: int):
    # Use your existing helper (same one used in dec_ops ensures correct mapping)
    return dec.infer_feature_indices(cfg, Fdim)

def _compute_budget_row(*, x_abs: torch.Tensor, levels: torch.Tensor, dx0: float, dy0: float, cfg: dict):
    """
    Compute global integrals on the mesh defined by `levels`.
    x_abs: (N,F) absolute conserved variables on THAT mesh.
    levels: (N,) refinement levels for that same mesh.
    Returns: area, mass, mom_x, mom_y, energy  (floats)
    """
    # Keep everything in float32 for MPS safety
    x = x_abs.detach().to(device="cpu", dtype=torch.float32)
    lev = levels.detach().to(device="cpu", dtype=torch.int64).view(-1)

    w = dec.cell_area_from_levels(
        lev,
        dx0=float(dx0),
        dy0=float(dy0),
        dtype=torch.float32,
        device=torch.device("cpu"),
        refine_ratio=_get_refine_ratio(cfg),
    )  # (N,)

    idx = _budget_feature_indices(cfg, int(x.shape[1]))
    rho = x[:, idx["rho"]]
    mx  = x[:, idx["mx"]]
    my  = x[:, idx["my"]]
    E   = x[:, idx["E"]]

    area   = float(w.sum().item())
    mass   = float((w * rho).sum().item())
    mom_x  = float((w * mx ).sum().item())
    mom_y  = float((w * my ).sum().item())
    energy = float((w * E  ).sum().item())
    return area, mass, mom_x, mom_y, energy

def _global_integrals_on_mesh(x_abs: torch.Tensor, area: torch.Tensor, cfg: dict):
    """
    x_abs: [N,F] absolute units (rho, mx, my, E assumed by infer_feature_indices)
    area:  [N]   cell area in physical units
    Returns python floats computed on CPU in float64 for stability (works on MPS).
    """
    idx = dec.infer_feature_indices(cfg, x_abs.size(1))
    rho = x_abs[:, idx["rho"]]
    mx  = x_abs[:, idx["mx"]]
    my  = x_abs[:, idx["my"]]
    E   = x_abs[:, idx["E"]]

    # compute on CPU/float64 to avoid MPS float64 limitations on-tensor ops
    a_cpu   = area.detach().cpu().double()
    rho_cpu = rho.detach().cpu().double()
    mx_cpu  = mx.detach().cpu().double()
    my_cpu  = my.detach().cpu().double()
    E_cpu   = E.detach().cpu().double()

    M  = (rho_cpu * a_cpu).sum().item()
    Px = (mx_cpu  * a_cpu).sum().item()
    Py = (my_cpu  * a_cpu).sum().item()
    Et = (E_cpu   * a_cpu).sum().item()

    return {"M": M, "Px": Px, "Py": Py, "E": Et}


def _budget_row(tag: str, step_k: int, dt_phys: float, budgets: dict):
    # flatten into a simple dict suitable for logging / CSV
    return {
        "tag": tag,
        "step_k": int(step_k),
        "dt_phys": float(dt_phys),
        "M": float(budgets["M"]),
        "Px": float(budgets["Px"]),
        "Py": float(budgets["Py"]),
        "E": float(budgets["E"]),
    }

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

def _dt_hat_feature_column(
    dt_hat: torch.Tensor | float,
    *,
    n_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return dt_hat as a per-node column of shape (N, 1)."""
    if torch.is_tensor(dt_hat):
        dt = dt_hat.to(device=device, dtype=dtype)
    else:
        dt = torch.tensor(float(dt_hat), device=device, dtype=dtype)

    if dt.numel() == 1:
        return dt.view(1, 1).expand(n_nodes, 1)
    if dt.dim() == 1 and dt.numel() == n_nodes:
        return dt.view(n_nodes, 1)
    if dt.dim() == 2 and tuple(dt.shape) == (n_nodes, 1):
        return dt

    raise RuntimeError(
        "Could not broadcast dt_hat to node feature column. "
        f"Expected scalar or (N,) / (N,1), got shape={tuple(dt.shape)} with N={n_nodes}."
    )


def _build_X(feat, centers, levels, cfg, dt_hat=None):
    """
    Build node features for FeatureNet:
      [ physics_feat ] (+ [x,y] if use_pos) (+ [level] if use_level)
      (+ [dt_hat_fixed] if features.build.use_dt_hat_fixed)

    feat:    (N, F) on whatever device we want to run the model on (cpu/cuda/mps)
    centers: (N, 2) (typically from precompute; may be on cpu)
    levels:  (N,) or (N,1)
    """
    build_cfg = cfg.get("features", {}).get("build", {})
    use_pos   = build_cfg.get("use_pos", True)
    use_level = build_cfg.get("use_level", True)
    use_dt_hat_fixed = bool(build_cfg.get("use_dt_hat_fixed", False))

    dev = feat.device
    Xs = [feat]  # already on dev

    if use_pos:
        Xs.append(centers.to(dev))

    if use_level:
        lvl = levels
        if lvl.dim() == 1:
            lvl = lvl.unsqueeze(-1)
        Xs.append(lvl.to(dev).to(feat.dtype))

    if use_dt_hat_fixed:
        if dt_hat is None:
            raise RuntimeError(
                "features.build.use_dt_hat_fixed=true but dt_hat was not provided to _build_X."
            )
        Xs.append(
            _dt_hat_feature_column(
                dt_hat,
                n_nodes=int(feat.size(0)),
                device=dev,
                dtype=feat.dtype,
            )
        )

    return torch.cat(Xs, dim=-1)

def _forward_main_head(
    model: FeatureNet,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
) -> torch.Tensor:
    if edge_attr is None:
        out = model(X, edge_index)
    else:
        out = model(X, edge_index, edge_attr=edge_attr)
    if isinstance(out, (list, tuple)):
        return out[0]
    return out
def _forward_main_head_with_edge_attr(model, x_in, edge_index, edge_attr=None, *, force_fp32: bool = True):
    """
    Backward compatible:
      - If model ignores edge_attr, it still works.
      - If force_fp32=True, we run THIS forward in fp32 (autocast disabled) to avoid fp16 NaNs.
    """
    if edge_attr is not None and not hasattr(_forward_main_head_with_edge_attr, "_printed"):
        _forward_main_head_with_edge_attr._printed = True
        print("[DEC-CHK] Forward called with edge_attr present.", flush=True)

    hang_dbg_once = not hasattr(_forward_main_head_with_edge_attr, "_hang_dbg_once")
    if hang_dbg_once:
        try:
            ei_shape = tuple(edge_index.shape) if torch.is_tensor(edge_index) else None
        except Exception:
            ei_shape = None
        try:
            ea_shape = tuple(edge_attr.shape) if torch.is_tensor(edge_attr) else None
        except Exception:
            ea_shape = None
        print(
            f"[HANG-DBG] entering _forward_main_head_with_edge_attr: "
            f"x_in={tuple(x_in.shape)} dtype={x_in.dtype} dev={x_in.device} "
            f"edge_index={ei_shape} edge_attr={ea_shape} force_fp32={bool(force_fp32)}",
            flush=True,
        )

    # --- choose autocast-off context safely ---
    if force_fp32:
        if x_in.is_cuda:
            autocast_off = torch.amp.autocast(device_type="cuda", enabled=False)
        else:
            # torch.autocast exists on newer torch; fall back to nullcontext if unavailable
            autocast_off = getattr(torch, "autocast", None)
            autocast_off = autocast_off(device_type=x_in.device.type, enabled=False) if autocast_off else nullcontext()
    else:
        autocast_off = nullcontext()

    # --- run forward ---
    with autocast_off:
        X = x_in.float() if force_fp32 else x_in
        EA = (edge_attr.float() if (force_fp32 and edge_attr is not None) else edge_attr)

        t_fwd0 = time.perf_counter() if hang_dbg_once else None
        if hang_dbg_once:
            print("[HANG-DBG] calling model forward...", flush=True)

        if EA is None:
            y = _forward_main_head(model, X, edge_index)
        else:
            try:
                # model supports edge_attr
                y = _forward_main_head(model, X, edge_index, edge_attr=EA)
            except TypeError:
                # model doesn’t accept edge_attr yet
                if hang_dbg_once:
                    print("[HANG-DBG] model forward rejected edge_attr; retrying without edge_attr.", flush=True)
                y = _forward_main_head(model, X, edge_index)

        if hang_dbg_once:
            dt_fwd = time.perf_counter() - t_fwd0
            print(f"[HANG-DBG] model forward returned in {dt_fwd:.3f}s", flush=True)
            _forward_main_head_with_edge_attr._hang_dbg_once = True

    # If you want to keep downstream memory low / match dtype expectations, cast back.
    # (Optionally clamp before casting if you ever see fp16 overflow.)
    if force_fp32 and (y.dtype != x_in.dtype):
        y = y.to(dtype=x_in.dtype)

    return y

def _get_bbox(cfg: Dict[str,Any]) -> Tuple[float,float,float,float]:
    if cfg.get("data", {}).get("bbox"):
        xmin, xmax, ymin, ymax = cfg["data"]["bbox"]
    else:
        dom = cfg.get("domain", {})
        xmin = float(dom.get("xmin", 0.0)); xmax = float(dom.get("xmax", 1.0))
        ymin = float(dom.get("ymin", 0.0)); ymax = float(dom.get("ymax", 1.0))
    return xmin, xmax, ymin, ymax


def _select_idw_backend(
    *,
    src_n: int,
    requested_chunk: int,
    out_device: torch.device,
) -> tuple[torch.device, int]:
    """
    Choose a safer device/chunk for IDW cdist blocks.
    - On MPS, run IDW on CPU to avoid large MPS cdist allocations.
    - Cap chunk so chunk*src_n pairwise matrix stays bounded.
    """
    out_dev = torch.device(out_device)
    idw_dev = torch.device("cpu") if out_dev.type == "mps" else out_dev

    if src_n <= 0:
        return idw_dev, max(1, int(requested_chunk))

    # Approximate cap on pairwise distance matrix elements (float32).
    # CPU can tolerate a larger temporary than GPU backends.
    max_pair_elems = 64_000_000 if idw_dev.type == "cpu" else 32_000_000
    cap_chunk = max(1, int(max_pair_elems // int(src_n)))
    eff_chunk = max(1, min(int(requested_chunk), cap_chunk))
    return idw_dev, eff_chunk


def _runtime_idw_backend_settings(
    cfg: Dict[str, Any],
    *,
    out_device: torch.device,
) -> tuple[str, Dict[str, Any]]:
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    idw_cfg = rt_cfg.get("idw", {}) or {}
    if not isinstance(idw_cfg, dict):
        raise ValueError("train.runtime_mesh.idw must be a JSON object when provided.")

    raw_backend = str(idw_cfg.get("backend", "exact")).strip().lower()
    backend_aliases = {
        "torch": "exact",
        "cdist": "exact",
        "exact": "exact",
        "flat": "faiss_flat",
        "faiss_flat": "faiss_flat",
        "faiss": "faiss_ivf",
        "ivf": "faiss_ivf",
        "faiss_ivf": "faiss_ivf",
        "ann": "faiss_ivf",
        "approx": "faiss_ivf",
    }
    backend = backend_aliases.get(raw_backend, raw_backend)
    if backend not in ("exact", "faiss_flat", "faiss_ivf"):
        raise ValueError(
            "train.runtime_mesh.idw.backend must be one of: exact, faiss_flat, faiss_ivf."
        )

    allow_fallback = bool(idw_cfg.get("allow_fallback_to_exact", True))
    faiss_nlist = max(1, int(idw_cfg.get("faiss_nlist", 256)))
    faiss_nprobe = max(1, int(idw_cfg.get("faiss_nprobe", 16)))
    faiss_cache = bool(idw_cfg.get("faiss_cache", True))
    faiss_cache_max_entries = max(1, int(idw_cfg.get("faiss_cache_max_entries", 4)))

    out_dev = torch.device(out_device)
    if backend in ("faiss_flat", "faiss_ivf"):
        if out_dev.type != "cuda":
            msg = (
                f"[IDW] backend={backend} requested but device={out_dev.type} is not CUDA; "
                "falling back to exact."
            )
            if allow_fallback:
                if not getattr(_runtime_idw_backend_settings, "_warn_non_cuda", False):
                    print(msg, flush=True)
                    _runtime_idw_backend_settings._warn_non_cuda = True
                backend = "exact"
            else:
                raise RuntimeError(msg + " Set allow_fallback_to_exact=true or use exact backend.")
        else:
            try:
                import faiss  # type: ignore
                if not hasattr(faiss, "StandardGpuResources"):
                    raise RuntimeError("installed faiss build does not expose CUDA GPU resources")
            except Exception as e:
                msg = (
                    f"[IDW] backend={backend} requested but faiss is unavailable ({e!r}); "
                    "falling back to exact."
                )
                if allow_fallback:
                    if not getattr(_runtime_idw_backend_settings, "_warn_no_faiss", False):
                        print(msg, flush=True)
                        _runtime_idw_backend_settings._warn_no_faiss = True
                    backend = "exact"
                else:
                    raise RuntimeError(
                        "train.runtime_mesh.idw.backend requests faiss, but import failed."
                    ) from e

    backend_kwargs: Dict[str, Any] = {}
    if backend in ("faiss_flat", "faiss_ivf"):
        backend_kwargs["faiss_nlist"] = int(faiss_nlist)
        backend_kwargs["faiss_nprobe"] = int(faiss_nprobe)
        backend_kwargs["faiss_cache"] = bool(faiss_cache)
        backend_kwargs["faiss_cache_max_entries"] = int(faiss_cache_max_entries)
    return backend, backend_kwargs


_RUNTIME_STEP_LOG_FIELDS = [
    "wall_time",
    "split",
    "epoch",
    "batch_idx",
    "step_k",
    "t_abs",
    "used_precomp_step0",
    "do_rebuild",
    "update_every_steps",
    "device",
    "n_state",
    "n_pred",
    "e_pred",
    "n_gt_src",
    "idw_dev_xin",
    "idw_chunk_xin",
    "n_src_xin",
    "n_dst_xin",
    "idw_dev_tgt",
    "idw_chunk_tgt",
    "n_src_tgt",
    "n_dst_tgt",
    "t_rebuild_s",
    "t_xin_map_s",
    "t_xtgt_map_s",
    "t_model_s",
    "t_step_total_s",
    "loss",
    "mae",
    "mem_alloc_mb",
    "mem_reserved_mb",
]


def _runtime_step_log_settings(cfg: Dict[str, Any]) -> tuple[bool, str, bool]:
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    step_log_cfg = rt_cfg.get("step_log", {}) or {}
    enabled = bool(step_log_cfg.get("enabled", False))
    path_default = os.path.join(cfg.get("train", {}).get("save_dir", "."), "runtime_step_log.csv")
    path = os.path.abspath(os.path.expanduser(str(step_log_cfg.get("path", path_default))))
    append = bool(step_log_cfg.get("append", False))
    return enabled, path, append


def _runtime_memory_snapshot_mb(device: torch.device) -> tuple[float, float]:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        didx = dev.index if dev.index is not None else torch.cuda.current_device()
        alloc = float(torch.cuda.memory_allocated(didx)) / (1024.0 * 1024.0)
        reserv = float(torch.cuda.memory_reserved(didx)) / (1024.0 * 1024.0)
        return alloc, reserv
    if dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        try:
            alloc = float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
            return alloc, float("nan")
        except Exception:
            return float("nan"), float("nan")
    return float("nan"), float("nan")


def _runtime_step_log_write(cfg: Dict[str, Any], row: Dict[str, Any]) -> None:
    enabled, path, append = _runtime_step_log_settings(cfg)
    if not enabled:
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if not hasattr(_runtime_step_log_write, "_initialized_paths"):
        _runtime_step_log_write._initialized_paths = set()

    init_set = _runtime_step_log_write._initialized_paths
    if path not in init_set:
        needs_header = (not append) or (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        if needs_header:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_RUNTIME_STEP_LOG_FIELDS)
                writer.writeheader()
        init_set.add(path)

    row_out = {k: row.get(k, "") for k in _RUNTIME_STEP_LOG_FIELDS}
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RUNTIME_STEP_LOG_FIELDS)
        writer.writerow(row_out)


def _runtime_step_log_write_summary(csv_path: str, summary_path: str) -> None:
    if not os.path.exists(csv_path):
        return

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        return

    metrics = [
        "t_rebuild_s",
        "t_xin_map_s",
        "t_xtgt_map_s",
        "t_model_s",
        "t_step_total_s",
        "n_state",
        "n_pred",
        "e_pred",
        "n_gt_src",
        "n_src_xin",
        "n_dst_xin",
        "n_src_tgt",
        "n_dst_tgt",
        "loss",
        "mae",
        "mem_alloc_mb",
        "mem_reserved_mb",
    ]

    groups: dict[str, list[dict[str, str]]] = {"all": rows}
    for r in rows:
        sp = str(r.get("split", "unknown"))
        groups.setdefault(sp, []).append(r)

    def _to_finite(vals: list[str]) -> list[float]:
        out = []
        for v in vals:
            try:
                x = float(v)
            except Exception:
                continue
            if np.isfinite(x):
                out.append(x)
        return out

    def _pct(values: list[float], p: float) -> float:
        if not values:
            return float("nan")
        xs = sorted(values)
        i = int(round((len(xs) - 1) * p))
        i = max(0, min(i, len(xs) - 1))
        return float(xs[i])

    parent = os.path.dirname(summary_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(summary_path, "w") as f:
        f.write(f"runtime_step_log: {csv_path}\n")
        f.write(f"rows: {len(rows)}\n\n")
        for gname in ["all"] + [k for k in groups.keys() if k != "all"]:
            grows = groups[gname]
            f.write(f"[{gname}] rows={len(grows)}\n")
            for m in metrics:
                vals = _to_finite([r.get(m, "") for r in grows])
                if not vals:
                    continue
                n = len(vals)
                mean = float(np.mean(vals))
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
                p95 = _pct(vals, 0.95)
                f.write(
                    f"  {m}: n={n} mean={mean:.6g} min={vmin:.6g} max={vmax:.6g} p95={p95:.6g}\n"
                )
            f.write("\n")


def _resolve_diagnostics_cfg(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    diagnostics = cfg.setdefault("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise ValueError("diagnostics must be a JSON object when provided.")

    loss_cfg = cfg.get("loss", {}) or {}

    entropy_cfg = diagnostics.setdefault("entropy", {})
    if not isinstance(entropy_cfg, dict):
        raise ValueError("diagnostics.entropy must be a JSON object when provided.")
    entropy_cfg.setdefault("enabled", True)
    entropy_cfg.setdefault("save_to_csv", True)

    admiss_cfg = diagnostics.setdefault("admissibility", {})
    if not isinstance(admiss_cfg, dict):
        raise ValueError("diagnostics.admissibility must be a JSON object when provided.")
    admiss_cfg.setdefault("enabled", True)
    admiss_cfg.setdefault("save_to_csv", True)
    admiss_cfg.setdefault("pre_sanitize", True)
    admiss_cfg.setdefault("post_sanitize", True)
    admiss_cfg.setdefault("rho_floor", loss_cfg.get("rho_floor", 1e-6))
    admiss_cfg.setdefault("eint_floor", 0.0)
    admiss_cfg.setdefault("p_floor", loss_cfg.get("p_floor", 0.0))

    shock_cfg = diagnostics.setdefault("shock_masked_entropy", {})
    if not isinstance(shock_cfg, dict):
        raise ValueError("diagnostics.shock_masked_entropy must be a JSON object when provided.")
    shock_cfg.setdefault("enabled", True)
    shock_cfg.setdefault("save_to_csv", True)
    shock_cfg.setdefault("source", "gt_density_gradient")
    shock_cfg.setdefault("top_fraction", 0.10)
    shock_cfg.setdefault("min_cells", 32)

    resolved = {
        "entropy": {
            "enabled": bool(entropy_cfg.get("enabled", True)),
            "save_to_csv": bool(entropy_cfg.get("save_to_csv", True)),
        },
        "admissibility": {
            "enabled": bool(admiss_cfg.get("enabled", True)),
            "save_to_csv": bool(admiss_cfg.get("save_to_csv", True)),
            "pre_sanitize": bool(admiss_cfg.get("pre_sanitize", True)),
            "post_sanitize": bool(admiss_cfg.get("post_sanitize", True)),
            "rho_floor": float(admiss_cfg.get("rho_floor", loss_cfg.get("rho_floor", 1e-6))),
            "eint_floor": float(admiss_cfg.get("eint_floor", 0.0)),
            "p_floor": float(admiss_cfg.get("p_floor", loss_cfg.get("p_floor", 0.0))),
        },
        "shock_masked_entropy": {
            "enabled": bool(shock_cfg.get("enabled", True)),
            "save_to_csv": bool(shock_cfg.get("save_to_csv", True)),
            "source": str(shock_cfg.get("source", "gt_density_gradient")).strip().lower(),
            "top_fraction": float(shock_cfg.get("top_fraction", 0.10)),
            "min_cells": int(shock_cfg.get("min_cells", 32)),
        },
    }

    admiss_resolved = resolved["admissibility"]
    if admiss_resolved["enabled"] and (not admiss_resolved["pre_sanitize"]) and (not admiss_resolved["post_sanitize"]):
        raise ValueError(
            "diagnostics.admissibility.enabled=true requires pre_sanitize=true or post_sanitize=true."
        )

    shock_resolved = resolved["shock_masked_entropy"]
    if shock_resolved["enabled"]:
        if not (0.0 < shock_resolved["top_fraction"] <= 1.0):
            raise ValueError("diagnostics.shock_masked_entropy.top_fraction must be in (0, 1].")
        if shock_resolved["min_cells"] < 1:
            raise ValueError("diagnostics.shock_masked_entropy.min_cells must be >= 1.")
        if shock_resolved["source"] != "gt_density_gradient":
            raise ValueError(
                "diagnostics.shock_masked_entropy.source currently supports only 'gt_density_gradient'."
            )

    return resolved


def _admissibility_prefixes(admiss_cfg: Dict[str, Any]) -> List[str]:
    prefixes: List[str] = []
    if bool(admiss_cfg.get("pre_sanitize", False)):
        prefixes.append("admiss_pre")
    if bool(admiss_cfg.get("post_sanitize", False)):
        prefixes.append("admiss_post")
    return prefixes


def _entropy_diagnostic_pair(
    pred_abs: torch.Tensor,
    gt_abs: torch.Tensor,
    pred_levels: torch.Tensor,
    cfg: Dict[str, Any],
    *,
    dx: float,
    dy: float,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Entropy diagnostics on the prediction mesh.
    total entropy uses the area integral of rho * s, where
      s ~ log(p) - gamma * log(rho)
    and therefore is an entropy proxy up to an additive constant.
    """
    if pred_abs.ndim != 2 or gt_abs.ndim != 2:
        raise RuntimeError(
            f"Entropy diagnostics expect [N,F] tensors, got pred={tuple(pred_abs.shape)} gt={tuple(gt_abs.shape)}"
        )
    if pred_abs.shape != gt_abs.shape:
        raise RuntimeError(
            f"Entropy diagnostic shape mismatch: pred={tuple(pred_abs.shape)} gt={tuple(gt_abs.shape)}"
        )

    levels = pred_levels.view(-1)
    w = dec.cell_area_from_levels(
        levels,
        dx0=float(dx),
        dy0=float(dy),
        dtype=pred_abs.dtype,
        device=pred_abs.device,
        refine_ratio=_get_refine_ratio(cfg),
    )
    idx = dec.infer_feature_indices(cfg, pred_abs.size(1))

    def _one(x_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rho = x_abs[:, idx["rho"]].abs().clamp_min(eps)
        s = dec.specific_entropy_from_conservative_state(x_abs, cfg, eps=eps, p_floor=eps)
        total = (w * (rho * s)).sum()
        mass = (w * rho).sum()
        return total, mass

    pred_total, pred_mass = _one(pred_abs)
    gt_total, gt_mass = _one(gt_abs)
    return {
        "entropy_pred_total": pred_total,
        "entropy_gt_total": gt_total,
        "entropy_pred_mass": pred_mass,
        "entropy_gt_mass": gt_mass,
    }


def _admissibility_diagnostic(
    x_abs: torch.Tensor,
    pred_levels: torch.Tensor,
    cfg: Dict[str, Any],
    *,
    dx: float,
    dy: float,
    rho_floor: float,
    eint_floor: float,
    p_floor: float,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    if x_abs.ndim != 2:
        raise RuntimeError(f"Admissibility diagnostics expect [N,F], got {tuple(x_abs.shape)}")

    levels = pred_levels.view(-1)
    w = dec.cell_area_from_levels(
        levels,
        dx0=float(dx),
        dy0=float(dy),
        dtype=x_abs.dtype,
        device=x_abs.device,
        refine_ratio=_get_refine_ratio(cfg),
    )
    idx = dec.infer_feature_indices(cfg, x_abs.size(1))
    rho = x_abs[:, idx["rho"]]
    mx = x_abs[:, idx["mx"]]
    my = x_abs[:, idx["my"]]
    E = x_abs[:, idx["E"]]

    finite_state = torch.isfinite(rho) & torch.isfinite(mx) & torch.isfinite(my) & torch.isfinite(E)
    eint = dec.specific_internal_energy_from_conservative_state(x_abs, cfg, eps=eps)
    p = dec.pressure_from_conservative_state(x_abs, cfg, eps=eps, clamp_min=None)

    rho_violation = (~finite_state) | (rho <= float(rho_floor))
    eint_violation = rho_violation | (~torch.isfinite(eint)) | (eint <= float(eint_floor))
    p_violation = rho_violation | (~torch.isfinite(p)) | (p <= float(p_floor))

    def _finite_min(t: torch.Tensor) -> torch.Tensor:
        finite_t = t[torch.isfinite(t)]
        if finite_t.numel() == 0:
            return t.new_tensor(float("nan"))
        return finite_t.min()

    return {
        "area_sum": w.sum(),
        "rho_violation_area_sum": (w * rho_violation.to(w.dtype)).sum(),
        "eint_violation_area_sum": (w * eint_violation.to(w.dtype)).sum(),
        "p_violation_area_sum": (w * p_violation.to(w.dtype)).sum(),
        "rho_min": _finite_min(rho),
        "eint_min": _finite_min(eint),
        "p_min": _finite_min(p),
    }


def _edge_geometry_from_centers(
    centers: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    dtype: torch.dtype,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if centers.ndim != 2 or centers.size(1) < 2:
        raise RuntimeError(f"Expected centers with shape [N,>=2], got {tuple(centers.shape)}")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise RuntimeError(f"Expected edge_index with shape [2,E], got {tuple(edge_index.shape)}")

    src = edge_index[0].long()
    dst = edge_index[1].long()
    xy = centers[:, :2].to(dtype=dtype)
    delta = xy.index_select(0, dst) - xy.index_select(0, src)
    nx = delta[:, 0]
    ny = delta[:, 1]
    dist = torch.sqrt((nx * nx + ny * ny).clamp_min(eps))
    return nx, ny, dist


def _shock_masked_entropy_diagnostic(
    pred_abs: torch.Tensor,
    gt_abs: torch.Tensor,
    pred_levels: torch.Tensor,
    pred_centers: torch.Tensor,
    pred_ei: torch.Tensor,
    cfg: Dict[str, Any],
    *,
    dx: float,
    dy: float,
    top_fraction: float,
    min_cells: int,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    if pred_abs.ndim != 2 or gt_abs.ndim != 2:
        raise RuntimeError(
            f"Shock-masked entropy diagnostics expect [N,F] tensors, got pred={tuple(pred_abs.shape)} gt={tuple(gt_abs.shape)}"
        )
    if pred_abs.shape != gt_abs.shape:
        raise RuntimeError(
            f"Shock-masked entropy shape mismatch: pred={tuple(pred_abs.shape)} gt={tuple(gt_abs.shape)}"
        )

    # NOTE:
    # Diagnostics are metrics-only (no grad path). On MPS we route to CPU to avoid
    # intermittent index_add_ placeholder errors; otherwise keep native device.
    diag_dev = torch.device("cpu") if pred_abs.device.type == "mps" else pred_abs.device
    pred_abs = pred_abs.detach().to(diag_dev)
    gt_abs = gt_abs.detach().to(diag_dev)
    pred_levels = pred_levels.detach().to(diag_dev)
    pred_centers = pred_centers.detach().to(diag_dev)
    pred_ei = pred_ei.detach().to(diag_dev) if pred_ei is not None else pred_ei

    levels = pred_levels.view(-1)
    w = dec.cell_area_from_levels(
        levels,
        dx0=float(dx),
        dy0=float(dy),
        dtype=pred_abs.dtype,
        device=pred_abs.device,
        refine_ratio=_get_refine_ratio(cfg),
    )
    total_area = w.sum()
    zero = pred_abs.new_tensor(0.0)

    if pred_abs.size(0) == 0 or pred_ei is None or pred_ei.numel() == 0:
        return {
            "shock_entropy_abs_diff_sum": zero,
            "shock_entropy_diff_sum": zero,
            "shock_entropy_mask_area_sum": zero,
            "shock_entropy_total_area_sum": total_area,
        }

    idx = dec.infer_feature_indices(cfg, gt_abs.size(1))
    rho_gt = gt_abs[:, idx["rho"]].reshape(-1, 1)
    nx, ny, edge_dist = _edge_geometry_from_centers(pred_centers, pred_ei, dtype=pred_abs.dtype, eps=eps)
    grad_rho = dec._compute_node_gradients_ls(rho_gt, pred_ei, nx, ny, edge_dist, eps=eps)
    grad_mag = torch.sqrt((grad_rho[:, 0, 0] * grad_rho[:, 0, 0] + grad_rho[:, 0, 1] * grad_rho[:, 0, 1]).clamp_min(0.0))

    valid_idx = torch.nonzero(torch.isfinite(grad_mag), as_tuple=False).view(-1)
    if valid_idx.numel() == 0:
        return {
            "shock_entropy_abs_diff_sum": zero,
            "shock_entropy_diff_sum": zero,
            "shock_entropy_mask_area_sum": zero,
            "shock_entropy_total_area_sum": total_area,
        }

    k = int(np.ceil(float(top_fraction) * float(valid_idx.numel())))
    k = max(int(min_cells), k)
    k = min(k, int(valid_idx.numel()))

    top_rel = torch.topk(grad_mag.index_select(0, valid_idx), k=k, largest=True).indices
    mask_idx = valid_idx.index_select(0, top_rel)
    mask = torch.zeros((pred_abs.size(0),), dtype=torch.bool, device=pred_abs.device)
    mask[mask_idx] = True

    s_pred = dec.specific_entropy_from_conservative_state(pred_abs, cfg, eps=eps, p_floor=eps)
    s_gt = dec.specific_entropy_from_conservative_state(gt_abs, cfg, eps=eps, p_floor=eps)
    diff = s_pred - s_gt
    mask = mask & torch.isfinite(diff)
    if not torch.any(mask):
        return {
            "shock_entropy_abs_diff_sum": zero,
            "shock_entropy_diff_sum": zero,
            "shock_entropy_mask_area_sum": zero,
            "shock_entropy_total_area_sum": total_area,
        }

    w_mask = w[mask]
    diff_mask = diff[mask]
    return {
        "shock_entropy_abs_diff_sum": (w_mask * diff_mask.abs()).sum(),
        "shock_entropy_diff_sum": (w_mask * diff_mask).sum(),
        "shock_entropy_mask_area_sum": w_mask.sum(),
        "shock_entropy_total_area_sum": total_area,
    }


def _new_entropy_accumulator() -> Dict[str, Any]:
    return {
        "num_steps": 0,
        "entropy_pred_total_sum": None,
        "entropy_gt_total_sum": None,
        "entropy_pred_mass_sum": None,
        "entropy_gt_mass_sum": None,
    }


def _accumulate_entropy_accumulator(acc: Dict[str, Any], diag: Dict[str, torch.Tensor]) -> None:
    acc["num_steps"] += 1
    for key in ("entropy_pred_total_sum", "entropy_gt_total_sum", "entropy_pred_mass_sum", "entropy_gt_mass_sum"):
        src_key = key.replace("_sum", "")
        val = diag[src_key].detach()
        if acc[key] is None:
            acc[key] = val.clone()
        else:
            acc[key] = acc[key] + val


def _finalize_entropy_accumulator(acc: Dict[str, Any], *, eps: float = 1e-12) -> Dict[str, float]:
    n_steps = int(acc.get("num_steps", 0))
    if n_steps <= 0:
        return {
            "entropy_num_steps": 0,
            "entropy_pred_total": float("nan"),
            "entropy_gt_total": float("nan"),
            "entropy_total_gap": float("nan"),
            "entropy_pred_specific_mean": float("nan"),
            "entropy_gt_specific_mean": float("nan"),
            "entropy_specific_mean_gap": float("nan"),
        }

    pred_total_sum_t = acc.get("entropy_pred_total_sum")
    gt_total_sum_t = acc.get("entropy_gt_total_sum")
    pred_mass_sum_t = acc.get("entropy_pred_mass_sum")
    gt_mass_sum_t = acc.get("entropy_gt_mass_sum")
    if pred_total_sum_t is None or gt_total_sum_t is None or pred_mass_sum_t is None or gt_mass_sum_t is None:
        raise RuntimeError("Entropy accumulator is incomplete despite num_steps > 0.")

    pred_total_sum = float(pred_total_sum_t.detach().cpu().item())
    gt_total_sum = float(gt_total_sum_t.detach().cpu().item())
    pred_mass_sum = max(float(pred_mass_sum_t.detach().cpu().item()), eps)
    gt_mass_sum = max(float(gt_mass_sum_t.detach().cpu().item()), eps)

    pred_total = pred_total_sum / float(n_steps)
    gt_total = gt_total_sum / float(n_steps)
    pred_mean = pred_total_sum / pred_mass_sum
    gt_mean = gt_total_sum / gt_mass_sum
    return {
        "entropy_num_steps": n_steps,
        "entropy_pred_total": pred_total,
        "entropy_gt_total": gt_total,
        "entropy_total_gap": pred_total - gt_total,
        "entropy_pred_specific_mean": pred_mean,
        "entropy_gt_specific_mean": gt_mean,
        "entropy_specific_mean_gap": pred_mean - gt_mean,
    }


def _new_admissibility_accumulator(admiss_cfg: Dict[str, Any]) -> Dict[str, Any]:
    acc: Dict[str, Any] = {"num_steps": 0}
    for prefix in _admissibility_prefixes(admiss_cfg):
        acc[f"{prefix}_area_sum"] = None
        acc[f"{prefix}_rho_violation_area_sum"] = None
        acc[f"{prefix}_eint_violation_area_sum"] = None
        acc[f"{prefix}_p_violation_area_sum"] = None
        acc[f"{prefix}_rho_min"] = None
        acc[f"{prefix}_eint_min"] = None
        acc[f"{prefix}_p_min"] = None
    return acc


def _running_min_update(prev: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
    new_detached = new.detach()
    if prev is None:
        return new_detached.clone()
    prev_finite = torch.isfinite(prev)
    new_finite = torch.isfinite(new_detached)
    both_finite = prev_finite & new_finite
    return torch.where(
        both_finite,
        torch.minimum(prev, new_detached),
        torch.where(prev_finite, prev, new_detached),
    )


def _accumulate_admissibility_accumulator(
    acc: Dict[str, Any],
    diag: Dict[str, torch.Tensor],
    *,
    prefix: str,
) -> None:
    for key in ("area_sum", "rho_violation_area_sum", "eint_violation_area_sum", "p_violation_area_sum"):
        acc_key = f"{prefix}_{key}"
        val = diag[key].detach()
        if acc[acc_key] is None:
            acc[acc_key] = val.clone()
        else:
            acc[acc_key] = acc[acc_key] + val

    for key in ("rho_min", "eint_min", "p_min"):
        acc_key = f"{prefix}_{key}"
        acc[acc_key] = _running_min_update(acc.get(acc_key), diag[key])


def _finalize_admissibility_accumulator(
    acc: Dict[str, Any],
    admiss_cfg: Dict[str, Any],
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    out: Dict[str, float] = {"admissibility_num_steps": int(acc.get("num_steps", 0))}
    for prefix in _admissibility_prefixes(admiss_cfg):
        area_sum_t = acc.get(f"{prefix}_area_sum")
        rho_viol_t = acc.get(f"{prefix}_rho_violation_area_sum")
        eint_viol_t = acc.get(f"{prefix}_eint_violation_area_sum")
        p_viol_t = acc.get(f"{prefix}_p_violation_area_sum")

        if area_sum_t is None or rho_viol_t is None or eint_viol_t is None or p_viol_t is None:
            out[f"{prefix}_rho_violation_frac"] = float("nan")
            out[f"{prefix}_eint_violation_frac"] = float("nan")
            out[f"{prefix}_p_violation_frac"] = float("nan")
        else:
            area_sum = max(float(area_sum_t.detach().cpu().item()), eps)
            out[f"{prefix}_rho_violation_frac"] = float(rho_viol_t.detach().cpu().item()) / area_sum
            out[f"{prefix}_eint_violation_frac"] = float(eint_viol_t.detach().cpu().item()) / area_sum
            out[f"{prefix}_p_violation_frac"] = float(p_viol_t.detach().cpu().item()) / area_sum

        for key in ("rho_min", "eint_min", "p_min"):
            min_t = acc.get(f"{prefix}_{key}")
            out[f"{prefix}_{key}"] = float("nan") if min_t is None else float(min_t.detach().cpu().item())

    return out


def _new_shock_entropy_accumulator() -> Dict[str, Any]:
    return {
        "num_steps": 0,
        "shock_entropy_abs_diff_sum": None,
        "shock_entropy_diff_sum": None,
        "shock_entropy_mask_area_sum": None,
        "shock_entropy_total_area_sum": None,
    }


def _accumulate_shock_entropy_accumulator(acc: Dict[str, Any], diag: Dict[str, torch.Tensor]) -> None:
    acc["num_steps"] += 1
    for key in (
        "shock_entropy_abs_diff_sum",
        "shock_entropy_diff_sum",
        "shock_entropy_mask_area_sum",
        "shock_entropy_total_area_sum",
    ):
        val = diag[key].detach()
        if acc[key] is None:
            acc[key] = val.clone()
        else:
            acc[key] = acc[key] + val


def _finalize_shock_entropy_accumulator(acc: Dict[str, Any], *, eps: float = 1e-12) -> Dict[str, float]:
    n_steps = int(acc.get("num_steps", 0))
    if n_steps <= 0:
        return {
            "shock_entropy_num_steps": 0,
            "shock_entropy_specific_mae": float("nan"),
            "shock_entropy_specific_bias": float("nan"),
            "shock_entropy_mask_area_frac": float("nan"),
        }

    mask_area_t = acc.get("shock_entropy_mask_area_sum")
    total_area_t = acc.get("shock_entropy_total_area_sum")
    abs_diff_t = acc.get("shock_entropy_abs_diff_sum")
    diff_t = acc.get("shock_entropy_diff_sum")
    if mask_area_t is None or total_area_t is None or abs_diff_t is None or diff_t is None:
        raise RuntimeError("Shock-masked entropy accumulator is incomplete despite num_steps > 0.")

    mask_area = float(mask_area_t.detach().cpu().item())
    total_area = max(float(total_area_t.detach().cpu().item()), eps)
    if mask_area <= eps:
        mae = float("nan")
        bias = float("nan")
    else:
        mae = float(abs_diff_t.detach().cpu().item()) / mask_area
        bias = float(diff_t.detach().cpu().item()) / mask_area

    return {
        "shock_entropy_num_steps": n_steps,
        "shock_entropy_specific_mae": mae,
        "shock_entropy_specific_bias": bias,
        "shock_entropy_mask_area_frac": mask_area / total_area,
    }
    

def _new_diagnostics_accumulator(diag_cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    acc: Dict[str, Any] = {}
    if diag_cfg["entropy"]["enabled"]:
        acc["entropy"] = _new_entropy_accumulator()
    if diag_cfg["admissibility"]["enabled"]:
        acc["admissibility"] = _new_admissibility_accumulator(diag_cfg["admissibility"])
    if diag_cfg["shock_masked_entropy"]["enabled"]:
        acc["shock_masked_entropy"] = _new_shock_entropy_accumulator()
    return acc


def _record_enabled_step_diagnostics(
    acc: Dict[str, Any],
    *,
    pred_raw_abs: torch.Tensor,
    pred_used_abs: torch.Tensor,
    gt_abs: torch.Tensor,
    pred_levels: torch.Tensor,
    pred_centers: torch.Tensor,
    pred_ei: torch.Tensor,
    cfg: Dict[str, Any],
    diag_cfg: Dict[str, Dict[str, Any]],
    dx: float,
    dy: float,
) -> None:
    if diag_cfg["entropy"]["enabled"]:
        diag = _entropy_diagnostic_pair(
            pred_used_abs,
            gt_abs,
            pred_levels,
            cfg,
            dx=float(dx),
            dy=float(dy),
        )
        _accumulate_entropy_accumulator(acc["entropy"], diag)

    if diag_cfg["admissibility"]["enabled"]:
        admiss_cfg = diag_cfg["admissibility"]
        acc["admissibility"]["num_steps"] += 1
        base_kwargs = {
            "pred_levels": pred_levels,
            "cfg": cfg,
            "dx": float(dx),
            "dy": float(dy),
            "rho_floor": float(admiss_cfg["rho_floor"]),
            "eint_floor": float(admiss_cfg["eint_floor"]),
            "p_floor": float(admiss_cfg["p_floor"]),
        }
        if admiss_cfg["pre_sanitize"]:
            diag_pre = _admissibility_diagnostic(pred_raw_abs, **base_kwargs)
            _accumulate_admissibility_accumulator(acc["admissibility"], diag_pre, prefix="admiss_pre")
        if admiss_cfg["post_sanitize"]:
            diag_post = _admissibility_diagnostic(pred_used_abs, **base_kwargs)
            _accumulate_admissibility_accumulator(acc["admissibility"], diag_post, prefix="admiss_post")

    if diag_cfg["shock_masked_entropy"]["enabled"]:
        shock_cfg = diag_cfg["shock_masked_entropy"]
        diag = _shock_masked_entropy_diagnostic(
            pred_used_abs,
            gt_abs,
            pred_levels,
            pred_centers,
            pred_ei,
            cfg,
            dx=float(dx),
            dy=float(dy),
            top_fraction=float(shock_cfg["top_fraction"]),
            min_cells=int(shock_cfg["min_cells"]),
        )
        _accumulate_shock_entropy_accumulator(acc["shock_masked_entropy"], diag)


def _finalize_diagnostics_accumulator(
    acc: Dict[str, Any],
    diag_cfg: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if diag_cfg["entropy"]["enabled"]:
        out.update(_finalize_entropy_accumulator(acc["entropy"]))
    if diag_cfg["admissibility"]["enabled"]:
        out.update(_finalize_admissibility_accumulator(acc["admissibility"], diag_cfg["admissibility"]))
    if diag_cfg["shock_masked_entropy"]["enabled"]:
        out.update(_finalize_shock_entropy_accumulator(acc["shock_masked_entropy"]))
    return out


def _diagnostics_csv_fields(diag_cfg: Dict[str, Dict[str, Any]]) -> List[str]:
    fields: List[str] = []
    if diag_cfg["entropy"]["enabled"] and diag_cfg["entropy"]["save_to_csv"]:
        fields.extend(
            [
                "entropy_pred_total",
                "entropy_gt_total",
                "entropy_total_gap",
                "entropy_pred_specific_mean",
                "entropy_gt_specific_mean",
                "entropy_specific_mean_gap",
            ]
        )
    if diag_cfg["admissibility"]["enabled"] and diag_cfg["admissibility"]["save_to_csv"]:
        for prefix in _admissibility_prefixes(diag_cfg["admissibility"]):
            fields.extend(
                [
                    f"{prefix}_rho_violation_frac",
                    f"{prefix}_eint_violation_frac",
                    f"{prefix}_p_violation_frac",
                    f"{prefix}_rho_min",
                    f"{prefix}_eint_min",
                    f"{prefix}_p_min",
                ]
            )
    if diag_cfg["shock_masked_entropy"]["enabled"] and diag_cfg["shock_masked_entropy"]["save_to_csv"]:
        fields.extend(
            [
                "shock_entropy_specific_mae",
                "shock_entropy_specific_bias",
                "shock_entropy_mask_area_frac",
            ]
        )
    return fields


def _format_diag_float(v: float) -> str:
    return f"{float(v):.3e}" if np.isfinite(v) else str(float(v))


def _format_split_diagnostic_summary(
    split: str,
    stats: Dict[str, Any] | None,
    diag_cfg: Dict[str, Dict[str, Any]],
) -> str | None:
    if stats is None:
        return None

    parts: List[str] = []
    if diag_cfg["entropy"]["enabled"]:
        s_pred = float(stats.get("entropy_pred_specific_mean", float("nan")))
        s_gt = float(stats.get("entropy_gt_specific_mean", float("nan")))
        s_gap = float(stats.get("entropy_specific_mean_gap", float("nan")))
        parts.append(
            f"Sent(mean)={_format_diag_float(s_pred)}/{_format_diag_float(s_gt)} (d={_format_diag_float(s_gap)})"
        )

    if diag_cfg["admissibility"]["enabled"]:
        admiss_parts: List[str] = []
        for prefix in _admissibility_prefixes(diag_cfg["admissibility"]):
            rho_v = float(stats.get(f"{prefix}_rho_violation_frac", float("nan")))
            eint_v = float(stats.get(f"{prefix}_eint_violation_frac", float("nan")))
            p_v = float(stats.get(f"{prefix}_p_violation_frac", float("nan")))
            label = "pre" if prefix.endswith("pre") else "post"
            admiss_parts.append(
                f"{label} rho/e/p={_format_diag_float(rho_v)}/{_format_diag_float(eint_v)}/{_format_diag_float(p_v)}"
            )
        if admiss_parts:
            parts.append("Adm(" + "; ".join(admiss_parts) + ")")

    if diag_cfg["shock_masked_entropy"]["enabled"]:
        shock_mae = float(stats.get("shock_entropy_specific_mae", float("nan")))
        shock_bias = float(stats.get("shock_entropy_specific_bias", float("nan")))
        shock_area = float(stats.get("shock_entropy_mask_area_frac", float("nan")))
        parts.append(
            "ShockS("
            f"mae={_format_diag_float(shock_mae)}, "
            f"bias={_format_diag_float(shock_bias)}, "
            f"area={_format_diag_float(shock_area)})"
        )

    if not parts:
        return None
    return f"[DIAG] {split}: " + " | ".join(parts)


def _cell_level_ij_from_centers(
    *,
    centers: torch.Tensor,
    levels: torch.Tensor,
    H: int,
    W: int,
    bbox: Tuple[float, float, float, float],
    refine_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recover integer (row, col) indices of each cell on its own level grid from centers.
    """
    if centers.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=centers.device)
        return empty, empty

    dev = centers.device
    lv = levels.view(-1).long()
    max_l = int(torch.clamp(lv.max(), min=0).item())
    rr = int(refine_ratio)
    rr_pows = torch.tensor([rr ** i for i in range(max_l + 1)], device=dev, dtype=torch.float32)
    ww_l = (float(W) * rr_pows).index_select(0, lv.clamp(0, max_l))
    hh_l = (float(H) * rr_pows).index_select(0, lv.clamp(0, max_l))

    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    xs = max(float(xmax - xmin), 1e-12)
    ys = max(float(ymax - ymin), 1e-12)

    # Inverse of center formula x = xmin + (i + 0.5) * xs / WW (same for y/HH).
    col_f = ((centers[:, 0].to(torch.float32) - xmin) * ww_l / xs) - 0.5
    row_f = ((centers[:, 1].to(torch.float32) - ymin) * hh_l / ys) - 0.5

    col = torch.round(col_f).to(torch.long)
    row = torch.round(row_f).to(torch.long)

    ww_i = ww_l.to(torch.long)
    hh_i = hh_l.to(torch.long)
    col = torch.minimum(torch.maximum(col, torch.zeros_like(col)), ww_i - 1)
    row = torch.minimum(torch.maximum(row, torch.zeros_like(row)), hh_i - 1)
    return row, col


def _cell_keys_level_ij(
    *,
    levels: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    H: int,
    W: int,
    refine_ratio: int,
) -> torch.Tensor:
    """
    Build a unique int64 key for (level,row,col), with level-specific row/col.
    """
    if levels.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=levels.device)

    lv = levels.view(-1).long()
    rr = int(refine_ratio)
    max_l = int(torch.clamp(lv.max(), min=0).item())
    h_max = int(H * (rr ** max_l))
    w_max = int(W * (rr ** max_l))
    h_max = max(1, h_max)
    w_max = max(1, w_max)

    key = ((lv * int(h_max)) + row.view(-1).long()) * int(w_max) + col.view(-1).long()
    return key.to(torch.long)


def _runtime_coarse_parent_counts(
    *,
    parents: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    p = torch.as_tensor(parents, dtype=torch.long).view(-1)
    cnt = torch.zeros(int(H) * int(W), dtype=torch.float32, device=p.device)
    if p.numel() == 0:
        return cnt
    p = p.clamp(0, int(H) * int(W) - 1)
    ones = torch.ones((int(p.numel()),), dtype=torch.float32, device=p.device)
    cnt.index_add_(0, p, ones)
    return cnt


def _runtime_build_coarse_rc_grid(H: int, W: int, device: torch.device | None = None) -> torch.Tensor:
    dev = device if device is not None else torch.device("cpu")
    rr = torch.arange(int(H), dtype=torch.float32, device=dev)
    cc = torch.arange(int(W), dtype=torch.float32, device=dev)
    gy, gx = torch.meshgrid(rr, cc, indexing="ij")
    return torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=1).contiguous()


def _runtime_fill_empty_interior_coarse_nearest(
    *,
    coarse_chw: torch.Tensor,
    parents: torch.Tensor,
    H: int,
    W: int,
    physical_parent_mask: torch.Tensor,
    rc_grid: torch.Tensor | None = None,
    chunk: int = 1024,
) -> torch.Tensor:
    coarse = torch.as_tensor(coarse_chw, dtype=torch.float32)
    if coarse.ndim != 3:
        raise ValueError(f"coarse_chw must be (C,H,W), got {tuple(coarse.shape)}")
    C, HH, WW = int(coarse.shape[0]), int(coarse.shape[1]), int(coarse.shape[2])
    if HH != int(H) or WW != int(W):
        raise ValueError(
            f"coarse_chw spatial shape mismatch: got {(HH, WW)} expected {(int(H), int(W))}"
        )

    phys = torch.as_tensor(
        physical_parent_mask,
        dtype=torch.bool,
        device=coarse.device,
    ).view(-1)
    if int(phys.numel()) != int(H) * int(W):
        raise ValueError(
            f"physical_parent_mask must have {int(H) * int(W)} entries, got {int(phys.numel())}"
        )

    counts = _runtime_coarse_parent_counts(parents=parents, H=int(H), W=int(W)).to(device=coarse.device)
    fill_flat = phys & (counts <= 0.0)
    valid_flat = phys & (counts > 0.0)
    if (not bool(fill_flat.any())) or (not bool(valid_flat.any())):
        return coarse

    valid_idx = torch.nonzero(valid_flat, as_tuple=False).view(-1)
    fill_idx = torch.nonzero(fill_flat, as_tuple=False).view(-1)
    if valid_idx.numel() == 0 or fill_idx.numel() == 0:
        return coarse

    if rc_grid is None:
        rc = _runtime_build_coarse_rc_grid(int(H), int(W), device=coarse.device)
    else:
        rc = torch.as_tensor(rc_grid, dtype=torch.float32, device=coarse.device)
        if rc.ndim != 2 or int(rc.shape[1]) != 2 or int(rc.shape[0]) != int(H) * int(W):
            raise ValueError(f"rc_grid must be (H*W,2), got {tuple(rc.shape)}")

    valid_rc = rc.index_select(0, valid_idx)
    fill_rc = rc.index_select(0, fill_idx)
    flat = coarse.view(C, int(H) * int(W))
    out = flat.clone()

    chunk = max(1, int(chunk))
    for s in range(0, int(fill_idx.numel()), chunk):
        e = min(s + chunk, int(fill_idx.numel()))
        q_idx = fill_idx[s:e]
        q_rc = fill_rc[s:e]
        d = q_rc[:, None, :] - valid_rc[None, :, :]
        d2 = (d * d).sum(dim=-1)
        nn = torch.argmin(d2, dim=1)
        src_idx = valid_idx.index_select(0, nn)
        out[:, q_idx] = out[:, src_idx]

    return out.view(C, int(H), int(W)).contiguous()


_RUNTIME_WEDGE_CONSTRAINTS_CACHE: Dict[str, Dict[str, Any]] = {}
_RUNTIME_BASE_MESH_CACHE: Dict[str, Dict[str, Any]] = {}


def _runtime_lookup_mask_values_by_level(
    mask_by_level: Dict[int, torch.Tensor],
    *,
    levels: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
) -> torch.Tensor:
    lv = levels.view(-1).long()
    rv = row.view(-1).long()
    cv = col.view(-1).long()
    out = torch.zeros(lv.shape[0], dtype=torch.bool, device=lv.device)
    if lv.numel() == 0:
        return out
    lmin = int(lv.min().item())
    lmax = int(lv.max().item())
    for Lint in range(lmin, lmax + 1):
        if Lint not in mask_by_level:
            continue
        sel = (lv == Lint)
        if not bool(sel.any()):
            continue
        M = mask_by_level[Lint]
        out[sel] = M[rv[sel], cv[sel]]
    return out


def _runtime_wedge_constraints_cache_key(
    *,
    wedge_path,
    cfg: Dict[str, Any],
    H: int,
    W: int,
    device: torch.device,
) -> str:
    verts = np.asarray(wedge_path.vertices, dtype=np.float64)
    vhash = hashlib.sha1(verts.tobytes()).hexdigest()[:20]
    bbox = _get_bbox(cfg)
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    rr = _get_refine_ratio(cfg)
    Lmax = int((cfg.get("policy", {}) or {}).get("max_level", 3))
    dev = torch.device(device)
    dkey = f"{dev.type}:{dev.index if dev.index is not None else -1}"
    return (
        f"v={vhash}|H={int(H)}|W={int(W)}|Lmax={int(Lmax)}|rr={int(rr)}|"
        f"bbox={xmin:.16g},{xmax:.16g},{ymin:.16g},{ymax:.16g}|dev={dkey}"
    )


def _get_runtime_wedge_constraints(
    *,
    cfg: Dict[str, Any],
    H: int,
    W: int,
    device: torch.device,
    wedge_path,
) -> Dict[str, Any]:
    key = _runtime_wedge_constraints_cache_key(
        wedge_path=wedge_path,
        cfg=cfg,
        H=H,
        W=W,
        device=device,
    )
    got = _RUNTIME_WEDGE_CONSTRAINTS_CACHE.get(key, None)
    if got is not None:
        return got

    pol = cfg.get("policy", {}) or {}
    Lmax = int(pol.get("max_level", 3))
    rr = _get_refine_ratio(cfg)
    bbox = _get_bbox(cfg)
    xspan = float(bbox[1] - bbox[0])
    yspan = float(bbox[3] - bbox[2])
    radius = float(pol.get("wedge_clip_radius", 1e-9 * max(xspan, yspan)))
    dev = torch.device(device)

    lookup = _get_wedge_clip_level_lookup(
        wedge_path=wedge_path,
        H=H,
        W=W,
        Lmax=Lmax,
        refine_ratio=rr,
        bbox=bbox,
        radius=radius,
    )
    full_by_level, intersect_by_level = _lookup_level_masks_on_device(lookup, dev)

    leaf_full: Dict[int, torch.Tensor] = {}
    parent_intersect: Dict[int, torch.Tensor] = {}
    parent_boundary: Dict[int, torch.Tensor] = {}

    for l in range(0, Lmax + 1):
        leaf_full[l] = full_by_level[l].to(device=dev, dtype=torch.bool)
    for L in range(1, Lmax + 1):
        l = L - 1
        inter = intersect_by_level[l].to(device=dev, dtype=torch.bool)
        full = full_by_level[l].to(device=dev, dtype=torch.bool)
        parent_intersect[L] = inter
        # Cells that intersect but are not fully inside are always refined.
        parent_boundary[L] = inter & (~full)

    out = {
        "max_level": int(Lmax),
        "refine_ratio": int(rr),
        "bbox": tuple(float(v) for v in bbox),
        "leaf_full": leaf_full,
        "parent_intersect": parent_intersect,
        "parent_boundary": parent_boundary,
    }
    _RUNTIME_WEDGE_CONSTRAINTS_CACHE[key] = out
    print(
        f"[RUNTIME-WEDGE] built static constraints (H={int(H)} W={int(W)} Lmax={int(Lmax)} rr={int(rr)})",
        flush=True,
    )
    return out


def _runtime_apply_wedge_constraints_to_masks(
    *,
    masks_by_level: Dict[int, torch.Tensor],
    constraints: Dict[str, Any],
) -> Dict[int, torch.Tensor]:
    rr = int(constraints["refine_ratio"])
    Lmax = int(constraints["max_level"])
    out: Dict[int, torch.Tensor] = {}

    for L in range(1, Lmax + 1):
        inter = constraints["parent_intersect"][L]
        boundary = constraints["parent_boundary"][L]

        m = masks_by_level.get(L, None)
        if m is None:
            m = torch.zeros_like(inter, dtype=torch.bool)
        else:
            m = m.to(device=inter.device, dtype=torch.bool)
        if tuple(m.shape) != tuple(inter.shape):
            raise RuntimeError(
                f"Runtime wedge constraint shape mismatch at L={L}: "
                f"mask={tuple(m.shape)} expected={tuple(inter.shape)}"
            )

        # Keep only intersecting parents and force boundary parents to refine.
        m = (m & inter) | boundary

        # Enforce hierarchy consistency.
        if L > 1:
            allow = F.interpolate(
                out[L - 1].float().unsqueeze(0).unsqueeze(0),
                scale_factor=float(rr),
                mode="nearest",
            )[0, 0].to(torch.bool)
            m = m & allow[: m.shape[0], : m.shape[1]]

        out[L] = m

    return out


def _runtime_filter_mesh_with_wedge_constraints(
    *,
    pred_centers: torch.Tensor,
    pred_levels: torch.Tensor,
    parent_flat: torch.Tensor,
    cfg: Dict[str, Any],
    H: int,
    W: int,
    device: torch.device,
    constraints: Dict[str, Any],
    timing_out: Dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = torch.device(device)
    timing = {"lookup_classify_s": 0.0, "edge_build_s": 0.0}

    rr = int(constraints["refine_ratio"])
    bbox = tuple(constraints["bbox"])
    leaf_full: Dict[int, torch.Tensor] = constraints["leaf_full"]

    t_cls0 = time.perf_counter()
    row, col = _cell_level_ij_from_centers(
        centers=pred_centers,
        levels=pred_levels,
        H=H,
        W=W,
        bbox=bbox,
        refine_ratio=rr,
    )
    lvl = pred_levels.view(-1).long()
    keep = _runtime_lookup_mask_values_by_level(leaf_full, levels=lvl, row=row, col=col)
    pred_centers = pred_centers[keep]
    pred_levels = pred_levels[keep]
    parent_flat = parent_flat[keep]
    timing["lookup_classify_s"] = float(time.perf_counter() - t_cls0)

    if pred_centers.numel() == 0:
        raise RuntimeError(
            "Runtime wedge constrained mesh became empty after full-cell filter."
        )

    t_edge0 = time.perf_counter()
    edge_method = str(cfg.get("edges", {}).get("method", "amr_local_knn")).lower()
    if ("face" in edge_method):
        pred_ei = build_amr_face_adjacency_edges(
            pred_centers,
            pred_levels,
            H,
            W,
            bbox=bbox,
            return_edge_attr=False,
            refine_ratio=rr,
        ).to(dev, dtype=torch.long)
    else:
        pred_ei = build_amr_local_knn_edges(
            pred_centers,
            parent_flat,
            H,
            W,
            k_local=int(cfg.get("edges", {}).get("k_local", 4)),
            max_local=int(cfg.get("edges", {}).get("max_local", 2048)),
        ).to(dev, dtype=torch.long)
    if pred_ei.numel() == 0 or int(pred_ei.shape[1]) == 0:
        raise RuntimeError("Empty edge_index after runtime wedge-constrained filtering.")
    timing["edge_build_s"] = float(time.perf_counter() - t_edge0)

    mask_pred_parent = _mask_from_parent_indices(parent_flat, H, W, dev)
    if isinstance(timing_out, dict):
        timing_out.clear()
        timing_out.update(timing)
    return pred_centers, pred_levels, parent_flat, pred_ei, mask_pred_parent


def _runtime_base_mesh_cache_key(
    *,
    mesh_spec_path: str,
    cfg: Dict[str, Any],
    H: int,
    W: int,
    device: torch.device,
) -> str:
    st = os.stat(mesh_spec_path)
    bbox = _get_bbox(cfg)
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    rr = _get_refine_ratio(cfg)
    Lmax = int((cfg.get("policy", {}) or {}).get("max_level", 3))
    dev = torch.device(device)
    dkey = f"{dev.type}:{dev.index if dev.index is not None else -1}"
    return (
        f"path={os.path.abspath(mesh_spec_path)}|mtime_ns={int(st.st_mtime_ns)}|size={int(st.st_size)}|"
        f"H={int(H)}|W={int(W)}|Lmax={int(Lmax)}|rr={int(rr)}|"
        f"bbox={xmin:.16g},{xmax:.16g},{ymin:.16g},{ymax:.16g}|dev={dkey}"
    )


def _runtime_centers_from_level_ij(
    *,
    levels: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    H: int,
    W: int,
    bbox: Tuple[float, float, float, float],
    refine_ratio: int,
) -> torch.Tensor:
    if levels.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=levels.device)

    dev = levels.device
    lv = levels.view(-1).long()
    rr = int(refine_ratio)
    max_l = int(torch.clamp(lv.max(), min=0).item())
    rr_pows = torch.tensor([rr ** i for i in range(max_l + 1)], device=dev, dtype=torch.float32)
    ww_l = (float(W) * rr_pows).index_select(0, lv.clamp(0, max_l))
    hh_l = (float(H) * rr_pows).index_select(0, lv.clamp(0, max_l))

    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    xs = float(xmax - xmin)
    ys = float(ymax - ymin)
    cx = float(xmin) + (col.to(torch.float32) + 0.5) * (xs / ww_l)
    cy = float(ymin) + (row.to(torch.float32) + 0.5) * (ys / hh_l)
    return torch.stack([cx, cy], dim=-1).to(dtype=torch.float32)


def _runtime_parents_from_level_ij(
    *,
    levels: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    H: int,
    W: int,
    refine_ratio: int,
) -> torch.Tensor:
    if levels.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=levels.device)
    lv = levels.view(-1).long()
    rr = int(refine_ratio)
    max_l = int(torch.clamp(lv.max(), min=0).item())
    rr_pows = torch.tensor([rr ** i for i in range(max_l + 1)], device=levels.device, dtype=torch.long)
    scale = rr_pows.index_select(0, lv.clamp(0, max_l))
    i0 = torch.div(col.view(-1).long(), scale, rounding_mode="floor")
    j0 = torch.div(row.view(-1).long(), scale, rounding_mode="floor")
    return (j0 * int(W) + i0).to(torch.long)


def _runtime_base_refine_masks_from_leaf_set(
    *,
    levels: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    H: int,
    W: int,
    refine_ratio: int,
    L_max: int,
) -> Dict[int, torch.Tensor]:
    rr = int(refine_ratio)
    lv = levels.view(-1).long()
    rv = row.view(-1).long()
    cv = col.view(-1).long()
    out: Dict[int, torch.Tensor] = {}

    for L in range(1, int(L_max) + 1):
        hL = int(H) * (rr ** (L - 1))
        wL = int(W) * (rr ** (L - 1))
        m = torch.zeros((hL, wL), dtype=torch.bool, device=lv.device)
        sel = (lv >= int(L))
        if bool(sel.any()):
            scale_pow = lv[sel] - int(L - 1)
            scale = torch.pow(
                torch.tensor(int(rr), dtype=torch.long, device=lv.device),
                scale_pow,
            )
            pr = torch.div(rv[sel], scale, rounding_mode="floor").clamp_(0, hL - 1)
            pc = torch.div(cv[sel], scale, rounding_mode="floor").clamp_(0, wL - 1)
            m[pr, pc] = True
        out[L] = m.to(torch.bool)
    return out


def _runtime_sort_unique_leaf_ij(
    *,
    levels: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    H: int,
    W: int,
    refine_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if levels.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=levels.device)
        return empty, empty, empty

    lv = levels.view(-1).long()
    rv = row.view(-1).long()
    cv = col.view(-1).long()

    # Use the original unique-dim path on backends that support it (CUDA/CPU),
    # and keep a key-sort fallback for MPS where unique_dim is not implemented.
    if lv.device.type != "mps":
        trip = torch.stack([lv, rv, cv], dim=1)
        trip = torch.unique(trip, dim=0)
        lv_u = trip[:, 0].contiguous()
        rv_u = trip[:, 1].contiguous()
        cv_u = trip[:, 2].contiguous()
        key = _cell_keys_level_ij(
            levels=lv_u,
            row=rv_u,
            col=cv_u,
            H=H,
            W=W,
            refine_ratio=refine_ratio,
        )
        order = torch.argsort(key)
        return lv_u[order], rv_u[order], cv_u[order]

    key = _cell_keys_level_ij(
        levels=lv,
        row=rv,
        col=cv,
        H=H,
        W=W,
        refine_ratio=refine_ratio,
    )
    order = torch.argsort(key)
    lv_s = lv[order]
    rv_s = rv[order]
    cv_s = cv[order]
    key_s = key[order]
    keep = torch.ones((key_s.numel(),), dtype=torch.bool, device=key_s.device)
    if key_s.numel() > 1:
        keep[1:] = key_s[1:] != key_s[:-1]
    return lv_s[keep], rv_s[keep], cv_s[keep]


def _runtime_refine_leaf_set_from_policy_masks(
    *,
    base_levels: torch.Tensor,
    base_row: torch.Tensor,
    base_col: torch.Tensor,
    masks_by_level: Dict[int, torch.Tensor],
    H: int,
    W: int,
    refine_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    if base_levels.numel() == 0:
        raise RuntimeError("Runtime starting base mesh is empty.")

    lv = base_levels.view(-1).long().clone()
    row = base_row.view(-1).long().clone()
    col = base_col.view(-1).long().clone()

    Lmax = max((int(k) for k in masks_by_level.keys()), default=0)
    for L in range(1, Lmax + 1):
        m = masks_by_level.get(L, None)
        if m is None:
            continue
        m = m.to(device=lv.device, dtype=torch.bool)
        exp_h = H * (rr ** (L - 1))
        exp_w = W * (rr ** (L - 1))
        if tuple(m.shape) != (int(exp_h), int(exp_w)):
            raise RuntimeError(
                f"Runtime mask shape mismatch at L={L}: got={tuple(m.shape)} expected={(int(exp_h), int(exp_w))}"
            )

        parent_level = L - 1
        sel_parent = (lv == parent_level)
        if not bool(sel_parent.any()):
            continue

        idx_parent = torch.nonzero(sel_parent, as_tuple=False).view(-1)
        refine_local = m[row[idx_parent], col[idx_parent]]
        if not bool(refine_local.any()):
            continue

        idx_refine = idx_parent[refine_local]
        keep = torch.ones((lv.shape[0],), dtype=torch.bool, device=lv.device)
        keep[idx_refine] = False

        pr = row[idx_refine]
        pc = col[idx_refine]
        n_ref = int(pr.numel())
        child_count = rr * rr

        offs = torch.arange(rr, device=lv.device, dtype=torch.long)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        oy = oy.reshape(1, child_count)
        ox = ox.reshape(1, child_count)

        child_row = pr.view(-1, 1) * rr + oy
        child_col = pc.view(-1, 1) * rr + ox
        child_lv = torch.full((n_ref, child_count), int(L), dtype=torch.long, device=lv.device)

        lv = torch.cat([lv[keep], child_lv.reshape(-1)], dim=0)
        row = torch.cat([row[keep], child_row.reshape(-1)], dim=0)
        col = torch.cat([col[keep], child_col.reshape(-1)], dim=0)

    return _runtime_sort_unique_leaf_ij(
        levels=lv,
        row=row,
        col=col,
        H=H,
        W=W,
        refine_ratio=rr,
    )


def _runtime_build_edges_from_mesh(
    *,
    pred_centers: torch.Tensor,
    pred_levels: torch.Tensor,
    parent_flat: torch.Tensor,
    cfg: Dict[str, Any],
    H: int,
    W: int,
    bbox: Tuple[float, float, float, float],
    refine_ratio: int,
    device: torch.device,
) -> torch.Tensor:
    dev = torch.device(device)
    edge_method = str(cfg.get("edges", {}).get("method", "amr_local_knn")).lower()
    if ("face" in edge_method):
        pred_ei = build_amr_face_adjacency_edges(
            pred_centers,
            pred_levels,
            H,
            W,
            bbox=bbox,
            return_edge_attr=False,
            refine_ratio=refine_ratio,
        ).to(dev, dtype=torch.long)
    else:
        pred_ei = build_amr_local_knn_edges(
            pred_centers,
            parent_flat,
            H,
            W,
            k_local=int(cfg.get("edges", {}).get("k_local", 4)),
            max_local=int(cfg.get("edges", {}).get("max_local", 2048)),
        ).to(dev, dtype=torch.long)
    if pred_ei.numel() == 0 or int(pred_ei.shape[1]) == 0:
        raise RuntimeError("Empty edge_index for runtime mesh.")
    return pred_ei


def _get_runtime_starting_mesh_base(
    *,
    cfg: Dict[str, Any],
    H: int,
    W: int,
    dx: float,
    dy: float,
    device: torch.device,
    mesh_spec_path: str,
) -> Dict[str, Any]:
    key = _runtime_base_mesh_cache_key(
        mesh_spec_path=mesh_spec_path,
        cfg=cfg,
        H=H,
        W=W,
        device=device,
    )
    got = _RUNTIME_BASE_MESH_CACHE.get(key, None)
    if got is not None:
        return got

    dev = torch.device(device)
    rr = _get_refine_ratio(cfg)
    Lmax = int((cfg.get("policy", {}) or {}).get("max_level", 3))
    bbox = _get_bbox(cfg)

    centers, levels, _parents, _ei, _mask_parent = _build_starting_mesh_from_spec(
        mesh_spec_path,
        cfg,
        H,
        W,
        float(dx),
        float(dy),
        device=dev,
    )
    centers = centers.to(device=dev, dtype=torch.float32)
    levels = levels.to(device=dev, dtype=torch.long).view(-1)

    row, col = _cell_level_ij_from_centers(
        centers=centers,
        levels=levels,
        H=H,
        W=W,
        bbox=bbox,
        refine_ratio=rr,
    )
    levels, row, col = _runtime_sort_unique_leaf_ij(
        levels=levels,
        row=row,
        col=col,
        H=H,
        W=W,
        refine_ratio=rr,
    )
    parents = _runtime_parents_from_level_ij(
        levels=levels,
        row=row,
        col=col,
        H=H,
        W=W,
        refine_ratio=rr,
    )
    mask_parent = _mask_from_parent_indices(parents, H, W, dev)
    base_refine_by_level = _runtime_base_refine_masks_from_leaf_set(
        levels=levels,
        row=row,
        col=col,
        H=H,
        W=W,
        refine_ratio=rr,
        L_max=Lmax,
    )

    out = {
        "levels": levels,
        "row": row,
        "col": col,
        "parents": parents,
        "parent_mask": mask_parent,
        "base_refine_by_level": {int(k): v.to(device=dev, dtype=torch.bool) for k, v in base_refine_by_level.items()},
    }
    _RUNTIME_BASE_MESH_CACHE[key] = out

    lv_cpu = levels.detach().cpu()
    if lv_cpu.numel() > 0:
        uniq, cnt = torch.unique(lv_cpu, return_counts=True)
        lv_summary = ", ".join([f"L{int(u)}={int(c)}" for u, c in zip(uniq.tolist(), cnt.tolist())])
    else:
        lv_summary = "empty"
    print(
        f"[RUNTIME-BASE-MESH] loaded from spec: cells={int(levels.numel())} ({lv_summary})",
        flush=True,
    )
    return out


def _runtime_build_mesh_from_starting_leaf_base(
    *,
    base_mesh: Dict[str, Any],
    masks_by_level: Dict[int, torch.Tensor],
    cfg: Dict[str, Any],
    H: int,
    W: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = torch.device(device)
    rr = _get_refine_ratio(cfg)
    bbox = _get_bbox(cfg)

    base_levels = base_mesh["levels"].to(device=dev, dtype=torch.long)
    base_row = base_mesh["row"].to(device=dev, dtype=torch.long)
    base_col = base_mesh["col"].to(device=dev, dtype=torch.long)

    pred_levels, pred_row, pred_col = _runtime_refine_leaf_set_from_policy_masks(
        base_levels=base_levels,
        base_row=base_row,
        base_col=base_col,
        masks_by_level=masks_by_level,
        H=H,
        W=W,
        refine_ratio=rr,
    )
    pred_centers = _runtime_centers_from_level_ij(
        levels=pred_levels,
        row=pred_row,
        col=pred_col,
        H=H,
        W=W,
        bbox=bbox,
        refine_ratio=rr,
    ).to(device=dev, dtype=torch.float32)
    parent_flat = _runtime_parents_from_level_ij(
        levels=pred_levels,
        row=pred_row,
        col=pred_col,
        H=H,
        W=W,
        refine_ratio=rr,
    ).to(device=dev, dtype=torch.long)
    pred_ei = _runtime_build_edges_from_mesh(
        pred_centers=pred_centers,
        pred_levels=pred_levels,
        parent_flat=parent_flat,
        cfg=cfg,
        H=H,
        W=W,
        bbox=bbox,
        refine_ratio=rr,
        device=dev,
    )
    mask_pred_parent = _mask_from_parent_indices(parent_flat, H, W, dev)
    return pred_centers, pred_levels, parent_flat, pred_ei, mask_pred_parent


def _map_pred_to_next_pred(pred_centers_src,
                           feats_src,
                           levels_src,
                           parents_src,
                           pred_centers_dst,
                           levels_dst,
                           parents_dst,
                           mask_pred_dst,
                           H, W, knn_k=8, chunk=8192,
                           *,
                           bbox: Tuple[float, float, float, float],
                           refine_ratio: int = 2,
                           knn_backend: str = "exact",
                           knn_backend_kwargs: Dict[str, Any] | None = None):
    """
    Map predicted features from mesh k -> k+1.

    - First: copy coarse cells where parent index is unchanged between src and dst.
    - Then: run IDW *only* for the remaining nodes.

    All tensors are moved to the device of feats_src.
    """
    dev = feats_src.device
    idw_dev, idw_chunk = _select_idw_backend(
        src_n=int(pred_centers_src.shape[0]),
        requested_chunk=int(chunk),
        out_device=dev,
    )

    pred_centers_src = pred_centers_src.to(dev)
    pred_centers_dst = pred_centers_dst.to(dev)
    levels_src       = levels_src.to(dev)
    levels_dst       = levels_dst.to(dev)
    parents_src      = parents_src.to(dev)
    parents_dst      = parents_dst.to(dev)
    mask_pred_dst    = mask_pred_dst.to(dev)

    feats_src = feats_src.to(dev)

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

    # ---- 1b) Exact same-cell copy for all levels ---------------------------
    # Match by recovered (level,row,col) key so unchanged fine cells bypass IDW too.
    if pred_centers_src.numel() > 0 and pred_centers_dst.numel() > 0:
        src_row, src_col = _cell_level_ij_from_centers(
            centers=pred_centers_src,
            levels=levels_src,
            H=H,
            W=W,
            bbox=bbox,
            refine_ratio=int(refine_ratio),
        )
        dst_row, dst_col = _cell_level_ij_from_centers(
            centers=pred_centers_dst,
            levels=levels_dst,
            H=H,
            W=W,
            bbox=bbox,
            refine_ratio=int(refine_ratio),
        )

        src_key = _cell_keys_level_ij(
            levels=levels_src,
            row=src_row,
            col=src_col,
            H=H,
            W=W,
            refine_ratio=int(refine_ratio),
        )
        dst_key = _cell_keys_level_ij(
            levels=levels_dst,
            row=dst_row,
            col=dst_col,
            H=H,
            W=W,
            refine_ratio=int(refine_ratio),
        )

        src_order = torch.argsort(src_key)
        src_key_sorted = src_key[src_order]
        src_idx_sorted = src_order

        dst_need_idx = need_idw.nonzero(as_tuple=True)[0]
        if dst_need_idx.numel() > 0:
            dst_need_key = dst_key[dst_need_idx]
            pos = torch.searchsorted(src_key_sorted, dst_need_key)
            valid = (pos >= 0) & (pos < src_key_sorted.numel())
            if valid.any():
                eq = torch.zeros_like(valid, dtype=torch.bool)
                eq[valid] = (src_key_sorted[pos[valid]] == dst_need_key[valid])
                if eq.any():
                    dst_keep = dst_need_idx[eq]
                    src_keep = src_idx_sorted[pos[eq]]
                    out[dst_keep] = feats_src[src_keep]
                    need_idw[dst_keep] = False

    # ---- 2) IDW for the remaining nodes -----------------------------------
    q_idx = need_idw.nonzero(as_tuple=True)[0]
    if q_idx.numel() > 0:
        backend_kwargs = dict(knn_backend_kwargs or {})
        q_pts = pred_centers_dst[q_idx].to(idw_dev, dtype=torch.float32)  # (Q,2)
        src_pts = pred_centers_src.to(idw_dev, dtype=torch.float32)
        src_feats = feats_src.to(idw_dev)
        idx_map, w_map = build_idw_map(
            q_pts,
            src_pts,
            k=knn_k,
            chunk=idw_chunk,
            backend=knn_backend,
            **backend_kwargs,
        )
        vals = apply_idw_map(idx_map, w_map, src_feats).to(dev)  # (Q,F)
        out[q_idx] = vals

    #if cfg.get("debug", {}).get("idw_stats", False):
    #print(f"[IDW stats] coarse_copied={copied}, "
    #        f"idw_points={int(q_idx.numel())}, N_dst={N_dst}")

    return out


def _mask_from_parent_indices(parents: torch.Tensor, H: int, W: int, device: torch.device) -> torch.Tensor:
    m = torch.zeros(H * W, dtype=torch.bool, device=device)
    if parents is not None and torch.is_tensor(parents) and parents.numel() > 0:
        p = parents.view(-1).long().clamp_(0, H * W - 1)
        m[p] = True
    return m.view(H, W)


@torch.no_grad()
def _map_gt_to_pred_mesh_once(
    *,
    src_centers: torch.Tensor,
    src_feats: torch.Tensor,
    pred_centers: torch.Tensor,
    knn_k: int = 8,
    chunk: int = 8192,
    knn_backend: str = "exact",
    knn_backend_kwargs: Dict[str, Any] | None = None,
) -> torch.Tensor:
    out_dev = pred_centers.device
    idw_dev, idw_chunk = _select_idw_backend(
        src_n=int(src_centers.shape[0]),
        requested_chunk=int(chunk),
        out_device=out_dev,
    )

    src_c = src_centers.to(idw_dev, dtype=torch.float32)
    src_f = src_feats.to(idw_dev, dtype=torch.float32)
    dst_c = pred_centers.to(idw_dev, dtype=torch.float32)

    backend_kwargs = dict(knn_backend_kwargs or {})
    idx_map, w_map = build_idw_map(
        dst_c,
        src_c,
        k=int(knn_k),
        chunk=int(idw_chunk),
        backend=knn_backend,
        **backend_kwargs,
    )
    mapped = apply_idw_map(idx_map, w_map, src_f)
    return mapped.to(out_dev)


def _runtime_mesh_backend(cfg: Dict[str, Any]) -> str:
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    raw = str(rt_cfg.get("policy_backend", "gradient")).strip().lower()
    if raw in ("gt_gradients", "gradients", "gradient_policy"):
        raw = "gradient"
    if raw in ("fast_gradient", "fast_gradients", "gradient_coarse", "coarse_gradient"):
        raw = "gradient_fast"
    if raw in ("cnn_policy",):
        raw = "cnn"
    if raw not in ("gradient", "gradient_fast", "cnn"):
        raise ValueError(
            "train.runtime_mesh.policy_backend must be "
            f"'gradient', 'gradient_fast', or 'cnn', got '{raw}'"
        )
    return raw


def _runtime_mesh_domain_mode(cfg: Dict[str, Any]) -> str:
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    raw = str(rt_cfg.get("domain_mode", "wedge_lookup")).strip().lower()
    if raw in ("wedge", "lookup", "wedge_constraints", "wedge_lookup_constraints"):
        raw = "wedge_lookup"
    if raw in ("starting", "starting_base", "starting_mesh_base", "base_mesh"):
        raw = "starting_mesh"
    if raw not in ("wedge_lookup", "starting_mesh"):
        raise ValueError(
            f"train.runtime_mesh.domain_mode must be 'wedge_lookup' or 'starting_mesh', got '{raw}'"
        )
    return raw


def _runtime_mesh_cnn_thresholds(
    cfg: Dict[str, Any],
    *,
    ckpt: Dict[str, Any] | None = None,
    cfg_Lmax: int | None = None,
    ckpt_path: str | None = None,
) -> tuple[float, Dict[int, float], str]:
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    cnn_cfg = rt_cfg.get("cnn", {}) or {}
    use_saved = bool(cnn_cfg.get("use_saved_thresholds", False))

    if not use_saved:
        thr_default = float(cnn_cfg.get("threshold_default", 0.5))
        raw_map = cnn_cfg.get("threshold_by_level", {}) or {}
        if not isinstance(raw_map, dict):
            raise ValueError("train.runtime_mesh.cnn.threshold_by_level must be a JSON object.")
        thr_by_level: Dict[int, float] = {}
        for k, v in raw_map.items():
            try:
                L = int(k)
            except Exception as e:
                raise ValueError(
                    f"Invalid level key in train.runtime_mesh.cnn.threshold_by_level: {k!r}"
                ) from e
            thr_by_level[L] = float(v)
        return thr_default, thr_by_level, "config"

    if not isinstance(ckpt, dict):
        raise RuntimeError(
            "train.runtime_mesh.cnn.use_saved_thresholds=true, but CNN checkpoint payload "
            "is missing or invalid."
        )

    meta = ckpt.get("train_meta", {})
    if not isinstance(meta, dict):
        meta = {}

    raw_saved_map = meta.get("sweep_best_threshold_by_level", None)

    saved_by_level: Dict[int, float] = {}
    if raw_saved_map is None:
        where = f" ({ckpt_path})" if ckpt_path else ""
        raise RuntimeError(
            "train.runtime_mesh.cnn.use_saved_thresholds=true, but checkpoint is missing "
            f"'train_meta.sweep_best_threshold_by_level'{where}."
        )
    if not isinstance(raw_saved_map, dict):
        raise RuntimeError(
            "train.runtime_mesh.cnn.use_saved_thresholds=true, but checkpoint field "
            "'train_meta.sweep_best_threshold_by_level' is not a dict."
        )
    for k, v in raw_saved_map.items():
        try:
            L = int(k)
        except Exception as e:
            raise RuntimeError(
                "train.runtime_mesh.cnn.use_saved_thresholds=true, but checkpoint has "
                f"invalid sweep threshold level key: {k!r}"
            ) from e
        saved_by_level[L] = float(v)

    if len(saved_by_level) == 0:
        where = f" ({ckpt_path})" if ckpt_path else ""
        raise RuntimeError(
            "train.runtime_mesh.cnn.use_saved_thresholds=true, but "
            f"'train_meta.sweep_best_threshold_by_level' is empty{where}."
        )

    if cfg_Lmax is None:
        cfg_Lmax = int(cfg.get("policy", {}).get("max_level", cfg.get("data", {}).get("L_max", 3)))
    missing = [int(L) for L in range(1, int(cfg_Lmax) + 1) if int(L) not in saved_by_level]
    if missing:
        where = f" ({ckpt_path})" if ckpt_path else ""
        raise RuntimeError(
            "train.runtime_mesh.cnn.use_saved_thresholds=true, but saved sweep thresholds do not "
            f"cover all levels 1..{int(cfg_Lmax)}. Missing levels: {missing}{where}."
        )

    # All levels are explicitly provided by sweep_best_threshold_by_level.
    # Keep a scalar for fallback path; it should not be used in normal operation.
    thr_default = float(saved_by_level.get(1, 0.5))
    return thr_default, saved_by_level, "checkpoint_sweep"


def _load_runtime_mesh_policy_from_cfg(
    cfg: Dict[str, Any],
    *,
    device: torch.device,
    H: int,
    W: int,
):
    backend = _runtime_mesh_backend(cfg)
    if backend != "cnn":
        return None

    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    cnn_cfg = rt_cfg.get("cnn", {}) or {}

    ckpt_raw = cnn_cfg.get("checkpoint_path", None)
    if not ckpt_raw:
        raise RuntimeError(
            "Runtime mesh backend is 'cnn' but train.runtime_mesh.cnn.checkpoint_path is empty."
        )
    ckpt_path = os.path.abspath(os.path.expanduser(str(ckpt_raw)))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Runtime mesh CNN checkpoint not found: {ckpt_path}")

    dev_spec = str(cnn_cfg.get("device", "same")).strip().lower()
    if dev_spec in ("", "same", "auto", "none"):
        policy_device = torch.device(device)
    else:
        policy_device = torch.device(dev_spec)

    try:
        from mesh_policy_cnn import MeshPolicyCNN
    except Exception as e:
        raise RuntimeError(
            "Runtime mesh backend is 'cnn' but mesh_policy_cnn.py could not be imported."
        ) from e

    ckpt = torch.load(ckpt_path, map_location=policy_device)
    model_args = ckpt.get("model_args", None) if isinstance(ckpt, dict) else None
    if not isinstance(model_args, dict):
        raise RuntimeError(
            f"Checkpoint {ckpt_path} is missing 'model_args'; expected artifact from train_mesh_policy.py."
        )

    model = MeshPolicyCNN(
        in_channels=int(model_args["in_channels"]),
        base_channels=int(model_args.get("base_channels", 48)),
        head_channels=int(model_args.get("head_channels", 8)),
        max_level=int(model_args["max_level"]),
        refine_ratio=int(model_args["refine_ratio"]),
        model_type=str(model_args.get("model_type", "upsample_heads")),
    ).to(policy_device)

    state = ckpt.get("model_state", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {ckpt_path} does not contain a valid model state dict.")
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cfg_rr = _get_refine_ratio(cfg)
    cfg_Lmax = int(cfg.get("policy", {}).get("max_level", cfg.get("data", {}).get("L_max", 3)))
    ckpt_rr = int(model_args.get("refine_ratio", cfg_rr))
    ckpt_Lmax = int(model_args.get("max_level", cfg_Lmax))
    if ckpt_rr != cfg_rr:
        raise RuntimeError(
            f"Runtime mesh CNN refine_ratio mismatch: checkpoint={ckpt_rr}, cfg={cfg_rr}"
        )
    if ckpt_Lmax != cfg_Lmax:
        raise RuntimeError(
            f"Runtime mesh CNN max_level mismatch: checkpoint={ckpt_Lmax}, cfg={cfg_Lmax}"
        )

    include_parent_mask = bool(cnn_cfg.get("include_parent_mask", True))
    include_coords = bool(cnn_cfg.get("include_coords", True))
    include_dt = bool(cnn_cfg.get("include_dt", False))
    dt_channel_mode = str(cnn_cfg.get("dt_channel_mode", "dt_hat")).strip().lower()
    if dt_channel_mode not in ("raw", "dt_hat"):
        raise ValueError(
            "train.runtime_mesh.cnn.dt_channel_mode must be 'raw' or 'dt_hat'. "
            f"Got {dt_channel_mode!r}"
        )

    coord_grid = None
    if include_coords:
        yy = torch.linspace(0.0, 1.0, H, device=policy_device, dtype=torch.float32)
        xx = torch.linspace(0.0, 1.0, W, device=policy_device, dtype=torch.float32)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        coord_grid = torch.stack([gx, gy], dim=0).contiguous()
    coarse_rc_grid = _runtime_build_coarse_rc_grid(H, W, device=policy_device)

    thr_default, thr_by_level, thr_source = _runtime_mesh_cnn_thresholds(
        cfg,
        ckpt=ckpt if isinstance(ckpt, dict) else None,
        cfg_Lmax=cfg_Lmax,
        ckpt_path=ckpt_path,
    )

    ctx = {
        "backend": "cnn",
        "model": model,
        "device": policy_device,
        "in_channels": int(model_args["in_channels"]),
        "model_type": str(model_args.get("model_type", "upsample_heads")),
        "max_level": ckpt_Lmax,
        "refine_ratio": ckpt_rr,
        "include_parent_mask": include_parent_mask,
        "include_coords": include_coords,
        "include_dt": include_dt,
        "dt_channel_mode": dt_channel_mode,
        "coord_grid": coord_grid,
        "coarse_rc_grid": coarse_rc_grid,
        "threshold_default": float(thr_default),
        "threshold_by_level": thr_by_level,
        "checkpoint_path": ckpt_path,
    }
    print(
        "[RUNTIME-MESH] loaded CNN policy:",
        f"ckpt={ckpt_path}",
        f"device={policy_device}",
        f"in_channels={ctx['in_channels']}",
        f"model_type={ctx['model_type']}",
        f"max_level={ctx['max_level']}",
        f"refine_ratio={ctx['refine_ratio']}",
        f"include_dt={bool(ctx['include_dt'])}",
        f"dt_channel_mode={ctx['dt_channel_mode']}",
        f"threshold_source={thr_source}",
        f"threshold_default={ctx['threshold_default']:.6g}",
    )
    if len(thr_by_level) > 0:
        print(
            "[RUNTIME-MESH] CNN threshold_by_level:",
            {int(k): float(v) for k, v in sorted(thr_by_level.items(), key=lambda kv: int(kv[0]))},
            flush=True,
        )
    return ctx


@torch.no_grad()
def _runtime_predict_masks_from_cnn(
    *,
    feat_policy: torch.Tensor,
    parents_t: torch.Tensor,
    mask_t_parent: torch.Tensor,
    centers_t: torch.Tensor,
    level_t: torch.Tensor,
    dt_phys: torch.Tensor | float | None,
    dt_ref: torch.Tensor | float | None,
    cfg: Dict[str, Any],
    H: int,
    W: int,
    device: torch.device,
    runtime_mesh_policy: Dict[str, Any],
    base_refine_by_level: Dict[int, torch.Tensor] | None = None,
    parent_mapping_mode: str = "dataset",
    fill_empty_interior: bool = False,
    physical_parent_mask: torch.Tensor | None = None,
    coarse_rc_grid: torch.Tensor | None = None,
) -> Dict[int, torch.Tensor]:
    if runtime_mesh_policy is None:
        raise RuntimeError("runtime_mesh_policy is None for CNN runtime mesh backend.")
    if runtime_mesh_policy.get("backend", None) != "cnn":
        raise RuntimeError("runtime_mesh_policy backend is not 'cnn'.")

    pdev = torch.device(runtime_mesh_policy["device"])
    model = runtime_mesh_policy["model"]
    rr = int(runtime_mesh_policy["refine_ratio"])
    Lmax = int(runtime_mesh_policy["max_level"])
    thr_default = float(runtime_mesh_policy["threshold_default"])
    thr_by_level: Dict[int, float] = runtime_mesh_policy["threshold_by_level"]

    parent_mode = str(parent_mapping_mode).strip().lower()
    if parent_mode in ("dataset", "legacy", "from_dataset"):
        parents_for_cnn = parents_t.to(feat_policy.device, dtype=torch.long).view(-1)
        mask_for_cnn = mask_t_parent.to(feat_policy.device, dtype=torch.bool)
    elif parent_mode in ("from_centers", "centers", "runtime_like", "coarse_from_centers"):
        row_t, col_t = _cell_level_ij_from_centers(
            centers=centers_t.to(feat_policy.device, dtype=torch.float32),
            levels=level_t.to(feat_policy.device, dtype=torch.long).view(-1),
            H=H,
            W=W,
            bbox=_get_bbox(cfg),
            refine_ratio=rr,
        )
        parents_for_cnn = _runtime_parents_from_level_ij(
            levels=level_t.to(feat_policy.device, dtype=torch.long).view(-1),
            row=row_t,
            col=col_t,
            H=H,
            W=W,
            refine_ratio=rr,
        ).view(-1)
        mask_for_cnn = _mask_from_parent_indices(parents_for_cnn, H, W, feat_policy.device)
    else:
        raise ValueError(
            f"Unknown runtime CNN parent_mapping_mode: {parent_mapping_mode!r}. "
            "Expected one of: dataset, from_centers"
        )

    coarse = coarse_aggregate_from_dynamic(feat_policy, parents_for_cnn, H, W)
    coarse = coarse.view(H, W, -1).permute(2, 0, 1).contiguous()  # (F,H,W)
    if bool(fill_empty_interior):
        phys = (
            torch.ones((H, W), dtype=torch.bool, device=coarse.device)
            if physical_parent_mask is None
            else torch.as_tensor(physical_parent_mask, dtype=torch.bool, device=coarse.device).view(H, W)
        )
        coarse = _runtime_fill_empty_interior_coarse_nearest(
            coarse_chw=coarse,
            parents=parents_for_cnn,
            H=H,
            W=W,
            physical_parent_mask=phys,
            rc_grid=coarse_rc_grid,
            chunk=1024,
        )

    channels = [coarse]
    if bool(runtime_mesh_policy.get("include_parent_mask", True)):
        channels.append(mask_for_cnn.to(coarse.device, dtype=torch.float32).unsqueeze(0))
    if bool(runtime_mesh_policy.get("include_coords", True)):
        coord_grid = runtime_mesh_policy.get("coord_grid", None)
        if coord_grid is None:
            raise RuntimeError("CNN runtime mesh policy expects coord channels, but coord_grid is missing.")
        channels.append(coord_grid.to(coarse.device, dtype=torch.float32))
    if bool(runtime_mesh_policy.get("include_dt", False)):
        if dt_phys is None:
            raise RuntimeError(
                "CNN runtime mesh include_dt=true but dt_phys was not provided."
            )
        dt_phys_t = _to_scalar_dt(dt_phys, device=coarse.device, dtype=torch.float32)
        mode = str(runtime_mesh_policy.get("dt_channel_mode", "dt_hat")).strip().lower()
        if mode == "dt_hat":
            if dt_ref is None:
                raise RuntimeError(
                    "CNN runtime mesh include_dt=true and dt_channel_mode='dt_hat' but dt_ref is missing."
                )
            dt_ref_t = _to_scalar_dt(dt_ref, device=coarse.device, dtype=torch.float32).clamp_min(1e-12)
            dt_val = dt_phys_t / dt_ref_t
        elif mode == "raw":
            dt_val = dt_phys_t
        else:
            raise RuntimeError(
                f"Unsupported runtime CNN dt_channel_mode={mode!r}. Expected 'raw' or 'dt_hat'."
            )
        dt_plane = torch.full(
            (1, int(H), int(W)),
            float(dt_val.item()),
            dtype=torch.float32,
            device=coarse.device,
        )
        channels.append(dt_plane)

    x_in = torch.cat(channels, dim=0).unsqueeze(0)  # (1,C,H,W)
    expected_in = int(runtime_mesh_policy.get("in_channels", -1))
    if expected_in > 0 and int(x_in.shape[1]) != expected_in:
        raise RuntimeError(
            f"CNN runtime mesh input channel mismatch: built={int(x_in.shape[1])}, expected={expected_in}. "
            "Check runtime_mesh.cnn.include_parent_mask/include_coords and feature channel setup."
        )

    logits_raw = model(x_in.to(pdev, dtype=torch.float32, non_blocking=True))
    if not isinstance(logits_raw, dict):
        raise RuntimeError(f"CNN runtime mesh model forward must return dict[int,Tensor], got {type(logits_raw)}")

    logits_by_level: Dict[int, torch.Tensor] = {}
    for k, v in logits_raw.items():
        logits_by_level[int(k)] = v

    masks_by_level: Dict[int, torch.Tensor] = {}
    for L in range(1, Lmax + 1):
        if L not in logits_by_level:
            raise RuntimeError(f"CNN runtime mesh missing logits for level L={L}")
        logits_L = logits_by_level[L]
        if logits_L.ndim != 4 or logits_L.shape[0] != 1 or logits_L.shape[1] != 1:
            raise RuntimeError(
                f"CNN runtime mesh logits for L={L} must be shape (1,1,H,W), got {tuple(logits_L.shape)}"
            )
        m = (torch.sigmoid(logits_L[:, 0]) >= float(thr_by_level.get(L, thr_default)))  # (1,h,w)
        h, w = int(m.shape[-2]), int(m.shape[-1])
        exp_h = H * (rr ** (L - 1))
        exp_w = W * (rr ** (L - 1))
        if (h, w) != (exp_h, exp_w):
            raise RuntimeError(
                f"CNN runtime mesh output shape mismatch at L={L}: got {(h, w)}, expected {(exp_h, exp_w)}"
            )

        if L > 1:
            prev = masks_by_level[L - 1]
            if (base_refine_by_level is not None) and ((L - 1) in base_refine_by_level):
                bprev = torch.as_tensor(
                    base_refine_by_level[L - 1],
                    dtype=torch.bool,
                    device=prev.device,
                )
                if tuple(bprev.shape) != tuple(prev.shape[-2:]):
                    bprev = F.interpolate(
                        bprev.float().unsqueeze(0).unsqueeze(0),
                        size=tuple(prev.shape[-2:]),
                        mode="nearest",
                    )[0, 0].to(torch.bool)
                prev = prev | bprev.unsqueeze(0)
            allow = F.interpolate(
                prev.float().unsqueeze(1),
                scale_factor=float(rr),
                mode="nearest",
            )[:, 0].bool()
            m = m & allow[:, :h, :w]
        masks_by_level[L] = m

    return {L: masks_by_level[L][0].to(device=device, dtype=torch.bool) for L in masks_by_level.keys()}


@torch.no_grad()
def _runtime_predict_masks_from_fast_gradients(
    *,
    centers_t: torch.Tensor,
    level_t: torch.Tensor,
    feat_policy: torch.Tensor,
    parents_t: torch.Tensor,
    mask_t_parent: torch.Tensor,
    cfg: Dict[str, Any],
    H: int,
    W: int,
    dx: float,
    dy: float,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Fast gradient backend for runtime mesh prediction.
    - Aggregate dynamic node features to coarse HxW parent grid.
    - Compute per-channel coarse-grid gradient magnitudes.
    - Combine channels and apply hierarchical refine masks with parent gating.

    This keeps the existing "gradient" backend unchanged and provides a fast
    alternative for timing/quality comparison.
    """
    dev = torch.device(device)
    pol = cfg.get("policy", {}) or {}
    rr = _get_refine_ratio(cfg)
    Lmax = int(pol.get("max_level", cfg.get("data", {}).get("L_max", 3)))

    p_pow = float(pol.get("p", 1.0))
    combine = str(pol.get("combine", "l2")).strip().lower()
    weights = pol.get("refine_weights", None)

    tau_by_level = pol.get("tau_by_level", None)
    tau_low_def = float(pol.get("tau_low", 0.02))
    tau_high_def = float(pol.get("tau_high", 0.03))

    mode = str(pol.get("hysteresis_mode", "absolute")).strip().lower()
    if mode not in ("absolute", "percentile"):
        raise ValueError(
            f"policy.hysteresis_mode must be 'absolute' or 'percentile', got {mode!r}"
        )

    pct_low_default = float(pol.get("percentile_low", 75.0))
    pct_high_default = float(pol.get("percentile_high", 90.0))
    pct_by_level = pol.get("percentiles_by_level", {})
    if not isinstance(pct_by_level, dict):
        pct_by_level = {}
    pct_mode = str(pct_by_level.get("selection", "auto")).strip().lower()
    if pct_mode not in ("auto", "global", "per_level", "per-level", "level", "levels"):
        raise ValueError(
            "policy.percentiles_by_level.selection must be one of: auto, global, per_level"
        )
    use_global_pct = (pct_mode == "global")
    use_level_pct = (pct_mode in ("per_level", "per-level", "level", "levels"))

    def _pct_level_entry(L: int):
        if L in pct_by_level:
            return pct_by_level[L]
        s = str(L)
        if s in pct_by_level:
            return pct_by_level[s]
        return None

    def _taus(L: int) -> tuple[float, float]:
        if isinstance(tau_by_level, dict):
            ent = tau_by_level.get(L, tau_by_level.get(str(L), None))
            if isinstance(ent, dict):
                return float(ent.get("low", tau_low_def)), float(ent.get("high", tau_high_def))
            if ent is not None:
                return tau_low_def, float(ent)
        return tau_low_def, tau_high_def

    coarse = coarse_aggregate_from_dynamic(feat_policy, parents_t, H, W).to(dev, dtype=torch.float32)
    Fdim = int(coarse.shape[1])
    idxs = [int(c) for c in _resolve_channel_indices(cfg) if 0 <= int(c) < Fdim]
    if len(idxs) == 0:
        idxs = list(range(Fdim))
    if len(idxs) == 0:
        idxs = [0]

    score_list: List[torch.Tensor] = []
    for c in idxs:
        score_list.append(_coarse_grad_mag(coarse[:, c], H, W, dx, dy, p=p_pow))
    S = torch.stack(score_list, dim=0)  # (Csel, H*W)

    if weights is not None:
        w = torch.as_tensor(weights, device=S.device, dtype=S.dtype)
        if w.numel() != S.size(0):
            if w.numel() < S.size(0):
                w = torch.cat([w, w.new_ones(S.size(0) - w.numel())])
            else:
                w = w[:S.size(0)]
    else:
        w = torch.ones(S.size(0), device=S.device, dtype=S.dtype)

    if combine == "max":
        s_ref = S.max(dim=0).values
    elif combine in ("sum", "weighted_sum"):
        s_ref = (w.view(-1, 1) * S).sum(dim=0)
    else:  # l2 / weighted_l2
        s_ref = torch.sqrt(((w.view(-1, 1) * S) ** 2).sum(dim=0))

    g_base = torch.nan_to_num(
        s_ref.view(H, W),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).to(dev)

    xmin, xmax, ymin, ymax = _get_bbox(cfg)
    masks_by_level: Dict[int, torch.Tensor] = {}

    # Reconstruct previous refine masks per level from current runtime mesh state.
    # prev_refine_by_level[L] lives on the parent grid for level L (H*rr^(L-1), W*rr^(L-1)).
    prev_refine_by_level: Dict[int, torch.Tensor] = {}
    try:
        lv_prev = level_t.to(dev, dtype=torch.long).view(-1)
        centers_prev = centers_t.to(dev, dtype=torch.float32)
        if lv_prev.numel() == int(centers_prev.shape[0]) and lv_prev.numel() > 0:
            row_prev, col_prev = _cell_level_ij_from_centers(
                centers=centers_prev,
                levels=lv_prev,
                H=H,
                W=W,
                bbox=(xmin, xmax, ymin, ymax),
                refine_ratio=rr,
            )
            for L in range(1, Lmax + 1):
                h_p = int(H * (rr ** (L - 1)))
                w_p = int(W * (rr ** (L - 1)))
                prev_mask_L = torch.zeros((h_p, w_p), dtype=torch.bool, device=dev)

                sel = (lv_prev >= int(L))
                if bool(sel.any()):
                    exp = (lv_prev[sel] - int(L - 1)).to(torch.long)
                    max_e = int(torch.clamp(exp.max(), min=0).item())
                    pow_tbl = torch.tensor(
                        [rr ** i for i in range(max_e + 1)],
                        device=dev,
                        dtype=torch.long,
                    )
                    scale = pow_tbl.index_select(0, exp.clamp(0, max_e))
                    prow = torch.div(row_prev[sel], scale, rounding_mode="floor")
                    pcol = torch.div(col_prev[sel], scale, rounding_mode="floor")
                    prow = prow.clamp_(0, h_p - 1)
                    pcol = pcol.clamp_(0, w_p - 1)
                    parent_flat = (prow * int(w_p) + pcol).view(-1).long()
                    if parent_flat.numel() > 0:
                        prev_mask_L.view(-1)[parent_flat] = True

                prev_refine_by_level[L] = prev_mask_L
    except Exception:
        # Fallback below keeps old behavior when previous masks cannot be reconstructed.
        prev_refine_by_level = {}

    for L in range(1, Lmax + 1):
        h_p = H * (rr ** (L - 1))
        w_p = W * (rr ** (L - 1))
        if (h_p, w_p) == (H, W):
            g_parent = g_base
        else:
            g_parent = F.interpolate(
                g_base[None, None],
                size=(h_p, w_p),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        g_parent = torch.nan_to_num(g_parent, nan=0.0, posinf=0.0, neginf=0.0)

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
            p_low = max(0.0, min(100.0, p_low))
            p_high = max(0.0, min(100.0, p_high))
            if p_low > p_high:
                p_low, p_high = p_high, p_low
            finite = torch.isfinite(g_parent)
            if finite.any():
                thr_low = torch.quantile(g_parent[finite], p_low / 100.0)
                thr_high = torch.quantile(g_parent[finite], p_high / 100.0)
            else:
                thr_low = torch.as_tensor(float("inf"), device=dev, dtype=g_parent.dtype)
                thr_high = torch.as_tensor(float("inf"), device=dev, dtype=g_parent.dtype)
        else:
            tau_low, tau_high = _taus(L)
            thr_low = torch.as_tensor(tau_low, device=dev, dtype=g_parent.dtype)
            thr_high = torch.as_tensor(tau_high, device=dev, dtype=g_parent.dtype)

        prev_mask = prev_refine_by_level.get(L, None)
        if prev_mask is None and L == 1:
            prev2d = mask_t_parent.to(dev, dtype=torch.float32).view(H, W)
            prev_mask = prev2d.bool()
        if prev_mask is not None:
            if prev_mask.shape != (h_p, w_p):
                prev_mask = F.interpolate(
                    prev_mask.float()[None, None],
                    size=(h_p, w_p),
                    mode="nearest",
                )[0, 0].bool()
            keep = prev_mask & (g_parent > thr_low)
            newr = (~prev_mask) & (g_parent > thr_high)
            M = keep | newr
        else:
            M = g_parent > thr_high

        if L > 1:
            allow = F.interpolate(
                masks_by_level[L - 1].float()[None, None],
                scale_factor=float(rr),
                mode="nearest",
            )[0, 0].bool()
            M = M & allow[:h_p, :w_p]

        # Optional dilation halo around refine mask.
        # Preferred config is cell-based halo:
        #   policy.dilate_cells_L{L} = n_cells
        # Optional convenience for finest level:
        #   policy.dilate_cells_lmax = n_cells (applies only when L==Lmax)
        # Backward compatible:
        #   policy.dilate_phys_L{L} = physical radius
        dxL_here = (xmax - xmin) / float(W * (rr ** (L - 1)))

        n_cells = None
        raw_cells = pol.get(f"dilate_cells_L{L}", None)
        if raw_cells is None and (L == Lmax):
            raw_cells = pol.get("dilate_cells_lmax", None)
        if raw_cells is not None:
            try:
                n_cells = float(raw_cells)
            except Exception:
                n_cells = None

        if n_cells is not None:
            r_phys = float(max(0.0, n_cells) * dxL_here)
        else:
            r_phys = float(pol.get(f"dilate_phys_L{L}", 0.0))

        if r_phys > 0:
            r_cells = max(1, int(round(r_phys / max(dxL_here, 1e-12))))
            k = 2 * r_cells + 1
            M = F.max_pool2d(
                M.float()[None, None],
                kernel_size=k,
                stride=1,
                padding=r_cells,
            )[0, 0].bool()

        masks_by_level[L] = M

    return {L: masks_by_level[L].to(device=device, dtype=torch.bool) for L in masks_by_level.keys()}


@torch.no_grad()
def _runtime_build_pred_mesh_from_state(
    *,
    centers_t: torch.Tensor,
    feat_t: torch.Tensor,
    level_t: torch.Tensor,
    parents_t: torch.Tensor,
    cfg: Dict[str, Any],
    H: int,
    W: int,
    dx: float,
    dy: float,
    device: torch.device,
    wedge_path,
    need_edge_attr: bool,
    runtime_mesh_policy: Dict[str, Any] | None = None,
    runtime_wedge_constraints: Dict[str, Any] | None = None,
    runtime_base_mesh: Dict[str, Any] | None = None,
    cnn_parent_mapping_mode: str = "dataset",
    cnn_fill_empty_interior: bool = False,
    dt_phys: torch.Tensor | float | None = None,
    dt_ref: torch.Tensor | float | None = None,
    timing_out: Dict[str, float] | None = None,
):
    """
    Build next-step predicted mesh from state gradients in runtime mode.
    Wedge clipping is mandatory for shock-ramp geometry.
    """
    t_build_t0 = time.perf_counter()
    timing: Dict[str, float] = {
        "mesh_predict_s": 0.0,
        "mesh_materialize_s": 0.0,
        "wedge_clip_s": 0.0,
        "wedge_lookup_classify_s": 0.0,
        "wedge_lookup_refine_s": 0.0,
        "wedge_edge_build_s": 0.0,
        "wedge_legacy_geom_s": 0.0,
        "edge_attr_s": 0.0,
        "total_s": 0.0,
    }

    dev = torch.device(device)
    rr = _get_refine_ratio(cfg)
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    domain_mode = _runtime_mesh_domain_mode(cfg)
    reset_each = bool(rt_cfg.get("reset_to_coarse_each_step", True))
    detach_policy_input = bool(rt_cfg.get("detach_policy_input", True))

    centers_t = centers_t.to(dev, dtype=torch.float32)
    feat_policy = (feat_t.detach() if detach_policy_input else feat_t).to(dev, dtype=torch.float32)
    level_t = level_t.to(dev, dtype=torch.long).view(-1)
    parents_t = parents_t.to(dev, dtype=torch.long).view(-1)

    if domain_mode == "starting_mesh":
        if runtime_base_mesh is None:
            raise RuntimeError(
                "Runtime mesh domain_mode='starting_mesh' requires runtime_base_mesh."
            )
    else:
        if wedge_path is None and runtime_wedge_constraints is None:
            raise RuntimeError("Runtime mesh mode requires wedge clipping; wedge_path is None.")

    if reset_each:
        if domain_mode == "starting_mesh":
            mask_t_parent = runtime_base_mesh["parent_mask"].to(device=dev, dtype=torch.bool)
        else:
            mask_t_parent = torch.zeros((H, W), dtype=torch.bool, device=dev)
    else:
        mask_t_parent = _mask_from_parent_indices(parents_t, H, W, dev)

    batch_like = {
        "centers_t": centers_t,
        "center_feat_t": feat_policy,
        "dyn_feat_t": feat_policy,
        "dyn_parents": parents_t,
        "mask_t": mask_t_parent.view(-1),
        "level_t": level_t,
    }
    policy_backend = _runtime_mesh_backend(cfg)
    t_policy_t0 = time.perf_counter()
    if policy_backend == "cnn":
        base_refine_by_level = None
        if (domain_mode == "starting_mesh") and (runtime_base_mesh is not None):
            base_refine_by_level = runtime_base_mesh.get("base_refine_by_level", None)
        cnn_physical_parent_mask = None
        if bool(cnn_fill_empty_interior):
            if (domain_mode == "starting_mesh") and (runtime_base_mesh is not None):
                cnn_physical_parent_mask = runtime_base_mesh.get("parent_mask", None)
            elif runtime_wedge_constraints is not None:
                parent_intersect = runtime_wedge_constraints.get("parent_intersect", {})
                if isinstance(parent_intersect, dict):
                    cnn_physical_parent_mask = parent_intersect.get(1, None)
        masks_pred_by_level = _runtime_predict_masks_from_cnn(
            feat_policy=feat_policy,
            parents_t=parents_t,
            mask_t_parent=mask_t_parent,
            centers_t=centers_t,
            level_t=level_t,
            dt_phys=dt_phys,
            dt_ref=dt_ref,
            cfg=cfg,
            H=H,
            W=W,
            device=dev,
            runtime_mesh_policy=runtime_mesh_policy,
            base_refine_by_level=base_refine_by_level,
            parent_mapping_mode=cnn_parent_mapping_mode,
            fill_empty_interior=bool(cnn_fill_empty_interior),
            physical_parent_mask=cnn_physical_parent_mask,
            coarse_rc_grid=runtime_mesh_policy.get("coarse_rc_grid", None) if isinstance(runtime_mesh_policy, dict) else None,
        )
    elif policy_backend == "gradient_fast":
        masks_pred_by_level = _runtime_predict_masks_from_fast_gradients(
            centers_t=centers_t,
            level_t=level_t,
            feat_policy=feat_policy,
            parents_t=parents_t,
            mask_t_parent=mask_t_parent,
            cfg=cfg,
            H=H,
            W=W,
            dx=dx,
            dy=dy,
            device=dev,
        )
    else:
        masks_pred_by_level = predict_masks_hierarchical_from_gt_gradients(
            batch_like,
            cfg,
            H,
            W,
            dx,
            dy,
            device=dev,
        )
    timing["mesh_predict_s"] = float(time.perf_counter() - t_policy_t0)

    if domain_mode == "starting_mesh":
        t_mesh_mat_t0 = time.perf_counter()
        pred_centers, pred_levels, parent_flat, pred_ei, mask_pred_parent = _runtime_build_mesh_from_starting_leaf_base(
            base_mesh=runtime_base_mesh,
            masks_by_level=masks_pred_by_level,
            cfg=cfg,
            H=H,
            W=W,
            device=dev,
        )
        timing["mesh_materialize_s"] = float(time.perf_counter() - t_mesh_mat_t0)
    else:
        xmin, xmax, ymin, ymax = _get_bbox(cfg)
        if runtime_wedge_constraints is not None:
            t_wref0 = time.perf_counter()
            masks_pred_by_level = _runtime_apply_wedge_constraints_to_masks(
                masks_by_level=masks_pred_by_level,
                constraints=runtime_wedge_constraints,
            )
            timing["wedge_lookup_refine_s"] = float(time.perf_counter() - t_wref0)

        t_mesh_mat_t0 = time.perf_counter()
        pred_centers, pred_levels, pred_parents, pred_ei = dynamic_cells_from_parent_masks(
            masks_pred_by_level,
            H,
            W,
            xmin,
            xmax,
            ymin,
            ymax,
            refine_ratio=rr,
        )
        timing["mesh_materialize_s"] = float(time.perf_counter() - t_mesh_mat_t0)
        parent_flat = pred_parents.view(-1).long()

        # Defer edge construction until after wedge-domain filtering.
        pred_ei = torch.empty((2, 0), dtype=torch.long, device=dev)
        t_wedge_t0 = time.perf_counter()
        if runtime_wedge_constraints is not None:
            wedge_fast_timing: Dict[str, float] = {}
            pred_centers, pred_levels, parent_flat, pred_ei, mask_pred_parent = _runtime_filter_mesh_with_wedge_constraints(
                pred_centers=pred_centers,
                pred_levels=pred_levels,
                parent_flat=parent_flat,
                cfg=cfg,
                H=H,
                W=W,
                device=dev,
                constraints=runtime_wedge_constraints,
                timing_out=wedge_fast_timing,
            )
            timing["wedge_lookup_classify_s"] = float(wedge_fast_timing.get("lookup_classify_s", 0.0))
            timing["wedge_edge_build_s"] = float(wedge_fast_timing.get("edge_build_s", 0.0))
            timing["wedge_legacy_geom_s"] = 0.0
        else:
            wedge_timing: Dict[str, float] = {}
            pred_centers, pred_levels, parent_flat, pred_ei, mask_pred_parent = _clip_pred_mesh_to_wedge(
                pred_centers=pred_centers,
                pred_levels=pred_levels,
                pred_parents=parent_flat,
                pred_ei=pred_ei,
                H=H,
                W=W,
                wedge_path=wedge_path,
                cfg=cfg,
                device=dev,
                timing_out=wedge_timing,
            )
            timing["wedge_lookup_classify_s"] = float(wedge_timing.get("lookup_classify_s", 0.0))
            timing["wedge_lookup_refine_s"] = float(wedge_timing.get("lookup_refine_s", 0.0))
            timing["wedge_edge_build_s"] = float(wedge_timing.get("edge_build_s", 0.0))
            timing["wedge_legacy_geom_s"] = float(wedge_timing.get("legacy_geom_s", 0.0))
        timing["wedge_clip_s"] = float(time.perf_counter() - t_wedge_t0)

    pred_centers = pred_centers.to(dev, dtype=torch.float32)
    pred_levels = pred_levels.to(dev, dtype=torch.long)
    parent_flat = parent_flat.to(dev, dtype=torch.long)
    pred_ei = pred_ei.to(dev, dtype=torch.long)
    mask_pred_parent = mask_pred_parent.to(dev, dtype=torch.bool)

    pred_ea = None
    if need_edge_attr:
        t_ea_t0 = time.perf_counter()
        edge_attr = dec_edge_attr_for_dyadic_quads(
            pred_centers.to("cpu", dtype=torch.float32),
            pred_levels.to("cpu", dtype=torch.int64),
            pred_ei.to("cpu", dtype=torch.int64),
            dx0=float(dx),
            dy0=float(dy),
            refine_ratio=rr,
        )
        pred_ea = edge_attr.to(dev, dtype=torch.float32)
        timing["edge_attr_s"] = float(time.perf_counter() - t_ea_t0)

    timing["total_s"] = float(time.perf_counter() - t_build_t0)
    if isinstance(timing_out, dict):
        timing_out.clear()
        timing_out.update(timing)

    return pred_centers, pred_levels, parent_flat, pred_ei, mask_pred_parent, pred_ea


def _require_list(batch, name):
    if name not in batch:
        raise RuntimeError(
            f"Batch is missing required key '{name}'. "
            f"This training loop is strict: precompute and include '{name}' in the collate."
        )
    return batch[name]

def _as_stat_tensor(v, *, device, dtype):
                    if v is None:
                        return None
                    if torch.is_tensor(v):
                        return v.to(device=device, dtype=dtype)
                    return torch.as_tensor(v, device=device, dtype=dtype)

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

def _sanitize_float_tensor(
    t: torch.Tensor | None,
    *,
    clip_abs: float = 0.0,
    fill_value: float = 0.0,
    nonneg: bool = False,
) -> torch.Tensor | None:
    """
    Make floating tensors safe for downstream ops:
      - replace NaN/Inf
      - optional magnitude clipping
    """
    if t is None or (not torch.is_tensor(t)):
        return t
    if not t.dtype.is_floating_point:
        return t

    c = float(clip_abs)
    if c > 0.0:
        t = torch.nan_to_num(t, nan=fill_value, posinf=c, neginf=-c)
        if nonneg:
            t = t.clamp(min=0.0, max=c)
        else:
            t = t.clamp(min=-c, max=c)
    else:
        t = torch.nan_to_num(t, nan=fill_value, posinf=fill_value, neginf=fill_value)
        if nonneg:
            t = t.clamp_min(0.0)
    return t

def _sanitize_ops_term(t: torch.Tensor | None, cfg: dict) -> torch.Tensor | None:
    """
    Guard DEC/MLS operator terms against catastrophic values.
    """
    loss = cfg.get("loss", {}) or {}
    clip_abs = float(loss.get("ops_term_clip", 0.0))
    return _sanitize_float_tensor(t, clip_abs=clip_abs, fill_value=0.0, nonneg=False)

def _sanitize_parc_extra_tensor(
    t: torch.Tensor | None,
    cfg: dict,
    *,
    post_adapter: bool = False,
) -> torch.Tensor | None:
    """
    Keep PARC feature block bounded even when adapter is disabled.
    """
    loss = cfg.get("loss", {}) or {}
    if post_adapter:
        clip_abs = float(loss.get("parc_feat_clip_post", loss.get("parc_feat_clip_pre", 0.0)))
    else:
        clip_abs = float(loss.get("parc_feat_clip_pre", 0.0))
    return _sanitize_float_tensor(t, clip_abs=clip_abs, fill_value=0.0, nonneg=False)

def _apply_rate_guardrails(y_rate: torch.Tensor, dt_hat: torch.Tensor | float, cfg: dict) -> torch.Tensor:
    """
    Optional clipping in model-rate space to limit rollout explosions.
    """
    train_cfg = cfg.get("train", {}) or {}
    rate_clip = float(train_cfg.get("rate_clip_norm", 0.0))
    delta_clip = float(train_cfg.get("delta_clip_norm", 0.0))

    y = y_rate
    if rate_clip > 0.0:
        y = y.clamp(min=-rate_clip, max=rate_clip)

    if delta_clip > 0.0:
        if torch.is_tensor(dt_hat):
            dt = dt_hat.to(device=y.device, dtype=y.dtype)
        else:
            dt = torch.tensor(float(dt_hat), device=y.device, dtype=y.dtype)
        dt_safe = dt.abs().clamp_min(1e-12)
        delta = (y * dt).clamp(min=-delta_clip, max=delta_clip)
        y = delta / dt_safe

    return y

def sanitize_state_for_ops(x_abs: torch.Tensor, cfg: dict, rho_floor=1e-6, E_floor=1e-6):
    loss = cfg.get("loss", {}) or {}
    u_clip = float(loss.get("u_clip", 1e3))
    rho_max = float(loss.get("ops_rho_max", loss.get("state_rho_max", 0.0)))
    E_max = float(loss.get("ops_E_max", loss.get("state_E_max", 0.0)))
    m_max = float(loss.get("ops_m_max", loss.get("state_m_max", 0.0)))

    idx = dec.infer_feature_indices(cfg, x_abs.size(1))
    rho = x_abs[:, idx["rho"]].clamp_min(rho_floor)
    E = x_abs[:, idx["E"]].clamp_min(E_floor)
    if rho_max > 0.0:
        rho = rho.clamp_max(rho_max)
    if E_max > 0.0:
        E = E.clamp_max(E_max)
    mx = x_abs[:, idx["mx"]]
    my = x_abs[:, idx["my"]]

    # enforce |u| <= u_clip by clamping momenta given rho
    if u_clip > 0:
        mx = mx.clamp(-u_clip * rho, u_clip * rho)
        my = my.clamp(-u_clip * rho, u_clip * rho)
    if m_max > 0.0:
        mx = mx.clamp(min=-m_max, max=m_max)
        my = my.clamp(min=-m_max, max=m_max)

    rho = _sanitize_float_tensor(rho, clip_abs=rho_max, fill_value=rho_floor, nonneg=True)
    E = _sanitize_float_tensor(E, clip_abs=E_max, fill_value=E_floor, nonneg=True)
    mx = _sanitize_float_tensor(mx, clip_abs=m_max, fill_value=0.0, nonneg=False)
    my = _sanitize_float_tensor(my, clip_abs=m_max, fill_value=0.0, nonneg=False)

    cols = [x_abs[:, j] for j in range(x_abs.size(1))]
    cols[idx["rho"]] = rho
    cols[idx["E"]] = E
    cols[idx["mx"]] = mx
    cols[idx["my"]] = my
    return torch.stack(cols, dim=1)

def _enforce_physical_state(
    x_abs: torch.Tensor,
    cfg: dict,
    rho_floor: float = 1e-6,
    E_floor: float = 1e-6,
):
    """
    Clamp rho and E using the same feature-index mapping as DEC/PARC.
    This avoids assuming channel order is [rho, mx, my, E].
    """
    if x_abs is None:
        return x_abs
    if x_abs.ndim != 2:
        raise ValueError(f"_enforce_physical_state expects [N,F], got {tuple(x_abs.shape)}")

    Fdim = int(x_abs.size(1))
    idx = dec.infer_feature_indices(cfg, Fdim)
    loss = cfg.get("loss", {}) or {}
    u_clip = float(loss.get("u_clip", 1e3))
    rho_max = float(loss.get("state_rho_max", 0.0))
    E_max = float(loss.get("state_E_max", 0.0))
    m_max = float(loss.get("state_m_max", 0.0))

    cols = [x_abs[:, j] for j in range(Fdim)]
    rho = cols[idx["rho"]].clamp_min(rho_floor)
    E = cols[idx["E"]].clamp_min(E_floor)
    if rho_max > 0.0:
        rho = rho.clamp_max(rho_max)
    if E_max > 0.0:
        E = E.clamp_max(E_max)

    mx = cols[idx["mx"]]
    my = cols[idx["my"]]
    if u_clip > 0.0:
        mx = mx.clamp(min=-u_clip * rho, max=u_clip * rho)
        my = my.clamp(min=-u_clip * rho, max=u_clip * rho)
    if m_max > 0.0:
        mx = mx.clamp(min=-m_max, max=m_max)
        my = my.clamp(min=-m_max, max=m_max)

    cols[idx["rho"]] = _sanitize_float_tensor(rho, clip_abs=rho_max, fill_value=rho_floor, nonneg=True)
    cols[idx["E"]] = _sanitize_float_tensor(E, clip_abs=E_max, fill_value=E_floor, nonneg=True)
    cols[idx["mx"]] = _sanitize_float_tensor(mx, clip_abs=m_max, fill_value=0.0, nonneg=False)
    cols[idx["my"]] = _sanitize_float_tensor(my, clip_abs=m_max, fill_value=0.0, nonneg=False)
    return torch.stack(cols, dim=1)

def _resolve_time_integrator(cfg: dict) -> str:
    """
    Resolve the rollout time integrator.
    Supported: "euler" (default), "rk4".
    """
    train_cfg = cfg.get("train", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    raw = str(train_cfg.get("time_integrator", loss_cfg.get("time_integrator", "euler"))).strip().lower()
    aliases = {
        "euler": "euler",
        "rk4": "rk4",
        "runge-kutta4": "rk4",
        "rungekutta4": "rk4",
    }
    if raw not in aliases:
        raise RuntimeError(f"Unsupported time integrator '{raw}'. Use 'euler' or 'rk4'.")
    return aliases[raw]

def _resolve_time_integrator_for_epoch(
    cfg: dict,
    *,
    epoch_idx: int | None,
    base_integrator: str | None = None,
) -> tuple[str, float]:
    """
    Epoch-aware integrator schedule.

    Returns:
      (effective_integrator, rk4_alpha)
      - effective_integrator: "euler" or "rk4"
      - rk4_alpha: blend weight in [0,1] used when base integrator is rk4.
    """
    base = _resolve_time_integrator(cfg) if base_integrator is None else str(base_integrator).lower()
    if base != "rk4":
        return base, 0.0

    train_cfg = cfg.get("train", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}

    start_epoch = int(train_cfg.get("rk4_start_epoch", loss_cfg.get("rk4_start_epoch", 1)))
    ramp_epochs = int(train_cfg.get("rk4_ramp_epochs", loss_cfg.get("rk4_ramp_epochs", 0)))

    if start_epoch < 1:
        raise RuntimeError(f"rk4_start_epoch must be >= 1, got {start_epoch}")
    if ramp_epochs < 0:
        raise RuntimeError(f"rk4_ramp_epochs must be >= 0, got {ramp_epochs}")

    # If epoch is unknown (e.g., final test call), use the configured base integrator.
    if epoch_idx is None:
        return "rk4", 1.0

    ep = int(epoch_idx)
    if ep < start_epoch:
        return "euler", 0.0

    if ramp_epochs == 0:
        return "rk4", 1.0

    alpha = float(ep - start_epoch) / float(ramp_epochs)
    alpha = max(0.0, min(1.0, alpha))
    if alpha <= 0.0:
        return "euler", 0.0
    return "rk4", alpha


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
    epoch_idx: int | None = None,
    runtime_mesh_policy: Dict[str, Any] | None = None,
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
    diag_cfg = _resolve_diagnostics_cfg(cfg)
    diagnostics_accum = _new_diagnostics_accumulator(diag_cfg)

    dbg = cfg.get("debug", {})
    nan_watch = bool(dbg.get("nan_watch", True))          # turn on/off
    nan_watch_first_only = bool(dbg.get("nan_first_only", True))
    hang_watch = bool(dbg.get("hang_watch", True))
    hang_watch_batches = int(dbg.get("hang_watch_batches", 4))
    if hang_watch_batches < 0:
        hang_watch_batches = 0
    # Effective enable: either master switch off, or 0 batches -> disabled.
    hang_watch_enabled = bool(hang_watch and (hang_watch_batches > 0))
    print_batch_time = bool(dbg.get("print_batch_time", False))
    print_runtime_mesh_batch_breakdown = bool(dbg.get("print_runtime_mesh_batch_breakdown", True))
    progress_every_batches_raw = int(dbg.get("progress_every_batches", 10))
    # 0 (or negative) disables periodic [PROGRESS] logging.
    progress_print_enabled = bool(progress_every_batches_raw > 0)
    progress_every_batches = max(1, int(progress_every_batches_raw))

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

    def _record_step_diagnostics(
        *,
        pred_raw_abs: torch.Tensor,
        pred_used_abs: torch.Tensor,
        gt_abs: torch.Tensor,
        pred_levels: torch.Tensor,
        pred_centers: torch.Tensor,
        pred_ei: torch.Tensor,
    ) -> None:
        _record_enabled_step_diagnostics(
            diagnostics_accum,
            pred_raw_abs=pred_raw_abs.detach(),
            pred_used_abs=pred_used_abs.detach(),
            gt_abs=gt_abs.detach(),
            pred_levels=pred_levels.detach(),
            pred_centers=pred_centers.detach(),
            pred_ei=pred_ei.detach(),
            cfg=cfg,
            diag_cfg=diag_cfg,
            dx=float(dx),
            dy=float(dy),
        )

    def _dump_state(tag, x):
        # expects channels [rho,mx,my,E] in first 4 or via infer_feature_indices
        idx_map = dec.infer_feature_indices(cfg, x.size(1))
        rho = x[:, idx_map["rho"]]
        mx  = x[:, idx_map["mx"]]
        my  = x[:, idx_map["my"]]
        E   = x[:, idx_map["E"]]
        rho_abs = rho.abs()

        print(f"\n[NAN-DBG][{tag}] x stats")
        _tstats("x", x)
        _tstats("rho", rho)
        print("  rho<=0 count:", int((rho <= 0).sum().item()))
        print("  |rho|<1e-6 count:", int((rho_abs < 1e-6).sum().item()))
        _tstats("mx", mx)
        _tstats("my", my)
        _tstats("E", E)

        rho_safe = rho_abs.clamp_min(1e-12)  # diagnostic only
        ux = mx / rho_safe
        uy = my / rho_safe
        _tstats("ux=mx/|rho|", ux)
        _tstats("uy=my/|rho|", uy)

    @torch.no_grad()
    def _enforce_physical_state_with_diag(x_abs: torch.Tensor, *, tag: str, step_k: int,
                                        rho_idx: int = 0, E_idx: int = 3,
                                        rho_floor: float = 1e-6, E_floor: float = 1e-6) -> torch.Tensor:
        """
        Prints how often rho/E are below floor BEFORE clamp, then applies your existing clamp helper.
        """
        # pre-clamp stats on the tensor you're about to clamp
        rho_pre = x_abs[:, rho_idx]
        E_pre   = x_abs[:, E_idx]

        negR_frac = (rho_pre < rho_floor).float().mean().item()
        negE_frac = (E_pre   < E_floor).float().mean().item()

        # only print if something is actually getting clamped (or toggle as you like)
        if (negR_frac > 0.0) or (negE_frac > 0.0):
            print(
                f"[CLAMP] step_k={step_k} tag={tag} "
                f"preclamp neg_rho_frac={negR_frac:.3e} neg_E_frac={negE_frac:.3e} "
                f"rho[min]={float(rho_pre.min()):+.3e} E[min]={float(E_pre.min()):+.3e}"
            )

        # now do the actual clamp using your existing helper
        return _enforce_physical_state(x_abs, cfg, rho_floor=rho_floor, E_floor=E_floor)

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
    use_huber = bool(cfg["loss"].get("use_huber", True))
    grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0))

    predict_type = str(cfg.get("model", {}).get("predict_type", "rate")).lower()
    if predict_type != "rate":
        raise RuntimeError(f"PARC/Variant-B implementation below assumes predict_type='rate', got '{predict_type}'")
    time_integrator = _resolve_time_integrator(cfg)
    _time_integrator_eff, rk4_alpha = _resolve_time_integrator_for_epoch(
        cfg, epoch_idx=epoch_idx, base_integrator=time_integrator
    )

    # DEC / PARC controls
    loss_cfg = cfg.get("loss", {}) or {}
    dec_use = bool(loss_cfg.get("dec", False))
    parc_use = bool(loss_cfg.get("parc", False) or loss_cfg.get("parc_inputs", False))
    use_adapter = bool(loss_cfg.get("parc_use_adapter", False))

    # Variant-B baseline strength (reuse your existing name)
    dec_blend_w = float(loss_cfg.get("blend_weight", 0.0))        # baseline weight (recommend 1.0 when parc_use)
    dec_resid_w = float(loss_cfg.get("residual_weight", 0.0))     # optional residual loss weight

    # Determine whether we must compute physics operators this step
    need_phy = parc_use or (dec_use and (dec_blend_w != 0.0 or dec_resid_w != 0.0))

    backend = str(loss_cfg.get("physics_backend", "dec")).lower()
    use_mls = (backend in ("mls", "moving_least_squares", "moving-least-squares"))

    if scaler is None and use_amp and device.type == "cuda":
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    amp_ctx = ((lambda: torch.amp.autocast(device_type="cuda", enabled=True))
               if use_amp and device.type == "cuda"
               else (lambda: torch.autocast("cpu", enabled=False)))

    runtime_mesh_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    runtime_mesh_enabled = bool(runtime_mesh_cfg.get("enabled", False))
    runtime_update_every = int(runtime_mesh_cfg.get("update_every_steps", 1))
    runtime_mesh_backend = _runtime_mesh_backend(cfg)
    runtime_domain_mode = _runtime_mesh_domain_mode(cfg)
    runtime_idw_backend = "exact"
    runtime_idw_backend_kwargs: Dict[str, Any] = {}
    runtime_bbox = _get_bbox(cfg)
    runtime_refine_ratio = _get_refine_ratio(cfg)
    runtime_need_edge_attr = bool(need_phy and (not use_mls))
    runtime_wedge_path = None
    runtime_wedge_constraints = None
    runtime_base_mesh = None
    if runtime_mesh_enabled:
        mesh_spec_path = cfg.get("mesh", {}).get("starting_mesh_path", None)
        if not mesh_spec_path:
            raise RuntimeError("Runtime mesh mode requires cfg['mesh']['starting_mesh_path'].")
        if runtime_domain_mode == "starting_mesh":
            runtime_base_mesh = _get_runtime_starting_mesh_base(
                cfg=cfg,
                H=H,
                W=W,
                dx=float(dx),
                dy=float(dy),
                device=device,
                mesh_spec_path=str(mesh_spec_path),
            )
        else:
            runtime_wedge_path = _load_wedge_path_from_spec(str(mesh_spec_path), cfg)
            runtime_wedge_constraints = _get_runtime_wedge_constraints(
                cfg=cfg,
                H=H,
                W=W,
                device=device,
                wedge_path=runtime_wedge_path,
            )
        runtime_idw_backend, runtime_idw_backend_kwargs = _runtime_idw_backend_settings(
            cfg,
            out_device=device,
        )
        if runtime_mesh_backend == "cnn" and runtime_mesh_policy is None:
            raise RuntimeError(
                "Runtime mesh backend is 'cnn' but runtime_mesh_policy was not loaded."
            )
        print(
            f"[RUNTIME-MESH] train loop active (backend={runtime_mesh_backend}, domain_mode={runtime_domain_mode}, "
            f"idw_backend={runtime_idw_backend}, update_every_steps={runtime_update_every})"
        )

    train_one_epoch_multi_step._printed_loss_components = False

    epoch_loop_t0 = time.perf_counter()
    try:
        total_batches = int(len(loader))
    except Exception:
        total_batches = -1
    hang_cap_notified = False

    def _maybe_print_batch_progress(batch_idx: int, batch_wall_s: float) -> None:
        bi = int(batch_idx) + 1
        if print_batch_time:
            if total_batches > 0:
                print(f"[BATCH-TIME] train batch {bi}/{total_batches} wall={batch_wall_s:.3f}s", flush=True)
            else:
                print(f"[BATCH-TIME] train batch {bi} wall={batch_wall_s:.3f}s", flush=True)
        if not progress_print_enabled:
            return
        if (bi % progress_every_batches) != 0:
            return
        elapsed = float(time.perf_counter() - epoch_loop_t0)
        avg = elapsed / float(max(bi, 1))
        if total_batches > 0:
            print(
                f"[PROGRESS] train batch {bi}/{total_batches} "
                f"batch_wall={batch_wall_s:.3f}s avg_batch={avg:.3f}s elapsed={elapsed:.1f}s",
                flush=True,
            )
        else:
            print(
                f"[PROGRESS] train batch {bi} "
                f"batch_wall={batch_wall_s:.3f}s avg_batch={avg:.3f}s elapsed={elapsed:.1f}s",
                flush=True,
            )

    def _maybe_print_runtime_mesh_batch_breakdown(
        batch_idx: int,
        batch_wall_s: float,
        *,
        t_mesh_predict_s: float,
        t_mesh_materialize_s: float,
        t_edge_attr_s: float,
        t_idw_remap_s: float,
        n_rebuilds: int,
    ) -> None:
        if (not runtime_mesh_enabled) or (not print_runtime_mesh_batch_breakdown):
            return
        bi = int(batch_idx) + 1
        total = float(max(batch_wall_s, 1e-12))
        accounted = (
            float(t_mesh_predict_s)
            + float(t_mesh_materialize_s)
            + float(t_edge_attr_s)
            + float(t_idw_remap_s)
        )
        other = max(0.0, total - accounted)

        def _pct(x: float) -> float:
            return 100.0 * float(x) / total

        if total_batches > 0:
            prefix = f"[RUNTIME-COST] train batch {bi}/{total_batches}"
        else:
            prefix = f"[RUNTIME-COST] train batch {bi}"

        print(
            f"{prefix} wall={total:.3f}s "
            f"| mesh_materialize={float(t_mesh_materialize_s):.3f}s ({_pct(t_mesh_materialize_s):.1f}%) "
            f"| idw_remap={float(t_idw_remap_s):.3f}s ({_pct(t_idw_remap_s):.1f}%) "
            f"| edge_attr={float(t_edge_attr_s):.3f}s ({_pct(t_edge_attr_s):.1f}%) "
            f"| other={other:.3f}s ({_pct(other):.1f}%) "
            f"| rebuilds={int(n_rebuilds)}",
            flush=True,
        )
        if total_batches > 0:
            mesh_prefix = f"[RUNTIME-MESH-PREDICT] train batch {bi}/{total_batches}"
        else:
            mesh_prefix = f"[RUNTIME-MESH-PREDICT] train batch {bi}"
        mesh_per_rebuild = float(t_mesh_predict_s) / float(max(1, int(n_rebuilds)))
        print(
            f"{mesh_prefix} total={float(t_mesh_predict_s):.3f}s ({_pct(t_mesh_predict_s):.1f}%) "
            f"| per_rebuild={mesh_per_rebuild:.3f}s | rebuilds={int(n_rebuilds)}",
            flush=True,
        )

    for batch_idx, batch in enumerate(loader):
        if hang_watch_enabled and (not hang_cap_notified) and (batch_idx == hang_watch_batches):
            print(
                f"[HANG-DBG] detailed hang logs reached configured limit: "
                f"debug.hang_watch_batches={hang_watch_batches}. "
                f"Set a larger value to continue per-step diagnostics.",
                flush=True,
            )
            hang_cap_notified = True

        batch_dbg = bool(hang_watch_enabled and (batch_idx < hang_watch_batches))
        batch_t0 = time.perf_counter()
        if batch_dbg:
            print(f"[HANG-DBG] batch fetched: idx={batch_idx}", flush=True)

        batch_rt_mesh_predict_s = 0.0
        batch_rt_mesh_materialize_s = 0.0
        batch_rt_edge_attr_s = 0.0
        batch_rt_idw_remap_s = 0.0
        batch_rt_rebuilds = 0

        dt_list = batch.get("dt_list", None)
        if dt_list is None:
            raise RuntimeError("Missing dt_list in batch. Ensure the active collate attaches dt_list.")

        # Base lists (always required)
        centers_list = _require_list(batch, "centers_list")
        feat_list = _require_list(batch, "feat_list")
        level_list = _require_list(batch, "level_list")
        parents_list = _require_list(batch, "parents_list")

        # Optional precompute lists (required only in strict/precompute mode).
        pred_centers_list = batch.get("pred_centers_list", None)
        pred_levels_list = batch.get("pred_levels_list", None)
        pred_parents_list = batch.get("pred_parents_list", None)
        pred_ei_list = batch.get("pred_ei_list", None)
        mask_pred_list = batch.get("mask_pred_list", None)
        pred_ea_list = batch.get("pred_ea_list", None) or batch.get("pred_edge_attr_list", None)
        feat_t_on_pred_list = batch.get("feat_t_on_pred_list", None)
        feat_tp1_on_pred_list = batch.get("feat_tp1_on_pred_list", None)
        pred2pred_idx_list = batch.get("pred2pred_idx_list", None)
        pred2pred_w_list = batch.get("pred2pred_w_list", None)

        K = len(centers_list)
        if K < 2:
            raise RuntimeError("window_size must be ≥ 2")

        dt_ref_scalar = batch.get("dt_ref", None)
        t_indices = batch.get("t_indices", None)

        has_precomp_lists = (
            (pred_centers_list is not None)
            and (pred_levels_list is not None)
            and (pred_parents_list is not None)
            and (pred_ei_list is not None)
            and (mask_pred_list is not None)
            and (feat_t_on_pred_list is not None)
            and (feat_tp1_on_pred_list is not None)
        )

        pred_lists = None
        if (not runtime_mesh_enabled) or has_precomp_lists:
            pred_centers_list = _require_list(batch, "pred_centers_list")
            pred_levels_list = _require_list(batch, "pred_levels_list")
            pred_parents_list = _require_list(batch, "pred_parents_list")
            pred_ei_list = _require_list(batch, "pred_ei_list")
            mask_pred_list = _require_list(batch, "mask_pred_list")
            feat_t_on_pred_list = _require_list(batch, "feat_t_on_pred_list")
            feat_tp1_on_pred_list = _require_list(batch, "feat_tp1_on_pred_list")

            if (not runtime_mesh_enabled) and need_phy and pred_ea_list is None:
                raise RuntimeError(
                    "PARC/DEC enabled but batch is missing pred_ea_list / pred_edge_attr_list. "
                    "Update H5 loader + CollateWithPrecompute to attach edge_attr per step."
                )

            pred_lists = (pred_centers_list, pred_levels_list, pred_parents_list, pred_ei_list, mask_pred_list)

        window_loss = 0.0
        window_mae = 0.0
        window_loss_graph = None

        opt.zero_grad(set_to_none=True)

        # Only DEC needs edge_attr. MLS does not.
        if (not runtime_mesh_enabled) and need_phy and (not use_mls) and (pred_ea_list is None):
            raise RuntimeError(
                "DEC/PARC physics enabled but batch is missing pred_ea_list / pred_edge_attr_list. "
                "Update loader + CollateWithPrecompute to attach edge_attr per step."
            )

        # ------------------------------------------------------------------
        # Move per-step tensors to GPU right before each _run_step
        # ------------------------------------------------------------------
        def _to_dev_nb(x, *, dtype=None):
            if x is None:
                return None
            if not torch.is_tensor(x):
                return x
            # Avoid needless copies when already correct
            if x.device == device and (dtype is None or x.dtype == dtype):
                return x
            return x.to(device=device, dtype=(dtype if dtype is not None else x.dtype), non_blocking=True)


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
            x_ops_abs: torch.Tensor | None = None,
        ):
            norm_in  = _maybe_norm(x_in_abs,  mu, sigma)
            norm_tgt = _maybe_norm(x_tgt_abs, mu, sigma)

            if _should_print() and nan_watch:
                print(f"[NAN-DBG] step={step_k} ---- norm checks ----")
                _tstats("x_in_abs", x_in_abs)
                _tstats("x_tgt_abs", x_tgt_abs)
                _tstats("norm_in", norm_in)
                _tstats("norm_tgt", norm_tgt)
                if sigma is not None and torch.is_tensor(sigma):
                    _tstats("sigma", sigma)
                if mu is not None and torch.is_tensor(mu):
                    _tstats("mu", mu)

                # assumes channels [rho, mx, my, E] in x_in_abs
                rho = x_in_abs[:, 0]
                mx  = x_in_abs[:, 1]
                my  = x_in_abs[:, 2]

                rho_safe = rho.abs().clamp_min(1e-12)
                ux = mx / rho_safe
                uy = my / rho_safe

                _tstats("rho", rho)
                print(f"[NAN-debug] |rho|<1e-6 count: {(rho.abs() < 1e-6).sum().item()} / {rho.numel()}")
                _tstats("ux(mx/rho_safe)", ux)
                _tstats("uy(my/rho_safe)", uy)

                _mark_printed()

            _assert_finite("norm_in", norm_in)
            _assert_finite("norm_tgt", norm_tgt)

            dt_ref_t = (torch.tensor(float(dt_ref_scalar), device=device, dtype=norm_in.dtype)
                        if dt_ref_scalar is not None else None)
            dt_hat = (dt_phys / dt_ref_t) if dt_ref_t is not None else dt_phys

            if _should_print() and nan_watch:
                print(f"[NAN-DBG] step={step_k} ---- dt checks ----")
                _tstats("dt_phys", dt_phys)
                _tstats("dt_ref_t", dt_ref_t if dt_ref_t is not None else None)
                _tstats("dt_hat", dt_hat)
                _mark_printed()

            _assert_finite("dt_hat", dt_hat)

            # ---------------------------
            # NAN DEBUG: dt and targets
            # ---------------------------
            if _should_print() and nan_watch:
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

            sigma_f32 = _as_stat_tensor(sigma, device=device, dtype=torch.float32)

            if sigma_f32 is not None:
                sigma_f32 = sigma_f32.clamp_min(1e-12)

            dt_phys_f32 = dt_phys.to(device=device, dtype=torch.float32)
            dt_ref_f32  = (dt_ref_t.to(device=device, dtype=torch.float32) if dt_ref_t is not None else None)

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
                if _should_print() and nan_watch:
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
                    #print("[DEC-CHK] Computing DEC operators for step", step_k)
                    with torch.autocast(device_type=device.type, enabled=False):

                        # Check whether advection should be applied at all steps or not
                        adv_all_steps  = bool(loss_cfg.get("adv_all_steps", True))
                        if not adv_all_steps:
                            adv_step_gate = (step_k == 0)
                        else:
                            adv_step_gate = True

                        adv_w  = float(loss_cfg.get("adv_weight", 1.0))
                        diff_w = float(loss_cfg.get("diff_weight", 1.0))

                        # Use your real keys (parc_include_adv/diff), defaulting True only if you want that behavior.
                        include_adv_cfg  = bool(loss_cfg.get("parc_include_adv", True))
                        include_diff_cfg = bool(loss_cfg.get("parc_include_diff", True))

                        # Compute flags for DEC operator evaluation this step
                        need_adv  = (adv_step_gate and include_adv_cfg  and (adv_w != 0.0))
                        need_diff = (include_diff_cfg and (diff_w != 0.0))

                        if cfg.get("debug", {}).get("print_ops_clamp", False):
                            _ = _enforce_physical_state_with_diag(
                                x_in_abs, tag="before_ops_sanitize", step_k=step_k, rho_floor=1e-6, E_floor=1e-6
                            )

                        #x_for_ops = _enforce_physical_state(x_in_abs, rho_floor=1e-6, E_floor=1e-6)
                        x_for_ops = sanitize_state_for_ops(x_in_abs, cfg, rho_floor=1e-6, E_floor=1e-6)
                        #x_ops_base = x_ops_abs if (x_ops_abs is not None) else x_in_abs
                        #x_for_ops = sanitize_state_for_ops(x_ops_base, cfg, rho_floor=1e-6, E_floor=1e-6)

                        # Triggered dump: only when state is already “dangerous”
                        if torch.is_tensor(x_in_abs):
                            danger = (
                                (~torch.isfinite(x_in_abs)).any()
                                or (x_in_abs.abs().max() > 1e6)          # tune threshold
                            )
                            if danger:
                                print(f"\n[NAN-DBG] step={step_k} TRIGGER: x_in_abs looks dangerous before ops")
                                _dump_state("x_in_abs", x_in_abs)

                        # after enforce_physical_state
                        if torch.is_tensor(x_for_ops):
                            danger2 = (
                                (~torch.isfinite(x_for_ops)).any()
                                or (x_for_ops.abs().max() > 1e6)
                            )
                            if danger2:
                                print(f"\n[NAN-DBG] step={step_k} TRIGGER: x_for_ops looks dangerous after enforce")
                                _dump_state("x_for_ops", x_for_ops)

                        with torch.autocast(device_type=device.type, enabled=False):
                            with torch.no_grad():
                                def _pick_mls_at_k(batch, k, N, E=None):
                                    """
                                    Pick an index kk near k such that:
                                    - grad_M_inv[kk].shape[0] == N
                                    - and if E is provided, grad_dX[kk].shape[0] == E
                                    Returns (kk, M_inv, dX, lap_w, ei_used)
                                    """
                                    M_list  = batch.get("mls_grad_M_inv_list", None)
                                    dX_list = batch.get("mls_grad_dX_list", None)
                                    w_list  = batch.get("mls_lap_weights_list", None) or batch.get("mls_lap_w_list", None)
                                    ei_list = batch.get("mls_grad_ei_used_list", None)  # strongly preferred

                                    if M_list is None or dX_list is None:
                                        raise RuntimeError("Missing MLS lists in batch (mls_grad_M_inv_list / mls_grad_dX_list).")

                                    for kk in (k, k + 1, k - 1):
                                        if not (0 <= kk < len(M_list)):
                                            continue
                                        M = M_list[kk]
                                        dX = dX_list[kk]
                                        if M is None or dX is None:
                                            continue
                                        if int(M.shape[0]) != int(N):
                                            continue

                                        ei_used = None
                                        if ei_list is not None and 0 <= kk < len(ei_list):
                                            ei_used = ei_list[kk]
                                            if ei_used is not None:
                                                # ensure E consistency with dX if possible
                                                if int(ei_used.shape[1]) != int(dX.shape[0]):
                                                    continue

                                        if E is not None and int(dX.shape[0]) != int(E):
                                            # if caller supplied E, enforce it
                                            continue

                                        lap_w = None
                                        if w_list is not None and 0 <= kk < len(w_list):
                                            lap_w = w_list[kk]

                                        return kk, M, dX, lap_w, ei_used

                                    # If we got here, show the nearby sizes to make the indexing bug obvious
                                    def _sz(x): 
                                        return None if x is None else tuple(x.shape)
                                    dbg = []
                                    for kk in (k, k+1, k-1):
                                        if 0 <= kk < len(M_list):
                                            dbg.append((kk, _sz(M_list[kk]), _sz(dX_list[kk]), _sz(ei_list[kk]) if ei_list is not None else None))
                                    raise RuntimeError(f"Could not align MLS geometry to x_abs(N={N}). Nearby shapes: {dbg}")

                                if use_mls:
                                    # FACE-ADJ edges (same ones used by DEC)
                                    r_adv_abs, r_diff_abs, area = mls_advdiff_terms_abs_faceadj(
                                        x_abs=x_for_ops.float(),          # your absolute state tensor on pred mesh
                                        pos=pred_centers,                # (N,2) for that mesh
                                        edge_index=pei.long(),           # (2,E) face-adj from precomp_rollout.h5
                                        levels=pred_levels,              # (N,)
                                        dx0=float(dx),
                                        dy0=float(dy),
                                        cfg=cfg,
                                        compute_adv=need_adv,
                                        compute_diff=need_diff,
                                    )

                                else:
                                    r_adv_abs, r_diff_abs, area = dec.dec_advdiff_terms_abs(
                                        #x_abs=x_in_abs.float(),
                                        x_abs=x_for_ops.float(),
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

                                r_adv_abs = _sanitize_ops_term(r_adv_abs, cfg)
                                r_diff_abs = _sanitize_ops_term(r_diff_abs, cfg)
                                area = _sanitize_float_tensor(area, clip_abs=0.0, fill_value=1.0, nonneg=True)

                                if not adv_step_gate:
                                    r_adv_abs = None

                                r_phy_abs = None
                                if (dec_blend_w != 0.0) or (dec_resid_w != 0.0):
                                    # Start from diffusion baseline (or zeros)
                                    if need_diff and (r_diff_abs is not None):
                                        base = diff_w * r_diff_abs
                                    else:
                                        # need a correctly-shaped zero tensor
                                        ref = r_diff_abs if (r_diff_abs is not None) else r_adv_abs
                                        if ref is None:
                                            # Shouldn't happen if need_phy, but keep safe
                                            ref = x_in_abs.float()
                                        base = torch.zeros_like(ref)

                                    # Add advection only at step 0 (because need_adv includes adv_step_gate)
                                    if need_adv and (r_adv_abs is not None):
                                        base = base + adv_w * r_adv_abs

                                    r_phy_abs = _sanitize_ops_term(base, cfg)

                                if cfg.get("debug", {}).get("stepA_star", False):
                                    if step_k >= int(cfg.get("debug", {}).get("stepA_star_kmin", 5)):
                                        stepA_check_given_state_vs_gt(
                                            tag=f"step_k={step_k} (TRAIN STATE)",
                                            x_in_abs=x_in_abs,
                                            x_tgt_abs=x_tgt_abs,
                                            pred_levels=pred_levels,
                                            pei=pei,
                                            pea=pea,
                                            dt=dt_phys,
                                            cfg=cfg,
                                            device=device,
                                            dx=float(dx),
                                            dy=float(dy),
                                            u_max=float(cfg.get("debug", {}).get("stepA_u_max", 1e3)),
                                        )
                                if cfg.get("debug", {}).get("stepA_teacher", False):
                                    if feat_t_on_pred_list is None:
                                        raise RuntimeError(
                                            "debug.stepA_teacher requires feat_t_on_pred_list from precompute collate."
                                        )
                                    j = step_k + 1  
                                    x_in_teacher = feat_t_on_pred_list[j].to(device)  # GT(t+k) on pred(t+k+1)

                                    stepA_check_given_state_vs_gt(
                                        tag=f"step_k={step_k} (GT STATE)",
                                        x_in_abs=x_in_teacher,
                                        x_tgt_abs=x_tgt_abs,
                                        pred_levels=pred_levels,
                                        pei=pei,
                                        pea=pea,
                                        dt=dt_phys,
                                        cfg=cfg,
                                        device=device,
                                        dx=float(dx),
                                        dy=float(dy),
                                        u_max=float(cfg.get("debug", {}).get("stepA_u_max", 1e3)),
                                    )

                                    # also print drift magnitude between rollout state and GT state on same mesh:
                                    drift = x_in_abs.float() - x_in_teacher.float()
                                    print(f"[STEP-B] step_k={step_k} drift absmax={float(drift.abs().max()):.3e} "
                                        f"drift meanabs={float(drift.abs().mean()):.3e}")


                        # Channel mask: use the same selection logic as PARC inputs
                        #sel = dec.parc_select_feature_indices(cfg, x_in_abs.size(1))  # list[int]
                        #ch_mask = torch.zeros((x_in_abs.size(1),), device=device, dtype=torch.float32)
                        #ch_mask[torch.as_tensor(sel, device=device)] = 1.0

                        Fdim = x_in_abs.size(1)
                        sel_adv  = dec.parc_select_feature_indices_adv(cfg, Fdim)
                        sel_diff = dec.parc_select_feature_indices_diff(cfg, Fdim)
                        sel = sorted(set(sel_adv + sel_diff))

                        ch_mask = torch.zeros((Fdim,), device=device, dtype=torch.float32)
                        if len(sel) > 0:
                            ch_mask[torch.as_tensor(sel, device=device)] = 1.0

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
                    #_assert_finite("r_adv_abs", r_adv_abs)
                    if r_phy_abs is not None:
                        _assert_finite("r_phy_abs", r_phy_abs)

                # ----- build node input X (PARC appends operator inputs) -----
                x_in = _build_X(norm_in, pred_centers, pred_levels, cfg, dt_hat=dt_hat)

                # Allow diffusion-only or advection-only PARC inputs:
                # If one operator term is missing (None), substitute a correctly-shaped zero tensor.
                parc_extra = None
                if parc_use:
                    if (r_adv_abs is None) and (r_diff_abs is None):
                        parc_extra = None
                    else:
                        # Pick a reference tensor to define shape/device
                        ref = r_diff_abs if (r_diff_abs is not None) else r_adv_abs
                        assert ref is not None, "Internal: ref should not be None here."

                        r_adv_in  = r_adv_abs  if (r_adv_abs  is not None) else torch.zeros_like(ref)
                        r_diff_in = r_diff_abs if (r_diff_abs is not None) else torch.zeros_like(ref)

                        # Optional: if you have a channel mask (e.g., rho+E only), apply it consistently
                        # so PARC inputs don't inject junk into channels you don't trust.
                        if ch_mask is not None:
                            cm = ch_mask.to(device=device, dtype=torch.float32).view(1, -1)
                            r_adv_in  = r_adv_in  * cm
                            r_diff_in = r_diff_in * cm

                        parc_extra = dec.parc_terms_to_node_inputs(
                            r_adv_in.to(device=device, dtype=torch.float32),
                            r_diff_in.to(device=device, dtype=torch.float32),
                            dt_phys=dt_phys.to(device=device, dtype=torch.float32),
                            dt_ref=(dt_ref_t.to(device=device, dtype=torch.float32) if dt_ref_t is not None else None),
                            #sigma=(sigma.to(device, dtype=torch.float32) if (sigma is not None and torch.is_tensor(sigma)) else None),
                            sigma=sigma_f32,
                            predict_type=predict_type,
                            cfg=cfg,
                            dtype=x_in.dtype,
                            detach=True,
                        )
                        parc_extra = _sanitize_parc_extra_tensor(parc_extra, cfg, post_adapter=False)

                    #if (parc_extra is not None) and (parc_extra.numel() > 0):
                        # parc_extra should already be dtype=x_in.dtype per your call, but keep this safe
                    #    x_in = torch.cat([x_in, parc_extra.to(dtype=x_in.dtype)], dim=1)
                    if parc_extra is not None and parc_extra.numel() > 0:
                        if hasattr(model, "parc_adapter") and (model.parc_adapter is not None) and use_adapter:
                            adv_active  = (r_adv_abs is not None)   # only step0 in your gating
                            diff_active = (r_diff_abs is not None)  # typically True if enabled

                            parc_extra = model.parc_adapter(
                                parc_extra,
                                update_adv_stats=adv_active,
                                update_diff_stats=diff_active,
                            )

                            # IMPORTANT: prevent normalized “fake adv” on steps where adv is gated off
                            if not adv_step_gate:
                                da = int(model.parc_adapter.dim_adv)
                                if da > 0:
                                    if da >= int(parc_extra.size(1)):
                                        parc_extra = torch.zeros_like(parc_extra)
                                    else:
                                        parc_extra = torch.cat(
                                            [
                                                torch.zeros_like(parc_extra[:, :da]),
                                                parc_extra[:, da:],
                                            ],
                                            dim=1,
                                        )

                        parc_extra = _sanitize_parc_extra_tensor(
                            parc_extra, cfg, post_adapter=bool(use_adapter)
                        )
                        x_in = torch.cat([x_in, parc_extra.to(dtype=x_in.dtype)], dim=1)


                    watch_steps = set(cfg.get("debug", {}).get("parc_scale_watch_steps", [0, 3, 6, 9]))
                    if cfg.get("debug", {}).get("parc_scale_watch", True) and (step_k in watch_steps):
                        with torch.no_grad():
                            _tstats("norm_in", norm_in)
                            _tstats("r_adv_abs", r_adv_abs)
                            _tstats("r_diff_abs", r_diff_abs)

                            # Convert ABS operator rates to "model units" (normalized-rate space)
                            sigma_f32 = _as_stat_tensor(sigma, device=device, dtype=torch.float32)
                            if sigma_f32 is not None:
                                sigma_f32 = sigma_f32.clamp_min(1e-12)

                            dt_ref_f32  = (dt_ref_t.to(device=device, dtype=torch.float32) if dt_ref_t is not None else None)
                            dt_phys_f32 = dt_phys.to(device=device, dtype=torch.float32)

                            if r_adv_abs is not None:
                                adv_model = dec.physics_to_model_units(
                                    r_adv_abs.to(dtype=torch.float32),
                                    dt_phys=dt_phys_f32,
                                    dt_ref=dt_ref_f32,
                                    sigma=sigma_f32,
                                    predict_type="rate",
                                )
                                _tstats("adv_model_units", adv_model)

                            if r_diff_abs is not None:
                                diff_model = dec.physics_to_model_units(
                                    r_diff_abs.to(dtype=torch.float32),
                                    dt_phys=dt_phys_f32,
                                    dt_ref=dt_ref_f32,
                                    sigma=sigma_f32,
                                    predict_type="rate",
                                )
                                _tstats("diff_model_units", diff_model)

                            _tstats("parc_extra", parc_extra)


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

                # one-time print guard
                if not hasattr(train_one_epoch_multi_step, "_printed_input_contract"):
                    train_one_epoch_multi_step._printed_input_contract = False

                if (not train_one_epoch_multi_step._printed_input_contract):
                    Fdim = x_in_abs.size(1)

                    # Base X is built from norm_in + geometry (pos, level, etc.)
                    base_X = _build_X(norm_in, pred_centers, pred_levels, cfg, dt_hat=dt_hat)

                    print("\n[BASE_X BREAKDOWN]")
                    print("  base_X shape:", tuple(base_X.shape))

                    # 1) state part (should be 4 channels when Fdim=4)
                    print("  assuming first Fdim columns are state:", Fdim)

                    state_block = base_X[:, :Fdim]
                    _tstats("  state_block(base_X[:, :Fdim])", state_block)

                    # label state channels using your infer_feature_indices mapping
                    idx = dec.infer_feature_indices(cfg, Fdim)
                    inv = {v: k for k, v in idx.items()}
                    for j in range(Fdim):
                        _tstats(f"  state[{j}]={inv.get(j, f'idx{j}')}", state_block[:, j])

                    # 2) the remainder (geometry / extras)
                    extra_block = base_X[:, Fdim:]
                    print("  extra_block dim:", int(extra_block.size(1)))
                    _tstats("  extra_block(base_X[:, Fdim:])", extra_block)

                    # 3) try to interpret extras as [pos(x,y?) , level?] based on tensors you have
                    # pred_centers is typically [N,2]
                    if torch.is_tensor(pred_centers):
                        print("  pred_centers shape:", tuple(pred_centers.shape))
                        _tstats("  pred_centers[:,0]", pred_centers[:, 0])
                        if pred_centers.size(1) > 1:
                            _tstats("  pred_centers[:,1]", pred_centers[:, 1])

                    if torch.is_tensor(pred_levels):
                        print("  pred_levels shape:", tuple(pred_levels.shape))
                        _tstats("  pred_levels", pred_levels.to(dtype=base_X.dtype))

                    # 4) correlation check: see if any extra column matches x, y, or level (up to scaling)
                    def _corr(a, b, eps=1e-12):
                        a = a.flatten().to(torch.float32)
                        b = b.flatten().to(torch.float32)
                        a = a - a.mean()
                        b = b - b.mean()
                        denom = (a.std() * b.std()).clamp_min(eps)
                        return float((a*b).mean() / denom)

                    if extra_block.numel() > 0:
                        for kcol in range(extra_block.size(1)):
                            col = extra_block[:, kcol]
                            msg = f"  extra_col[{kcol}] stats:"
                            print(msg, "min/max =", float(col.min().cpu()), float(col.max().cpu()))
                            if torch.is_tensor(pred_centers):
                                cx = pred_centers[:, 0].to(device=col.device, dtype=col.dtype)
                                print(f"    corr(extra[{kcol}], center_x) =", _corr(col, cx))
                                if pred_centers.size(1) > 1:
                                    cy = pred_centers[:, 1].to(device=col.device, dtype=col.dtype)
                                    print(f"    corr(extra[{kcol}], center_y) =", _corr(col, cy))
                            if torch.is_tensor(pred_levels):
                                lv = pred_levels.to(device=col.device, dtype=col.dtype)
                                print(f"    corr(extra[{kcol}], level)    =", _corr(col, lv))

                    base_dim = base_X.size(1)
                    total_dim = x_in.size(1)

                    # Selections used for PARC inputs
                    sel_adv  = dec.parc_select_feature_indices_adv(cfg, Fdim)
                    sel_diff = dec.parc_select_feature_indices_diff(cfg, Fdim)

                    idx = dec.infer_feature_indices(cfg, Fdim)
                    inv = {v: k for k, v in idx.items()}  # index->name

                    def _names(sel):
                        return [inv.get(int(i), f"idx{i}") for i in sel]

                    print("\n[INPUT-CONTRACT]")
                    print("  Fdim(state channels):", Fdim)
                    print("  base_X dim:", base_dim)
                    print("  parc_extra dim:", 0 if parc_extra is None else int(parc_extra.size(1)))
                    print("  total x_in dim:", total_dim)
                    print("  adv sel idx:", sel_adv, "names:", _names(sel_adv))
                    print("  diff sel idx:", sel_diff, "names:", _names(sel_diff))

                    # sanity: expected parc dim = len(sel_adv)+len(sel_diff) (given include flags)
                    loss = cfg.get("loss", {}) or {}
                    inc_adv = bool(loss.get("parc_include_adv", True))
                    inc_diff = bool(loss.get("parc_include_diff", True))
                    expected_parc = (len(sel_adv) if inc_adv else 0) + (len(sel_diff) if inc_diff else 0)
                    got_parc = 0 if parc_extra is None else int(parc_extra.size(1))
                    print("  expected parc_extra dim:", expected_parc, "got:", got_parc)

                    # check model expected input dim (FeatureNet first layer)
                    try:
                        # For SAGEConv, lin_l has in_channels = model.in_channels
                        # but FeatureNet might store it differently; try both:
                        model_in = getattr(model, "in_channels", None)
                        if model_in is None and hasattr(model, "convs") and len(model.convs) > 0 and hasattr(model.convs[0], "lin_l"):
                            model_in = model.convs[0].lin_l.in_features
                        print("  model expects in_channels:", model_in)
                    except Exception as e:
                        print("  [WARN] couldn't infer model in_channels:", repr(e))

                    train_one_epoch_multi_step._printed_input_contract = True

                    if parc_extra is not None and parc_extra.numel() > 0:
                        print("[INPUT-CONTRACT] parc_extra stats:")
                        _tstats("parc_extra", parc_extra)

                        # split the block into adv and diff pieces to confirm ordering
                        la = len(dec.parc_select_feature_indices_adv(cfg, x_in_abs.size(1)))
                        ld = len(dec.parc_select_feature_indices_diff(cfg, x_in_abs.size(1)))

                        off = 0
                        if la > 0:
                            _tstats("parc_adv_block", parc_extra[:, off:off+la])
                            off += la
                        if ld > 0:
                            _tstats("parc_diff_block", parc_extra[:, off:off+ld])


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

                if need_phy and (dec_blend_w != 0.0) and (r_phy_abs is not None):
                    # compute baseline in fp32 (and without autocast) for stability
                    with torch.autocast(device_type=device.type, enabled=False):
                        r_phy_f32 = r_phy_abs.to(device=device, dtype=torch.float32)

                        phy_units_f32 = dec.physics_to_model_units(
                            r_phy_f32,
                            dt_phys=dt_phys_f32,
                            dt_ref=dt_ref_f32,
                            sigma=sigma_f32,              # robust: tensor or None
                            predict_type=predict_type,
                        )

                        # Mask baseline to selected channels
                        if ch_mask is not None:
                            phy_units_f32 = phy_units_f32 * ch_mask.view(1, -1).to(device=device, dtype=torch.float32)

                    # Cast back to model dtype and add
                    phy_units = phy_units_f32.to(dtype=y_corr.dtype)
                    y_pred = y_corr + dec_blend_w * phy_units

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

                r_phy_for_loss = r_phy_abs

                # Optional higher-order integration in normalized-rate space.
                if rk4_alpha > 0.0:
                    def _predict_rate_for_norm_state(
                        norm_state: torch.Tensor,
                        *,
                        update_adapter_stats: bool,
                    ):
                        x_state_abs = _maybe_denorm(norm_state, mu, sigma)
                        r_adv_stage = r_diff_stage = r_phy_stage = None
                        ch_mask_stage = ch_mask

                        if need_phy:
                            with torch.autocast(device_type=device.type, enabled=False):
                                adv_all_steps_stage = bool(loss_cfg.get("adv_all_steps", True))
                                adv_step_gate_stage = (step_k == 0) if (not adv_all_steps_stage) else True

                                adv_w_stage = float(loss_cfg.get("adv_weight", 1.0))
                                diff_w_stage = float(loss_cfg.get("diff_weight", 1.0))
                                include_adv_cfg_stage = bool(loss_cfg.get("parc_include_adv", True))
                                include_diff_cfg_stage = bool(loss_cfg.get("parc_include_diff", True))
                                need_adv_stage = (adv_step_gate_stage and include_adv_cfg_stage and (adv_w_stage != 0.0))
                                need_diff_stage = (include_diff_cfg_stage and (diff_w_stage != 0.0))

                                x_for_ops_stage = sanitize_state_for_ops(x_state_abs, cfg, rho_floor=1e-6, E_floor=1e-6)

                                if use_mls:
                                    r_adv_stage, r_diff_stage, _ = mls_advdiff_terms_abs_faceadj(
                                        x_abs=x_for_ops_stage.float(),
                                        pos=pred_centers,
                                        edge_index=pei.long(),
                                        levels=pred_levels,
                                        dx0=float(dx),
                                        dy0=float(dy),
                                        cfg=cfg,
                                        compute_adv=need_adv_stage,
                                        compute_diff=need_diff_stage,
                                    )
                                else:
                                    r_adv_stage, r_diff_stage, _ = dec.dec_advdiff_terms_abs(
                                        x_abs=x_for_ops_stage.float(),
                                        edge_index=pei.long(),
                                        pred_ea=pea.float(),
                                        levels=pred_levels.long().to(device),
                                        dx0=float(dx),
                                        dy0=float(dy),
                                        cfg=cfg,
                                        compute_adv=need_adv_stage,
                                        compute_diff=need_diff_stage,
                                    )

                                r_adv_stage = _sanitize_ops_term(r_adv_stage, cfg)
                                r_diff_stage = _sanitize_ops_term(r_diff_stage, cfg)

                                if not adv_step_gate_stage:
                                    r_adv_stage = None

                                if ch_mask_stage is None:
                                    Fdim_stage = x_state_abs.size(1)
                                    sel_adv_stage = dec.parc_select_feature_indices_adv(cfg, Fdim_stage)
                                    sel_diff_stage = dec.parc_select_feature_indices_diff(cfg, Fdim_stage)
                                    sel_stage = sorted(set(sel_adv_stage + sel_diff_stage))
                                    ch_mask_stage = torch.zeros((Fdim_stage,), device=device, dtype=torch.float32)
                                    if len(sel_stage) > 0:
                                        ch_mask_stage[torch.as_tensor(sel_stage, device=device)] = 1.0

                                if (dec_blend_w != 0.0) or (dec_resid_w != 0.0):
                                    if need_diff_stage and (r_diff_stage is not None):
                                        base_stage = diff_w_stage * r_diff_stage
                                    else:
                                        ref_stage = r_diff_stage if (r_diff_stage is not None) else r_adv_stage
                                        if ref_stage is None:
                                            ref_stage = x_state_abs.float()
                                        base_stage = torch.zeros_like(ref_stage)

                                    if need_adv_stage and (r_adv_stage is not None):
                                        base_stage = base_stage + adv_w_stage * r_adv_stage
                                    r_phy_stage = _sanitize_ops_term(base_stage, cfg)

                        x_stage_in = _build_X(norm_state, pred_centers, pred_levels, cfg, dt_hat=dt_hat)
                        if parc_use and ((r_adv_stage is not None) or (r_diff_stage is not None)):
                            ref_stage = r_diff_stage if (r_diff_stage is not None) else r_adv_stage
                            r_adv_in_stage = r_adv_stage if (r_adv_stage is not None) else torch.zeros_like(ref_stage)
                            r_diff_in_stage = r_diff_stage if (r_diff_stage is not None) else torch.zeros_like(ref_stage)

                            if ch_mask_stage is not None:
                                cm_stage = ch_mask_stage.to(device=device, dtype=torch.float32).view(1, -1)
                                r_adv_in_stage = r_adv_in_stage * cm_stage
                                r_diff_in_stage = r_diff_in_stage * cm_stage

                            parc_extra_stage = dec.parc_terms_to_node_inputs(
                                r_adv_in_stage.to(device=device, dtype=torch.float32),
                                r_diff_in_stage.to(device=device, dtype=torch.float32),
                                dt_phys=dt_phys_f32,
                                dt_ref=dt_ref_f32,
                                sigma=sigma_f32,
                                predict_type=predict_type,
                                cfg=cfg,
                                dtype=x_stage_in.dtype,
                                detach=True,
                            )
                            parc_extra_stage = _sanitize_parc_extra_tensor(parc_extra_stage, cfg, post_adapter=False)

                            if parc_extra_stage is not None and parc_extra_stage.numel() > 0:
                                if hasattr(model, "parc_adapter") and (model.parc_adapter is not None) and use_adapter:
                                    adv_active_stage = (r_adv_stage is not None)
                                    diff_active_stage = (r_diff_stage is not None)
                                    parc_extra_stage = model.parc_adapter(
                                        parc_extra_stage,
                                        update_adv_stats=bool(adv_active_stage and update_adapter_stats),
                                        update_diff_stats=bool(diff_active_stage and update_adapter_stats),
                                    )

                                    adv_all_steps_stage = bool(loss_cfg.get("adv_all_steps", True))
                                    adv_step_gate_stage = (step_k == 0) if (not adv_all_steps_stage) else True
                                    if not adv_step_gate_stage:
                                        da = int(model.parc_adapter.dim_adv)
                                        if da > 0:
                                            if da >= int(parc_extra_stage.size(1)):
                                                parc_extra_stage = torch.zeros_like(parc_extra_stage)
                                            else:
                                                parc_extra_stage = torch.cat(
                                                    [
                                                        torch.zeros_like(parc_extra_stage[:, :da]),
                                                        parc_extra_stage[:, da:],
                                                    ],
                                                    dim=1,
                                                )

                                parc_extra_stage = _sanitize_parc_extra_tensor(
                                    parc_extra_stage, cfg, post_adapter=bool(use_adapter)
                                )
                                x_stage_in = torch.cat([x_stage_in, parc_extra_stage.to(dtype=x_stage_in.dtype)], dim=1)

                        y_corr_stage = _forward_main_head_with_edge_attr(model, x_stage_in, pei, edge_attr=pea)
                        y_stage = y_corr_stage

                        if need_phy and (dec_blend_w != 0.0) and (r_phy_stage is not None):
                            with torch.autocast(device_type=device.type, enabled=False):
                                phy_units_stage = dec.physics_to_model_units(
                                    r_phy_stage.to(device=device, dtype=torch.float32),
                                    dt_phys=dt_phys_f32,
                                    dt_ref=dt_ref_f32,
                                    sigma=sigma_f32,
                                    predict_type=predict_type,
                                )
                                if ch_mask_stage is not None:
                                    phy_units_stage = phy_units_stage * ch_mask_stage.view(1, -1).to(
                                        device=device, dtype=torch.float32
                                    )
                            y_stage = y_corr_stage + dec_blend_w * phy_units_stage.to(dtype=y_corr_stage.dtype)

                        return y_stage, r_phy_stage

                    k1 = y_pred
                    k2, r_phy_k2 = _predict_rate_for_norm_state(norm_in + (0.5 * dt_hat) * k1, update_adapter_stats=False)
                    k3, r_phy_k3 = _predict_rate_for_norm_state(norm_in + (0.5 * dt_hat) * k2, update_adapter_stats=False)
                    k4, r_phy_k4 = _predict_rate_for_norm_state(norm_in + dt_hat * k3, update_adapter_stats=False)

                    y_pred_rk4 = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

                    r_phy_rk4 = None
                    if (r_phy_abs is not None) and (r_phy_k2 is not None) and (r_phy_k3 is not None) and (r_phy_k4 is not None):
                        r_phy_rk4 = (r_phy_abs + 2.0 * r_phy_k2 + 2.0 * r_phy_k3 + r_phy_k4) / 6.0

                    if rk4_alpha >= (1.0 - 1e-12):
                        y_pred = y_pred_rk4
                        if r_phy_rk4 is not None:
                            r_phy_for_loss = r_phy_rk4
                    else:
                        y_pred = (1.0 - rk4_alpha) * k1 + rk4_alpha * y_pred_rk4
                        if (r_phy_abs is not None) and (r_phy_rk4 is not None):
                            r_phy_for_loss = (1.0 - rk4_alpha) * r_phy_abs + rk4_alpha * r_phy_rk4

                    _assert_finite("y_pred_rk_sched", y_pred)

                # ----- supervision target -----
                delta_target = norm_tgt - norm_in
                rate_target = delta_target / dt_hat.clamp_min(1e-12)

                if _should_print():
                    print(f"[NAN-DBG] step={step_k} ---- rate_target checks ----")
                    _tstats("delta_target", delta_target)
                    _tstats("rate_target", rate_target)
                    _mark_printed()

                _assert_finite("rate_target", rate_target)

                watch_steps = set(cfg.get("debug", {}).get("parc_scale_watch_steps", [0, 3, 6, 9]))
                if cfg.get("debug", {}).get("parc_scale_watch", True) and (step_k in watch_steps):
                    with torch.no_grad():
                        _tstats("rate_target", rate_target)
                        _tstats("y_pred", y_pred)

                y_pred = _apply_rate_guardrails(y_pred, dt_hat, cfg)


                # ==========================================================
                # FAST NEXT DIAGNOSTIC (RUN ONCE, STEP 0 ONLY)
                #   - norm-space call: best for dt scale + channel permutation
                #   - phys-space call: best to detect normalization mismatch
                # NOTE: do NOT pass the AMP GradScaler "scaler" here.
                # ==========================================================
                center_loss = (F.huber_loss(y_pred, rate_target, delta=huber_delta)
                               if use_huber else F.mse_loss(y_pred, rate_target))

                y_pred_abs = _maybe_denorm(norm_in + y_pred * dt_hat, mu, sigma)

                # ---------------- DEBUG: oracle check ----------------
                if cfg.get("debug", {}).get("oracle_watch", False):
                    Fdim = int(x_in_abs.size(1))
                    idx = dec.infer_feature_indices(cfg, Fdim)
                    rho_idx = int(idx.get("rho", 0))
                    E_idx   = int(idx.get("E", 3))

                    # dt_hat should already be a scalar tensor; make it robust anyway
                    dt_oracle = dt_hat
                    if not torch.is_tensor(dt_oracle):
                        dt_oracle = torch.tensor(float(dt_oracle), device=device, dtype=norm_in.dtype)

                    oracle_debug_step(
                        step_k=step_k,
                        x_in_norm=norm_in,        # (N,F) normalized state at time t
                        x_tgt_norm=norm_tgt,      # (N,F) normalized teacher at time t+1
                        pred_norm=y_pred,         # (N,F) the rate you actually integrate (includes baseline if enabled)
                        dt=dt_oracle,             # scalar tensor
                        x_roll_abs=y_pred_abs,    # (N,F) absolute next state produced by integration
                        x_tgt_abs=x_tgt_abs,      # (N,F) absolute teacher next state
                        mu=mu,
                        sigma=sigma,
                        centers=(pred_centers if torch.is_tensor(pred_centers) else None),
                        levels=(pred_levels if torch.is_tensor(pred_levels) else None),
                        feature_names=("rho", "mx", "my", "E"),
                        rho_idx=rho_idx,
                        E_idx=E_idx,
                        rho_floor=1e-6,
                        E_floor=1e-6,
                        topk=int(cfg.get("debug", {}).get("oracle_topk", 10)),
                    )
                # ---------------- END DEBUG ----------------

                # ----- optional existing regularizers -----
                lap_loss = (laplacian_smoothness(y_pred, pei) if lap_w > 0 else y_pred.new_zeros(()))
                tmp_loss = (temporal_consistency(x_in, norm_tgt) if tmp_w > 0 else y_pred.new_zeros(()))

                # ----- optional physics residual loss (delta-form) -----
                phy_loss = y_pred.new_zeros(())
                if need_phy and dec_resid_w > 0.0 and (r_phy_for_loss is not None):
                    with torch.autocast(device_type=device.type, enabled=False):
                        phy_loss = dec.physics_residual_loss_delta(
                            y_pred_abs=y_pred_abs.float(),
                            x_in_abs=x_in_abs.float(),
                            dt_phys=dt_phys.float(),          # <-- correct dt for this step
                            r_phy_abs=r_phy_for_loss.float(),
                            area=area.float(),
                            #sigma=(sigma.to(device, dtype=torch.float32) if (sigma is not None and torch.is_tensor(sigma)) else None),
                            sigma=sigma_f32,
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

        if runtime_mesh_enabled:
            runtime_knn_k = int(cfg.get("train", {}).get("knn_k", cfg.get("loss", {}).get("interp_k", 8)))
            runtime_chunk = int(cfg.get("speed", {}).get("interp_chunk", 8192))

            state_centers = _to_dev_nb(centers_list[0], dtype=torch.float32)
            state_levels = _to_dev_nb(level_list[0], dtype=torch.long).view(-1)
            state_parents = _to_dev_nb(parents_list[0], dtype=torch.long).view(-1)
            state_feat = _to_dev_nb(feat_list[0], dtype=torch.float32)

            active_pred_centers = None
            active_pred_levels = None
            active_pred_parents = None
            active_pred_ei = None
            active_pred_mask = None
            active_pred_ea = None

            runtime_has_step0_precomp = (
                (pred_lists is not None)
                and (feat_t_on_pred_list is not None)
                and (feat_tp1_on_pred_list is not None)
                and (len(feat_t_on_pred_list) > 1)
                and (len(feat_tp1_on_pred_list) > 1)
                and (runtime_domain_mode != "starting_mesh")
            )

            for k in range(0, K - 1):
                step_wall_t0 = time.perf_counter()
                used_precomp_step0 = False
                do_rebuild = False
                t_rebuild_s = 0.0
                t_xin_map_s = 0.0
                t_xtgt_map_s = 0.0
                t_model_s = 0.0
                idw_dev_xin = ""
                idw_chunk_xin = -1
                n_src_xin = -1
                n_dst_xin = -1
                idw_dev_tgt = ""
                idw_chunk_tgt = -1
                n_src_tgt = -1
                n_dst_tgt = -1
                dtk = _to_scalar_dt(dt_list[k], device=device, dtype=state_feat.dtype)

                if (k == 0) and runtime_has_step0_precomp:
                    used_precomp_step0 = True
                    pred_centers_1, pred_levels_1, pred_parents_1, pred_ei_1, mask_pred_1 = _pred_mesh_for_step_strict(
                        0, pred_lists=pred_lists
                    )
                    active_pred_centers = _to_dev_nb(pred_centers_1, dtype=torch.float32)
                    active_pred_levels = _to_dev_nb(pred_levels_1, dtype=torch.long).view(-1)
                    active_pred_parents = _to_dev_nb(pred_parents_1, dtype=torch.long).view(-1)
                    active_pred_ei = _to_dev_nb(pred_ei_1, dtype=torch.long)
                    active_pred_mask = _to_dev_nb(mask_pred_1, dtype=torch.bool)
                    pred_ea_step0 = pred_ea_list[1] if (pred_ea_list is not None and len(pred_ea_list) > 1) else None
                    active_pred_ea = _to_dev_nb(pred_ea_step0, dtype=torch.float32)

                    if runtime_need_edge_attr and active_pred_ea is None:
                        active_pred_ea = dec_edge_attr_for_dyadic_quads(
                            active_pred_centers.to("cpu", dtype=torch.float32),
                            active_pred_levels.to("cpu", dtype=torch.int64),
                            active_pred_ei.to("cpu", dtype=torch.int64),
                            dx0=float(dx),
                            dy0=float(dy),
                            refine_ratio=_get_refine_ratio(cfg),
                        ).to(device=device, dtype=torch.float32)

                    x_in_abs = _to_dev_nb(feat_t_on_pred_list[1], dtype=torch.float32)
                    x_tgt_abs = _to_dev_nb(feat_tp1_on_pred_list[1], dtype=torch.float32)
                else:
                    step_idx = k + 1
                    do_rebuild = (k == 0) or ((step_idx % runtime_update_every) == 0)

                    if do_rebuild:
                        t_rebuild_t0 = time.perf_counter()
                        rebuild_timing: Dict[str, float] = {}
                        (
                            active_pred_centers,
                            active_pred_levels,
                            active_pred_parents,
                            active_pred_ei,
                            active_pred_mask,
                            active_pred_ea,
                        ) = _runtime_build_pred_mesh_from_state(
                            centers_t=state_centers,
                            feat_t=state_feat,
                            level_t=state_levels,
                            parents_t=state_parents,
                            cfg=cfg,
                            H=H,
                            W=W,
                            dx=float(dx),
                            dy=float(dy),
                            device=device,
                            wedge_path=runtime_wedge_path,
                            need_edge_attr=runtime_need_edge_attr,
                            runtime_mesh_policy=runtime_mesh_policy,
                            runtime_wedge_constraints=runtime_wedge_constraints,
                            runtime_base_mesh=runtime_base_mesh,
                            cnn_parent_mapping_mode=("from_centers" if (k == 0) else "dataset"),
                            cnn_fill_empty_interior=bool(k == 0),
                            dt_phys=dtk,
                            dt_ref=dt_ref_scalar,
                            timing_out=rebuild_timing,
                        )
                        t_rebuild_s = time.perf_counter() - t_rebuild_t0
                        batch_rt_rebuilds += 1
                        batch_rt_mesh_predict_s += float(rebuild_timing.get("mesh_predict_s", 0.0))
                        batch_rt_mesh_materialize_s += float(rebuild_timing.get("mesh_materialize_s", 0.0))
                        batch_rt_edge_attr_s += float(rebuild_timing.get("edge_attr_s", 0.0))

                        if k == 0:
                            n_src_xin = int(state_centers.shape[0])
                            n_dst_xin = int(active_pred_centers.shape[0])
                            idw_dev_xin_t, idw_chunk_xin = _select_idw_backend(
                                src_n=n_src_xin,
                                requested_chunk=int(runtime_chunk),
                                out_device=device,
                            )
                            idw_dev_xin = idw_dev_xin_t.type
                            t_xin_t0 = time.perf_counter()
                            x_in_abs = _map_gt_to_pred_mesh_once(
                                src_centers=state_centers,
                                src_feats=state_feat,
                                pred_centers=active_pred_centers,
                                knn_k=runtime_knn_k,
                                chunk=runtime_chunk,
                                knn_backend=runtime_idw_backend,
                                knn_backend_kwargs=runtime_idw_backend_kwargs,
                            )
                            t_xin_map_s = time.perf_counter() - t_xin_t0
                        else:
                            n_src_xin = int(state_centers.shape[0])
                            n_dst_xin = int(active_pred_centers.shape[0])
                            idw_dev_xin_t, idw_chunk_xin = _select_idw_backend(
                                src_n=n_src_xin,
                                requested_chunk=int(runtime_chunk),
                                out_device=device,
                            )
                            idw_dev_xin = idw_dev_xin_t.type
                            t_xin_t0 = time.perf_counter()
                            x_in_abs = _map_pred_to_next_pred(
                                pred_centers_src=state_centers,
                                feats_src=state_feat,
                                levels_src=state_levels,
                                parents_src=state_parents,
                                pred_centers_dst=active_pred_centers,
                                levels_dst=active_pred_levels,
                                parents_dst=active_pred_parents,
                                mask_pred_dst=active_pred_mask.view(-1),
                                H=H,
                                W=W,
                                knn_k=runtime_knn_k,
                                chunk=runtime_chunk,
                                bbox=runtime_bbox,
                                refine_ratio=runtime_refine_ratio,
                                knn_backend=runtime_idw_backend,
                                knn_backend_kwargs=runtime_idw_backend_kwargs,
                            )
                            t_xin_map_s = time.perf_counter() - t_xin_t0
                    else:
                        if active_pred_centers is None:
                            raise RuntimeError("Runtime mesh state is not initialized before a reuse step.")
                        x_in_abs = state_feat

                    gt_centers_tp1 = _to_dev_nb(centers_list[k + 1], dtype=torch.float32)
                    gt_feat_tp1 = _to_dev_nb(feat_list[k + 1], dtype=torch.float32)
                    n_src_tgt = int(gt_centers_tp1.shape[0])
                    n_dst_tgt = int(active_pred_centers.shape[0])
                    idw_dev_tgt_t, idw_chunk_tgt = _select_idw_backend(
                        src_n=n_src_tgt,
                        requested_chunk=int(runtime_chunk),
                        out_device=device,
                    )
                    idw_dev_tgt = idw_dev_tgt_t.type
                    t_xtgt_t0 = time.perf_counter()
                    x_tgt_abs = _map_gt_to_pred_mesh_once(
                        src_centers=gt_centers_tp1,
                        src_feats=gt_feat_tp1,
                        pred_centers=active_pred_centers,
                        knn_k=runtime_knn_k,
                        chunk=runtime_chunk,
                        knn_backend=runtime_idw_backend,
                        knn_backend_kwargs=runtime_idw_backend_kwargs,
                    )
                    t_xtgt_map_s = time.perf_counter() - t_xtgt_t0

                batch_rt_idw_remap_s += float(t_xin_map_s + t_xtgt_map_s)

                dtk = dtk.to(device=device, dtype=x_in_abs.dtype, non_blocking=True)

                t_model_t0 = time.perf_counter()
                if batch_dbg:
                    print(
                        f"[HANG-DBG] train entering _run_step: step={k} "
                        f"N={int(x_in_abs.shape[0]) if torch.is_tensor(x_in_abs) else -1} "
                        f"E={int(active_pred_ei.shape[1]) if (torch.is_tensor(active_pred_ei) and active_pred_ei.ndim == 2) else -1}",
                        flush=True,
                    )
                loss_k, y_pred_abs_k, _ = _run_step(
                    step_k=k,
                    pred_centers=active_pred_centers,
                    pred_levels=active_pred_levels,
                    pred_parents=active_pred_parents,
                    pred_ei=active_pred_ei,
                    pred_ea=active_pred_ea,
                    x_in_abs=x_in_abs,
                    x_tgt_abs=x_tgt_abs,
                    dt_phys=dtk,
                    dt_ref_scalar=dt_ref_scalar,
                    x_ops_abs=None,
                )
                if batch_dbg:
                    print(f"[HANG-DBG] train _run_step returned: step={k}", flush=True)
                t_model_s = time.perf_counter() - t_model_t0

                window_loss += float(loss_k.detach().cpu())
                step_mae = float(torch.mean(torch.abs(y_pred_abs_k.detach() - x_tgt_abs)).cpu())
                window_mae += step_mae
                n_steps += 1
                window_loss_graph = loss_k if window_loss_graph is None else (window_loss_graph + loss_k)

                state_feat = _enforce_physical_state(y_pred_abs_k, cfg, rho_floor=1e-6, E_floor=1e-6)
                _record_step_diagnostics(
                    pred_raw_abs=y_pred_abs_k,
                    pred_used_abs=state_feat,
                    gt_abs=x_tgt_abs,
                    pred_levels=active_pred_levels,
                    pred_centers=active_pred_centers,
                    pred_ei=active_pred_ei,
                )
                state_centers = active_pred_centers
                state_levels = active_pred_levels
                state_parents = active_pred_parents

                step_t_abs = -1
                if t_indices is not None:
                    step_t_abs = int(t_indices[k + 1].item()) if torch.is_tensor(t_indices) else int(t_indices[k + 1])
                mem_alloc_mb, mem_reserved_mb = _runtime_memory_snapshot_mb(device)
                _runtime_step_log_write(
                    cfg,
                    {
                        "wall_time": datetime.datetime.now().isoformat(timespec="seconds"),
                        "split": "train",
                        "epoch": (int(epoch_idx) if epoch_idx is not None else -1),
                        "batch_idx": int(batch_idx),
                        "step_k": int(k),
                        "t_abs": int(step_t_abs),
                        "used_precomp_step0": int(used_precomp_step0),
                        "do_rebuild": int(do_rebuild),
                        "update_every_steps": int(runtime_update_every),
                        "device": str(device),
                        "n_state": int(x_in_abs.shape[0]) if torch.is_tensor(x_in_abs) else -1,
                        "n_pred": int(active_pred_centers.shape[0]) if torch.is_tensor(active_pred_centers) else -1,
                        "e_pred": int(active_pred_ei.shape[1]) if (torch.is_tensor(active_pred_ei) and active_pred_ei.ndim == 2) else -1,
                        "n_gt_src": int(centers_list[k + 1].shape[0]) if torch.is_tensor(centers_list[k + 1]) else -1,
                        "idw_dev_xin": idw_dev_xin,
                        "idw_chunk_xin": int(idw_chunk_xin),
                        "n_src_xin": int(n_src_xin),
                        "n_dst_xin": int(n_dst_xin),
                        "idw_dev_tgt": idw_dev_tgt,
                        "idw_chunk_tgt": int(idw_chunk_tgt),
                        "n_src_tgt": int(n_src_tgt),
                        "n_dst_tgt": int(n_dst_tgt),
                        "t_rebuild_s": float(t_rebuild_s),
                        "t_xin_map_s": float(t_xin_map_s),
                        "t_xtgt_map_s": float(t_xtgt_map_s),
                        "t_model_s": float(t_model_s),
                        "t_step_total_s": float(time.perf_counter() - step_wall_t0),
                        "loss": float(loss_k.detach().cpu()),
                        "mae": float(step_mae),
                        "mem_alloc_mb": float(mem_alloc_mb),
                        "mem_reserved_mb": float(mem_reserved_mb),
                    },
                )

            # ===== backward + step once per window =====
            if window_loss_graph is not None:
                if batch_dbg:
                    print("[HANG-DBG] starting backward+step", flush=True)
                t_bw0 = time.perf_counter() if batch_dbg else None
                if scaler is not None:
                    scaler.scale(window_loss_graph).backward()
                    if batch_dbg:
                        print(f"[HANG-DBG] backward done (scaled) in {time.perf_counter() - t_bw0:.3f}s", flush=True)
                    t_clip0 = time.perf_counter() if batch_dbg else None
                    scaler.unscale_(opt)  # <-- required before clipping
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if batch_dbg:
                        print(f"[HANG-DBG] unscale+clip done in {time.perf_counter() - t_clip0:.3f}s", flush=True)
                    t_step0 = time.perf_counter() if batch_dbg else None
                    scaler.step(opt)
                    scaler.update()
                    if batch_dbg:
                        print(f"[HANG-DBG] optimizer step+update done in {time.perf_counter() - t_step0:.3f}s", flush=True)
                else:
                    window_loss_graph.backward()
                    if batch_dbg:
                        print(f"[HANG-DBG] backward done in {time.perf_counter() - t_bw0:.3f}s", flush=True)
                    t_clip0 = time.perf_counter() if batch_dbg else None
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if batch_dbg:
                        print(f"[HANG-DBG] clip done in {time.perf_counter() - t_clip0:.3f}s", flush=True)
                    t_step0 = time.perf_counter() if batch_dbg else None
                    opt.step()
                    if batch_dbg:
                        print(f"[HANG-DBG] optimizer step done in {time.perf_counter() - t_step0:.3f}s", flush=True)
            else:
                opt.step()
                if batch_dbg:
                    print("[HANG-DBG] no graph loss; optimizer step only", flush=True)

            total_loss_accum += window_loss
            mae_accum += window_mae
            batch_wall = float(time.perf_counter() - batch_t0)
            _maybe_print_batch_progress(batch_idx, batch_wall)
            _maybe_print_runtime_mesh_batch_breakdown(
                batch_idx,
                batch_wall,
                t_mesh_predict_s=batch_rt_mesh_predict_s,
                t_mesh_materialize_s=batch_rt_mesh_materialize_s,
                t_edge_attr_s=batch_rt_edge_attr_s,
                t_idw_remap_s=batch_rt_idw_remap_s,
                n_rebuilds=batch_rt_rebuilds,
            )
            if batch_dbg:
                print(
                    f"[HANG-DBG] batch complete: idx={batch_idx} "
                    f"wall={batch_wall:.3f}s",
                    flush=True,
                )
            continue

        # ======================
        # STEP 0 (k=0)
        # ======================
        k = 0
        pred_centers_1, pred_levels_1, pred_parents_1, pred_ei_1, mask_pred_1 = _pred_mesh_for_step_strict(
            k, pred_lists=pred_lists
        )
        pred_ea_1 = pred_ea_list[k + 1] if (pred_ea_list is not None) else None

        # --- Option A: move ONLY step-0 tensors to GPU here ---
        pred_centers_1 = _to_dev_nb(pred_centers_1)
        pred_levels_1  = _to_dev_nb(pred_levels_1)
        pred_parents_1 = _to_dev_nb(pred_parents_1)
        pred_ei_1      = _to_dev_nb(pred_ei_1)
        mask_pred_1    = _to_dev_nb(mask_pred_1)
        pred_ea_1      = _to_dev_nb(pred_ea_1)

        x_in_abs0  = _to_dev_nb(feat_t_on_pred_list[k + 1])
        x_tgt_abs0 = _to_dev_nb(feat_tp1_on_pred_list[k + 1])

        dt0 = dt_list[0]
        if torch.is_tensor(dt0):
            dt0 = dt0.to(device=device, dtype=x_in_abs0.dtype, non_blocking=True)
        else:
            dt0 = torch.tensor(float(dt0), device=device, dtype=x_in_abs0.dtype)

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

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # [STEP-C] DRIFT WATCH (place HERE for step 0, before _run_step)
        if cfg.get("debug", {}).get("drift_watch", False):
            x_teacher = x_in_abs0.float()
            x_roll    = x_in_abs0.float()  # for step0 rollout == teacher by construction
            drift = (x_roll - x_teacher).abs()

            drift_mean = drift.mean(dim=0)
            drift_max  = drift.max(dim=0).values
            print(f"[DRIFT-C] step_k=0 (pre-run) mean={drift_mean.tolist()} max={drift_max.tolist()}")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ---- DEBUG: normalization check on real training tensors (prints once) ----
        if cfg.get("debug", {}).get("print_norm_sanity_once", True) and (not hasattr(train_one_epoch_multi_step, "_printed_norm_sanity")):
            train_one_epoch_multi_step._printed_norm_sanity = True

            print("\n[NORM-SANITY] step0 tensors on pred mesh (k+1)")
            _tstats("x_in_abs0", x_in_abs0)
            _tstats("x_tgt_abs0", x_tgt_abs0)

            norm_in0  = _maybe_norm(x_in_abs0,  mu, sigma)
            norm_tgt0 = _maybe_norm(x_tgt_abs0, mu, sigma)

            _tstats("norm_in0", norm_in0)
            _tstats("norm_tgt0", norm_tgt0)

            if mu is None or sigma is None:
                print("[NORM-SANITY] mu/sigma are None (normalization disabled).")
            else:
                _tstats("mu", mu)
                _tstats("sigma", sigma)

                # check how close to 0 mean / unit scale the normalized values look (rough)
                try:
                    m = norm_in0.mean(dim=0).detach().cpu()
                    s = norm_in0.std(dim=0, unbiased=False).detach().cpu()
                    print("[NORM-SANITY] norm_in0 per-channel mean:", m.numpy())
                    print("[NORM-SANITY] norm_in0 per-channel std :", s.numpy())
                except Exception as e:
                    print("[NORM-SANITY][WARN] couldn't compute per-channel mean/std:", repr(e))

        window_loss += float(loss0.detach().cpu())
        window_mae  += float(torch.mean(torch.abs(y_pred_abs0.detach() - x_tgt_abs0)).cpu())
        n_steps += 1
        window_loss_graph = loss0 if window_loss_graph is None else (window_loss_graph + loss0)

        # chain absolute prediction forward
        #pred_feats_k = y_pred_abs0
        #pred_feats_k = _enforce_physical_state(y_pred_abs0, rho_floor=1e-6, E_floor=1e-6)
        #pred_feats_k = _enforce_physical_state_with_diag(
        #    y_pred_abs0, tag="rollout_chain", step_k=0, rho_floor=1e-6, E_floor=1e-6
        #)
        pred_feats_k = _enforce_physical_state(y_pred_abs0, cfg, rho_floor=1e-6, E_floor=1e-6)
        _record_step_diagnostics(
            pred_raw_abs=y_pred_abs0,
            pred_used_abs=pred_feats_k,
            gt_abs=x_tgt_abs0,
            pred_levels=pred_levels_1,
            pred_centers=pred_centers_1,
            pred_ei=pred_ei_1,
        )

        # ======================
        # STEPS 1..K-2
        # ======================
        for k in range(1, K - 1):
            #print(f"\nStep {k}")
            pred_centers_next, pred_levels_next, pred_parents_next, pred_ei_next, mask_pred_next = _pred_mesh_for_step_strict(k, pred_lists=pred_lists)
            pred_ea_next = pred_ea_list[k + 1] if (pred_ea_list is not None) else None

            # map pred(k)->pred(k+1)
            #idx_km1 = pred2pred_idx_list[k-1].to(device)
            #w_km1   = pred2pred_w_list[k-1].to(device)
            idx_km1 = pred2pred_idx_list[k - 1].to(device=device, non_blocking=True)
            w_km1   = pred2pred_w_list[k - 1].to(device=device, non_blocking=True)

            x_in_abs = apply_precomputed_idw_map(idx_km1, w_km1, pred_feats_k).to(device)

            # teacher state on the SAME mesh (pred mesh at j=k+1)
            #x_in_teacher = feat_t_on_pred_list[k + 1].to(device)
            x_in_teacher = _to_dev_nb(feat_t_on_pred_list[k + 1])

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # [STEP-C] DRIFT WATCH (place HERE, right after x_in_abs is defined)
            if cfg.get("debug", {}).get("drift_watch", False):
                drift = (x_in_abs.float() - x_in_teacher.float()).abs()  # (N,F)

                drift_mean = drift.mean(dim=0)
                drift_p95 = torch.quantile(drift.detach().to("cpu"), 0.95, dim=0).to(drift.device)
                drift_max  = drift.max(dim=0).values

                print(f"[DRIFT-C] step_k={k} (pre-run) mean={drift_mean.tolist()} "
                    f"p95={drift_p95.tolist()} max={drift_max.tolist()}")

                # optional top-k worst nodes
                node_score = drift.sum(dim=1)
                topk = min(10, node_score.numel())
                vals, idxs = torch.topk(node_score, k=topk, largest=True)
                print(f"[DRIFT-C] step_k={k} top-{topk} nodes by L1 drift:")
                for r in range(topk):
                    ii = int(idxs[r].item())
                    v  = float(vals[r].item())
                    xy = pred_centers_next[ii].tolist() if torch.is_tensor(pred_centers_next) else None
                    lv = int(pred_levels_next[ii].item()) if torch.is_tensor(pred_levels_next) else None
                    print(f"  node={ii:6d} score={v:.3e} xy={xy} level={lv}")

                    if torch.is_tensor(x_in_abs) and torch.is_tensor(x_in_teacher):
                        d = (x_in_abs[ii] - x_in_teacher[ii]).detach().to("cpu")
                        xs = x_in_abs[ii].detach().to("cpu")
                        xt = x_in_teacher[ii].detach().to("cpu")

                        # channels assumed [rho, mx, my, E]
                        print(f"    drift[rho,mx,my,E]={d.tolist()}")
                        print(f"    roll [rho,mx,my,E]={xs.tolist()}")
                        print(f"    teach[rho,mx,my,E]={xt.tolist()}")

                        # velocity diagnostic (avoid div0)
                        rho_r = float(xs[0])
                        rho_t = float(xt[0])
                        mx_r, my_r = float(xs[1]), float(xs[2])
                        mx_t, my_t = float(xt[1]), float(xt[2])

                        urx = mx_r / max(abs(rho_r), 1e-12)
                        ury = my_r / max(abs(rho_r), 1e-12)
                        utx = mx_t / max(abs(rho_t), 1e-12)
                        uty = my_t / max(abs(rho_t), 1e-12)

                        print(f"    u_roll=[{urx:.3e},{ury:.3e}]  u_teach=[{utx:.3e},{uty:.3e}]")
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            #x_tgt_abs = feat_tp1_on_pred_list[k + 1].to(device)
            x_tgt_abs = _to_dev_nb(feat_tp1_on_pred_list[k + 1])

            #dtk = dt_list[k]
            #dtk = dtk.to(device=device, dtype=x_in_abs.dtype) if torch.is_tensor(dtk) else torch.tensor(float(dtk), device=device, dtype=x_in_abs.dtype)
            dtk = dt_list[k]
            if torch.is_tensor(dtk):
                dtk = dtk.to(device=device, dtype=x_in_abs.dtype, non_blocking=True)
            else:
                dtk = torch.tensor(float(dtk), device=device, dtype=x_in_abs.dtype)

            use_ops_teacher = bool(cfg.get("debug", {}).get("teacher_force_ops", False))

            # --- Move ONLY this step's mesh tensors to GPU here ---
            pred_centers_next = _to_dev_nb(pred_centers_next)
            pred_levels_next  = _to_dev_nb(pred_levels_next)
            pred_parents_next = _to_dev_nb(pred_parents_next)
            pred_ei_next      = _to_dev_nb(pred_ei_next)
            mask_pred_next    = _to_dev_nb(mask_pred_next)
            pred_ea_next      = _to_dev_nb(pred_ea_next)

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
                x_ops_abs=(x_in_teacher if (use_ops_teacher and k > 0) else None),
            )

            window_loss += float(loss_k.detach().cpu())
            window_mae  += float(torch.mean(torch.abs(y_pred_abs_k.detach() - x_tgt_abs)).cpu())
            n_steps += 1
            window_loss_graph = window_loss_graph + loss_k

            #pred_feats_k = y_pred_abs_k
            #pred_feats_k = _enforce_physical_state(y_pred_abs_k, rho_floor=1e-6, E_floor=1e-6)
            #pred_feats_k = _enforce_physical_state_with_diag(
            #    y_pred_abs_k, tag="rollout_chain", step_k=k, rho_floor=1e-6, E_floor=1e-6
            #)
            pred_feats_k = _enforce_physical_state(y_pred_abs_k, cfg, rho_floor=1e-6, E_floor=1e-6)
            _record_step_diagnostics(
                pred_raw_abs=y_pred_abs_k,
                pred_used_abs=pred_feats_k,
                gt_abs=x_tgt_abs,
                pred_levels=pred_levels_next,
                pred_centers=pred_centers_next,
                pred_ei=pred_ei_next,
            )

        # ===== backward + step once per window =====
        if window_loss_graph is not None:
            if scaler is not None:
                scaler.scale(window_loss_graph).backward()
                scaler.unscale_(opt)  # <-- required before clipping
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                window_loss_graph.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
        else:
            opt.step()

        total_loss_accum += window_loss
        mae_accum        += window_mae
        _maybe_print_batch_progress(batch_idx, float(time.perf_counter() - batch_t0))

    denom = max(n_steps, 1)
    stats = {
        "num_windows": len(loader),
        "num_steps": n_steps,
        **_finalize_diagnostics_accumulator(diagnostics_accum, diag_cfg),
    }
    return total_loss_accum / denom, mae_accum / denom, stats

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
    collect_example_edges: bool = False,
    budget_csv_path: str | None = None,
    write_budgets: bool = False,
    epoch_idx: int | None = None,
    runtime_mesh_policy: Dict[str, Any] | None = None,
    infer_only: bool = False,
):

    model.eval()

    idx_map = None

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
    use_huber = bool(cfg["loss"].get("use_huber", True))

    predict_type = str(cfg.get("model", {}).get("predict_type", "rate")).lower()
    if predict_type != "rate":
        raise RuntimeError(f"PARC/Variant-B implementation below assumes predict_type='rate', got '{predict_type}'")
    time_integrator = _resolve_time_integrator(cfg)
    _time_integrator_eff, rk4_alpha = _resolve_time_integrator_for_epoch(
        cfg, epoch_idx=epoch_idx, base_integrator=time_integrator
    )

    loss_cfg = cfg.get("loss", {}) or {}
    dec_use = bool(loss_cfg.get("dec", False))
    parc_use = bool(loss_cfg.get("parc", False) or loss_cfg.get("parc_inputs", False))
    use_adapter = bool(loss_cfg.get("parc_use_adapter", False))

    dec_blend_w = float(loss_cfg.get("blend_weight", 0.0))
    dec_resid_w = float(loss_cfg.get("residual_weight", 0.0))

    need_phy = parc_use or (dec_use and (dec_blend_w != 0.0 or dec_resid_w != 0.0))

    backend = str(loss_cfg.get("physics_backend", "dec")).lower()
    use_mls = (backend in ("mls", "moving_least_squares", "moving-least-squares"))

    amp_ctx = ((lambda: torch.amp.autocast(device_type="cuda", enabled=True))
               if use_amp and device.type == "cuda"
               else (lambda: torch.autocast("cpu", enabled=False)))

    runtime_mesh_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    runtime_mesh_enabled = bool(runtime_mesh_cfg.get("enabled", False))
    runtime_infer_only = bool(infer_only)
    runtime_update_every = int(runtime_mesh_cfg.get("update_every_steps", 1))
    runtime_mesh_backend = _runtime_mesh_backend(cfg)
    runtime_domain_mode = _runtime_mesh_domain_mode(cfg)
    runtime_idw_backend = "exact"
    runtime_idw_backend_kwargs: Dict[str, Any] = {}
    runtime_bbox = _get_bbox(cfg)
    runtime_refine_ratio = _get_refine_ratio(cfg)
    runtime_need_edge_attr = bool(need_phy and (not use_mls))
    runtime_wedge_path = None
    runtime_wedge_constraints = None
    runtime_base_mesh = None
    if runtime_infer_only and (not runtime_mesh_enabled):
        print(
            "[RUNTIME-MESH][WARN] infer_only requested, but runtime_mesh.enabled=false; running full eval mode.",
            flush=True,
        )
        runtime_infer_only = False
    if runtime_mesh_enabled:
        mesh_spec_path = cfg.get("mesh", {}).get("starting_mesh_path", None)
        if not mesh_spec_path:
            raise RuntimeError("Runtime mesh mode requires cfg['mesh']['starting_mesh_path'].")
        if runtime_domain_mode == "starting_mesh":
            runtime_base_mesh = _get_runtime_starting_mesh_base(
                cfg=cfg,
                H=H,
                W=W,
                dx=float(dx),
                dy=float(dy),
                device=device,
                mesh_spec_path=str(mesh_spec_path),
            )
        else:
            runtime_wedge_path = _load_wedge_path_from_spec(str(mesh_spec_path), cfg)
            runtime_wedge_constraints = _get_runtime_wedge_constraints(
                cfg=cfg,
                H=H,
                W=W,
                device=device,
                wedge_path=runtime_wedge_path,
            )
        runtime_idw_backend, runtime_idw_backend_kwargs = _runtime_idw_backend_settings(
            cfg,
            out_device=device,
        )
        if runtime_mesh_backend == "cnn" and runtime_mesh_policy is None:
            raise RuntimeError(
                "Runtime mesh backend is 'cnn' but runtime_mesh_policy was not loaded."
            )
        print(
            f"[RUNTIME-MESH] eval loop active (backend={runtime_mesh_backend}, domain_mode={runtime_domain_mode}, "
            f"idw_backend={runtime_idw_backend}, update_every_steps={runtime_update_every}, "
            f"infer_only={int(runtime_infer_only)})"
        )
        if runtime_infer_only:
            print(
                "[RUNTIME-MESH] infer_only enabled: skipping GT->pred remap and target-based eval metrics/loss.",
                flush=True,
            )

    # ---- metric accumulators (weighted by cell area) ----
    step_wsum = []
    step_mae_num = []
    step_mse_num = []
    step_gt2_num = []
    by_t = {}

    total_loss_accum = 0.0
    n_steps_total = 0
    examples = [] if collect_examples else None
    diag_cfg = _resolve_diagnostics_cfg(cfg)
    diagnostics_accum = _new_diagnostics_accumulator(diag_cfg)

    # ---- budgets (CSV) ----
    budget_rows = []
    budget_cfg = cfg.get("eval", {}) or {}

    # Where to save:
    # - In training runs you probably use train.save_dir
    # - In rollout script you can override via cfg["eval"]["budgets_csv_path"]
    budgets_csv_path = budget_cfg.get("budgets_csv_path", None)
    if budgets_csv_path is None:
        save_dir = cfg.get("train", {}).get("save_dir", ".")
        budgets_csv_path = os.path.join(save_dir, "rollout_budgets.csv")

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
        w = dec.cell_area_from_levels(
            pred_levels,
            dx0=float(dx),
            dy0=float(dy),
            dtype=dtype,
            device=dev,
            refine_ratio=_get_refine_ratio(cfg),
        )  # [N]
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

    def _record_step_diagnostics(
        *,
        pred_raw_abs: torch.Tensor,
        pred_used_abs: torch.Tensor,
        gt_abs: torch.Tensor,
        pred_levels: torch.Tensor,
        pred_centers: torch.Tensor,
        pred_ei: torch.Tensor,
    ) -> None:
        _record_enabled_step_diagnostics(
            diagnostics_accum,
            pred_raw_abs=pred_raw_abs.detach(),
            pred_used_abs=pred_used_abs.detach(),
            gt_abs=gt_abs.detach(),
            pred_levels=pred_levels.detach(),
            pred_centers=pred_centers.detach(),
            pred_ei=pred_ei.detach(),
            cfg=cfg,
            diag_cfg=diag_cfg,
            dx=float(dx),
            dy=float(dy),
        )

    def _append_example_step(
        *,
        step_idx: int,
        pred_centers,
        pred_levels,
        pred_parents,
        pred_ei,
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
        ex_rec = {
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
        }
        if collect_example_edges and (pred_ei is not None):
            ex_rec["pred_ei"] = pred_ei.detach().cpu()
        examples.append(ex_rec)

    def _budget_integrals(x_abs: torch.Tensor, pred_levels: torch.Tensor):
        """
        x_abs: [N,F] absolute units on pred mesh at t+1
        pred_levels: [N] levels for pred mesh (for area weights)
        Returns dict of global integrals.
        """
        nonlocal idx_map
        if x_abs.ndim != 2:
            raise RuntimeError(f"x_abs must be [N,F], got {tuple(x_abs.shape)}")
        N, Fdim = x_abs.shape
        if idx_map is None:
            idx_map = dec.infer_feature_indices(cfg, Fdim)  # expects keys like rho,mx,my,E

        # area weights on pred mesh
        w = dec.cell_area_from_levels(
            pred_levels, dx0=float(dx), dy0=float(dy),
            dtype=x_abs.dtype, device=x_abs.device,
            refine_ratio=_get_refine_ratio(cfg),
        )  # [N]

        rho = x_abs[:, idx_map["rho"]]
        mx  = x_abs[:, idx_map["mx"]]
        my  = x_abs[:, idx_map["my"]]
        E   = x_abs[:, idx_map["E"]]

        # global integrals over domain (pred mesh)
        out = {
            "mass":   (w * rho).sum(),
            "mom_x":  (w * mx ).sum(),
            "mom_y":  (w * my ).sum(),
            "energy": (w * E  ).sum(),
            "area":   w.sum(),
        }
        # move to python floats
        return {k: float(v.detach().cpu().item()) for k, v in out.items()}

    def _append_budget_row(*, t_abs: int, step_k: int, kind: str, x_abs: torch.Tensor, pred_levels: torch.Tensor):
        d = _budget_integrals(x_abs, pred_levels)
        budget_rows.append({
            "t_abs": int(t_abs),
            "step_k": int(step_k),
            "kind": str(kind),  # "gt" or "pred"
            **d,
        })


    # ==========================================================
    # Main eval loop
    # ==========================================================

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            dt_list = batch.get("dt_list", None)
            if dt_list is None:
                raise RuntimeError("Missing dt_list in batch. Ensure the active collate attaches dt_list.")

            centers_list  = _require_list(batch, "centers_list")
            feat_list     = _require_list(batch, "feat_list")
            level_list    = _require_list(batch, "level_list")
            parents_list  = _require_list(batch, "parents_list")

            pred_centers_list = batch.get("pred_centers_list", None)
            pred_levels_list = batch.get("pred_levels_list", None)
            pred_parents_list = batch.get("pred_parents_list", None)
            pred_ei_list = batch.get("pred_ei_list", None)
            mask_pred_list = batch.get("mask_pred_list", None)
            pred_ea_list = batch.get("pred_ea_list", None) or batch.get("pred_edge_attr_list", None)

            feat_t_on_pred_list = batch.get("feat_t_on_pred_list", None)
            feat_tp1_on_pred_list = batch.get("feat_tp1_on_pred_list", None)

            pred2pred_idx_list = batch.get("pred2pred_idx_list", None)
            pred2pred_w_list   = batch.get("pred2pred_w_list", None)

            t_indices = batch.get("t_indices", None)

            K = len(centers_list)
            if K < 2:
                raise RuntimeError("window_size must be ≥ 2")

            dt_ref_scalar = batch.get("dt_ref", None)

            has_precomp_lists = (
                (pred_centers_list is not None)
                and (pred_levels_list is not None)
                and (pred_parents_list is not None)
                and (pred_ei_list is not None)
                and (mask_pred_list is not None)
                and (feat_t_on_pred_list is not None)
                and (feat_tp1_on_pred_list is not None)
            )

            pred_lists = None
            if (not runtime_mesh_enabled) or has_precomp_lists:
                pred_centers_list = _require_list(batch, "pred_centers_list")
                pred_levels_list = _require_list(batch, "pred_levels_list")
                pred_parents_list = _require_list(batch, "pred_parents_list")
                pred_ei_list = _require_list(batch, "pred_ei_list")
                mask_pred_list = _require_list(batch, "mask_pred_list")
                feat_t_on_pred_list = _require_list(batch, "feat_t_on_pred_list")
                feat_tp1_on_pred_list = _require_list(batch, "feat_tp1_on_pred_list")

                if (not runtime_mesh_enabled) and need_phy and pred_ea_list is None:
                    raise RuntimeError(
                        "PARC/DEC enabled but batch is missing pred_ea_list / pred_edge_attr_list."
                    )

                pred_lists = (pred_centers_list, pred_levels_list, pred_parents_list, pred_ei_list, mask_pred_list)

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
                x_tgt_abs: torch.Tensor | None,
                dt_phys,
            ):
                norm_in  = _maybe_norm(x_in_abs,  mu, sigma)
                norm_tgt = _maybe_norm(x_tgt_abs, mu, sigma) if x_tgt_abs is not None else None

                dt_ref_t = (torch.tensor(float(dt_ref_scalar), device=device, dtype=norm_in.dtype)
                            if dt_ref_scalar is not None else None)
                dt_hat = (dt_phys / dt_ref_t) if dt_ref_t is not None else dt_phys

                sigma_f32 = _as_stat_tensor(sigma, device=device, dtype=torch.float32)
                if sigma_f32 is not None:
                    sigma_f32 = sigma_f32.clamp_min(1e-12)

                dt_phys_f32 = dt_phys.to(device=device, dtype=torch.float32)
                dt_ref_f32  = (dt_ref_t.to(device=device, dtype=torch.float32) if dt_ref_t is not None else None)

                with amp_ctx():
                    pei = pred_ei.to(device) if torch.is_tensor(pred_ei) else pred_ei
                    pea = pred_ea.to(device) if (pred_ea is not None and torch.is_tensor(pred_ea)) else pred_ea

                    r_adv_abs = r_diff_abs = r_phy_abs = area = None
                    ch_mask = None

                    if need_phy:
                        with torch.autocast(device_type=device.type, enabled=False):

                            # Check whether advection should be applied at all steps or not
                            adv_all_steps  = bool(loss_cfg.get("adv_all_steps", True))
                            if not adv_all_steps:
                                adv_step_gate = (step_k == 0)
                            else:
                                adv_step_gate = True

                            adv_w  = float(loss_cfg.get("adv_weight", 1.0))
                            diff_w = float(loss_cfg.get("diff_weight", 1.0))

                            include_adv_cfg  = bool(loss_cfg.get("parc_include_adv", True))
                            include_diff_cfg = bool(loss_cfg.get("parc_include_diff", True))

                            # Compute flags for DEC evaluation
                            need_adv  = (adv_step_gate and include_adv_cfg and (adv_w != 0.0))
                            need_diff = (include_diff_cfg and (diff_w != 0.0))

                            # If you have the sanitize helper in eval too, use it (recommended)
                            x_for_ops = sanitize_state_for_ops(x_in_abs, cfg, rho_floor=1e-6, E_floor=1e-6)
                            #x_for_ops = x_in_abs

                            if use_mls:
                                # MLS physics terms computed on the SAME face-adj edges as DEC uses
                                r_adv_abs, r_diff_abs, area = mls_advdiff_terms_abs_faceadj(
                                    x_abs=x_for_ops.float(),
                                    pos=pred_centers.to(device=device, dtype=torch.float32),  # (N,2)
                                    edge_index=pei.long(),                                   # (2,E) face-adj
                                    levels=pred_levels.to(device=device),                    # (N,)
                                    dx0=float(dx),
                                    dy0=float(dy),
                                    cfg=cfg,
                                    compute_adv=need_adv,
                                    compute_diff=need_diff,
                                )
                            else:
                                r_adv_abs, r_diff_abs, area = dec.dec_advdiff_terms_abs(
                                    x_abs=x_for_ops.float(),
                                    edge_index=pei.long(),
                                    pred_ea=pea.float(),
                                    levels=pred_levels.long().to(device),
                                    dx0=float(dx),
                                    dy0=float(dy),
                                    cfg=cfg,
                                    compute_adv=need_adv,
                                    compute_diff=need_diff,
                                )

                            r_adv_abs = _sanitize_ops_term(r_adv_abs, cfg)
                            r_diff_abs = _sanitize_ops_term(r_diff_abs, cfg)
                            area = _sanitize_float_tensor(area, clip_abs=0.0, fill_value=1.0, nonneg=True)

                            # Belt-and-suspenders: ensure adv absent for k>0
                            if not adv_step_gate:
                                r_adv_abs = None

                            # Build channel mask from the SAME selector used for PARC inputs
                            #sel = dec.parc_select_feature_indices(cfg, x_in_abs.size(1))
                            #ch_mask = torch.zeros((x_in_abs.size(1),), device=device, dtype=torch.float32)
                            #ch_mask[torch.as_tensor(sel, device=device)] = 1.0
                            Fdim = x_in_abs.size(1)
                            sel_adv  = dec.parc_select_feature_indices_adv(cfg, Fdim)
                            sel_diff = dec.parc_select_feature_indices_diff(cfg, Fdim)
                            sel = sorted(set(sel_adv + sel_diff))

                            ch_mask = torch.zeros((Fdim,), device=device, dtype=torch.float32)
                            if len(sel) > 0:
                                ch_mask[torch.as_tensor(sel, device=device)] = 1.0

                            # Build r_phy_abs safely (never 0*inf / None arithmetic)
                            r_phy_abs = None
                            if (dec_blend_w != 0.0) or (dec_resid_w != 0.0):
                                if need_diff and (r_diff_abs is not None):
                                    base = diff_w * r_diff_abs
                                else:
                                    ref = r_diff_abs if (r_diff_abs is not None) else r_adv_abs
                                    if ref is None:
                                        ref = x_in_abs.float()
                                    base = torch.zeros_like(ref)

                                if need_adv and (r_adv_abs is not None):
                                    base = base + adv_w * r_adv_abs

                                r_phy_abs = _sanitize_ops_term(base, cfg)

                    # Build node inputs
                    x_in = _build_X(norm_in, pred_centers, pred_levels, cfg, dt_hat=dt_hat)

                    parc_extra = None
                    if parc_use:
                        if (r_adv_abs is None) and (r_diff_abs is None):
                            parc_extra = None
                        else:
                            ref = r_diff_abs if (r_diff_abs is not None) else r_adv_abs
                            assert ref is not None

                            r_adv_in  = r_adv_abs  if (r_adv_abs  is not None) else torch.zeros_like(ref)
                            r_diff_in = r_diff_abs if (r_diff_abs is not None) else torch.zeros_like(ref)

                            # keep PARC channels consistent with baseline/residual
                            if ch_mask is not None:
                                cm = ch_mask.view(1, -1).to(device=device, dtype=torch.float32)
                                r_adv_in  = r_adv_in  * cm
                                r_diff_in = r_diff_in * cm

                            parc_extra = dec.parc_terms_to_node_inputs(
                                r_adv_in.to(device=device, dtype=torch.float32),
                                r_diff_in.to(device=device, dtype=torch.float32),
                                dt_phys=dt_phys_f32,
                                dt_ref=dt_ref_f32,
                                sigma=sigma_f32,
                                predict_type=predict_type,
                                cfg=cfg,
                                dtype=x_in.dtype,
                                detach=True,
                            )
                            parc_extra = _sanitize_parc_extra_tensor(parc_extra, cfg, post_adapter=False)

                        #if (parc_extra is not None) and (parc_extra.numel() > 0):
                        #    x_in = torch.cat([x_in, parc_extra.to(dtype=x_in.dtype)], dim=1)
                        if parc_extra is not None and parc_extra.numel() > 0:
                            if hasattr(model, "parc_adapter") and (model.parc_adapter is not None) and use_adapter:
                                adv_active  = (r_adv_abs is not None)   # only step0 in your gating
                                diff_active = (r_diff_abs is not None)  # typically True if enabled

                                parc_extra = model.parc_adapter(
                                    parc_extra,
                                    update_adv_stats=adv_active,
                                    update_diff_stats=diff_active,
                                )

                                # IMPORTANT: prevent normalized “fake adv” on steps where adv is gated off
                                if not adv_step_gate:
                                    da = int(model.parc_adapter.dim_adv)
                                    if da > 0:
                                        if da >= int(parc_extra.size(1)):
                                            parc_extra = torch.zeros_like(parc_extra)
                                        else:
                                            parc_extra = torch.cat(
                                                [
                                                    torch.zeros_like(parc_extra[:, :da]),
                                                    parc_extra[:, da:],
                                                ],
                                                dim=1,
                                            )

                            parc_extra = _sanitize_parc_extra_tensor(
                                parc_extra, cfg, post_adapter=bool(use_adapter)
                            )
                            x_in = torch.cat([x_in, parc_extra.to(dtype=x_in.dtype)], dim=1)


                    y_corr = _forward_main_head_with_edge_attr(model, x_in, pei, edge_attr=pea)

                    y_pred = y_corr
                    if need_phy and (dec_blend_w != 0.0) and (r_phy_abs is not None):
                        # baseline in fp32 for stability
                        with torch.autocast(device_type=device.type, enabled=False):
                            phy_units_f32 = dec.physics_to_model_units(
                                r_phy_abs.to(dtype=torch.float32),
                                dt_phys=(dt_phys.to(dtype=torch.float32) if torch.is_tensor(dt_phys) else dt_phys),
                                dt_ref=(dt_ref_t.to(dtype=torch.float32) if (dt_ref_t is not None and torch.is_tensor(dt_ref_t)) else dt_ref_t),
                                #sigma=(sigma.to(device, dtype=torch.float32) if (sigma is not None and torch.is_tensor(sigma)) else None),
                                sigma=sigma_f32,
                                predict_type=predict_type,
                            )
                            phy_units_f32 = phy_units_f32 * ch_mask.view(1, -1).to(dtype=torch.float32)
                        phy_units = phy_units_f32.to(dtype=y_corr.dtype)
                        y_pred = y_corr + dec_blend_w * phy_units

                    r_phy_for_loss = r_phy_abs

                    if rk4_alpha > 0.0:
                        def _predict_rate_for_norm_state_eval(norm_state: torch.Tensor):
                            x_state_abs = _maybe_denorm(norm_state, mu, sigma)
                            r_adv_stage = r_diff_stage = r_phy_stage = None
                            ch_mask_stage = ch_mask

                            if need_phy:
                                with torch.autocast(device_type=device.type, enabled=False):
                                    adv_all_steps_stage = bool(loss_cfg.get("adv_all_steps", True))
                                    adv_step_gate_stage = (step_k == 0) if (not adv_all_steps_stage) else True
                                    adv_w_stage = float(loss_cfg.get("adv_weight", 1.0))
                                    diff_w_stage = float(loss_cfg.get("diff_weight", 1.0))
                                    include_adv_cfg_stage = bool(loss_cfg.get("parc_include_adv", True))
                                    include_diff_cfg_stage = bool(loss_cfg.get("parc_include_diff", True))
                                    need_adv_stage = (adv_step_gate_stage and include_adv_cfg_stage and (adv_w_stage != 0.0))
                                    need_diff_stage = (include_diff_cfg_stage and (diff_w_stage != 0.0))

                                    x_for_ops_stage = sanitize_state_for_ops(x_state_abs, cfg, rho_floor=1e-6, E_floor=1e-6)

                                    if use_mls:
                                        r_adv_stage, r_diff_stage, _ = mls_advdiff_terms_abs_faceadj(
                                            x_abs=x_for_ops_stage.float(),
                                            pos=pred_centers.to(device=device, dtype=torch.float32),
                                            edge_index=pei.long(),
                                            levels=pred_levels.to(device=device),
                                            dx0=float(dx),
                                            dy0=float(dy),
                                            cfg=cfg,
                                            compute_adv=need_adv_stage,
                                            compute_diff=need_diff_stage,
                                        )
                                    else:
                                        r_adv_stage, r_diff_stage, _ = dec.dec_advdiff_terms_abs(
                                            x_abs=x_for_ops_stage.float(),
                                            edge_index=pei.long(),
                                            pred_ea=pea.float(),
                                            levels=pred_levels.long().to(device),
                                            dx0=float(dx),
                                            dy0=float(dy),
                                            cfg=cfg,
                                            compute_adv=need_adv_stage,
                                            compute_diff=need_diff_stage,
                                        )

                                    r_adv_stage = _sanitize_ops_term(r_adv_stage, cfg)
                                    r_diff_stage = _sanitize_ops_term(r_diff_stage, cfg)

                                    if not adv_step_gate_stage:
                                        r_adv_stage = None

                                    if ch_mask_stage is None:
                                        Fdim_stage = x_state_abs.size(1)
                                        sel_adv_stage = dec.parc_select_feature_indices_adv(cfg, Fdim_stage)
                                        sel_diff_stage = dec.parc_select_feature_indices_diff(cfg, Fdim_stage)
                                        sel_stage = sorted(set(sel_adv_stage + sel_diff_stage))
                                        ch_mask_stage = torch.zeros((Fdim_stage,), device=device, dtype=torch.float32)
                                        if len(sel_stage) > 0:
                                            ch_mask_stage[torch.as_tensor(sel_stage, device=device)] = 1.0

                                    if (dec_blend_w != 0.0) or (dec_resid_w != 0.0):
                                        if need_diff_stage and (r_diff_stage is not None):
                                            base_stage = diff_w_stage * r_diff_stage
                                        else:
                                            ref_stage = r_diff_stage if (r_diff_stage is not None) else r_adv_stage
                                            if ref_stage is None:
                                                ref_stage = x_state_abs.float()
                                            base_stage = torch.zeros_like(ref_stage)
                                        if need_adv_stage and (r_adv_stage is not None):
                                            base_stage = base_stage + adv_w_stage * r_adv_stage
                                        r_phy_stage = _sanitize_ops_term(base_stage, cfg)

                            x_stage_in = _build_X(norm_state, pred_centers, pred_levels, cfg, dt_hat=dt_hat)
                            if parc_use and ((r_adv_stage is not None) or (r_diff_stage is not None)):
                                ref_stage = r_diff_stage if (r_diff_stage is not None) else r_adv_stage
                                r_adv_in_stage = r_adv_stage if (r_adv_stage is not None) else torch.zeros_like(ref_stage)
                                r_diff_in_stage = r_diff_stage if (r_diff_stage is not None) else torch.zeros_like(ref_stage)

                                if ch_mask_stage is not None:
                                    cm_stage = ch_mask_stage.view(1, -1).to(device=device, dtype=torch.float32)
                                    r_adv_in_stage = r_adv_in_stage * cm_stage
                                    r_diff_in_stage = r_diff_in_stage * cm_stage

                                parc_extra_stage = dec.parc_terms_to_node_inputs(
                                    r_adv_in_stage.to(device=device, dtype=torch.float32),
                                    r_diff_in_stage.to(device=device, dtype=torch.float32),
                                    dt_phys=dt_phys_f32,
                                    dt_ref=dt_ref_f32,
                                    sigma=sigma_f32,
                                    predict_type=predict_type,
                                    cfg=cfg,
                                    dtype=x_stage_in.dtype,
                                    detach=True,
                                )
                                parc_extra_stage = _sanitize_parc_extra_tensor(
                                    parc_extra_stage, cfg, post_adapter=False
                                )
                                if parc_extra_stage is not None and parc_extra_stage.numel() > 0:
                                    if hasattr(model, "parc_adapter") and (model.parc_adapter is not None) and use_adapter:
                                        parc_extra_stage = model.parc_adapter(
                                            parc_extra_stage,
                                            update_adv_stats=False,
                                            update_diff_stats=False,
                                        )

                                        adv_all_steps_stage = bool(loss_cfg.get("adv_all_steps", True))
                                        adv_step_gate_stage = (step_k == 0) if (not adv_all_steps_stage) else True
                                        if not adv_step_gate_stage:
                                            da = int(model.parc_adapter.dim_adv)
                                            if da > 0:
                                                if da >= int(parc_extra_stage.size(1)):
                                                    parc_extra_stage = torch.zeros_like(parc_extra_stage)
                                                else:
                                                    parc_extra_stage = torch.cat(
                                                        [
                                                            torch.zeros_like(parc_extra_stage[:, :da]),
                                                            parc_extra_stage[:, da:],
                                                        ],
                                                        dim=1,
                                                    )

                                    parc_extra_stage = _sanitize_parc_extra_tensor(
                                        parc_extra_stage, cfg, post_adapter=bool(use_adapter)
                                    )
                                    x_stage_in = torch.cat([x_stage_in, parc_extra_stage.to(dtype=x_stage_in.dtype)], dim=1)

                            y_corr_stage = _forward_main_head_with_edge_attr(model, x_stage_in, pei, edge_attr=pea)
                            y_stage = y_corr_stage

                            if need_phy and (dec_blend_w != 0.0) and (r_phy_stage is not None):
                                with torch.autocast(device_type=device.type, enabled=False):
                                    phy_units_stage = dec.physics_to_model_units(
                                        r_phy_stage.to(dtype=torch.float32),
                                        dt_phys=dt_phys_f32,
                                        dt_ref=dt_ref_f32,
                                        sigma=sigma_f32,
                                        predict_type=predict_type,
                                    )
                                    if ch_mask_stage is not None:
                                        phy_units_stage = phy_units_stage * ch_mask_stage.view(1, -1).to(dtype=torch.float32)
                                y_stage = y_corr_stage + dec_blend_w * phy_units_stage.to(dtype=y_corr_stage.dtype)

                            return y_stage, r_phy_stage

                        k1 = y_pred
                        k2, r_phy_k2 = _predict_rate_for_norm_state_eval(norm_in + (0.5 * dt_hat) * k1)
                        k3, r_phy_k3 = _predict_rate_for_norm_state_eval(norm_in + (0.5 * dt_hat) * k2)
                        k4, r_phy_k4 = _predict_rate_for_norm_state_eval(norm_in + dt_hat * k3)
                        y_pred_rk4 = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

                        r_phy_rk4 = None
                        if (r_phy_abs is not None) and (r_phy_k2 is not None) and (r_phy_k3 is not None) and (r_phy_k4 is not None):
                            r_phy_rk4 = (r_phy_abs + 2.0 * r_phy_k2 + 2.0 * r_phy_k3 + r_phy_k4) / 6.0

                        if rk4_alpha >= (1.0 - 1e-12):
                            y_pred = y_pred_rk4
                            if r_phy_rk4 is not None:
                                r_phy_for_loss = r_phy_rk4
                        else:
                            y_pred = (1.0 - rk4_alpha) * k1 + rk4_alpha * y_pred_rk4
                            if (r_phy_abs is not None) and (r_phy_rk4 is not None):
                                r_phy_for_loss = (1.0 - rk4_alpha) * r_phy_abs + rk4_alpha * r_phy_rk4

                    y_pred = _apply_rate_guardrails(y_pred, dt_hat, cfg)

                    # Absolute prediction update in fp32 for stability.
                    with torch.autocast(device_type=device.type, enabled=False):
                        dt_hat_f32 = dt_hat.to(dtype=torch.float32)
                        y_pred_f32 = y_pred.to(dtype=torch.float32)
                        y_pred_abs = _maybe_denorm(
                            norm_in.to(dtype=torch.float32) + y_pred_f32 * dt_hat_f32,
                            mu, sigma
                        )

                    center_loss = y_pred.new_zeros(())
                    tmp_loss = y_pred.new_zeros(())
                    if norm_tgt is not None:
                        with torch.autocast(device_type=device.type, enabled=False):
                            delta_target_f32 = (norm_tgt - norm_in).to(dtype=torch.float32)
                            dt_hat_f32 = dt_hat.to(dtype=torch.float32)
                            rate_target_f32 = delta_target_f32 / dt_hat_f32.clamp_min(1e-12)
                            center_loss = (
                                F.huber_loss(y_pred_f32, rate_target_f32, delta=huber_delta)
                                if use_huber else F.mse_loss(y_pred_f32, rate_target_f32)
                            ).to(dtype=y_pred.dtype)
                        if tmp_w > 0:
                            tmp_loss = temporal_consistency(x_in, norm_tgt)

                    lap_loss = y_pred.new_zeros(())
                    if (norm_tgt is not None) and (lap_w > 0):
                        lap_loss = laplacian_smoothness(y_pred, pei)

                    phy_loss = y_pred.new_zeros(())
                    if (
                        (norm_tgt is not None)
                        and need_phy
                        and dec_resid_w > 0.0
                        and (r_phy_for_loss is not None)
                    ):
                        with torch.autocast(device_type=device.type, enabled=False):
                            phy_loss = dec.physics_residual_loss_delta(
                                y_pred_abs=y_pred_abs.float(),
                                x_in_abs=x_in_abs.float(),
                                dt_phys=dt_phys.float(),
                                r_phy_abs=r_phy_for_loss.float(),
                                area=area.float(),
                                #sigma=(sigma.to(device, dtype=torch.float32) if (sigma is not None and torch.is_tensor(sigma)) else None),
                                sigma=sigma_f32,
                                channel_mask=ch_mask,
                            ).to(dtype=y_pred.dtype)

                    loss_step = center_loss + lap_w * lap_loss + tmp_w * tmp_loss + dec_resid_w * phy_loss
                    if norm_tgt is None:
                        loss_step = y_pred.new_zeros(())

                return loss_step, y_pred_abs

            if runtime_mesh_enabled:
                runtime_knn_k = int(cfg.get("train", {}).get("knn_k", cfg.get("loss", {}).get("interp_k", 8)))
                runtime_chunk = int(cfg.get("speed", {}).get("interp_chunk", 8192))

                state_centers = centers_list[0].to(device=device, dtype=torch.float32)
                state_levels = level_list[0].to(device=device, dtype=torch.long).view(-1)
                state_parents = parents_list[0].to(device=device, dtype=torch.long).view(-1)
                state_feat = feat_list[0].to(device=device, dtype=torch.float32)

                active_pred_centers = None
                active_pred_levels = None
                active_pred_parents = None
                active_pred_ei = None
                active_pred_mask = None
                active_pred_ea = None

                runtime_has_step0_precomp = (
                    (pred_lists is not None)
                    and (feat_t_on_pred_list is not None)
                    and (feat_tp1_on_pred_list is not None)
                    and (len(feat_t_on_pred_list) > 1)
                    and (len(feat_tp1_on_pred_list) > 1)
                    and (runtime_domain_mode != "starting_mesh")
                )

                for k in range(0, K - 1):
                    step_wall_t0 = time.perf_counter()
                    used_precomp_step0 = False
                    do_rebuild = False
                    t_rebuild_s = 0.0
                    t_xin_map_s = 0.0
                    t_xtgt_map_s = 0.0
                    t_model_s = 0.0
                    idw_dev_xin = ""
                    idw_chunk_xin = -1
                    n_src_xin = -1
                    n_dst_xin = -1
                    idw_dev_tgt = ""
                    idw_chunk_tgt = -1
                    n_src_tgt = -1
                    n_dst_tgt = -1
                    dtk = _to_scalar_dt(dt_list[k], device=device, dtype=state_feat.dtype)

                    if (k == 0) and runtime_has_step0_precomp:
                        used_precomp_step0 = True
                        pred_centers_1, pred_levels_1, pred_parents_1, pred_ei_1, mask_pred_1 = _pred_mesh_for_step_strict(
                            0, pred_lists=pred_lists
                        )
                        active_pred_centers = pred_centers_1.to(device=device, dtype=torch.float32)
                        active_pred_levels = pred_levels_1.to(device=device, dtype=torch.long).view(-1)
                        active_pred_parents = pred_parents_1.to(device=device, dtype=torch.long).view(-1)
                        active_pred_ei = pred_ei_1.to(device=device, dtype=torch.long)
                        active_pred_mask = mask_pred_1.to(device=device, dtype=torch.bool)
                        pred_ea_step0 = pred_ea_list[1] if (pred_ea_list is not None and len(pred_ea_list) > 1) else None
                        active_pred_ea = (
                            pred_ea_step0.to(device=device, dtype=torch.float32)
                            if torch.is_tensor(pred_ea_step0)
                            else None
                        )

                        if runtime_need_edge_attr and active_pred_ea is None:
                            active_pred_ea = dec_edge_attr_for_dyadic_quads(
                                active_pred_centers.to("cpu", dtype=torch.float32),
                                active_pred_levels.to("cpu", dtype=torch.int64),
                                active_pred_ei.to("cpu", dtype=torch.int64),
                                dx0=float(dx),
                                dy0=float(dy),
                                refine_ratio=_get_refine_ratio(cfg),
                            ).to(device=device, dtype=torch.float32)

                        x_in_abs = feat_t_on_pred_list[1].to(device=device, dtype=torch.float32)
                        x_tgt_abs = (
                            None
                            if runtime_infer_only
                            else feat_tp1_on_pred_list[1].to(device=device, dtype=torch.float32)
                        )
                    else:
                        step_idx = k + 1
                        do_rebuild = (k == 0) or ((step_idx % runtime_update_every) == 0)

                        if do_rebuild:
                            t_rebuild_t0 = time.perf_counter()
                            (
                                active_pred_centers,
                                active_pred_levels,
                                active_pred_parents,
                                active_pred_ei,
                                active_pred_mask,
                                active_pred_ea,
                            ) = _runtime_build_pred_mesh_from_state(
                                centers_t=state_centers,
                                feat_t=state_feat,
                                level_t=state_levels,
                                parents_t=state_parents,
                                cfg=cfg,
                                H=H,
                                W=W,
                                dx=float(dx),
                                dy=float(dy),
                                device=device,
                                wedge_path=runtime_wedge_path,
                                need_edge_attr=runtime_need_edge_attr,
                                runtime_mesh_policy=runtime_mesh_policy,
                                runtime_wedge_constraints=runtime_wedge_constraints,
                                runtime_base_mesh=runtime_base_mesh,
                                cnn_parent_mapping_mode=("from_centers" if (k == 0) else "dataset"),
                                cnn_fill_empty_interior=bool(k == 0),
                                dt_phys=dtk,
                                dt_ref=dt_ref_scalar,
                            )
                            t_rebuild_s = time.perf_counter() - t_rebuild_t0

                            if k == 0:
                                n_src_xin = int(state_centers.shape[0])
                                n_dst_xin = int(active_pred_centers.shape[0])
                                idw_dev_xin_t, idw_chunk_xin = _select_idw_backend(
                                    src_n=n_src_xin,
                                    requested_chunk=int(runtime_chunk),
                                    out_device=device,
                                )
                                idw_dev_xin = idw_dev_xin_t.type
                                t_xin_t0 = time.perf_counter()
                                x_in_abs = _map_gt_to_pred_mesh_once(
                                    src_centers=state_centers,
                                    src_feats=state_feat,
                                    pred_centers=active_pred_centers,
                                    knn_k=runtime_knn_k,
                                    chunk=runtime_chunk,
                                    knn_backend=runtime_idw_backend,
                                    knn_backend_kwargs=runtime_idw_backend_kwargs,
                                )
                                t_xin_map_s = time.perf_counter() - t_xin_t0
                            else:
                                n_src_xin = int(state_centers.shape[0])
                                n_dst_xin = int(active_pred_centers.shape[0])
                                idw_dev_xin_t, idw_chunk_xin = _select_idw_backend(
                                    src_n=n_src_xin,
                                    requested_chunk=int(runtime_chunk),
                                    out_device=device,
                                )
                                idw_dev_xin = idw_dev_xin_t.type
                                t_xin_t0 = time.perf_counter()
                                x_in_abs = _map_pred_to_next_pred(
                                    pred_centers_src=state_centers,
                                    feats_src=state_feat,
                                    levels_src=state_levels,
                                    parents_src=state_parents,
                                    pred_centers_dst=active_pred_centers,
                                    levels_dst=active_pred_levels,
                                    parents_dst=active_pred_parents,
                                    mask_pred_dst=active_pred_mask.view(-1),
                                    H=H,
                                    W=W,
                                    knn_k=runtime_knn_k,
                                    chunk=runtime_chunk,
                                    bbox=runtime_bbox,
                                    refine_ratio=runtime_refine_ratio,
                                    knn_backend=runtime_idw_backend,
                                    knn_backend_kwargs=runtime_idw_backend_kwargs,
                                )
                                t_xin_map_s = time.perf_counter() - t_xin_t0
                        else:
                            if active_pred_centers is None:
                                raise RuntimeError("Runtime mesh state is not initialized before a reuse step.")
                            x_in_abs = state_feat

                        if runtime_infer_only:
                            x_tgt_abs = None
                        else:
                            gt_centers_tp1 = centers_list[k + 1].to(device=device, dtype=torch.float32)
                            gt_feat_tp1 = feat_list[k + 1].to(device=device, dtype=torch.float32)
                            n_src_tgt = int(gt_centers_tp1.shape[0])
                            n_dst_tgt = int(active_pred_centers.shape[0])
                            idw_dev_tgt_t, idw_chunk_tgt = _select_idw_backend(
                                src_n=n_src_tgt,
                                requested_chunk=int(runtime_chunk),
                                out_device=device,
                            )
                            idw_dev_tgt = idw_dev_tgt_t.type
                            t_xtgt_t0 = time.perf_counter()
                            x_tgt_abs = _map_gt_to_pred_mesh_once(
                                src_centers=gt_centers_tp1,
                                src_feats=gt_feat_tp1,
                                pred_centers=active_pred_centers,
                                knn_k=runtime_knn_k,
                                chunk=runtime_chunk,
                                knn_backend=runtime_idw_backend,
                                knn_backend_kwargs=runtime_idw_backend_kwargs,
                            )
                            t_xtgt_map_s = time.perf_counter() - t_xtgt_t0

                    dtk = dtk.to(device=device, dtype=x_in_abs.dtype)

                    t_model_t0 = time.perf_counter()
                    loss_k, y_pred_abs_k = _run_step_eval(
                        step_k=k,
                        pred_centers=active_pred_centers,
                        pred_levels=active_pred_levels,
                        pred_parents=active_pred_parents,
                        pred_ei=active_pred_ei,
                        pred_ea=active_pred_ea,
                        x_in_abs=x_in_abs,
                        x_tgt_abs=x_tgt_abs,
                        dt_phys=dtk,
                    )
                    t_model_s = time.perf_counter() - t_model_t0

                    total_loss_accum += float(loss_k.detach().cpu())
                    n_steps_total += 1

                    t_absk = None
                    if t_indices is not None:
                        t_absk = int(t_indices[k + 1].item()) if torch.is_tensor(t_indices) else int(t_indices[k + 1])

                    if x_tgt_abs is not None:
                        _accumulate_metrics(
                            k=k,
                            t_abs=t_absk,
                            pred_abs=y_pred_abs_k,
                            gt_abs=x_tgt_abs,
                            pred_levels=active_pred_levels,
                        )
                        step_mae = float(torch.mean(torch.abs(y_pred_abs_k.detach() - x_tgt_abs)).cpu())
                    else:
                        step_mae = float("nan")
                    mem_alloc_mb, mem_reserved_mb = _runtime_memory_snapshot_mb(device)
                    loss_log = "" if runtime_infer_only else float(loss_k.detach().cpu())
                    mae_log = "" if runtime_infer_only else float(step_mae)
                    n_gt_src_log = (
                        -1 if runtime_infer_only
                        else (int(centers_list[k + 1].shape[0]) if torch.is_tensor(centers_list[k + 1]) else -1)
                    )
                    _runtime_step_log_write(
                        cfg,
                        {
                            "wall_time": datetime.datetime.now().isoformat(timespec="seconds"),
                            "split": "eval",
                            "epoch": (int(epoch_idx) if epoch_idx is not None else -1),
                            "batch_idx": int(batch_idx),
                            "step_k": int(k),
                            "t_abs": int(t_absk) if t_absk is not None else -1,
                            "used_precomp_step0": int(used_precomp_step0),
                            "do_rebuild": int(do_rebuild),
                            "update_every_steps": int(runtime_update_every),
                            "device": str(device),
                            "n_state": int(x_in_abs.shape[0]) if torch.is_tensor(x_in_abs) else -1,
                            "n_pred": int(active_pred_centers.shape[0]) if torch.is_tensor(active_pred_centers) else -1,
                            "e_pred": int(active_pred_ei.shape[1]) if (torch.is_tensor(active_pred_ei) and active_pred_ei.ndim == 2) else -1,
                            "n_gt_src": n_gt_src_log,
                            "idw_dev_xin": idw_dev_xin,
                            "idw_chunk_xin": int(idw_chunk_xin),
                            "n_src_xin": int(n_src_xin),
                            "n_dst_xin": int(n_dst_xin),
                            "idw_dev_tgt": idw_dev_tgt,
                            "idw_chunk_tgt": int(idw_chunk_tgt),
                            "n_src_tgt": int(n_src_tgt),
                            "n_dst_tgt": int(n_dst_tgt),
                            "t_rebuild_s": float(t_rebuild_s),
                            "t_xin_map_s": float(t_xin_map_s),
                            "t_xtgt_map_s": float(t_xtgt_map_s),
                            "t_model_s": float(t_model_s),
                            "t_step_total_s": float(time.perf_counter() - step_wall_t0),
                            "loss": loss_log,
                            "mae": mae_log,
                            "mem_alloc_mb": float(mem_alloc_mb),
                            "mem_reserved_mb": float(mem_reserved_mb),
                        },
                    )

                    if write_budgets and (t_absk is not None) and (not runtime_infer_only):
                        gt_tp1_abs = feat_list[k + 1].to(device)
                        gt_tp1_lev = level_list[k + 1].to(device)

                        area, mass, mom_x, mom_y, energy = _compute_budget_row(
                            x_abs=gt_tp1_abs,
                            levels=gt_tp1_lev,
                            dx0=float(dx),
                            dy0=float(dy),
                            cfg=cfg,
                        )
                        budget_rows.append({
                            "t_abs": int(t_absk),
                            "step_k": int(k),
                            "kind": "gt_on_gt_mesh_tp1",
                            "mass": mass,
                            "mom_x": mom_x,
                            "mom_y": mom_y,
                            "energy": energy,
                            "area": area,
                        })

                        if k == 0:
                            t_abs_t = int(t_indices[0].item()) if (t_indices is not None and torch.is_tensor(t_indices)) else (int(t_indices[0]) if t_indices is not None else None)
                            if t_abs_t is not None:
                                gt_t_abs = feat_list[0].to(device)
                                gt_t_lev = level_list[0].to(device)
                                area, mass, mom_x, mom_y, energy = _compute_budget_row(
                                    x_abs=gt_t_abs,
                                    levels=gt_t_lev,
                                    dx0=float(dx),
                                    dy0=float(dy),
                                    cfg=cfg,
                                )
                                budget_rows.append({
                                    "t_abs": int(t_abs_t),
                                    "step_k": -1,
                                    "kind": "gt_on_gt_mesh_t",
                                    "mass": mass,
                                    "mom_x": mom_x,
                                    "mom_y": mom_y,
                                    "energy": energy,
                                    "area": area,
                                })

                    _append_example_step(
                        step_idx=k,
                        pred_centers=active_pred_centers,
                        pred_levels=active_pred_levels,
                        pred_parents=active_pred_parents,
                        pred_ei=active_pred_ei,
                        y_pred_step_abs=y_pred_abs_k,
                        centers_list=centers_list,
                        feat_list=feat_list,
                        level_list=level_list,
                        parents_list=parents_list,
                        batch=batch,
                    )

                    if write_budgets and (t_absk is not None):
                        if x_tgt_abs is not None:
                            _append_budget_row(
                                t_abs=t_absk,
                                step_k=k,
                                kind="gt_on_pred_mesh_tp1",
                                x_abs=x_tgt_abs,
                                pred_levels=active_pred_levels,
                            )
                        _append_budget_row(
                            t_abs=t_absk,
                            step_k=k,
                            kind="pred_on_pred_mesh_tp1",
                            x_abs=y_pred_abs_k,
                            pred_levels=active_pred_levels,
                        )

                    state_feat = _enforce_physical_state(y_pred_abs_k, cfg, rho_floor=1e-6, E_floor=1e-6)
                    if x_tgt_abs is not None:
                        _record_step_diagnostics(
                            pred_raw_abs=y_pred_abs_k,
                            pred_used_abs=state_feat,
                            gt_abs=x_tgt_abs,
                            pred_levels=active_pred_levels,
                            pred_centers=active_pred_centers,
                            pred_ei=active_pred_ei,
                        )
                    state_centers = active_pred_centers
                    state_levels = active_pred_levels
                    state_parents = active_pred_parents

                continue

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

            if write_budgets and (t_abs0 is not None):
                # GT(t+1) on GT mesh
                gt_tp1_abs = feat_list[1].to(device)
                gt_tp1_lev = level_list[1].to(device)

                area, mass, mom_x, mom_y, energy = _compute_budget_row(
                    x_abs=gt_tp1_abs, levels=gt_tp1_lev,
                    dx0=float(dx), dy0=float(dy), cfg=cfg
                )
                budget_rows.append({
                    "t_abs": int(t_abs0),
                    "step_k": 0,
                    "kind": "gt_on_gt_mesh_tp1",
                    "mass": mass, "mom_x": mom_x, "mom_y": mom_y, "energy": energy, "area": area,
                })

                # (optional) GT(t) on GT mesh
                t_abs_t = int(t_indices[0].item()) if (t_indices is not None and torch.is_tensor(t_indices)) else (int(t_indices[0]) if t_indices is not None else None)
                if t_abs_t is not None:
                    gt_t_abs = feat_list[0].to(device)
                    gt_t_lev = level_list[0].to(device)

                    area, mass, mom_x, mom_y, energy = _compute_budget_row(
                        x_abs=gt_t_abs, levels=gt_t_lev,
                        dx0=float(dx), dy0=float(dy), cfg=cfg
                    )
                    budget_rows.append({
                        "t_abs": int(t_abs_t),
                        "step_k": -1,  # label GT(t) as "pre-step"
                        "kind": "gt_on_gt_mesh_t",
                        "mass": mass, "mom_x": mom_x, "mom_y": mom_y, "energy": energy, "area": area,
                    })


            _append_example_step(
                step_idx=0,
                pred_centers=pred_centers_1,
                pred_levels=pred_levels_1,
                pred_parents=pred_parents_1,
                pred_ei=pred_ei_1,
                y_pred_step_abs=y_pred_abs0,
                centers_list=centers_list,
                feat_list=feat_list,
                level_list=level_list,
                parents_list=parents_list,
                batch=batch,
            )

            if write_budgets and (t_abs0 is not None):
                _append_budget_row(t_abs=t_abs0, step_k=0, kind="gt_on_pred_mesh_tp1",   x_abs=x_tgt_abs0,  pred_levels=pred_levels_1)
                _append_budget_row(t_abs=t_abs0, step_k=0, kind="pred_on_pred_mesh_tp1", x_abs=y_pred_abs0, pred_levels=pred_levels_1)


            #pred_feats_k = y_pred_abs0
            pred_feats_k = _enforce_physical_state(y_pred_abs0, cfg, rho_floor=1e-6, E_floor=1e-6)
            _record_step_diagnostics(
                pred_raw_abs=y_pred_abs0,
                pred_used_abs=pred_feats_k,
                gt_abs=x_tgt_abs0,
                pred_levels=pred_levels_1,
                pred_centers=pred_centers_1,
                pred_ei=pred_ei_1,
            )

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

                if write_budgets and (t_absk is not None):
                    gt_tp1_abs = feat_list[k + 1].to(device)
                    gt_tp1_lev = level_list[k + 1].to(device)

                    area, mass, mom_x, mom_y, energy = _compute_budget_row(
                        x_abs=gt_tp1_abs, levels=gt_tp1_lev,
                        dx0=float(dx), dy0=float(dy), cfg=cfg
                    )
                    budget_rows.append({
                        "t_abs": int(t_absk),
                        "step_k": int(k),
                        "kind": "gt_on_gt_mesh_tp1",
                        "mass": mass, "mom_x": mom_x, "mom_y": mom_y, "energy": energy, "area": area,
                    })


                _append_example_step(
                    step_idx=k,
                    pred_centers=pred_centers_next,
                    pred_levels=pred_levels_next,
                    pred_parents=pred_parents_next,
                    pred_ei=pred_ei_next,
                    y_pred_step_abs=y_pred_abs_k,
                    centers_list=centers_list,
                    feat_list=feat_list,
                    level_list=level_list,
                    parents_list=parents_list,
                    batch=batch,
                )

                if write_budgets and (t_absk is not None):
                    _append_budget_row(t_abs=t_absk, step_k=k, kind="gt_on_pred_mesh_tp1",   x_abs=x_tgt_abs,   pred_levels=pred_levels_next)
                    _append_budget_row(t_abs=t_absk, step_k=k, kind="pred_on_pred_mesh_tp1", x_abs=y_pred_abs_k, pred_levels=pred_levels_next)

                # keep your existing propagation behavior
                #pred_feats_k = _enforce_physical_state(y_pred_abs_k, rho_floor=1e-6, E_floor=1e-6)
                pred_feats_k = _enforce_physical_state(y_pred_abs_k, cfg, rho_floor=1e-6, E_floor=1e-6)
                _record_step_diagnostics(
                    pred_raw_abs=y_pred_abs_k,
                    pred_used_abs=pred_feats_k,
                    gt_abs=x_tgt_abs,
                    pred_levels=pred_levels_next,
                    pred_centers=pred_centers_next,
                    pred_ei=pred_ei_next,
                )


    # ---- finalize metrics ----
    eps = 1e-12
    S = len(step_wsum)
    if S == 0:
        if runtime_infer_only:
            avg_loss = total_loss_accum / max(n_steps_total, 1)
            stats = {
                "num_windows": len(loader),
                "num_steps": n_steps_total,
                "infer_only": True,
                **_finalize_diagnostics_accumulator(diagnostics_accum, diag_cfg),
                "maew_by_rollout_step": [],
                "rell2w_by_rollout_step": [],
                "maew_feat_by_rollout_step": None,
                "rell2w_feat_by_rollout_step": None,
                "t_values": [],
                "maew_by_t": None,
                "rell2w_by_t": None,
                "maew_feat_by_t": None,
                "rell2w_feat_by_t": None,
            }
            if collect_examples:
                stats["examples"] = examples
            return avg_loss, stats
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

    # -----------------------------
    # Write budgets CSV (once per eval call)
    # -----------------------------
    
    if write_budgets and (budget_csv_path is not None):
        os.makedirs(os.path.dirname(budget_csv_path) or ".", exist_ok=True)
        fieldnames = ["t_abs", "step_k", "kind", "mass", "mom_x", "mom_y", "energy", "area"]
        with open(budget_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(budget_rows)

    stats = {
        "num_windows": len(loader),
        "num_steps": n_steps_total,
        **_finalize_diagnostics_accumulator(diagnostics_accum, diag_cfg),
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


def build_model_from_cfg(cfg, device):
    use_cols = cfg.get("features", {}).get("use_columns", [0, 1, 2, 3])
    Fdim = int(cfg.get("features", {}).get("num_features", len(use_cols)))
    model_cfg = cfg.get("model", {}) or {}

    b = cfg.get("features", {}).get("build", {})
    in_ch = Fdim + (2 if b.get("use_pos", True) else 0) + (1 if b.get("use_level", True) else 0)
    if bool(b.get("use_dt_hat_fixed", False)):
        in_ch += 1

    loss = cfg.get("loss", {}) or {}
    parc_on = bool(loss.get("parc", False) or loss.get("parc_inputs", False))
    if parc_on:
        in_ch += dec.parc_extra_in_channels(cfg, Fdim)

    out_ch = Fdim
    conv_type = str(model_cfg.get("conv_type", "sage")).strip().lower()
    edge_dim = int(model_cfg.get("edge_dim", 5)) if conv_type in ("gine", "nnconv") else None
    nnconv_hidden = int(model_cfg.get("nnconv_hidden", model_cfg.get("hidden", 128)))

    model = FeatureNet(
        in_channels=in_ch,
        out_channels=out_ch,
        hidden=int(model_cfg.get("hidden", 128)),
        layers=int(model_cfg.get("layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        make_score_head=True,
        conv_type=conv_type,
        edge_dim=edge_dim,
        nnconv_hidden=nnconv_hidden,
        use_skip=bool(model_cfg.get("use_skip", False)),
        skip_type=str(model_cfg.get("skip_type", "block")),
        use_layernorm=bool(model_cfg.get("use_layernorm", False)),
        layernorm_eps=float(model_cfg.get("layernorm_eps", 1e-6)),
    ).to(device)

    # -------------------------
    # Optional PARC adapter
    # -------------------------
    use_adapter = bool(loss.get("parc_use_adapter", False))

    if parc_on and use_adapter:
        la = len(dec.parc_select_feature_indices_adv(cfg, Fdim))
        ld = len(dec.parc_select_feature_indices_diff(cfg, Fdim))

        model.parc_adapter = ParcFeatureAdapter(
            la, ld,
            use_norm=bool(loss.get("parc_feat_norm", True)),
            clip_pre=float(loss.get("parc_feat_clip_pre", 50.0)),
            clip_post=float(loss.get("parc_feat_clip_post", 10.0)),
            momentum=float(loss.get("parc_feat_norm_momentum", 0.02)),
            per_channel_gates=bool(loss.get("parc_gate_per_channel", True)),
            gate_init=float(loss.get("parc_gate_init", -3.0)),
        ).to(device)
    else:
        model.parc_adapter = None

    return model


# ------------------------------ Main ------------------------------

def main(config_path: str | None = None, out_dir: str | None = None):
    # -------- load & normalize config --------
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config_feature_first.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    if out_dir is not None:
        out_dir_abs = os.path.abspath(os.path.expanduser(str(out_dir)))
        cfg.setdefault("train", {})["save_dir"] = out_dir_abs
        print(f"[CLI] Overriding train.save_dir -> {out_dir_abs}")

    # Canonical level settings:
    # - policy.max_level controls AMR mask hierarchy / clipping depth.
    # - data.L_max is kept in sync when missing.
    # - eval.fine_L is auto-derived from policy.max_level (no separate tuning).
    data_cfg = cfg.setdefault("data", {})
    policy_cfg = cfg.setdefault("policy", {})
    eval_cfg = cfg.setdefault("eval", {})
    features_cfg = cfg.setdefault("features", {})
    norm_cfg = features_cfg.setdefault("normalization", {})

    if "max_level" not in policy_cfg:
        policy_cfg["max_level"] = int(data_cfg.get("L_max", 3))
    if "L_max" not in data_cfg:
        data_cfg["L_max"] = int(policy_cfg.get("max_level", 3))

    lmax_policy = int(policy_cfg.get("max_level", 3))
    lmax_data = int(data_cfg.get("L_max", lmax_policy))
    if lmax_policy != lmax_data:
        print(
            f"[CFG][WARN] policy.max_level ({lmax_policy}) != data.L_max ({lmax_data}). "
            "Using policy.max_level for AMR operations.",
            flush=True,
        )
    eval_cfg["fine_L"] = int(lmax_policy)
    print(
        f"[CFG] effective levels: policy.max_level={lmax_policy}, data.L_max={lmax_data}, eval.fine_L={int(eval_cfg['fine_L'])}",
        flush=True,
    )

    momentum_sigma_mode = _resolve_momentum_sigma_mode(cfg)
    norm_cfg.setdefault(
        "_comment_momentum_sigma_mode",
        "independent (per-channel std) or shared_rms (tie x/y momentum sigmas by RMS).",
    )
    norm_cfg["momentum_sigma_mode"] = momentum_sigma_mode
    print(f"[NORM-CFG] momentum_sigma_mode={momentum_sigma_mode}")

    idx = dec.infer_feature_indices(cfg, 4)
    print("[IDX]", idx)
    assert idx["rho"] == 0 and idx["mx"] == 1 and idx["my"] == 2 and idx["E"] == 3

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
        loss.setdefault("channels", ["rho", "x_momentum", "y_momentum", "E"])  # default physics channels

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

    train_cfg = cfg.setdefault("train", {})
    train_cfg.setdefault("validation_every_epochs", 1)
    train_cfg.setdefault("checkpoint_every_epochs", 1)
    train_cfg.setdefault("mesh_mode", "predicted")

    mesh_mode = str(train_cfg.get("mesh_mode", "predicted")).strip().lower()
    if mesh_mode not in ("predicted", "uniform"):
        raise ValueError(
            f"train.mesh_mode must be 'predicted' or 'uniform', got '{mesh_mode}'"
        )
    train_cfg["mesh_mode"] = mesh_mode

    # Runtime mesh defaults (infrastructure only in this commit)
    runtime_mesh_cfg = train_cfg.setdefault("runtime_mesh", {})
    runtime_mesh_cfg.setdefault("enabled", False)
    runtime_mesh_cfg.setdefault("detach_policy_input", True)
    runtime_mesh_cfg.setdefault("reset_to_coarse_each_step", True)
    runtime_mesh_cfg.setdefault("refine_only", True)
    runtime_mesh_cfg.setdefault("warm_start_from_precompute", True)
    runtime_mesh_cfg.setdefault("update_every_steps", 1)
    runtime_mesh_cfg.setdefault("max_cells_per_step", 400_000)
    runtime_mesh_cfg.setdefault("policy_backend", "gradient")
    runtime_mesh_cfg.setdefault("domain_mode", "wedge_lookup")
    runtime_mesh_cfg.setdefault("wedge_clip_use_lookup", True)
    runtime_mesh_cfg.setdefault("wedge_clip_lookup_fallback", False)
    runtime_mesh_idw_cfg = runtime_mesh_cfg.setdefault("idw", {})
    if not isinstance(runtime_mesh_idw_cfg, dict):
        raise ValueError("train.runtime_mesh.idw must be a JSON object when provided.")
    runtime_mesh_idw_cfg.setdefault("backend", "exact")
    runtime_mesh_idw_cfg.setdefault("faiss_nlist", 256)
    runtime_mesh_idw_cfg.setdefault("faiss_nprobe", 16)
    runtime_mesh_idw_cfg.setdefault("faiss_cache", True)
    runtime_mesh_idw_cfg.setdefault("faiss_cache_max_entries", 4)
    runtime_mesh_idw_cfg.setdefault("allow_fallback_to_exact", True)
    runtime_mesh_cnn_cfg = runtime_mesh_cfg.setdefault("cnn", {})
    if not isinstance(runtime_mesh_cnn_cfg, dict):
        raise ValueError("train.runtime_mesh.cnn must be a JSON object when provided.")
    runtime_mesh_cnn_cfg.setdefault("checkpoint_path", "")
    runtime_mesh_cnn_cfg.setdefault("include_parent_mask", True)
    runtime_mesh_cnn_cfg.setdefault("include_coords", True)
    runtime_mesh_cnn_cfg.setdefault("include_dt", False)
    runtime_mesh_cnn_cfg.setdefault("dt_channel_mode", "dt_hat")
    runtime_mesh_cnn_cfg.setdefault("threshold_default", 0.5)
    runtime_mesh_cnn_cfg.setdefault("threshold_by_level", {})
    runtime_mesh_cnn_cfg.setdefault("use_saved_thresholds", False)
    runtime_mesh_cnn_cfg.setdefault("device", "same")
    step_log_cfg = runtime_mesh_cfg.setdefault("step_log", {})
    if not isinstance(step_log_cfg, dict):
        raise ValueError("train.runtime_mesh.step_log must be a JSON object when provided.")
    step_log_cfg.setdefault("enabled", False)
    step_log_cfg.setdefault("path", os.path.join(cfg.get("train", {}).get("save_dir", "."), "runtime_step_log.csv"))
    step_log_cfg.setdefault("append", False)
    diag_cfg_main = _resolve_diagnostics_cfg(cfg)

    runtime_mesh_enabled = bool(runtime_mesh_cfg.get("enabled", False))
    runtime_update_every = int(runtime_mesh_cfg.get("update_every_steps", 1))
    if runtime_update_every < 1:
        raise ValueError("train.runtime_mesh.update_every_steps must be >= 1.")

    if runtime_mesh_enabled:
        runtime_backend = _runtime_mesh_backend(cfg)
        runtime_domain_mode = _runtime_mesh_domain_mode(cfg)
        if not bool(runtime_mesh_cfg.get("reset_to_coarse_each_step", True)):
            raise RuntimeError(
                "Runtime mesh mode requires reset_to_coarse_each_step=true "
                "(interpreted as reset-to-starting-mesh when domain_mode='starting_mesh')."
            )
        if not bool(runtime_mesh_cfg.get("refine_only", True)):
            raise RuntimeError("Runtime mesh mode requires refine_only=true for this workflow.")
        if runtime_domain_mode == "wedge_lookup":
            if not bool(runtime_mesh_cfg.get("wedge_clip_use_lookup", True)):
                raise RuntimeError(
                    "Runtime mesh mode requires train.runtime_mesh.wedge_clip_use_lookup=true "
                    "(lookup-only wedge clipping)."
                )
            if bool(runtime_mesh_cfg.get("wedge_clip_lookup_fallback", False)):
                raise RuntimeError(
                    "Runtime mesh mode requires train.runtime_mesh.wedge_clip_lookup_fallback=false "
                    "(no fallback path)."
                )
        if runtime_backend == "cnn":
            ckpt_raw = runtime_mesh_cnn_cfg.get("checkpoint_path", "")
            ckpt_path = os.path.abspath(os.path.expanduser(str(ckpt_raw))) if ckpt_raw else ""
            if not ckpt_path:
                raise RuntimeError(
                    "Runtime mesh backend 'cnn' requires train.runtime_mesh.cnn.checkpoint_path."
                )
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Runtime mesh CNN checkpoint not found: {ckpt_path}")
            cfg["train"]["runtime_mesh"]["cnn"]["checkpoint_path"] = ckpt_path

        mesh_path_cfg = cfg.get("mesh", {}).get("starting_mesh_path", None)
        if not mesh_path_cfg:
            raise RuntimeError(
                "Runtime mesh mode requires cfg['mesh']['starting_mesh_path'] to point to wedge_mesh_spec.pt."
            )
        mesh_path = os.path.abspath(os.path.expanduser(str(mesh_path_cfg)))
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Runtime mesh mode expected mesh spec at: {mesh_path}")
        cfg.setdefault("mesh", {})["starting_mesh_path"] = mesh_path
        if bool(step_log_cfg.get("enabled", False)):
            _enabled, step_log_path, _append = _runtime_step_log_settings(cfg)
            print(f"[RUNTIME-MESH] step log enabled: {step_log_path}")

    # Ensure dataset sees the intended columns
    if cfg.get("data", {}).get("feature_idx") and not cfg.get("features", {}).get("use_columns"):
        cfg.setdefault("features", {}).setdefault("use_columns", cfg["data"]["feature_idx"])

    #device = pick_device(cfg.get("train", {}).get("device", "auto"))
    raw_dev = cfg.get("device", "cpu")
    device = torch.device(raw_dev)
    #print(f"[INFO] Using device: {device}")
    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    H = int(cfg["data"].get("H", 64)); W = int(cfg["data"].get("W", 64))
    xmin, xmax, ymin, ymax = _get_bbox(cfg)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H

    runtime_mesh_policy = None
    if runtime_mesh_enabled:
        runtime_mesh_policy = _load_runtime_mesh_policy_from_cfg(
            cfg,
            device=device,
            H=H,
            W=W,
        )

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

    # After model is created, before optimizer is created:
    loss_cfg = cfg.get("loss", {}) or {}
    parc_use = bool(loss_cfg.get("parc", False) or loss_cfg.get("parc_inputs", False))
    use_adapter = bool(loss_cfg.get("parc_use_adapter", False))

    if parc_use and use_adapter:
        Fdim = int(cfg.get("features", {}).get("num_features", 4))  # your state size
        la = len(dec.parc_select_feature_indices_adv(cfg, Fdim))
        ld = len(dec.parc_select_feature_indices_diff(cfg, Fdim))

        adapter = ParcFeatureAdapter(
            la, ld,
            use_norm=bool(loss_cfg.get("parc_feat_norm", True)),
            clip_pre=float(loss_cfg.get("parc_feat_clip_pre", 10.0)),
            clip_post=float(loss_cfg.get("parc_feat_clip_post", 10.0)),
            momentum=float(loss_cfg.get("parc_feat_norm_momentum", 0.02)),
            per_channel_gates=bool(loss_cfg.get("parc_gate_per_channel", True)),
            gate_init=float(loss_cfg.get("parc_gate_init", -5.0)),
        ).to(device)

        model.parc_adapter = adapter

    def _get_lr(optimizer):
            return optimizer.param_groups[0]["lr"]

    # 1) Build modules first
    opt_groups = [{"params": model.parameters(), "lr": float(cfg["train"]["lr"])}]

    if cfg.get("train", {}).get("interp_type", "standard") == "gnn":
        raise NotImplementedError(
            "train.interp_type='gnn' is not available in this branch: "
            "ProlongationHead/RestrictionHead are not defined. "
            "Use 'standard' or 'knn', or add/import those modules first."
        )

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

    # Optional raw-snapshot subset for fast precompute/debug runs.
    # These are absolute indices into the loaded sequence and are inclusive.
    pre_t_start = train_cfg.get("precompute_t_start", None)
    pre_t_end = train_cfg.get("precompute_t_end", None)
    if pre_t_start is not None or pre_t_end is not None:
        n_total = len(data_list)
        if n_total < 2:
            raise ValueError(f"Need at least 2 raw snapshots, found {n_total}.")

        start = 0 if pre_t_start is None else int(pre_t_start)
        end = (n_total - 1) if pre_t_end is None else int(pre_t_end)

        if start < 0 or end < 0:
            raise ValueError(
                "train.precompute_t_start / train.precompute_t_end must be >= 0 when provided. "
                f"Got start={start}, end={end}."
            )
        if start >= n_total:
            raise ValueError(
                f"train.precompute_t_start={start} is out of range for n_total={n_total}."
            )
        if end >= n_total:
            raise ValueError(
                f"train.precompute_t_end={end} is out of range for n_total={n_total}."
            )
        if end < start:
            raise ValueError(
                f"Invalid precompute range: end ({end}) < start ({start})."
            )

        # inclusive -> Python slice end-exclusive
        data_list = data_list[start : end + 1]
        if len(data_list) < 2:
            raise ValueError(
                "Selected precompute range is too small. Need at least two snapshots "
                f"to form one transition, got {len(data_list)} (start={start}, end={end})."
            )

        # Persist resolved values for provenance in saved config/meta.
        train_cfg["precompute_t_start"] = int(start)
        train_cfg["precompute_t_end"] = int(end)
        print(
            f"[PRECOMP-RANGE] using raw snapshots [{start}, {end}] "
            f"(count={len(data_list)} of original {n_total})"
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
    n_train = int(round(train_frac * T))
    n_val   = int(round(val_frac   * T))
    # keep the rest for test
    train_idx = idxs[:n_train]
    val_idx   = idxs[n_train:n_train + n_val]
    test_idx  = idxs[n_train + n_val:]

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    test_ds  = Subset(full_ds, test_idx.tolist())

    precomp = None
    runtime_warm_start_precomp = bool(runtime_mesh_cfg.get("warm_start_from_precompute", True))
    use_precomp_collate = (not runtime_mesh_enabled) or runtime_warm_start_precomp

    if use_precomp_collate:
        mesh_mode = str(cfg.get("train", {}).get("mesh_mode", "predicted")).strip().lower()
        cache_path = cfg["train"].get("precomp_cache_path", None)
        force_recompute = bool(cfg["train"].get("precomp_force_recompute", False))
        if mesh_mode == "uniform":
            if cache_path is None:
                cache_dir = cfg.get("train", {}).get("cache_dir", cfg.get("train", {}).get("save_dir", "."))
                cache_path = os.path.join(cache_dir, "precomp_uniform.h5")
            print(f"[PRECOMP] train.mesh_mode=uniform: static-mesh precompute with cache_path={cache_path}")
            precomp = precompute_uniform_mesh_in_memory(
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
        else:
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

        precomp = move_precomp_to_device(precomp, device="cpu")

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
            if (not runtime_mesh_enabled) and isinstance(precomp, dict):
                if ("pred_edge_attr" not in precomp) and ("pred_ea" not in precomp):
                    raise RuntimeError(
                        "DEC enabled but precomp is missing pred_ea_list/pred_edge_attr_list. "
                        "Update H5->precomp loader and CollateWithPrecompute to read/store pred_edge_attr."
                    )

        if runtime_mesh_enabled:
            print(
                "[RUNTIME-MESH] warm-start enabled: using precomputed step-0 meshes/maps, "
                "then runtime remeshing for later steps."
            )

        collate = CollateWithPrecompute(precomp, dt_transitions=dt_transitions, dt_ref=dt_ref)
    else:
        print(
            "[RUNTIME-MESH] enabled: skipping precompute and using dt-only collate."
        )
        collate = CollateWithDtOnly(dt_transitions=dt_transitions, dt_ref=dt_ref)

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

    # --- Step A debug (ONE batch, ONE step) ---
    if cfg.get("debug", {}).get("run_stepA", False):
        batch0 = next(iter(train_loader))  # or val_loader
        _ = stepA_check_ops_vs_gt_delta(
            batch0,
            cfg,
            device,
            dx=float(dx),
            dy=float(dy),
            step_k=int(cfg.get("debug", {}).get("stepA_k", 0)),
            u_max=float(cfg.get("debug", {}).get("stepA_u_max", 1e3)),
            rho_floor=1e-6,
            E_floor=1e-6,
        )
        raise SystemExit("[STEP-A] done")

     # ---- Normalization stats (μ/σ) ----
    feats_cfg = cfg.get("features", {})
    do_norm = bool(feats_cfg.get("normalize", True))
    #mu, sigma = _as_mu_sigma(cfg, device)
    mu = sigma = None

    if do_norm and (mu is None or sigma is None):
        # compute from training loader on CPU/GPU (device already set)
        mu, sigma = _compute_norm_stats_from_loader(
            train_loader,
            device,
            momentum_sigma_mode=momentum_sigma_mode,
            momentum_x_idx=int(idx["mx"]),
            momentum_y_idx=int(idx["my"]),
        )
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

    if sigma is not None:
        sigma = _apply_momentum_sigma_mode(
            sigma,
            mode=momentum_sigma_mode,
            momentum_x_idx=int(idx["mx"]),
            momentum_y_idx=int(idx["my"]),
        )
        cfg.setdefault("features", {}).setdefault("norm_stats", {})
        cfg["features"]["norm_stats"]["sigma"] = sigma.detach().cpu().tolist()

    # --- ABS really means ABS check (one batch) ---
    if cfg.get("debug", {}).get("abs_means_abs_check", True):
        batch0 = next(iter(train_loader))
        abs_means_abs_check(cfg=cfg, batch=batch0, mu=mu, sigma=sigma, device=device, tag="ABS-MEANS-ABS/TRAIN")

    if cfg.get("debug", {}).get("abs_means_abs_check_val", False):
        batchv = next(iter(val_loader))
        abs_means_abs_check(cfg=cfg, batch=batchv, mu=mu, sigma=sigma, device=device, tag="ABS-MEANS-ABS/VAL")
    
    # ---- DEBUG: normalization stats sanity (prints once) ----
    print("\n[NORM-STATS] computed from train_loader")
    print("  mu:", mu.detach().cpu().numpy())
    print("  sigma:", sigma.detach().cpu().numpy())
    if momentum_sigma_mode == "shared_rms":
        mx_i = int(idx["mx"])
        my_i = int(idx["my"])
        print(
            "  shared momentum sigma:",
            float(sigma[mx_i].detach().cpu()),
            f"(mx idx={mx_i}, my idx={my_i})",
        )
    print("  sigma min/max:", float(sigma.min().detach().cpu()), float(sigma.max().detach().cpu()))
    print("  any sigma<=0:", bool((sigma <= 0).any().detach().cpu()))
    print("  any nonfinite mu:", bool((~torch.isfinite(mu)).any().detach().cpu()))
    print("  any nonfinite sigma:", bool((~torch.isfinite(sigma)).any().detach().cpu()))

    # -------- training loop --------
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    log_csv = os.path.join(save_dir, "train_log.csv")
    last_ckpt_path = os.path.join(save_dir, "last_model.pt")

    # Save config once at startup.
    save_config(save_dir, cfg)
    print(f"[INFO] Saved config once: {os.path.join(save_dir, 'config.json')}")

    checkpoint_every = int(cfg.get("train", {}).get("checkpoint_every_epochs", 1))
    if checkpoint_every < 0:
        raise ValueError("train.checkpoint_every_epochs must be >= 0.")
    if checkpoint_every == 0:
        print("[INFO] Periodic last-model checkpointing disabled (checkpoint_every_epochs=0).")
    else:
        print(
            f"[INFO] Periodic last-model checkpointing every {checkpoint_every} epoch(s): {last_ckpt_path}"
        )

    def _build_last_checkpoint_payload(epoch: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model.state_dict(),
            "epoch": int(epoch),
            "norm_stats": {
                "mu": None if mu is None else mu.detach().cpu().tolist(),
                "sigma": None if sigma is None else sigma.detach().cpu().tolist(),
            },
            "cfg": cfg,
        }
        # Optional states for robust restart/resume tooling.
        payload["optimizer"] = opt.state_dict()
        if scheduler is not None:
            try:
                payload["scheduler"] = scheduler.state_dict()
            except Exception:
                pass
        return payload

    diag_log_fields = _diagnostics_csv_fields(diag_cfg_main)
    log_fields = ["epoch", "split", "loss", "mae", *diag_log_fields]

    def _log_row(
        *,
        epoch: int,
        split: str,
        loss_val: float,
        mae_val: float | None,
        stats: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        row = {
            "epoch": int(epoch),
            "split": str(split),
            "loss": float(loss_val),
            "mae": ("" if mae_val is None else float(mae_val)),
        }
        for key in diag_log_fields:
            row[key] = ""
        if stats is not None:
            for key in diag_log_fields:
                if key in stats and stats.get(key, None) is not None:
                    row[key] = float(stats[key])
        return row

    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    print("[INFO] Starting training...")
    total_epochs = int(cfg["train"]["epochs"])
    val_every = int(cfg.get("train", {}).get("validation_every_epochs", 1))
    if val_every < 1:
        raise ValueError("train.validation_every_epochs must be >= 1.")
    print(f"[INFO] Validation cadence: every {val_every} epoch(s) + final epoch.")

    best_val = float("inf")
    TR, VL = [], []
    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        #batch = next(iter(train_loader))

        # unpack: loss, mae, stats
        tr_loss, tr_mae, tr_stats = train_one_epoch_multi_step(
            model, train_loader, opt, cfg, device,
            H=H, W=W, dx=dx, dy=dy, mu=mu, sigma=sigma, epoch_idx=epoch,
            runtime_mesh_policy=runtime_mesh_policy,
        )
        run_validation = ((epoch % val_every) == 0) or (epoch == total_epochs)
        vl_loss = float("nan")
        vl_stats = None
        if run_validation:
            vl_loss, vl_stats = evaluate_one_epoch_multi_step(
                model, val_loader, cfg, device,
                H=H, W=W, dx=dx, dy=dy, mu=mu, sigma=sigma,
                collect_examples=False, epoch_idx=epoch,
                runtime_mesh_policy=runtime_mesh_policy,
            )

        TR.append(tr_loss)
        VL.append(vl_loss if run_validation else float("nan"))
        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(_log_row(epoch=epoch, split="train", loss_val=tr_loss, mae_val=tr_mae, stats=tr_stats))
            if run_validation:
                writer.writerow(_log_row(epoch=epoch, split="val", loss_val=vl_loss, mae_val=None, stats=vl_stats))

        # track best model by validation loss
        if run_validation and (vl_loss < best_val):
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

        if scheduler is not None and run_validation:
            scheduler.step(vl_loss)

        if run_validation:
            print(
                f"[INFO] Epoch {epoch:03d}: "
                f"train {tr_loss:.6f} (MAE {tr_mae:.6f}) | "
                f"val {vl_loss:.6f} | "
                f"{dt:.1f}s | lr={_get_lr(opt):.3e}"
            )
            tr_diag_line = _format_split_diagnostic_summary("train", tr_stats, diag_cfg_main)
            if tr_diag_line is not None:
                print(tr_diag_line)
            vl_diag_line = _format_split_diagnostic_summary("val", vl_stats, diag_cfg_main)
            if vl_diag_line is not None:
                print(vl_diag_line)
        else:
            next_val_epoch = epoch + (val_every - (epoch % val_every))
            if next_val_epoch > total_epochs:
                next_val_epoch = total_epochs
            print(
                f"[INFO] Epoch {epoch:03d}: "
                f"train {tr_loss:.6f} (MAE {tr_mae:.6f}) | "
                f"val skipped (every {val_every}; next {next_val_epoch:03d}) | "
                f"{dt:.1f}s | lr={_get_lr(opt):.3e}"
            )
            tr_diag_line = _format_split_diagnostic_summary("train", tr_stats, diag_cfg_main)
            if tr_diag_line is not None:
                print(tr_diag_line)

        do_periodic_ckpt = (checkpoint_every > 0) and (
            (epoch % checkpoint_every) == 0 or (epoch == total_epochs)
        )
        if do_periodic_ckpt:
            torch.save(_build_last_checkpoint_payload(epoch), last_ckpt_path)
            print(f"[INFO] Saved last checkpoint at epoch {epoch:03d}: {last_ckpt_path}")

    # Always write final last-model checkpoint when training loop completes.
    torch.save(_build_last_checkpoint_payload(total_epochs), last_ckpt_path)
    print(f"[INFO] Saved final last checkpoint at epoch {total_epochs:03d}: {last_ckpt_path}")

    plot_loss_curves(
        os.path.join(save_dir, "loss_curves.png"),
        list(range(1, len(TR) + 1)),
        TR, VL
    )

    # -------- final test evaluation --------
    print("test_loader length:", len(test_loader))
    print("=== Running test evaluation ===")
    test_loss, _test_stats = evaluate_one_epoch_multi_step(
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
        collect_examples=False,
        runtime_mesh_policy=runtime_mesh_policy,
    )

    print(f"[TEST] loss={test_loss:.4e}")
    test_diag_line = _format_split_diagnostic_summary("test", _test_stats, diag_cfg_main)
    if test_diag_line is not None:
        print(test_diag_line.replace("[DIAG]", "[TEST-DIAG]", 1))

    step_log_enabled, step_log_path, _step_log_append = _runtime_step_log_settings(cfg)
    if step_log_enabled:
        summary_path = os.path.splitext(step_log_path)[0] + "_summary.txt"
        try:
            _runtime_step_log_write_summary(step_log_path, summary_path)
            print(f"[RUNTIME-MESH] step log summary written: {summary_path}")
        except Exception as e:
            print(f"[RUNTIME-MESH][WARN] failed to write step log summary: {e!r}")

                
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override cfg['train']['save_dir'] for this run.",
    )
    args = ap.parse_args()
    print("Running main...", flush=True)
    main(args.config, args.out_dir)
