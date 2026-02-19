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
import os, io, json, time, zipfile, random, hashlib, sys, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch import optim
from torch.amp import autocast
from torch_geometric.data import Data
from contextlib import nullcontext, contextmanager
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

from dataset import CellRefineTemporalDataset, CellRefineWindowDataset, \
                    preprocess_timesteps_once
from models import FeatureNet, ParcFeatureAdapter
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
from utils.fast_next_diag import maybe_run_fast_next_diag
#from utils.mls import SolveGradientsLST, SolveWeightLST2d, apply_laplacian
import utils.mls as mls



debug_once = False

'''
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
'''
@torch.no_grad()
def _compute_norm_stats_from_loader(loader, device):
    n_total = 0
    mean = None
    M2 = None

    for batch in loader:
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

'''
# IMPORTANT for dynamic AMR: disable caching-by-geometry (otherwise memory grows forever)
_MLS_GRAD = mls.SolveGradientsLST(
    cache_by_geometry=False,
    use_2hop_extension=True,
    use_neighbor_damping=True,
    damping_alpha=0.5,
)

_MLS_LAPW = mls.SolveWeightLST2d(
    polynomial_order=2,
    cache_by_geometry=False,
    use_2hop_extension=True,
    use_neighbor_damping=True,
    damping_alpha=0.5,
)

_MLS_ADV  = mls.AdvectionMLS(_MLS_GRAD)
_MLS_DIFF = mls.DiffusionMLS(_MLS_LAPW)

@torch.no_grad()
def mls_advdiff_terms_abs(
    *,
    x_abs: torch.Tensor,          # (N,F) absolute state on current pred mesh
    pos: torch.Tensor,            # (N,2) centers
    edge_index: torch.Tensor,     # (2,E)
    levels: torch.Tensor,         # (N,)
    dx0: float,
    dy0: float,
    cfg: dict,
    compute_adv: bool,
    compute_diff: bool,
):
    loss_cfg = cfg.get("loss", {}) or {}
    rho_floor = float(loss_cfg.get("rho_floor", 1e-6))
    u_clip    = float(loss_cfg.get("u_clip", 1000.0))
    nu        = float(loss_cfg.get("nu", 0.0))

    # run MLS ops on CPU if requested (helps GPU memory)
    ops_dev = torch.device(loss_cfg.get("mls_ops_device", "cpu"))

    x_ops  = x_abs.to(device=ops_dev, dtype=torch.float32)
    pos_ops = pos.to(device=ops_dev, dtype=torch.float32)
    ei_ops  = edge_index.to(device=ops_dev, dtype=torch.long)

    data = Data(pos=pos_ops, edge_index=ei_ops)

    # velocity from conservative state (assumes [rho,mx,my,E] layout)
    idx = dec.infer_feature_indices(cfg, x_ops.size(1))
    rho = x_ops[:, idx["rho"]].abs().clamp_min(rho_floor)
    mx  = x_ops[:, idx["mx"]]
    my  = x_ops[:, idx["my"]]
    vel = torch.stack([mx / rho, my / rho], dim=1).clamp(-u_clip, u_clip)  # (N,2)

    r_adv = None
    if compute_adv:
        r_adv = _MLS_ADV(x_ops, vel, data)  # (N,F)

    r_diff = None
    if compute_diff:
        r_diff = _MLS_DIFF(x_ops, data)     # (N,F)
        if nu != 0.0:
            r_diff = nu * r_diff

    # area for compatibility with your residual loss codepath
    area = dec.cell_area_from_levels(
        levels.long().to(device=x_abs.device),
        dx0=float(dx0),
        dy0=float(dy0),
        dtype=torch.float32,
        device=x_abs.device,
    )

    # move outputs back to the training device
    if r_adv is not None:
        r_adv = r_adv.to(device=x_abs.device, dtype=torch.float32)
    if r_diff is not None:
        r_diff = r_diff.to(device=x_abs.device, dtype=torch.float32)

    return r_adv, r_diff, area


def mls_terms_from_precomp(
    *,
    x_abs: torch.Tensor,            # (N,F)
    edge_index: torch.Tensor,       # (2,E)
    grad_M_inv: torch.Tensor,       # (N,2,2)
    grad_dX: torch.Tensor,          # (E,2)
    lap_w: torch.Tensor | None,     # (E,) or None
    cfg: dict,
    compute_adv: bool,
    compute_diff: bool,
):
    """
    MPS-safe MLS advection/diffusion terms using precomputed geometry.

    Key changes vs your version:
      - Avoid 3D in-place index_add_ on MPS by flattening (E,2,F)->(E,2F) and using out-of-place index_add.
      - Also avoid in-place index_add_ for diffusion on MPS (optional but recommended).
      - Ensure row/col and grad_dX/lap_w live on the same device as x_abs.
    """
    loss_cfg = cfg.get("loss", {}) or {}
    rho_floor = float(loss_cfg.get("rho_floor", 1e-6))
    u_clip    = float(loss_cfg.get("u_clip", 1000.0))

    dev = x_abs.device
    dtype = x_abs.dtype

    # velocity from conservative state (assumes [rho,mx,my,E] layout)
    idx = dec.infer_feature_indices(cfg, x_abs.size(1))
    rho = x_abs[:, idx["rho"]].abs().clamp_min(rho_floor)
    mx  = x_abs[:, idx["mx"]]
    my  = x_abs[:, idx["my"]]
    vel = torch.stack([mx / rho, my / rho], dim=1).clamp(-u_clip, u_clip)  # (N,2)

    # edge indices on-device
    edge_index = edge_index.to(dev, dtype=torch.long)
    row, col = edge_index[0], edge_index[1]

    N = int(x_abs.size(0))
    F = int(x_abs.size(1))

    r_adv = None
    if compute_adv:
        # Bring geometry to device/dtype
        dX = grad_dX.to(dev, dtype=torch.float32)  # keep dX as float32 for stability
        M_inv = grad_M_inv.to(dev, dtype=torch.float32)

        # du uses model dtype; compute in float32 to match geometry + avoid MPS dtype oddities
        du = (x_abs[col] - x_abs[row]).to(torch.float32)                    # (E,F)
        V_edge = dX.unsqueeze(2) * du.unsqueeze(1)                          # (E,2,F) float32

        if dev.type == "mps":
            # MPS workaround: flatten and out-of-place index_add
            V_edge2 = V_edge.reshape(V_edge.size(0), -1).contiguous()       # (E,2F)
            V_node2 = torch.zeros((N, V_edge2.size(1)), device=dev, dtype=V_edge2.dtype)
            V_node2 = V_node2.index_add(0, row, V_edge2)                    # (N,2F)
            V_node = V_node2.view(N, 2, F)                                  # (N,2,F)
        else:
            V_node = torch.zeros((N, 2, F), device=dev, dtype=V_edge.dtype)
            V_node.index_add_(0, row, V_edge)

        grads = torch.einsum("nij,njf->nif", M_inv, V_node)                 # (N,2,F) float32
        r_adv_f32 = vel[:, 0:1].to(torch.float32) * grads[:, 0, :] + vel[:, 1:2].to(torch.float32) * grads[:, 1, :]
        r_adv = r_adv_f32.to(dtype)                                         # back to model dtype

    r_diff = None
    if compute_diff and (lap_w is not None):
        w = lap_w.to(dev, dtype=torch.float32)                              # (E,)
        du = (x_abs[col] - x_abs[row]).to(torch.float32)                    # (E,F)
        contrib = w.unsqueeze(1) * du                                       # (E,F) float32

        if dev.type == "mps":
            # MPS: prefer out-of-place index_add
            r_diff_f32 = torch.zeros((N, F), device=dev, dtype=contrib.dtype).index_add(0, row, contrib)
        else:
            r_diff_f32 = torch.zeros((N, F), device=dev, dtype=contrib.dtype)
            r_diff_f32.index_add_(0, row, contrib)

        nu = float(loss_cfg.get("nu", 0.0))
        if nu != 0.0:
            r_diff_f32 = nu * r_diff_f32

        r_diff = r_diff_f32.to(dtype)

    return r_adv, r_diff
'''
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

    w = dec.cell_area_from_levels(lev, dx0=float(dx0), dy0=float(dy0),
                                  dtype=torch.float32, device=torch.device("cpu"))  # (N,)

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
'''
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
'''
def _forward_main_head_with_edge_attr(model, x_in, edge_index, edge_attr=None, *, force_fp32: bool = True):
    """
    Backward compatible:
      - If model ignores edge_attr, it still works.
      - If force_fp32=True, we run THIS forward in fp32 (autocast disabled) to avoid fp16 NaNs.
    """
    if edge_attr is not None and not hasattr(_forward_main_head_with_edge_attr, "_printed"):
        _forward_main_head_with_edge_attr._printed = True
        print("[DEC-CHK] Forward called with edge_attr present.")

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

        if EA is None:
            y = _forward_main_head(model, X, edge_index)
        else:
            try:
                # model supports edge_attr
                y = _forward_main_head(model, X, edge_index, edge_attr=EA)
            except TypeError:
                # model doesn’t accept edge_attr yet
                y = _forward_main_head(model, X, edge_index)

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

def sanitize_state_for_ops(x_abs: torch.Tensor, cfg: dict, rho_floor=1e-6, E_floor=1e-6):
    loss = cfg.get("loss", {}) or {}
    u_clip = float(loss.get("u_clip", 1e3))

    idx = dec.infer_feature_indices(cfg, x_abs.size(1))
    out = x_abs.clone()

    rho = out[:, idx["rho"]].clamp_min(rho_floor)
    E   = out[:, idx["E"]].clamp_min(E_floor)

    mx = out[:, idx["mx"]]
    my = out[:, idx["my"]]

    # enforce |u| <= u_clip by clamping momenta given rho
    if u_clip > 0:
        mx = mx.clamp(-u_clip * rho, u_clip * rho)
        my = my.clamp(-u_clip * rho, u_clip * rho)

    out[:, idx["rho"]] = rho
    out[:, idx["E"]]   = E
    out[:, idx["mx"]]  = mx
    out[:, idx["my"]]  = my
    return out

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

    x = x_abs.clone()
    x[:, idx["rho"]] = x[:, idx["rho"]].clamp_min(rho_floor)
    x[:, idx["E"]]   = x[:, idx["E"]].clamp_min(E_floor)
    return x


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

    diag_state = {"norm": {}, "phys": {}}

    model.train()
    total_loss_accum = 0.0
    mae_accum = 0.0
    n_steps = 0

    dbg = cfg.get("debug", {})
    nan_watch = bool(dbg.get("nan_watch", True))          # turn on/off
    nan_watch_first_only = bool(dbg.get("nan_first_only", True))

    budget_rows = []

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
        return _enforce_physical_state(x_abs, rho_floor=rho_floor, E_floor=E_floor)

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

    predict_type = str(cfg.get("model", {}).get("predict_type", "rate")).lower()
    if predict_type != "rate":
        raise RuntimeError(f"PARC/Variant-B implementation below assumes predict_type='rate', got '{predict_type}'")

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

    # For MLS, use CPU ops to avoid GPU OOMs
    mls_ops_device = torch.device(loss_cfg.get("mls_ops_device", "cpu"))
    mls_scale_by_dt = bool(loss_cfg.get("mls_scale_by_dt", False))

    IDX = {"rho": 0, "mx": 1, "my": 2, "E": 3}  # for velocity extraction inside MLS

    if scaler is None and use_amp and device.type == "cuda":
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    amp_ctx = ((lambda: torch.amp.autocast(device_type="cuda", enabled=True))
               if use_amp and device.type == "cuda"
               else (lambda: torch.autocast("cpu", enabled=False)))

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

        # Only DEC needs edge_attr. MLS does not.
        if need_phy and (not use_mls) and (pred_ea_list is None):
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

            # area on the pred mesh for THIS step
            # (use float32 on device; we move to CPU in the integrals function anyway)
            area_pred = dec.cell_area_from_levels(
                pred_levels.long().to(device),
                dx0=float(dx),
                dy0=float(dy),
                dtype=torch.float32,
                device=device,
            )  # [N]

            dbg_cfg = cfg.get("debug", {}) or {}
            budget_watch = bool(dbg_cfg.get("budget_watch", True))
            budget_every = int(dbg_cfg.get("budget_every", 1))  # set to >1 to reduce spam

            sigma_f32 = _as_stat_tensor(sigma, device=device, dtype=torch.float32)
            mu_f32    = _as_stat_tensor(mu,    device=device, dtype=torch.float32)  # if you ever need it here

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

                                '''
                                if use_mls:
                                    r_adv_abs, r_diff_abs, area = mls_advdiff_terms_abs(
                                        x_abs=x_for_ops.float(),
                                        pos=pred_centers,
                                        edge_index=pei.long(),
                                        levels=pred_levels,
                                        dx0=float(dx),
                                        dy0=float(dy),
                                        cfg=cfg,
                                        compute_adv=need_adv,
                                        compute_diff=need_diff,
                                    )
                                '''
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

                                    r_phy_abs = base

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

                """
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
                """
                # ----- build node input X (PARC appends operator inputs) -----
                x_in = _build_X(norm_in, pred_centers, pred_levels, cfg)

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

                    #if (parc_extra is not None) and (parc_extra.numel() > 0):
                        # parc_extra should already be dtype=x_in.dtype per your call, but keep this safe
                    #    x_in = torch.cat([x_in, parc_extra.to(dtype=x_in.dtype)], dim=1)
                    '''
                    if parc_extra is not None and parc_extra.numel() > 0:
                        if hasattr(model, "parc_adapter") and (model.parc_adapter is not None) and use_adapter:
                            parc_extra = model.parc_adapter(parc_extra)                       
                        x_in = torch.cat([x_in, parc_extra.to(dtype=x_in.dtype)], dim=1)
                    '''
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
                                    parc_extra[:, :da] = 0.0

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
                    base_X = _build_X(norm_in, pred_centers, pred_levels, cfg)

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


                # ==========================================================
                # FAST NEXT DIAGNOSTIC (RUN ONCE, STEP 0 ONLY)
                #   - norm-space call: best for dt scale + channel permutation
                #   - phys-space call: best to detect normalization mismatch
                # NOTE: do NOT pass the AMP GradScaler "scaler" here.
                # ==========================================================
                '''
                if step_k == 0:
                    print("\n[FAST NEXT DIAGNOSTIC] Running fast next diagnostic at step 0")
                    # (A) Norm-space: consistent with your actual loss space (y_pred vs rate_target)
                    #     This is the cleanest place to detect:
                    #       - dt vs 1/dt vs delta-vs-rate bugs (w.r.t dt_hat)
                    #       - channel permutation
                    maybe_run_fast_next_diag(
                        cfg=cfg,
                        state=diag_state["norm"],
                        pred_rate=y_pred,          # (N,F) predicted RATE in *normalized* space
                        gt_t_feats=norm_in,        # (N,F) GT(t) in *normalized* space
                        gt_tp1_feats=norm_tgt,     # (N,F) GT(t+1) in *normalized* space
                        dt=dt_hat,                # the exact dt used for rate_target + integration
                        mu=None,                  # IMPORTANT: already normalized
                        sigma=None,               # IMPORTANT: already normalized
                        scaler=None,              # IMPORTANT: do NOT pass AMP GradScaler
                        feature_names=["rho", "mx", "my", "E"],
                        tag="FASTDIAG/norm_space/step0",
                    )

                    # (B) Abs-space GT + mu/sigma provided:
                    #     This is the fastest way to detect a normalization mismatch:
                    #       “pred lives in normalized space, but you’re comparing/plotting in abs (or vice versa)”
                    maybe_run_fast_next_diag(
                        cfg=cfg,
                        state=diag_state["phys"],
                        pred_rate=y_pred,          # still normalized-rate output
                        gt_t_feats=x_in_abs,       # (N,F) GT(t) in *absolute* space (step0 is teacher by construction)
                        gt_tp1_feats=x_tgt_abs,    # (N,F) GT(t+1) in *absolute* space
                        dt=dt_hat,                # keep dt consistent with your supervision/integration path
                        mu=mu,
                        sigma=sigma,
                        scaler=None,              # IMPORTANT: do NOT pass AMP GradScaler
                        feature_names=["rho", "mx", "my", "E"],
                        tag="FASTDIAG/abs_space_with_mu_sigma/step0",
                    )
                '''

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
                if need_phy and dec_resid_w > 0.0 and (r_phy_abs is not None):
                    with torch.autocast(device_type=device.type, enabled=False):
                        phy_loss = dec.physics_residual_loss_delta(
                            y_pred_abs=y_pred_abs.float(),
                            x_in_abs=x_in_abs.float(),
                            dt_phys=dt_phys.float(),          # <-- correct dt for this step
                            r_phy_abs=r_phy_abs.float(),
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

        '''
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
        '''
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

        # ======================
        # STEPS 1..K-2
        # ======================
        for k in range(1, K - 1):
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

            '''
            # ---- DEBUG: IDW map sanity (prints once when blowup starts) ----
            if (k >= 6) and (not hasattr(train_one_epoch_multi_step, "_printed_idw_sanity")):
                train_one_epoch_multi_step._printed_idw_sanity = True
                print(f"\n[IDW-SANITY] k={k}")
                _tstats("pred_feats_k (before IDW)", pred_feats_k)

                w = w_km1
                _tstats("w_km1", w)
                # row sums
                row_sum = w.sum(dim=1)
                _tstats("w row_sum", row_sum)

                print("  w min:", float(w.min().detach().cpu()), "max:", float(w.max().detach().cpu()))
                print("  row_sum min:", float(row_sum.min().detach().cpu()), "max:", float(row_sum.max().detach().cpu()))
                print("  any w<0:", bool((w < 0).any().detach().cpu()))
                print("  any row_sum<=0:", bool((row_sum <= 0).any().detach().cpu()))
                print("  any nonfinite w:", bool((~torch.isfinite(w)).any().detach().cpu()))
            '''
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

        # ===== backward + step once per window =====
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if window_loss_graph is not None:
            '''
            if scaler is not None:
                scaler.scale(window_loss_graph).backward()
                scaler.step(opt)
                scaler.update()
            else:
                window_loss_graph.backward()
                opt.step()
            '''
            if scaler is not None:
                scaler.scale(window_loss_graph).backward()
                scaler.unscale_(opt)  # <-- required before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                window_loss_graph.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            opt.step()

        total_loss_accum += window_loss
        mae_accum        += window_mae

    out_dir = cfg["train"]["save_dir"]
    budget_path = os.path.join(out_dir, "train_budgets.csv")

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
    budget_csv_path: str | None = None,
    write_budgets: bool = False,
):

    model.eval()

    budget_rows = []  
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

    # ---- metric accumulators (weighted by cell area) ----
    step_wsum = []
    step_mae_num = []
    step_mse_num = []
    step_gt2_num = []
    by_t = {}

    total_loss_accum = 0.0
    n_steps_total = 0
    examples = [] if collect_examples else None

    # ---- budgets (CSV) ----
    budget_rows = []
    budget_cfg = cfg.get("eval", {}) or {}
    save_budgets = bool(budget_cfg.get("save_budgets_csv", True))

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
            dtype=x_abs.dtype, device=x_abs.device
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

                                r_phy_abs = base

                    # Build node inputs
                    x_in = _build_X(norm_in, pred_centers, pred_levels, cfg)

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

                        #if (parc_extra is not None) and (parc_extra.numel() > 0):
                        #    x_in = torch.cat([x_in, parc_extra.to(dtype=x_in.dtype)], dim=1)
                        '''
                        if parc_extra is not None and parc_extra.numel() > 0:
                            if hasattr(model, "parc_adapter") and (model.parc_adapter is not None) and use_adapter:
                                parc_extra = model.parc_adapter(parc_extra)
                            x_in = torch.cat([x_in, parc_extra.to(dtype=x_in.dtype)], dim=1)
                        '''
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
                                        parc_extra[:, :da] = 0.0                                

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

                    # targets / loss in fp32 for stability
                    with torch.autocast(device_type=device.type, enabled=False):
                        delta_target_f32 = (norm_tgt - norm_in).to(dtype=torch.float32)
                        dt_hat_f32 = dt_hat.to(dtype=torch.float32)
                        rate_target_f32 = delta_target_f32 / dt_hat_f32.clamp_min(1e-12)

                        y_pred_f32 = y_pred.to(dtype=torch.float32)
                        center_loss = (F.huber_loss(y_pred_f32, rate_target_f32, delta=huber_delta)
                                        if use_huber else F.mse_loss(y_pred_f32, rate_target_f32))

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
                                dt_phys=dt_phys.float(),
                                r_phy_abs=r_phy_abs.float(),
                                area=area.float(),
                                #sigma=(sigma.to(device, dtype=torch.float32) if (sigma is not None and torch.is_tensor(sigma)) else None),
                                sigma=sigma_f32,
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

'''
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

    amp_ctx = ((lambda: torch.amp.autocast(device_type="cuda", enabled=True))
               if use_amp and device.type == "cuda"
               else (lambda: torch.autocast("cpu", enabled=False)))

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
                    """
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
                    """
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
                    """
                    delta_target = norm_tgt - norm_in
                    rate_target = delta_target / dt_hat.clamp_min(1e-12)

                    center_loss = (F.huber_loss(y_pred, rate_target, delta=huber_delta)
                                   if use_huber else F.l1_loss(y_pred, rate_target))

                    y_pred_abs = _maybe_denorm(norm_in + y_pred * dt_hat, mu, sigma)
                    """
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

                #pred_feats_k = y_pred_abs_k
                pred_feats_k = _enforce_physical_state(y_pred_abs_k, rho_floor=1e-6, E_floor=1e-6)

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
'''

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

'''
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
'''
def build_model_from_cfg(cfg, device):
    use_cols = cfg.get("features", {}).get("use_columns", [0, 1, 2, 3])
    Fdim = int(cfg.get("features", {}).get("num_features", len(use_cols)))

    b = cfg.get("features", {}).get("build", {})
    in_ch = Fdim + (2 if b.get("use_pos", True) else 0) + (1 if b.get("use_level", True) else 0)

    loss = cfg.get("loss", {}) or {}
    parc_on = bool(loss.get("parc", False) or loss.get("parc_inputs", False))
    if parc_on:
        in_ch += dec.parc_extra_in_channels(cfg, Fdim)

    out_ch = Fdim

    model = FeatureNet(
        in_channels=in_ch,
        out_channels=out_ch,
        hidden=int(cfg.get("model", {}).get("hidden", 128)),
        layers=int(cfg.get("model", {}).get("layers", 3)),
        dropout=float(cfg.get("model", {}).get("dropout", 0.1)),
        make_score_head=True,
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

def main(config_path: str | None = None):
    # -------- load & normalize config --------
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config_feature_first.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

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
    print("  sigma min/max:", float(sigma.min().detach().cpu()), float(sigma.max().detach().cpu()))
    print("  any sigma<=0:", bool((sigma <= 0).any().detach().cpu()))
    print("  any nonfinite mu:", bool((~torch.isfinite(mu)).any().detach().cpu()))
    print("  any nonfinite sigma:", bool((~torch.isfinite(sigma)).any().detach().cpu()))

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
        #batch = next(iter(train_loader))

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

    # Save config file                                                                                                                                                        
    save_config(os.path.join(cfg["train"]["save_dir"]), cfg)
                
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config")
    args = ap.parse_args()
    print("Running main...", flush=True)
    main(args.config)

