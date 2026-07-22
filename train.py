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
from collections import OrderedDict
import inspect
import os, io, json, time, zipfile, random, sys, csv, datetime, re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch import optim
from torch_geometric.data import Data
from contextlib import nullcontext
import numpy as np

from dataset import CellRefineWindowDataset
from models import FeatureNet, FluxGraphNetModel, MeshGraphNetModel, ParcFeatureAdapter, SAGEConvModel
from amr_policy import coarse_aggregate_from_dynamic, predict_masks_hierarchical_from_gt_gradients
            
from plots import plot_loss_curves, amr_composite_to_finest_grid
from utils_geom import build_idw_map, apply_idw_map, apply_precomputed_idw_map, dynamic_cells_from_parent_masks

from pretrain import (
    precompute_uniform_mesh_in_memory,
    CollateWithPrecompute,
    CollateWithUniformStaticNoPrecomp,
    build_amr_face_adjacency_edges,
    build_amr_local_knn_edges,
    dec_edge_attr_for_dyadic_quads,
    _clip_pred_mesh_to_wedge,
)

from utils.precomp_h5 import LazyPrecompH5
from utils.chunk_sidecar import ChunkSidecarH5, derive_sidecar_path
import utils.dec_ops as dec
#from utils.mls import SolveGradientsLST, SolveWeightLST2d, apply_laplacian
import utils.mls as mls

try:
    import h5py
except Exception:
    h5py = None



debug_once = False

def _get_refine_ratio(cfg: dict) -> int:
    pol = cfg.get("policy", {}) or {}
    mesh = cfg.get("mesh", {}) or {}
    rr = pol.get("refine_ratio", mesh.get("refine_ratio", 2))
    rr = int(rr)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {rr}")
    return rr


def _cfg_bool_strict(value: Any, *, key: str) -> bool:
    """
    Parse a config bool safely.
    Accepts bool, 0/1, and common true/false strings.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in ("1", "true", "yes", "y", "on"):
            return True
        if raw in ("0", "false", "no", "n", "off"):
            return False
    raise ValueError(
        f"{key} must be a boolean (or 0/1, true/false string). Got: {value!r}"
    )


def _runtime_mesh_enabled_from_cfg(cfg: Dict[str, Any]) -> bool:
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    return _cfg_bool_strict(rt_cfg.get("enabled", False), key="train.runtime_mesh.enabled")


def _physics_inputs_enabled_from_loss_cfg(loss_cfg: Dict[str, Any] | None) -> bool:
    loss = loss_cfg or {}
    raw = loss.get("physics_inputs_enabled", True)
    return _cfg_bool_strict(raw, key="loss.physics_inputs_enabled")


def _physics_inputs_active_from_loss_cfg(loss_cfg: Dict[str, Any] | None) -> bool:
    return _physics_inputs_enabled_from_loss_cfg(loss_cfg)


def _enforce_cartesian_project_contract(cfg: Dict[str, Any]) -> None:
    """
    Physics-Aware-cartesian is intentionally fixed-mesh:
      - train.mesh_mode must be 'uniform'
      - train.runtime_mesh.enabled must be false
    """
    train_cfg = cfg.setdefault("train", {})
    mesh_mode = str(train_cfg.get("mesh_mode", "uniform")).strip().lower()
    if mesh_mode != "uniform":
        raise RuntimeError(
            "Cartesian project contract violation: train.mesh_mode must be 'uniform'. "
            f"Got {mesh_mode!r}."
        )
    rt_cfg = train_cfg.setdefault("runtime_mesh", {})
    rt_enabled = _runtime_mesh_enabled_from_cfg(cfg)
    if rt_enabled:
        raise RuntimeError(
            "Cartesian project contract violation: train.runtime_mesh.enabled must be false. "
            "Runtime remeshing paths are disabled in this repo."
        )
    # Normalize value so downstream code never sees string-like booleans.
    rt_cfg["enabled"] = False

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
    print_progress: bool = True,
    progress_every: int = 25,
):
    dev = torch.device(device)
    start_t = time.time()
    total_batches = None
    try:
        total_batches = int(len(loader))
    except Exception:
        total_batches = None

    if print_progress:
        total_txt = str(total_batches) if total_batches is not None else "unknown"
        print(
            f"[NORM] Computing normalization stats on device={dev} "
            f"(batches={total_txt}, progress_every={max(1, int(progress_every))}).",
            flush=True,
        )

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

        if print_progress:
            b_done = int(batch_idx) + 1
            pe = max(1, int(progress_every))
            should_print = (
                (b_done == 1)
                or (b_done % pe == 0)
                or (total_batches is not None and b_done >= total_batches)
            )
            if should_print:
                elapsed = max(time.time() - start_t, 1e-9)
                rate = float(b_done) / elapsed
                if total_batches is not None:
                    remaining = max(0, int(total_batches) - b_done)
                    eta_s = float(remaining) / max(rate, 1e-9)
                    eta_txt = f"{eta_s:.1f}s"
                    prog_txt = f"{b_done}/{int(total_batches)}"
                else:
                    eta_txt = "unknown"
                    prog_txt = f"{b_done}/?"

                cuda_txt = ""
                if dev.type == "cuda" and torch.cuda.is_available():
                    didx = dev.index if dev.index is not None else torch.cuda.current_device()
                    mem_alloc_mb = float(torch.cuda.memory_allocated(didx)) / (1024.0 * 1024.0)
                    mem_res_mb = float(torch.cuda.memory_reserved(didx)) / (1024.0 * 1024.0)
                    cuda_txt = (
                        f" | cuda_mem_alloc_mb={mem_alloc_mb:.1f} "
                        f"cuda_mem_reserved_mb={mem_res_mb:.1f}"
                    )

                print(
                    f"[NORM] progress {prog_txt} | elapsed={elapsed:.1f}s "
                    f"| rate={rate:.2f} batches/s | eta={eta_txt}{cuda_txt}",
                    flush=True,
                )

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
    if print_progress:
        elapsed = max(time.time() - start_t, 1e-9)
        print(
            f"[NORM] Done in {elapsed:.1f}s (samples={int(n_total)} features={int(mean.numel())}).",
            flush=True,
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
    advection_type = str(loss_cfg.get("advection_type", "scalar")).strip().lower()
    if compute_adv and advection_type != "scalar":
        raise RuntimeError(
            "MLS backend only supports loss.advection_type='scalar'. "
            f"Got advection_type={advection_type!r}. "
            "Use loss.physics_backend='dec' for Euler advection."
        )
    rho_floor = float(loss_cfg.get("rho_floor", 1e-6))
    u_clip    = float(loss_cfg.get("u_clip", 1000.0))
    nu        = float(loss_cfg.get("nu", 0.0))

    # Choose where MLS runs (CPU recommended on MPS; avoids scatter/index_add headaches)
    ops_dev = torch.device(loss_cfg.get("mls_ops_device", "cpu"))

    x_ops   = x_abs.to(device=ops_dev, dtype=torch.float32)
    pos_ops = pos.to(device=ops_dev, dtype=torch.float32)
    ei_ops  = edge_index.to(device=ops_dev, dtype=torch.long)

    data = Data(pos=pos_ops, edge_index=ei_ops)

    # velocity from whichever state representation is configured
    vel = dec.compute_velocity_from_state(x_ops, cfg, eps=rho_floor).clamp(-u_clip, u_clip)  # (N,2)

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

    # -------- sanitize state for ops --------
    x_ops = sanitize_state_for_ops(x_in_abs.clone(), cfg, rho_floor=rho_floor, E_floor=E_floor)

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

    # sanitize state for ops
    x_ops = sanitize_state_for_ops(x_in_abs.clone(), cfg, rho_floor=rho_floor, E_floor=E_floor)

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


def _print_training_device_report(cfg: Dict[str, Any], device: torch.device) -> None:
    """
    Print a concise startup report indicating whether training will run on GPU.
    """
    dev = torch.device(device)
    requested = str(cfg.get("device", "cpu"))
    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    gpu_active = dev.type in ("cuda", "mps")

    print(
        f"[DEVICE] requested={requested!r} selected={str(dev)!r} "
        f"gpu_active={'yes' if gpu_active else 'no'} backend={dev.type}",
        flush=True,
    )
    print(
        f"[DEVICE] availability: cuda={cuda_available} mps={mps_available}",
        flush=True,
    )

    if dev.type == "cuda":
        if not cuda_available:
            print(
                "[DEVICE][WARN] selected CUDA device but torch.cuda.is_available() is False.",
                flush=True,
            )
            return
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        total_gb = float(props.total_memory) / (1024.0 ** 3)
        print(
            f"[DEVICE] CUDA index={int(idx)} name={name} "
            f"cc={int(props.major)}.{int(props.minor)} vram_gb={total_gb:.2f}",
            flush=True,
        )
    elif dev.type == "mps":
        print("[DEVICE] Apple Metal Performance Shaders (MPS) backend active.", flush=True)
    else:
        print("[DEVICE] CPU backend active (no GPU acceleration).", flush=True)


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


class _SegmentedPrecompSeq:
    def __init__(self, owner: "SegmentedPrecompView", key: str):
        self.owner = owner
        self.key = key

    def __len__(self):
        return int(self.owner.T)

    def __getitem__(self, t: int):
        return self.owner._get(self.key, t)


class SegmentedPrecompView(dict):
    """
    Dict-like precomp view that stitches multiple source-specific precomp objects
    into one global-timeline precomp for CollateWithPrecompute.
    """

    _SEQ_KEYS = (
        "pred_centers",
        "pred_levels",
        "pred_parents",
        "pred_ei",
        "pred_edge_attr",
        "mask_pred",
        "feat_t_on_pred",
        "feat_tp1_on_pred",
        "pred2pred_idx",
        "pred2pred_w",
    )

    def __init__(self, *, segments: List[Dict[str, Any]], T: int, H: int, W: int):
        super().__init__()
        self.T = int(T)
        self.H = int(H)
        self.W = int(W)
        self._segments = sorted(list(segments), key=lambda s: int(s["t_start"]))
        self._starts = [int(s["t_start"]) for s in self._segments]
        self._ends = [int(s["t_end"]) for s in self._segments]

        for k in self._SEQ_KEYS:
            super().__setitem__(k, _SegmentedPrecompSeq(self, k))

        # Optional scalar metadata passthrough (best effort).
        layout_val = None
        for seg in self._segments:
            pre = seg.get("precomp", None)
            if pre is None:
                continue
            v = pre.get("pred_edge_attr_layout", None) if isinstance(pre, dict) else None
            if callable(v):
                try:
                    v = v()
                except Exception:
                    v = None
            if hasattr(v, "get"):
                try:
                    v = v.get()
                except Exception:
                    pass
            if isinstance(v, str) and v != "":
                layout_val = v
                break
        super().__setitem__("pred_edge_attr_layout", layout_val)

    def _locate_segment(self, t: int) -> Dict[str, Any] | None:
        if t < 0 or t >= self.T:
            return None
        # Linear scan is fine here (small #sources). Keep it explicit.
        for i, (s, e) in enumerate(zip(self._starts, self._ends)):
            if s <= t <= e:
                return self._segments[i]
        return None

    def _get(self, key: str, t: int):
        seg = self._locate_segment(int(t))
        if seg is None:
            return None
        pre = seg.get("precomp", None)
        if pre is None:
            return None
        t_local = int(t) - int(seg["t_start"])
        seq = pre.get(key, None) if isinstance(pre, dict) else None
        if seq is None:
            return None
        try:
            return seq[t_local]
        except Exception:
            return None

    def close(self):
        for seg in self._segments:
            pre = seg.get("precomp", None)
            close_fn = getattr(pre, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

def _compute_budget_row(*, x_abs: torch.Tensor, levels: torch.Tensor | None, dx0: float, dy0: float, cfg: dict):
    """
    Compute global integrals on the mesh defined by `levels`.
    x_abs: (N,F) absolute conserved variables on THAT mesh.
    levels: (N,) refinement levels for that same mesh.
    Returns: area, mass, mom_x, mom_y, energy  (floats)
    """
    # Keep everything in float32 for MPS safety
    x = x_abs.detach().to(device="cpu", dtype=torch.float32)
    if levels is None:
        w = x.new_full((x.size(0),), float(dx0) * float(dy0))
    else:
        lev = levels.detach().to(device="cpu", dtype=torch.int64).view(-1)
        w = dec.cell_area_from_levels(
            lev,
            dx0=float(dx0),
            dy0=float(dy0),
            dtype=torch.float32,
            device=torch.device("cpu"),
            refine_ratio=_get_refine_ratio(cfg),
        )  # (N,)

    state = dec.state_views(x, cfg)
    rho = state["rho"]
    mx = state["mx"]
    my = state["my"]
    E = state["E_tot"]

    area   = float(w.sum().item())
    mass   = float((w * rho).sum().item())
    mom_x  = float((w * mx ).sum().item())
    mom_y  = float((w * my ).sum().item())
    energy = float((w * E  ).sum().item())
    return area, mass, mom_x, mom_y, energy

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


def _canonical_gradient_norm_mode(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in ("none", "off", "false", "0", ""):
        return "none"
    if s in ("log1p", "log", "log1p_only"):
        return "log1p"
    if s in ("zscore", "standardize", "standardise", "std"):
        return "zscore"
    if s in ("log1p_zscore", "log1p-zscore", "log+zscore", "log_zscore"):
        return "log1p_zscore"
    raise ValueError(
        f"Unknown runtime CNN gradient_norm_mode: {raw!r}. "
        "Expected one of: auto, none, log1p, zscore, log1p_zscore"
    )


def _canonical_feature_norm_mode(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in ("none", "off", "false", "0", ""):
        return "none"
    if s in ("zscore", "standardize", "standardise", "std"):
        return "zscore"
    raise ValueError(
        f"Unknown runtime CNN feature_norm_mode: {raw!r}. "
        "Expected one of: auto, none, zscore"
    )


def _parse_feature_norm_stats(
    raw,
    *,
    n_channels: int | None = None,
) -> Dict[str, List[float]] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    if "mean" not in raw or "std" not in raw:
        return None
    mean_raw = raw["mean"]
    std_raw = raw["std"]
    if not isinstance(mean_raw, (list, tuple)) or not isinstance(std_raw, (list, tuple)):
        return None
    if len(mean_raw) != len(std_raw):
        return None
    if n_channels is not None and len(mean_raw) != int(n_channels):
        return None
    mean_out: List[float] = []
    std_out: List[float] = []
    for m, s in zip(mean_raw, std_raw):
        mf = float(m)
        sf = float(s)
        if (not math.isfinite(mf)) or (not math.isfinite(sf)):
            return None
        mean_out.append(mf)
        std_out.append(max(sf, 1e-8))
    return {"mean": mean_out, "std": std_out}


def _apply_feature_norm_chw(
    x_chw: torch.Tensor,
    *,
    mode: str,
    stats: Dict[str, List[float]] | None,
    require_stats: bool,
) -> torch.Tensor:
    m = _canonical_feature_norm_mode(mode)
    x = torch.as_tensor(x_chw, dtype=torch.float32)
    if m != "zscore":
        return x.contiguous()

    parsed = _parse_feature_norm_stats(stats, n_channels=int(x.shape[0]))
    if parsed is None:
        if require_stats:
            raise RuntimeError(
                "runtime CNN feature_norm_mode='zscore' requires feature_norm_stats "
                f"with per-channel mean/std (n_channels={int(x.shape[0])})."
            )
        return x.contiguous()

    mean_t = torch.as_tensor(parsed["mean"], dtype=torch.float32, device=x.device).view(-1, 1, 1)
    std_t = torch.as_tensor(parsed["std"], dtype=torch.float32, device=x.device).view(-1, 1, 1)
    return ((x - mean_t) / std_t).contiguous()


def _parse_gradient_norm_stats(raw) -> Dict[str, Dict[str, float]] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Dict[str, float]] = {}
    for key in ("rho", "E"):
        node = raw.get(key, None)
        if not isinstance(node, dict):
            return None
        if "mean" not in node or "std" not in node:
            return None
        mean_v = float(node["mean"])
        std_v = float(node["std"])
        if (not math.isfinite(mean_v)) or (not math.isfinite(std_v)):
            return None
        out[key] = {"mean": float(mean_v), "std": float(max(std_v, 1e-8))}
    return out


def _apply_gradient_norm_flat(
    g_flat: torch.Tensor,
    *,
    mode: str,
    mean: float | None,
    std: float | None,
    require_stats: bool,
) -> torch.Tensor:
    m = _canonical_gradient_norm_mode(mode)
    x = torch.as_tensor(g_flat, dtype=torch.float32)
    if m in ("log1p", "log1p_zscore"):
        x = torch.log1p(x.clamp_min(0.0))
    if m in ("zscore", "log1p_zscore"):
        if mean is None or std is None:
            if require_stats:
                raise RuntimeError(
                    f"runtime CNN gradient_norm_mode={m!r} requires stats, but none were provided."
                )
            return x
        x = (x - float(mean)) / max(float(std), 1e-8)
    return x


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
    pressure_aux_w=0.0,
    pressure_aux_loss=None,
    pressure_consistency_w=0.0,
    pressure_consistency_loss=None,
    y_nn=None,
    y_pred=None,
):
    dtv = _t2f(dt_hat)
    c   = _t2f(center_loss)
    ll  = _t2f(lap_loss)
    tl  = _t2f(tmp_loss)
    pl  = _t2f(phy_loss)
    pal = _t2f(pressure_aux_loss)
    pcl = _t2f(pressure_consistency_loss)
    tot = _t2f(total_loss)

    lap_term = (float(lap_w) * (ll if ll is not None else 0.0)) if lap_w else 0.0
    tmp_term = (float(tmp_w) * (tl if tl is not None else 0.0)) if tmp_w else 0.0
    phy_term = (float(dec_resid_w) * (pl if pl is not None else 0.0)) if dec_resid_w else 0.0
    paux_term = (
        float(pressure_aux_w) * (pal if pal is not None else 0.0)
    ) if pressure_aux_w else 0.0
    pcons_term = (
        float(pressure_consistency_w) * (pcl if pcl is not None else 0.0)
    ) if pressure_consistency_w else 0.0

    yn = None if y_nn is None else _t2f(y_nn.abs().mean())
    yp = None if y_pred is None else _t2f(y_pred.abs().mean())

    print(
        f"[LOSS] {tag} k={step_k:02d} dt_hat={dtv:.4e} | "
        f"center={_fmt(c)} | "
        f"lap={_fmt(lap_term)} (w={lap_w:.2e}, raw={_fmt(ll)}) | "
        f"tmp={_fmt(tmp_term)} (w={tmp_w:.2e}, raw={_fmt(tl)}) | "
        f"dec_resid={_fmt(phy_term)} (w={dec_resid_w:.2e}, raw={_fmt(pl)}) | "
        f"p_aux={_fmt(paux_term)} (w={pressure_aux_w:.2e}, raw={_fmt(pal)}) | "
        f"p_cons={_fmt(pcons_term)} (w={pressure_consistency_w:.2e}, raw={_fmt(pcl)}) | "
        f"TOTAL={_fmt(tot)} | "
        f"dec_use={int(bool(dec_use))} dec_blend_w={dec_blend_w:.2e} | "
        f"|y_nn|={_fmt(yn)} | |y_pred|={_fmt(yp)}"
    )


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


def _include_dt_hat_from_build_cfg(build_cfg: Dict[str, Any] | None) -> bool:
    build_cfg = build_cfg or {}
    if "include_dt_hat" in build_cfg:
        return _cfg_bool_strict(
            build_cfg.get("include_dt_hat"),
            key="features.build.include_dt_hat",
        )
    if "use_dt_hat_fixed" in build_cfg:
        return _cfg_bool_strict(
            build_cfg.get("use_dt_hat_fixed"),
            key="features.build.use_dt_hat_fixed",
        )
    return False


def _to_scalar_float(v: Any) -> float | None:
    if v is None:
        return None
    if torch.is_tensor(v):
        if v.numel() != 1:
            return None
        return float(v.detach().cpu().item())
    try:
        return float(v)
    except Exception:
        return None


def _extract_angle_from_mapping(d: Dict[str, Any]) -> float | None:
    if not isinstance(d, dict):
        return None
    angle_keys = (
        "ramp_angle_deg",
        "ramp_angle",
        "angle_deg",
        "angle",
        "theta_deg",
        "theta",
    )
    for k in angle_keys:
        if k in d:
            val = _to_scalar_float(d.get(k))
            if val is not None:
                return float(val)

    # Nested metadata dicts are common in data payloads.
    for mk in ("meta", "metadata", "attrs", "info"):
        mv = d.get(mk, None)
        if isinstance(mv, dict):
            val = _extract_angle_from_mapping(mv)
            if val is not None:
                return float(val)
    return None


def _extract_angle_from_data_obj(data_obj: Any) -> float | None:
    """
    Try to read a ramp-angle-like scalar from a loaded .pt payload.
    Returns degrees when found.
    """
    if isinstance(data_obj, dict):
        val = _extract_angle_from_mapping(data_obj)
        if val is not None:
            return float(val)
        for k in ("snapshots", "steps", "time_steps", "sequence", "data_list"):
            if k in data_obj and isinstance(data_obj[k], list) and len(data_obj[k]) > 0:
                return _extract_angle_from_data_obj(data_obj[k][0])
        return None

    if isinstance(data_obj, (list, tuple)):
        if len(data_obj) == 0:
            return None
        return _extract_angle_from_data_obj(data_obj[0])

    # PyG Data / custom objects
    for k in ("ramp_angle_deg", "ramp_angle", "angle_deg", "angle", "theta_deg", "theta"):
        if hasattr(data_obj, k):
            val = _to_scalar_float(getattr(data_obj, k))
            if val is not None:
                return float(val)
    if hasattr(data_obj, "meta"):
        meta = getattr(data_obj, "meta")
        if isinstance(meta, dict):
            return _extract_angle_from_mapping(meta)
    return None


def _extract_angle_from_path(path_like: str | None) -> float | None:
    if not path_like:
        return None
    # Parse from filename only (not parent directories).
    s = os.path.basename(str(path_like))
    patterns = (
        r"(?:^|[_\-])ramp[-_]?angle[-_]?(-?\d+(?:\.\d+)?)",
        r"(?:^|[_\-])angle[-_]?(-?\d+(?:\.\d+)?)",
    )
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m is not None:
            try:
                return float(m.group(1))
            except Exception:
                continue

    # Fallback for shock-ramp DMR files named like:
    #   DMR_8_0_60__125x250.h5  -> pressure/token 8_0, ramp angle 60 deg
    #   DMR_5_5_70__125x250.h5  -> pressure/token 5_5, ramp angle 70 deg
    m = re.search(
        r"^DMR[_\-]"
        r"(-?\d+(?:[pP]\d+|\.\d+)?)"
        r"[_\-]"
        r"(-?\d+(?:[pP]\d+|\.\d+)?)"
        r"[_\-]"
        r"(-?\d+(?:[pP]\d+|\.\d+)?)"
        r"(?=(?:[_\-]{1,2}|\.|$))",
        s,
        flags=re.IGNORECASE,
    )
    if m is not None:
        angle = _parse_compact_decimal_token(m.group(3))
        if angle is not None:
            return float(angle)
    return None


def _parse_compact_decimal_token(tok: str | None) -> float | None:
    if tok is None:
        return None
    s = str(tok).strip()
    if s == "":
        return None
    # Support compact decimal style used in folder names: "5p5" -> 5.5
    if ("p" in s.lower()) and ("." not in s):
        # Keep a possible leading sign untouched.
        if s[0] in "+-":
            s = s[0] + s[1:].replace("p", ".").replace("P", ".")
        else:
            s = s.replace("p", ".").replace("P", ".")
    try:
        return float(s)
    except Exception:
        return None


def _dmr_case_key(mach: float, shock_angle: float) -> Tuple[float, float]:
    return (round(float(mach), 8), round(float(shock_angle), 8))


def _combine_dmr_mach_tokens(int_token: str | None, frac_token: str | None) -> float | None:
    """
    DMR filenames encode Mach as two underscore-separated tokens, e.g.
    DMR_7_5_50__125x250.h5 -> Mach=7.5, ShockAngle=50.
    """
    if int_token is None or frac_token is None:
        return None
    left = str(int_token).strip()
    right = str(frac_token).strip()
    if left == "" or right == "":
        return None

    compact_int_frac = (
        re.fullmatch(r"[+-]?\d+", left) is not None
        and re.fullmatch(r"\d+", right) is not None
    )
    if compact_int_frac:
        sign = ""
        digits = left
        if digits[0] in "+-":
            sign = digits[0]
            digits = digits[1:]
        try:
            return float(f"{sign}{digits}.{right}")
        except Exception:
            return None

    # Fallback for less compact names, e.g. DMR_7p5_0_50 or DMR_7.5_0_50.
    mach = _parse_compact_decimal_token(left)
    frac = _parse_compact_decimal_token(right)
    if mach is None:
        return None
    if frac is None or abs(float(frac)) <= 1e-12:
        return float(mach)
    return float(mach)


def _extract_dmr_case_from_path(path_like: str | None) -> Tuple[float, float] | None:
    if not path_like:
        return None
    s = os.path.basename(str(path_like))
    m = re.search(
        r"^DMR[_\-]"
        r"(-?\d+(?:[pP]\d+|\.\d+)?)"
        r"[_\-]"
        r"(-?\d+(?:[pP]\d+|\.\d+)?)"
        r"[_\-]"
        r"(-?\d+(?:[pP]\d+|\.\d+)?)"
        r"(?=(?:[_\-]{1,2}|\.|$))",
        s,
        flags=re.IGNORECASE,
    )
    if m is None:
        return None
    mach = _combine_dmr_mach_tokens(m.group(1), m.group(2))
    shock_angle = _parse_compact_decimal_token(m.group(3))
    if mach is None or shock_angle is None:
        return None
    return float(mach), float(shock_angle)


def _csv_row_value(row: Dict[str, Any], aliases: Sequence[str]) -> Any:
    normalized = {
        str(k).strip().lower().replace(" ", "").replace("_", ""): v
        for k, v in row.items()
    }
    for alias in aliases:
        key = str(alias).strip().lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


def _resolve_dt_csv_path(cfg: Dict[str, Any]) -> str | None:
    data_cfg = cfg.get("data", {}) or {}
    raw = None
    for key in (
        "dt_csv_path",
        "dt_table_path",
        "parameter_study_csv",
        "dt_parameter_csv",
    ):
        if data_cfg.get(key, None) is not None:
            raw = data_cfg.get(key)
            break
    if raw is None:
        return None
    raw_s = str(raw).strip()
    if raw_s == "":
        return None
    return os.path.abspath(os.path.expanduser(raw_s))


def _load_dt_parameter_table(csv_path: str) -> Dict[Tuple[float, float], float]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Configured data.dt_csv_path does not exist: {csv_path}"
        )

    table: Dict[Tuple[float, float], float] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"dt CSV has no header row: {csv_path}")
        for row_idx, row in enumerate(reader, start=2):
            mach_raw = _csv_row_value(row, ("Mach", "mach", "M"))
            angle_raw = _csv_row_value(
                row,
                ("ShockAngle", "shock_angle", "angle", "Angle", "Shock Angle"),
            )
            dt_raw = _csv_row_value(
                row,
                ("Timestep", "time_step", "dt", "delta_t", "DeltaT"),
            )
            if mach_raw is None or angle_raw is None or dt_raw is None:
                raise ValueError(
                    f"dt CSV row {row_idx} must contain Mach, ShockAngle, and Timestep columns."
                )
            try:
                mach = float(mach_raw)
                angle = float(angle_raw)
                dt = float(dt_raw)
            except Exception as exc:
                raise ValueError(
                    f"Could not parse dt CSV row {row_idx}: {row!r}"
                ) from exc
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError(
                    f"dt CSV row {row_idx} has invalid Timestep={dt_raw!r}; expected finite positive dt."
                )
            key = _dmr_case_key(mach, angle)
            prev = table.get(key, None)
            if prev is not None and abs(float(prev) - dt) > 1e-12:
                raise ValueError(
                    f"dt CSV has conflicting Timestep values for Mach={mach:g}, "
                    f"ShockAngle={angle:g}: {prev:g} vs {dt:g}."
                )
            table[key] = float(dt)

    if len(table) == 0:
        raise ValueError(f"dt CSV contains no data rows: {csv_path}")
    return table


def _dt_known_case_summary(table: Dict[Tuple[float, float], float], *, limit: int = 10) -> str:
    keys = sorted(table.keys())
    shown = ", ".join(f"(Mach={m:g}, ShockAngle={a:g})" for m, a in keys[:limit])
    if len(keys) > limit:
        shown += f", ... ({len(keys)} total)"
    return shown


def _build_dt_transitions_from_cfg(
    cfg: Dict[str, Any],
    data_list: Sequence[Any],
    *,
    step_source_ids: Sequence[int] | None = None,
    source_records: Sequence[Dict[str, Any]] | None = None,
    source_paths: Sequence[str] | None = None,
    log_fn: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-transition physical dt values.

    Required mode: data.dt_csv_path points to a parameter-study CSV containing
    Mach, ShockAngle, and Timestep. The DMR case is parsed from each input path.
    """
    n = int(len(data_list))
    if n < 2:
        raise RuntimeError(f"Need at least 2 snapshots to build dt transitions, got {n}.")

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)
        else:
            print(msg, flush=True)

    if step_source_ids is None:
        src_ids_np = np.zeros(n, dtype=np.int64)
    else:
        if len(step_source_ids) != n:
            raise RuntimeError(
                f"step_source_ids length ({len(step_source_ids)}) must match data_list length ({n})."
            )
        src_ids_np = np.asarray(step_source_ids, dtype=np.int64)

    dt_csv_path = _resolve_dt_csv_path(cfg)
    if dt_csv_path is None:
        raise RuntimeError(
            "data.dt_csv_path is required because snapshot time values in the DMR HDF5 files "
            "are not physical time steps. Add this under the data block, for example: "
            '"dt_csv_path": "cache/parameter_study.csv".'
        )

    eps_dt = 1e-12
    dt_list: List[float] = []

    table = _load_dt_parameter_table(dt_csv_path)
    cfg.setdefault("data", {})["dt_csv_path"] = dt_csv_path

    records: List[Dict[str, Any]] = []
    if source_records is not None:
        records = [dict(r) for r in source_records]
    elif source_paths is not None:
        records = [
            {"source_id": int(i), "pt_path": os.path.abspath(os.path.expanduser(str(p)))}
            for i, p in enumerate(source_paths)
        ]
    else:
        raise RuntimeError(
            "data.dt_csv_path is configured, but no source path information was provided "
            "for matching DMR files to CSV rows."
        )

    dt_by_source: Dict[int, float] = {}
    case_by_source: Dict[int, Tuple[float, float]] = {}
    for fallback_sid, rec in enumerate(records):
        sid = int(rec.get("source_id", fallback_sid))
        path = rec.get("pt_path", rec.get("path", None))
        case = _extract_dmr_case_from_path(path)
        if case is None:
            raise RuntimeError(
                "Could not parse DMR Mach/ShockAngle from input path while using data.dt_csv_path. "
                f"path={path!r}, csv={dt_csv_path}"
            )
        key = _dmr_case_key(case[0], case[1])
        if key not in table:
            raise RuntimeError(
                "No Timestep entry in data.dt_csv_path for "
                f"Mach={case[0]:g}, ShockAngle={case[1]:g} parsed from {path!r}. "
                f"Known cases: {_dt_known_case_summary(table)}"
            )
        dt_by_source[sid] = float(table[key])
        case_by_source[sid] = (float(case[0]), float(case[1]))

    valid_same_src: List[float] = []
    for i in range(n - 1):
        sid0 = int(src_ids_np[i])
        sid1 = int(src_ids_np[i + 1])
        if sid0 == sid1:
            if sid0 not in dt_by_source:
                raise RuntimeError(
                    f"No dt CSV mapping was built for source_id={sid0}; "
                    f"available source ids: {sorted(dt_by_source.keys())}"
                )
            dti = float(dt_by_source[sid0])
            valid_same_src.append(dti)
            dt_list.append(dti)
        else:
            dt_list.append(float("nan"))

    if len(valid_same_src) > 0:
        boundary_placeholder_dt = float(np.median(np.asarray(valid_same_src, dtype=np.float64)))
    else:
        boundary_placeholder_dt = float(np.median(np.asarray(list(dt_by_source.values()), dtype=np.float64)))
    dt_list = [
        boundary_placeholder_dt if (not np.isfinite(v) or (v <= eps_dt)) else float(v)
        for v in dt_list
    ]
    dt_arr = np.asarray(dt_list, dtype=np.float64)
    dt_transitions = torch.as_tensor(dt_arr, dtype=torch.float32).contiguous()
    dt_ref = dt_transitions.median()

    cfg.setdefault("data", {})["dt_source"] = "parameter_study_csv"
    cfg["data"]["dt_ref"] = float(dt_ref.detach().cpu().item())

    per_source_items = [
        f"source_id={sid}: Mach={case_by_source[sid][0]:g}, "
        f"ShockAngle={case_by_source[sid][1]:g}, dt={dt_by_source[sid]:.9g}"
        for sid in sorted(dt_by_source.keys())
    ]
    per_source = "; ".join(per_source_items[:8])
    if len(per_source_items) > 8:
        per_source += f"; ... ({len(per_source_items)} sources total)"
    _log(f"[DT] Using physical per-trajectory dt from data.dt_csv_path={dt_csv_path}")
    _log(f"[DT] {per_source}")
    _log(
        "[DT] transitions="
        f"{len(dt_list)} dt_ref={float(dt_ref):.9g} "
        f"min/median/max=({float(np.min(dt_arr)):.9g}, "
        f"{float(np.median(dt_arr)):.9g}, {float(np.max(dt_arr)):.9g})"
    )
    return dt_transitions, dt_ref


def _extract_pressure_from_path(path_like: str | None) -> float | None:
    if not path_like:
        return None
    # Parse from filename only (not parent directories), matching user request.
    s = os.path.basename(str(path_like))
    patterns = (
        r"(?:^|[_\-])p[-_]?(-?\d+(?:[pP]\d+|\.\d+)?)",
        r"(?:^|[_\-])pressure[-_]?(-?\d+(?:[pP]\d+|\.\d+)?)",
    )
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m is not None:
            v = _parse_compact_decimal_token(m.group(1))
            if v is not None:
                return float(v)
    return None

def _load_series_from_h5_path(h5_path: str):
    """
    Load a uniform-grid snapshot sequence from an HDF5 file with datasets:
      - snapshots: [T,H,W,C]
      - time: [T]
      - xy: [H,W,2]
      - channel_names: [C]
    Returns list[dict] snapshots compatible with CellRefineWindowDataset raw-series path.
    """
    if h5py is None:
        raise RuntimeError(
            "Input file is HDF5, but h5py is not available in this environment."
        )

    p = os.path.abspath(os.path.expanduser(str(h5_path)))
    with h5py.File(p, "r") as f:
        h5_attrs: Dict[str, Any] = {}
        for k, v in f.attrs.items():
            if isinstance(v, np.ndarray) and v.shape == ():
                v = v.item()
            if isinstance(v, (bytes, np.bytes_)):
                v = v.decode("utf-8")
            h5_attrs[str(k)] = v
        if "coarsen_meta" in f:
            for k, v in f["coarsen_meta"].attrs.items():
                if isinstance(v, np.ndarray) and v.shape == ():
                    v = v.item()
                if isinstance(v, (bytes, np.bytes_)):
                    v = v.decode("utf-8")
                h5_attrs.setdefault(str(k), v)

        required = ("snapshots", "time", "xy", "channel_names")
        missing = [k for k in required if k not in f]
        if missing:
            raise RuntimeError(
                f"HDF5 file '{p}' is missing required datasets: {missing}"
            )

        d_snap = f["snapshots"]
        d_time = f["time"]
        d_xy = f["xy"]
        d_names = f["channel_names"]

        if d_snap.ndim != 4:
            raise RuntimeError(
                f"Expected snapshots to be rank-4 [T,H,W,C], got shape={tuple(d_snap.shape)}"
            )
        if d_xy.ndim != 3 or int(d_xy.shape[-1]) < 2:
            raise RuntimeError(
                f"Expected xy to be [H,W,2], got shape={tuple(d_xy.shape)}"
            )

        T, Hf, Wf, C = [int(v) for v in d_snap.shape]
        if int(d_time.shape[0]) != T:
            raise RuntimeError(
                f"time length ({int(d_time.shape[0])}) does not match snapshots T ({T})"
            )
        if int(d_xy.shape[0]) != Hf or int(d_xy.shape[1]) != Wf:
            raise RuntimeError(
                "xy shape does not match snapshots spatial shape: "
                f"xy={tuple(d_xy.shape)} vs snapshots[H,W]=({Hf},{Wf})"
            )
        if int(d_names.shape[0]) != C:
            raise RuntimeError(
                f"channel_names length ({int(d_names.shape[0])}) does not match snapshots C ({C})"
            )
        if T < 2:
            raise RuntimeError(f"Need at least 2 snapshots in '{p}', got {T}.")

        channel_names: List[str] = []
        for raw in d_names[()]:
            if isinstance(raw, (bytes, np.bytes_)):
                channel_names.append(raw.decode("utf-8"))
            else:
                channel_names.append(str(raw))

        xy_np = np.asarray(d_xy[..., :2], dtype=np.float32)
        pos_shared = torch.from_numpy(xy_np.reshape(-1, 2))
        row_idx, col_idx = np.indices((Hf, Wf), dtype=np.int64)
        ij_shared = torch.from_numpy(
            np.stack([row_idx, col_idx], axis=-1).reshape(-1, 2)
        )
        times_np = np.asarray(d_time[()], dtype=np.float64).reshape(-1)

        seq: List[Dict[str, Any]] = []
        for t in range(T):
            snap_np = np.asarray(d_snap[t], dtype=np.float32).reshape(-1, C)
            x_t = torch.from_numpy(snap_np)
            snap: Dict[str, Any] = {
                "x": x_t,
                "pos": pos_shared,
                "ij": ij_shared,
                "time": float(times_np[t]),
                "channel_names": channel_names,
                "H": int(Hf),
                "W": int(Wf),
            }
            if h5_attrs:
                snap["attrs"] = h5_attrs
            seq.append(snap)

    return p, seq


def _load_series_from_pt_path(pt_path: str):
    """
    Load a snapshot sequence from:
      - .pt/.pth
      - .zip containing .pt/.pth
      - .h5/.hdf5 (uniform grid snapshots)
    """
    p = os.path.abspath(os.path.expanduser(str(pt_path)))
    if not os.path.exists(p):
        raise FileNotFoundError(f"Input data path not found: {p}")

    if p.lower().endswith((".h5", ".hdf5")):
        return _load_series_from_h5_path(p)

    if p.endswith(".zip"):
        with zipfile.ZipFile(p, "r") as zf:
            member = next(m for m in zf.namelist() if m.endswith(".pt") or m.endswith(".pth"))
            with zf.open(member, "r") as fh:
                buf = io.BytesIO(fh.read())
        obj = torch.load(buf, weights_only=False)
    else:
        obj = torch.load(p, weights_only=False)

    if not isinstance(obj, list):
        raise RuntimeError(
            f"Expected loaded data at '{p}' to be a list of snapshots; got {type(obj)}."
        )
    if len(obj) < 2:
        raise RuntimeError(f"Need at least 2 snapshots in '{p}', got {len(obj)}.")
    return p, obj


def _resolve_pt_path_list(cfg: Dict[str, Any]) -> List[str]:
    data_cfg = cfg.get("data", {}) or {}
    raw_multi = data_cfg.get("pt_paths", None)
    raw_single = data_cfg.get("pt_path", None)
    recursive = _cfg_bool_strict(
        data_cfg.get("pt_path_recursive", False),
        key="data.pt_path_recursive",
    )
    supported_exts = (".pt", ".pth", ".zip", ".h5", ".hdf5")

    paths_raw: List[Any]
    if isinstance(raw_multi, (list, tuple)) and len(raw_multi) > 0:
        paths_raw = list(raw_multi)
    elif isinstance(raw_single, (list, tuple)) and len(raw_single) > 0:
        paths_raw = list(raw_single)
    elif raw_single is not None:
        paths_raw = [raw_single]
    else:
        raise RuntimeError("cfg['data']['pt_path'] (or data.pt_paths) must be provided.")

    out: List[str] = []
    for pr in paths_raw:
        if pr is None:
            continue
        p = os.path.abspath(os.path.expanduser(str(pr)))
        if os.path.isdir(p):
            found: List[str] = []
            if recursive:
                for root, _, files in os.walk(p):
                    for fn in files:
                        if fn.lower().endswith(supported_exts):
                            found.append(os.path.join(root, fn))
            else:
                for fn in os.listdir(p):
                    fp = os.path.join(p, fn)
                    if os.path.isfile(fp) and fn.lower().endswith(supported_exts):
                        found.append(fp)

            found = sorted(os.path.abspath(x) for x in found)
            if len(found) == 0:
                raise RuntimeError(
                    f"Directory data source contains no supported files {supported_exts}: {p}"
                )
            out.extend(found)
        else:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Input data path not found: {p}")
            out.append(p)

    # Preserve first-seen order while dropping duplicates.
    dedup: List[str] = []
    seen: set[str] = set()
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    if len(dedup) == 0:
        raise RuntimeError("No valid entries found in data.pt_path / data.pt_paths.")
    return dedup


def _resolve_mesh_spec_path_for_input(
    mesh_spec_root_or_file: str | None,
    *,
    input_pt_path: str,
) -> str | None:
    """
    Resolve angle-specific mesh spec:
      - if mesh path points to a file -> return that file
      - if mesh path points to a directory -> choose .pt file whose filename angle
        matches the input filename angle
    """
    if mesh_spec_root_or_file is None:
        return None

    root = os.path.abspath(os.path.expanduser(str(mesh_spec_root_or_file)))
    if os.path.isfile(root):
        return root
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"mesh.starting_mesh_path must be an existing file or directory, got: {root}"
        )

    spec_files = sorted(
        os.path.join(root, fn)
        for fn in os.listdir(root)
        if fn.lower().endswith((".pt", ".pth"))
    )
    if len(spec_files) == 0:
        raise RuntimeError(f"No .pt/.pth wedge mesh spec files found in directory: {root}")
    if len(spec_files) == 1:
        return spec_files[0]

    src_angle = _extract_angle_from_path(input_pt_path)
    if src_angle is None:
        raise RuntimeError(
            "Could not infer input ramp angle from input filename for mesh-spec directory matching: "
            f"{os.path.basename(str(input_pt_path))}. "
            "Use filenames containing 'angle-XX' / 'angle_XX', or set mesh.starting_mesh_path to a file."
        )

    tol = 1e-6
    exact = []
    for sp in spec_files:
        a = _extract_angle_from_path(sp)
        if (a is not None) and (abs(float(a) - float(src_angle)) <= tol):
            exact.append(sp)

    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError(
            "Ambiguous wedge mesh spec match in directory "
            f"{root} for input angle={src_angle:g}: {exact}"
        )

    available = sorted(
        {
            float(a)
            for a in (_extract_angle_from_path(sp) for sp in spec_files)
            if a is not None
        }
    )
    raise RuntimeError(
        f"No wedge mesh spec filename in '{root}' matches input angle={src_angle:g} "
        f"(from {os.path.basename(str(input_pt_path))}). "
        f"Available spec angles: {available if len(available) > 0 else 'none'}."
    )


def _extract_xy_from_snapshot_like(snapshot: Any) -> torch.Tensor | None:
    """
    Return centers (N,2) from a snapshot-like object when available.
    """
    keys = ("pos", "xy", "centers", "pos_phys")
    if isinstance(snapshot, dict):
        for k in keys:
            if k in snapshot:
                v = snapshot.get(k)
                t = torch.as_tensor(v) if v is not None else None
                if torch.is_tensor(t) and t.ndim == 2 and t.size(1) >= 2 and t.numel() > 0:
                    return t[:, :2].to(torch.float32)
        return None

    for k in keys:
        if hasattr(snapshot, k):
            v = getattr(snapshot, k)
            t = torch.as_tensor(v) if v is not None else None
            if torch.is_tensor(t) and t.ndim == 2 and t.size(1) >= 2 and t.numel() > 0:
                return t[:, :2].to(torch.float32)
    return None


def _cell_edge_bbox_from_centers(
    centers_xy: torch.Tensor,
    *,
    H: int,
    W: int,
) -> tuple[float, float, float, float, float, float]:
    """
    HDF5 uniform snapshots store cell centers. Convert center min/max to the
    physical cell-edge bbox so dx/dy and parent indexing use the true domain.

    Returns (xmin_edge, xmax_edge, ymin_edge, ymax_edge, dx_center, dy_center).
    """
    if (not torch.is_tensor(centers_xy)) or centers_xy.ndim != 2 or centers_xy.size(1) < 2:
        raise RuntimeError(
            "Expected centers tensor with shape (N,2+) when deriving cell-edge bbox."
        )
    pts = centers_xy[:, :2].detach().to(device="cpu", dtype=torch.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    xb0 = float(x.min().item())
    xb1 = float(x.max().item())
    yb0 = float(y.min().item())
    yb1 = float(y.max().item())

    def _spacing(vals: torch.Tensor, n_expected: int, span: float) -> float:
        uniq = torch.unique(vals)
        uniq, _ = torch.sort(uniq)
        if uniq.numel() > 1:
            diffs = torch.diff(uniq)
            diffs = diffs[diffs > 1e-12]
            if diffs.numel() > 0:
                return float(torch.median(diffs).item())
        if n_expected > 1:
            return float(span / float(n_expected - 1))
        return 0.0

    dx_c = _spacing(x, int(W), xb1 - xb0)
    dy_c = _spacing(y, int(H), yb1 - yb0)
    return (
        xb0 - 0.5 * dx_c,
        xb1 + 0.5 * dx_c,
        yb0 - 0.5 * dy_c,
        yb1 + 0.5 * dy_c,
        dx_c,
        dy_c,
    )


def _fit_ramp_line_from_centers(
    centers_xy: torch.Tensor,
    *,
    low_y_quantile: float = 0.12,
    min_points: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a line from near-bottom boundary points:
      - point p0 on line (2,)
      - unit normal n (2,), oriented so median signed distance over all points is positive
    """
    if (not torch.is_tensor(centers_xy)) or centers_xy.ndim != 2 or centers_xy.size(1) < 2:
        raise RuntimeError(
            f"Cannot fit ramp line: expected centers (N,2), got "
            f"{None if not torch.is_tensor(centers_xy) else tuple(centers_xy.shape)}."
        )
    pts = centers_xy[:, :2].detach().to(device="cpu", dtype=torch.float64)
    n_all = int(pts.size(0))
    if n_all < 4:
        raise RuntimeError(f"Cannot fit ramp line: need >=4 centers, got {n_all}.")

    q = float(low_y_quantile)
    q = min(max(q, 0.01), 0.49)
    y = pts[:, 1]
    y_thr = torch.quantile(y, q)
    sel = (y <= y_thr)
    if int(sel.sum().item()) < int(max(4, min_points)):
        k = int(min(max(4, min_points), n_all))
        idx = torch.argsort(y)[:k]
        fit_pts = pts.index_select(0, idx)
    else:
        fit_pts = pts[sel]

    p0 = fit_pts.mean(dim=0)  # (2,)
    centered = fit_pts - p0
    # principal direction
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    tang = vh[0]
    tang = tang / tang.norm().clamp_min(1e-12)
    n = torch.stack([-tang[1], tang[0]], dim=0)
    n = n / n.norm().clamp_min(1e-12)

    # orient sign so bulk of cells lie on positive side
    signed_all = (pts - p0.view(1, 2)) @ n.view(2, 1)
    med = torch.median(signed_all.view(-1))
    if float(med.item()) < 0.0:
        n = -n

    return p0.to(torch.float32), n.to(torch.float32)


def _build_ramp_feature_context(
    cfg: Dict[str, Any],
    *,
    data_obj: Any,
    pt_path: str | None,
    mesh_spec_path: str | None,
) -> Dict[str, Any]:
    build_cfg = cfg.get("features", {}).get("build", {}) or {}
    include_ramp_angle = bool(build_cfg.get("include_ramp_angle", False))
    include_signed_dist = bool(build_cfg.get("include_signed_distance_to_ramp", False))

    ctx: Dict[str, Any] = {
        "include_ramp_angle": include_ramp_angle,
        "include_signed_distance_to_ramp": include_signed_dist,
    }
    if not (include_ramp_angle or include_signed_dist):
        return ctx

    # 1) angle (degrees): explicit config -> data payload -> file path -> mesh path
    angle_deg = _to_scalar_float(
        build_cfg.get("ramp_angle_deg", build_cfg.get("ramp_angle_degrees", None))
    )
    if angle_deg is None:
        angle_deg = _extract_angle_from_data_obj(data_obj)
    if angle_deg is None:
        angle_deg = _extract_angle_from_path(pt_path)
    if angle_deg is None:
        angle_deg = _extract_angle_from_path(mesh_spec_path)
    if angle_deg is not None:
        angle_deg = float(angle_deg)
        ctx["angle_deg"] = angle_deg

    # 2) signed-distance line fit from centers in first snapshot
    if include_signed_dist:
        first_snapshot = None
        if isinstance(data_obj, list) and len(data_obj) > 0:
            first_snapshot = data_obj[0]
        elif isinstance(data_obj, dict):
            for k in ("snapshots", "steps", "time_steps", "sequence", "data_list"):
                vv = data_obj.get(k, None)
                if isinstance(vv, list) and len(vv) > 0:
                    first_snapshot = vv[0]
                    break
            if first_snapshot is None:
                first_snapshot = data_obj
        else:
            first_snapshot = data_obj

        xy0 = _extract_xy_from_snapshot_like(first_snapshot)
        if xy0 is None:
            raise RuntimeError(
                "features.build.include_signed_distance_to_ramp=true but could not find centers "
                "(pos/xy/centers) in loaded data to fit ramp line."
            )

        q = float(build_cfg.get("signed_distance_low_y_quantile", 0.12))
        mpts = int(build_cfg.get("signed_distance_min_points", 32))
        p0, n = _fit_ramp_line_from_centers(
            xy0,
            low_y_quantile=q,
            min_points=mpts,
        )
        ctx["distance_point"] = p0
        ctx["distance_normal"] = n

        # Optional scaling for O(1) feature values
        normalize_dist = bool(build_cfg.get("signed_distance_normalize", True))
        dist_scale = 1.0
        if normalize_dist:
            bbox = cfg.get("data", {}).get("bbox", None)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x0, x1, y0, y1 = map(float, bbox)
                dist_scale = float(np.hypot(x1 - x0, y1 - y0))
            else:
                xy_cpu = xy0.detach().to(torch.float64).cpu()
                x0 = float(xy_cpu[:, 0].min().item()); x1 = float(xy_cpu[:, 0].max().item())
                y0 = float(xy_cpu[:, 1].min().item()); y1 = float(xy_cpu[:, 1].max().item())
                dist_scale = float(np.hypot(x1 - x0, y1 - y0))
            if dist_scale <= 0.0:
                dist_scale = 1.0
        ctx["distance_scale"] = float(dist_scale)

    # 3) ramp-angle feature value
    if include_ramp_angle:
        units = str(build_cfg.get("ramp_angle_units", "radians")).strip().lower()
        if units not in ("degrees", "degree", "deg", "radians", "radian", "rad"):
            raise ValueError(
                "features.build.ramp_angle_units must be one of "
                "{radians, degrees} (aliases allowed)."
            )
        if angle_deg is None:
            raise RuntimeError(
                "features.build.include_ramp_angle=true but no ramp angle could be inferred. "
                "Set features.build.ramp_angle_deg explicitly or ensure the data/paths include angle metadata."
            )
        angle_feature = float(angle_deg)
        if units in ("radians", "radian", "rad"):
            angle_feature = float(np.deg2rad(angle_feature))
        ctx["angle_feature_value"] = float(angle_feature)
        ctx["angle_units"] = "radians" if units in ("radians", "radian", "rad") else "degrees"

    return ctx


def _ramp_feature_source_id_for_t(
    ramp_feature_ctx: Dict[str, Any] | None,
    step_t_abs: int | None,
) -> int | None:
    if ramp_feature_ctx is None or step_t_abs is None:
        return None
    src_by_t = ramp_feature_ctx.get("source_id_by_t", None)
    if src_by_t is None:
        return None
    try:
        ti = int(step_t_abs)
        if isinstance(src_by_t, (list, tuple)):
            if 0 <= ti < len(src_by_t):
                return int(src_by_t[ti])
            return None
        if isinstance(src_by_t, dict):
            if ti in src_by_t:
                return int(src_by_t[ti])
            ks = str(ti)
            return int(src_by_t[ks]) if ks in src_by_t else None
    except Exception:
        return None
    return None


def _position_feature_mode(cfg: Dict[str, Any]) -> str:
    build_cfg = cfg.get("features", {}).get("build", {}) or {}
    if not _cfg_bool_strict(build_cfg.get("use_pos", True), key="features.build.use_pos"):
        return "none"

    raw = str(build_cfg.get("position_mode", "xy")).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "xy": "xy",
        "x_y": "xy",
        "pos": "xy",
        "position": "xy",
        "positions": "xy",
        "coords": "xy",
        "coordinates": "xy",
        "absolute": "xy",
        "absolute_xy": "xy",
        "boundary": "boundary_distances",
        "boundary_distance": "boundary_distances",
        "boundary_distances": "boundary_distances",
        "distance_to_boundary": "boundary_distances",
        "distances_to_boundary": "boundary_distances",
        "distance_to_boundaries": "boundary_distances",
        "distances_to_boundaries": "boundary_distances",
        "boundary_distance_clipped": "boundary_distances",
        "boundary_distances_clipped": "boundary_distances",
        "clipped_boundary_distances": "boundary_distances",
    }
    if raw not in aliases:
        raise ValueError(
            "features.build.position_mode must be one of {xy, boundary_distances, none}; "
            f"got {build_cfg.get('position_mode')!r}."
        )
    return aliases[raw]


def _position_feature_dim(cfg: Dict[str, Any]) -> int:
    mode = _position_feature_mode(cfg)
    if mode == "none":
        return 0
    if mode == "xy":
        return 2
    if mode == "boundary_distances":
        return 4
    raise RuntimeError(f"Unexpected position feature mode: {mode}")


def _build_position_features(
    centers: torch.Tensor,
    cfg: Dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    mode = _position_feature_mode(cfg)
    if mode == "none":
        return None
    if centers is None or (not torch.is_tensor(centers)) or centers.ndim != 2 or centers.size(1) < 2:
        raise RuntimeError(
            f"features.build.position_mode='{mode}' requires centers tensor with shape (N,2)."
        )

    xy = centers[:, :2].to(device=device, dtype=dtype)
    if mode == "xy":
        return xy

    xmin, xmax, ymin, ymax = _get_bbox(cfg)
    xmin_t = torch.tensor(float(xmin), device=device, dtype=dtype)
    xmax_t = torch.tensor(float(xmax), device=device, dtype=dtype)
    ymin_t = torch.tensor(float(ymin), device=device, dtype=dtype)
    ymax_t = torch.tensor(float(ymax), device=device, dtype=dtype)

    x = xy[:, 0:1]
    y = xy[:, 1:2]
    d_left = x - xmin_t
    d_right = xmax_t - x
    d_bottom = y - ymin_t
    d_top = ymax_t - y
    out = torch.cat([d_left, d_right, d_bottom, d_top], dim=1)

    build_cfg = cfg.get("features", {}).get("build", {}) or {}
    if bool(build_cfg.get("boundary_distance_normalize", True)):
        lx = max(float(xmax) - float(xmin), 1e-12)
        ly = max(float(ymax) - float(ymin), 1e-12)
        scale = torch.tensor([lx, lx, ly, ly], device=device, dtype=dtype).view(1, 4)
        out = out / scale

    clip_raw = build_cfg.get("boundary_distance_clip", None)
    if clip_raw is not None:
        clip = float(clip_raw)
        if clip <= 0.0:
            raise ValueError("features.build.boundary_distance_clip must be > 0 when provided.")
        out = out.clamp(min=0.0, max=clip)
        if bool(build_cfg.get("boundary_distance_clip_normalize", True)):
            out = out / clip

    return out


def _build_X(
    feat,
    centers,
    levels,
    cfg,
    dt_hat=None,
    ramp_feature_ctx: Dict[str, Any] | None = None,
    step_t_abs: int | None = None,
):
    """
    Build node features for FeatureNet:
      [ physics_feat ] (+ position features if use_pos) (+ [level] if use_level)
      (+ [dt_hat] if features.build.include_dt_hat)

    feat:    (N, F) on whatever device we want to run the model on (cpu/cuda/mps)
    centers: (N, 2) (typically from precompute; may be on cpu)
    levels:  (N,) or (N,1)
    """
    build_cfg = cfg.get("features", {}).get("build", {})
    use_level = build_cfg.get("use_level", True)
    include_dt_hat = _include_dt_hat_from_build_cfg(build_cfg)
    include_ramp_angle = bool(build_cfg.get("include_ramp_angle", False))
    include_signed_dist = bool(build_cfg.get("include_signed_distance_to_ramp", False))

    dev = feat.device
    Xs = [feat]  # already on dev

    pos_feat = _build_position_features(centers, cfg, device=dev, dtype=feat.dtype)
    if pos_feat is not None:
        Xs.append(pos_feat)

    if use_level:
        lvl = levels
        if lvl.dim() == 1:
            lvl = lvl.unsqueeze(-1)
        Xs.append(lvl.to(dev).to(feat.dtype))

    if include_dt_hat:
        if dt_hat is None:
            raise RuntimeError(
                "features.build.include_dt_hat=true but dt_hat was not provided to _build_X."
            )
        Xs.append(
            _dt_hat_feature_column(
                dt_hat,
                n_nodes=int(feat.size(0)),
                device=dev,
                dtype=feat.dtype,
            )
        )

    if include_ramp_angle:
        if ramp_feature_ctx is None:
            raise RuntimeError(
                "features.build.include_ramp_angle=true but ramp_feature_ctx was not provided."
            )
        aval = None
        sid = _ramp_feature_source_id_for_t(ramp_feature_ctx, step_t_abs)
        by_source = ramp_feature_ctx.get("angle_feature_by_source", None)
        if (sid is not None) and isinstance(by_source, dict):
            aval = by_source.get(int(sid), None)
            if aval is None:
                aval = by_source.get(str(int(sid)), None)
        if aval is None:
            by_t = ramp_feature_ctx.get("angle_feature_by_t", None)
            if isinstance(by_t, (list, tuple)) and (step_t_abs is not None):
                ti = int(step_t_abs)
                if 0 <= ti < len(by_t):
                    aval = by_t[ti]
            elif isinstance(by_t, dict) and (step_t_abs is not None):
                ti = int(step_t_abs)
                aval = by_t.get(ti, by_t.get(str(ti), None))
        if aval is None:
            aval = ramp_feature_ctx.get("angle_feature_value", None)
        if aval is None:
            raise RuntimeError(
                "features.build.include_ramp_angle=true but angle_feature_value is missing. "
                "Set features.build.ramp_angle_deg or provide angle metadata in data/path."
            )
        if not getattr(_build_X, "_printed_ramp_angle_input", False):
            angle_units = str(ramp_feature_ctx.get("angle_units", "radians")).lower()
            angle_deg_by_source = ramp_feature_ctx.get("angle_deg_by_source", None)
            deg_val = None
            if (sid is not None) and isinstance(angle_deg_by_source, dict):
                deg_val = angle_deg_by_source.get(int(sid), None)
                if deg_val is None:
                    deg_val = angle_deg_by_source.get(str(int(sid)), None)
            if deg_val is None:
                angle_deg = ramp_feature_ctx.get("angle_deg", None)
                if angle_deg is not None:
                    deg_val = float(angle_deg)
            feature_val = float(aval)
            if deg_val is None:
                deg_val = float(np.rad2deg(feature_val)) if angle_units == "radians" else feature_val
            col_idx = sum(
                int(x.size(1))
                for x in Xs
                if torch.is_tensor(x) and x.ndim == 2
            )
            print(
                "[INPUT-FEAT] ramp_angle appended "
                f"column={col_idx}, feature={feature_val:.6g}, units={angle_units}, "
                f"deg={float(deg_val):.6g}, source_id={sid}, step_t_abs={step_t_abs}",
                flush=True,
            )
            setattr(_build_X, "_printed_ramp_angle_input", True)
        a = torch.tensor(float(aval), device=dev, dtype=feat.dtype).view(1, 1)
        Xs.append(a.expand(int(feat.size(0)), 1))

    if include_signed_dist:
        if centers is None or (not torch.is_tensor(centers)) or centers.ndim != 2 or centers.size(1) < 2:
            raise RuntimeError(
                "features.build.include_signed_distance_to_ramp=true requires centers tensor with shape (N,2)."
            )
        if ramp_feature_ctx is None:
            ramp_feature_ctx = {}
        p0 = ramp_feature_ctx.get("distance_point", None)
        n = ramp_feature_ctx.get("distance_normal", None)
        dscale = float(ramp_feature_ctx.get("distance_scale", 1.0))
        sid = _ramp_feature_source_id_for_t(ramp_feature_ctx, step_t_abs)
        by_source = ramp_feature_ctx.get("distance_by_source", None)
        if (sid is not None) and isinstance(by_source, dict):
            src_pack = by_source.get(int(sid), by_source.get(str(int(sid)), None))
            if isinstance(src_pack, dict):
                p0 = src_pack.get("point", p0)
                n = src_pack.get("normal", n)
                dscale = float(src_pack.get("scale", dscale))
        if (p0 is None) or (n is None):
            # Last-resort fallback: fit from this step's centers.
            p0, n = _fit_ramp_line_from_centers(
                centers.detach().to(device="cpu", dtype=torch.float32),
                low_y_quantile=float(build_cfg.get("signed_distance_low_y_quantile", 0.12)),
                min_points=int(build_cfg.get("signed_distance_min_points", 32)),
            )
        p0_t = torch.as_tensor(p0, dtype=feat.dtype, device=dev).view(1, 2)
        n_t = torch.as_tensor(n, dtype=feat.dtype, device=dev).view(2, 1)
        signed = (centers[:, :2].to(device=dev, dtype=feat.dtype) - p0_t) @ n_t
        signed = signed.view(-1, 1)
        if dscale <= 0.0:
            dscale = 1.0
        if bool(build_cfg.get("signed_distance_normalize", True)):
            signed = signed / float(dscale)
        Xs.append(signed)

    return torch.cat(Xs, dim=-1)

def _forward_main_head(
    model: FeatureNet,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
) -> torch.Tensor:
    feat_only_supported = getattr(model, "_feat_only_forward_supported", None)
    if feat_only_supported is None:
        try:
            sig = inspect.signature(model.forward)
            params = sig.parameters
            feat_only_supported = ("return_score" in params) and ("return_hidden" in params)
        except Exception:
            feat_only_supported = False
        try:
            setattr(model, "_feat_only_forward_supported", bool(feat_only_supported))
        except Exception:
            pass

    if edge_attr is None:
        if feat_only_supported:
            out = model(X, edge_index, return_score=False, return_hidden=False)
        else:
            out = model(X, edge_index)
    else:
        if feat_only_supported:
            out = model(
                X,
                edge_index,
                edge_attr=edge_attr,
                return_score=False,
                return_hidden=False,
            )
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


def _runtime_mesh_plot_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    plot_cfg = rt_cfg.get("mesh_plot", {}) or {}
    out_default = os.path.join(cfg.get("train", {}).get("save_dir", "."), "runtime_mesh_plots")
    split_raw = str(plot_cfg.get("split", "train")).strip().lower()
    if split_raw in ("all",):
        split_raw = "both"
    if split_raw not in ("train", "eval", "both"):
        raise ValueError(
            "train.runtime_mesh.mesh_plot.split must be one of {train, eval, both}. "
            f"Got: {split_raw!r}"
        )
    every_rebuilds = int(plot_cfg.get("every_rebuilds", 1))
    if every_rebuilds < 1:
        raise ValueError("train.runtime_mesh.mesh_plot.every_rebuilds must be >= 1.")
    max_plots = int(plot_cfg.get("max_plots", 20))
    if max_plots < 0:
        raise ValueError("train.runtime_mesh.mesh_plot.max_plots must be >= 0.")
    dpi = int(plot_cfg.get("dpi", 180))
    if dpi < 32:
        raise ValueError("train.runtime_mesh.mesh_plot.dpi must be >= 32.")
    line_width = float(plot_cfg.get("line_width", 0.2))
    if line_width <= 0.0:
        raise ValueError("train.runtime_mesh.mesh_plot.line_width must be > 0.")
    predictor_input_max_channels = int(plot_cfg.get("predictor_input_max_channels", 4))
    if predictor_input_max_channels < 1:
        raise ValueError("train.runtime_mesh.mesh_plot.predictor_input_max_channels must be >= 1.")
    gt_feature_max_channels = int(plot_cfg.get("gt_feature_max_channels", 4))
    if gt_feature_max_channels < 1:
        raise ValueError("train.runtime_mesh.mesh_plot.gt_feature_max_channels must be >= 1.")
    gt_feature_indices_raw = plot_cfg.get("gt_feature_indices", [])
    if gt_feature_indices_raw is None:
        gt_feature_indices_raw = []
    if not isinstance(gt_feature_indices_raw, (list, tuple)):
        raise ValueError("train.runtime_mesh.mesh_plot.gt_feature_indices must be a list of integers.")
    gt_feature_indices: List[int] = []
    for v in gt_feature_indices_raw:
        try:
            gt_feature_indices.append(int(v))
        except Exception:
            raise ValueError(
                f"train.runtime_mesh.mesh_plot.gt_feature_indices contains non-integer value: {v!r}"
            )
    feat_names_cfg = cfg.get("features", {}).get("names", None)
    if not isinstance(feat_names_cfg, (list, tuple)):
        feat_names_cfg = cfg.get("features", {}).get("dataset_order", None)
    if not isinstance(feat_names_cfg, (list, tuple)):
        feat_names_cfg = []
    feature_names = [str(nm) for nm in feat_names_cfg]
    return {
        "enabled": bool(plot_cfg.get("enabled", False)),
        "out_dir": os.path.abspath(os.path.expanduser(str(plot_cfg.get("out_dir", out_default)))),
        "split": split_raw,
        "every_rebuilds": every_rebuilds,
        "max_plots": max_plots,
        "dpi": dpi,
        "line_width": line_width,
        "show_wedge": bool(plot_cfg.get("show_wedge", True)),
        "plot_predictor_inputs": bool(plot_cfg.get("plot_predictor_inputs", False)),
        "predictor_input_max_channels": predictor_input_max_channels,
        "plot_gt_features": bool(plot_cfg.get("plot_gt_features", False)),
        "gt_feature_max_channels": gt_feature_max_channels,
        "gt_feature_indices": gt_feature_indices,
        "feature_names": feature_names,
    }


def _runtime_mesh_plot_wants_split(settings: Dict[str, Any], split: str) -> bool:
    target = str(settings.get("split", "train")).strip().lower()
    s = str(split).strip().lower()
    if target == "both":
        return s in ("train", "eval")
    return s == target


def _save_runtime_pred_mesh_plot(
    *,
    out_path: str,
    pred_centers: torch.Tensor,
    pred_levels: torch.Tensor,
    H: int,
    W: int,
    bbox: Tuple[float, float, float, float],
    refine_ratio: int,
    title: str,
    wedge_path=None,
    show_wedge: bool = True,
    dpi: int = 180,
    line_width: float = 0.2,
) -> None:
    # Local import to avoid forcing matplotlib on non-plot runs.
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    centers = pred_centers.detach().to("cpu", dtype=torch.float32).numpy()
    levels = pred_levels.detach().view(-1).to("cpu", dtype=torch.int64).numpy()
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise RuntimeError(f"pred_centers must be (N,2), got {centers.shape}")
    if levels.ndim != 1 or levels.shape[0] != centers.shape[0]:
        raise RuntimeError(f"pred_levels must be (N,), got levels={levels.shape} for centers={centers.shape}")
    if centers.shape[0] == 0:
        raise RuntimeError("Cannot plot empty predicted mesh.")

    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {rr}")
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    scale = np.power(float(rr), levels.astype(np.float32))
    hx = (dx0 / scale) * 0.5
    hy = (dy0 / scale) * 0.5
    x = centers[:, 0]
    y = centers[:, 1]
    x0 = x - hx
    x1 = x + hx
    y0 = y - hy
    y1 = y + hy

    n = centers.shape[0]
    segs = np.empty((n * 4, 2, 2), dtype=np.float32)
    segs[0::4, 0, 0] = x0; segs[0::4, 0, 1] = y0
    segs[0::4, 1, 0] = x1; segs[0::4, 1, 1] = y0
    segs[1::4, 0, 0] = x0; segs[1::4, 0, 1] = y1
    segs[1::4, 1, 0] = x1; segs[1::4, 1, 1] = y1
    segs[2::4, 0, 0] = x0; segs[2::4, 0, 1] = y0
    segs[2::4, 1, 0] = x0; segs[2::4, 1, 1] = y1
    segs[3::4, 0, 0] = x1; segs[3::4, 0, 1] = y0
    segs[3::4, 1, 0] = x1; segs[3::4, 1, 1] = y1

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.0, 6.5), dpi=int(dpi))
    lc = LineCollection(segs, linewidths=float(line_width), colors="k", alpha=0.92)
    ax.add_collection(lc)
    if show_wedge and (wedge_path is not None) and hasattr(wedge_path, "vertices"):
        vv = np.asarray(wedge_path.vertices, dtype=np.float32)
        if vv.ndim == 2 and vv.shape[1] == 2 and vv.shape[0] >= 2:
            ax.plot(vv[:, 0], vv[:, 1], color="tab:red", linewidth=1.2, alpha=0.95)
    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(float(ymin), float(ymax))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(str(title))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _runtime_predictor_input_to_panels(
    *,
    predictor_input: Dict[str, Any] | None,
    max_channels: int,
) -> tuple[str, list[tuple[str, np.ndarray]]]:
    if (predictor_input is None) or (not isinstance(predictor_input, dict)):
        return "unknown", []
    backend = str(predictor_input.get("backend", "unknown"))
    panels: list[tuple[str, np.ndarray]] = []

    if backend == "gradient_fast":
        coarse_hwf = predictor_input.get("coarse_hwf", None)
        selected_idx = predictor_input.get("selected_idx", [])
        g_base_hxw = predictor_input.get("g_base_hxw", None)
        if torch.is_tensor(coarse_hwf) and coarse_hwf.ndim == 3:
            Fdim = int(coarse_hwf.shape[2])
            idxs = [int(i) for i in selected_idx] if isinstance(selected_idx, (list, tuple)) else list(range(Fdim))
            if len(idxs) == 0:
                idxs = list(range(Fdim))
            for c in idxs[: int(max_channels)]:
                if 0 <= int(c) < Fdim:
                    panels.append((f"coarse[c={int(c)}]", coarse_hwf[:, :, int(c)].detach().cpu().numpy()))
        if torch.is_tensor(g_base_hxw) and g_base_hxw.ndim == 2:
            panels.append(("combined_grad_score", g_base_hxw.detach().cpu().numpy()))
        return backend, panels

    if backend == "cnn":
        x_in_chw = predictor_input.get("x_in_chw", None)
        channel_names = predictor_input.get("channel_names", None)
        if torch.is_tensor(x_in_chw) and x_in_chw.ndim == 3:
            C = int(x_in_chw.shape[0])
            if not isinstance(channel_names, (list, tuple)):
                channel_names = [f"input[{i}]" for i in range(C)]
            for c in range(min(int(max_channels), C)):
                nm = str(channel_names[c]) if c < len(channel_names) else f"input[{c}]"
                panels.append((nm, x_in_chw[c].detach().cpu().numpy()))
        return backend, panels

    if backend == "gradient":
        G = predictor_input.get("G", None)
        pooled = predictor_input.get("pooled_up", None)
        if isinstance(G, dict):
            for L in sorted(G.keys(), key=lambda v: int(v)):
                v = G[L]
                if torch.is_tensor(v) and v.ndim == 2:
                    panels.append((f"G[L={int(L)}]", v.detach().cpu().numpy()))
                if len(panels) >= int(max_channels):
                    break
        if isinstance(pooled, dict):
            for L in sorted(pooled.keys(), key=lambda v: int(v)):
                v = pooled[L]
                if torch.is_tensor(v) and v.ndim == 2:
                    panels.append((f"pooled_up[L={int(L)}]", v.detach().cpu().numpy()))
                if len(panels) >= int(max_channels) + 1:
                    break
        return backend, panels

    return backend, panels


def _save_runtime_predictor_input_plot(
    *,
    out_path: str,
    predictor_input: Dict[str, Any] | None,
    title_prefix: str,
    bbox: Tuple[float, float, float, float],
    dpi: int = 180,
    max_channels: int = 4,
    wedge_path=None,
    show_wedge: bool = True,
) -> bool:
    backend, panels = _runtime_predictor_input_to_panels(
        predictor_input=predictor_input,
        max_channels=int(max_channels),
    )
    if len(panels) == 0:
        return False

    # Local import to avoid forcing matplotlib on non-plot runs.
    import matplotlib.pyplot as plt

    n = int(len(panels))
    ncols = min(3, n)
    nrows = int(np.ceil(float(n) / float(max(1, ncols))))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.0 * ncols, 3.8 * nrows), dpi=int(dpi))
    axs_arr = np.atleast_1d(axs).reshape(-1)
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]

    for i, (nm, img) in enumerate(panels):
        ax = axs_arr[i]
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 2:
            ax.axis("off")
            ax.set_title(f"{nm} (invalid shape)")
            continue
        im = ax.imshow(
            arr,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="viridis",
            aspect="auto",
        )
        if show_wedge and (wedge_path is not None) and hasattr(wedge_path, "vertices"):
            vv = np.asarray(wedge_path.vertices, dtype=np.float32)
            if vv.ndim == 2 and vv.shape[1] == 2 and vv.shape[0] >= 2:
                ax.plot(vv[:, 0], vv[:, 1], color="tab:red", linewidth=0.8, alpha=0.9)
        ax.set_title(str(nm))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    for j in range(n, axs_arr.size):
        axs_arr[j].axis("off")

    fig.suptitle(f"{title_prefix} | predictor_input backend={backend}", fontsize=10)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return True


def _runtime_gt_feature_to_panels(
    *,
    gt_centers: torch.Tensor,
    gt_levels: torch.Tensor,
    gt_feat: torch.Tensor,
    H: int,
    W: int,
    bbox: Tuple[float, float, float, float],
    refine_ratio: int,
    max_channels: int,
    feature_names: Sequence[str] | None = None,
    feature_indices: Sequence[int] | None = None,
) -> list[tuple[str, np.ndarray]]:
    if (not torch.is_tensor(gt_centers)) or (not torch.is_tensor(gt_levels)) or (not torch.is_tensor(gt_feat)):
        return []

    centers = gt_centers.detach().to("cpu", dtype=torch.float32)
    levels = gt_levels.detach().view(-1).to("cpu", dtype=torch.long)
    feat = gt_feat.detach().to("cpu", dtype=torch.float32)

    if centers.ndim != 2 or int(centers.shape[1]) != 2:
        return []
    if feat.ndim != 2:
        return []
    if int(levels.numel()) != int(centers.shape[0]) or int(feat.shape[0]) != int(centers.shape[0]):
        return []

    Fdim = int(feat.shape[1])
    if Fdim <= 0:
        return []

    if isinstance(feature_indices, (list, tuple)) and len(feature_indices) > 0:
        idxs = [int(i) for i in feature_indices if 0 <= int(i) < Fdim]
    else:
        idxs = list(range(Fdim))
    if len(idxs) == 0:
        return []
    # Keep order, drop duplicates, cap channel count.
    seen: set[int] = set()
    sel: list[int] = []
    for c in idxs:
        cc = int(c)
        if cc in seen:
            continue
        seen.add(cc)
        sel.append(cc)
        if len(sel) >= int(max_channels):
            break
    if len(sel) == 0:
        return []

    img_flat, valid_flat, HH, WW = amr_composite_to_finest_grid(
        centers=centers,
        levels=levels,
        values=feat,
        H=int(H),
        W=int(W),
        bbox=bbox,
        refine_ratio=int(refine_ratio),
    )
    if (not torch.is_tensor(img_flat)) or (not torch.is_tensor(valid_flat)):
        return []
    if int(img_flat.numel()) <= 0 or int(valid_flat.numel()) <= 0:
        return []

    img = img_flat.view(int(HH), int(WW), -1).to("cpu", dtype=torch.float32).numpy()
    valid = valid_flat.view(int(HH), int(WW)).to("cpu", dtype=torch.bool).numpy()

    names = list(feature_names) if isinstance(feature_names, (list, tuple)) else []
    panels: list[tuple[str, np.ndarray]] = []
    for c in sel:
        nm = str(names[c]) if (0 <= c < len(names)) else f"feat[{int(c)}]"
        arr = img[:, :, int(c)]
        arr = np.where(valid, arr, np.nan).astype(np.float32, copy=False)
        panels.append((nm, arr))
    return panels


def _save_runtime_gt_feature_plot(
    *,
    out_path: str,
    title_prefix: str,
    t_label: str,
    gt_centers: torch.Tensor,
    gt_levels: torch.Tensor,
    gt_feat: torch.Tensor,
    H: int,
    W: int,
    bbox: Tuple[float, float, float, float],
    refine_ratio: int,
    dpi: int = 180,
    max_channels: int = 4,
    feature_names: Sequence[str] | None = None,
    feature_indices: Sequence[int] | None = None,
    wedge_path=None,
    show_wedge: bool = True,
) -> bool:
    try:
        panels = _runtime_gt_feature_to_panels(
            gt_centers=gt_centers,
            gt_levels=gt_levels,
            gt_feat=gt_feat,
            H=int(H),
            W=int(W),
            bbox=bbox,
            refine_ratio=int(refine_ratio),
            max_channels=int(max_channels),
            feature_names=feature_names,
            feature_indices=feature_indices,
        )
    except Exception as e:
        print(f"[RUNTIME-MESH][PLOT][WARN] GT feature plot failed while preparing panels: {e!r}", flush=True)
        return False

    if len(panels) == 0:
        return False

    # Local import to avoid forcing matplotlib on non-plot runs.
    import matplotlib.pyplot as plt

    n = int(len(panels))
    ncols = min(3, n)
    nrows = int(np.ceil(float(n) / float(max(1, ncols))))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.0 * ncols, 3.8 * nrows), dpi=int(dpi))
    axs_arr = np.atleast_1d(axs).reshape(-1)
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]

    for i, (nm, img) in enumerate(panels):
        ax = axs_arr[i]
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 2:
            ax.axis("off")
            ax.set_title(f"{nm} (invalid shape)")
            continue
        marr = np.ma.masked_invalid(arr)
        im = ax.imshow(
            marr,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="viridis",
            aspect="auto",
        )
        if show_wedge and (wedge_path is not None) and hasattr(wedge_path, "vertices"):
            vv = np.asarray(wedge_path.vertices, dtype=np.float32)
            if vv.ndim == 2 and vv.shape[1] == 2 and vv.shape[0] >= 2:
                ax.plot(vv[:, 0], vv[:, 1], color="tab:red", linewidth=0.8, alpha=0.9)
        ax.set_title(str(nm))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    for j in range(n, axs_arr.size):
        axs_arr[j].axis("off")

    fig.suptitle(f"{title_prefix} | GT features ({t_label})", fontsize=10)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return True


def _runtime_mesh_plot_maybe_save(
    *,
    settings: Dict[str, Any],
    state: Dict[str, int],
    split: str,
    epoch_idx: int | None,
    batch_idx: int,
    step_k: int,
    t_abs: int,
    pred_centers: torch.Tensor,
    pred_levels: torch.Tensor,
    H: int,
    W: int,
    bbox: Tuple[float, float, float, float],
    refine_ratio: int,
    wedge_path=None,
    predictor_input: Dict[str, Any] | None = None,
    gt_snapshots: List[Dict[str, Any]] | None = None,
) -> None:
    state["rebuilds"] = int(state.get("rebuilds", 0)) + 1
    if not bool(settings.get("enabled", False)):
        return
    if not _runtime_mesh_plot_wants_split(settings, split):
        return
    every = int(settings.get("every_rebuilds", 1))
    rebuild_idx = int(state["rebuilds"])
    if ((rebuild_idx - 1) % every) != 0:
        return
    max_plots = int(settings.get("max_plots", 20))
    saved_so_far = int(state.get("saved", 0))
    if (max_plots > 0) and (saved_so_far >= max_plots):
        return

    uniq, cnt = torch.unique(pred_levels.detach().to("cpu", dtype=torch.long), return_counts=True)
    lv_summary = ", ".join([f"L{int(u)}={int(c)}" for u, c in zip(uniq.tolist(), cnt.tolist())])
    ep_str = f"{int(epoch_idx):04d}" if epoch_idx is not None else "na"
    file_name = (
        f"{str(split).lower()}_ep{ep_str}_b{int(batch_idx):05d}_k{int(step_k):03d}"
        f"_t{int(t_abs):05d}_rb{int(rebuild_idx):05d}.png"
    )
    out_dir = str(settings.get("out_dir", "."))
    out_path = os.path.join(out_dir, file_name)
    title = (
        f"runtime mesh | split={str(split).lower()} ep={ep_str} "
        f"batch={int(batch_idx)} step={int(step_k)} t={int(t_abs)} "
        f"N={int(pred_levels.numel())} ({lv_summary})"
    )
    _save_runtime_pred_mesh_plot(
        out_path=out_path,
        pred_centers=pred_centers,
        pred_levels=pred_levels,
        H=H,
        W=W,
        bbox=bbox,
        refine_ratio=refine_ratio,
        title=title,
        wedge_path=wedge_path,
        show_wedge=bool(settings.get("show_wedge", True)),
        dpi=int(settings.get("dpi", 180)),
        line_width=float(settings.get("line_width", 0.2)),
    )
    if bool(settings.get("plot_predictor_inputs", False)):
        in_path = out_path[:-4] + "_input.png" if out_path.lower().endswith(".png") else (out_path + "_input.png")
        saved_in = _save_runtime_predictor_input_plot(
            out_path=in_path,
            predictor_input=predictor_input,
            title_prefix=title,
            bbox=bbox,
            dpi=int(settings.get("dpi", 180)),
            max_channels=int(settings.get("predictor_input_max_channels", 4)),
            wedge_path=wedge_path,
            show_wedge=bool(settings.get("show_wedge", True)),
        )
        if saved_in:
            print(
                f"[RUNTIME-MESH][PLOT] saved {in_path}",
                flush=True,
            )
    if bool(settings.get("plot_gt_features", False)) and isinstance(gt_snapshots, list):
        for si, snap in enumerate(gt_snapshots):
            if not isinstance(snap, dict):
                continue
            gt_centers = snap.get("centers", None)
            gt_levels = snap.get("levels", None)
            gt_feat = snap.get("feat", None)
            if (not torch.is_tensor(gt_centers)) or (not torch.is_tensor(gt_levels)) or (not torch.is_tensor(gt_feat)):
                continue
            raw_tag = str(snap.get("tag", f"gt_{si}")).strip()
            if raw_tag == "":
                raw_tag = f"gt_{si}"
            safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_tag)
            t_snap = snap.get("t_abs", None)
            t_label = "t=?"
            t_suffix = ""
            if t_snap is not None:
                try:
                    t_i = int(t_snap)
                    t_label = f"t={t_i}"
                    t_suffix = f"_t{t_i:05d}"
                except Exception:
                    t_label = f"t={t_snap}"
            gt_path = (
                out_path[:-4] + f"_{safe_tag}{t_suffix}.png"
                if out_path.lower().endswith(".png")
                else (out_path + f"_{safe_tag}{t_suffix}.png")
            )
            saved_gt = _save_runtime_gt_feature_plot(
                out_path=gt_path,
                title_prefix=title,
                t_label=t_label,
                gt_centers=gt_centers,
                gt_levels=gt_levels,
                gt_feat=gt_feat,
                H=H,
                W=W,
                bbox=bbox,
                refine_ratio=refine_ratio,
                dpi=int(settings.get("dpi", 180)),
                max_channels=int(settings.get("gt_feature_max_channels", 4)),
                feature_names=settings.get("feature_names", []),
                feature_indices=settings.get("gt_feature_indices", []),
                wedge_path=wedge_path,
                show_wedge=bool(settings.get("show_wedge", True)),
            )
            if saved_gt:
                print(
                    f"[RUNTIME-MESH][PLOT] saved {gt_path}",
                    flush=True,
                )
    state["saved"] = saved_so_far + 1
    print(
        f"[RUNTIME-MESH][PLOT] saved {out_path}",
        flush=True,
    )


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


def _cell_level_ij_from_centers(
    *,
    centers: torch.Tensor,
    levels: torch.Tensor,
    H: int,
    W: int,
    bbox: Tuple[float, float, float, float],
    refine_ratio: int,
    index_mode: str = "round",
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

    mode = str(index_mode).strip().lower()
    if mode == "round":
        # Inverse of center formula x = xmin + (i + 0.5) * xs / WW (same for y/HH).
        col_f = ((centers[:, 0].to(torch.float32) - xmin) * ww_l / xs) - 0.5
        row_f = ((centers[:, 1].to(torch.float32) - ymin) * hh_l / ys) - 0.5
        col = torch.round(col_f).to(torch.long)
        row = torch.round(row_f).to(torch.long)
    elif mode == "floor":
        # More robust to tiny center jitters around half-cell offsets.
        col_f = ((centers[:, 0].to(torch.float32) - xmin) * ww_l / xs)
        row_f = ((centers[:, 1].to(torch.float32) - ymin) * hh_l / ys)
        col = torch.floor(col_f).to(torch.long)
        row = torch.floor(row_f).to(torch.long)
    else:
        raise ValueError(f"Unsupported index_mode={index_mode!r}; expected 'round' or 'floor'.")

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


def _runtime_multires_lookup_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    raw = rt_cfg.get("multires_gt_lookup", {})
    if isinstance(raw, bool):
        raw = {"enabled": bool(raw)}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("train.runtime_mesh.multires_gt_lookup must be a bool or JSON object.")

    out = dict(raw)
    out.setdefault("enabled", False)
    out.setdefault("directory", "")
    out.setdefault("level_files", {})
    out.setdefault("fallback_to_idw", True)
    out.setdefault("max_cached_feature_steps", 16)
    return out


class _RuntimeMultiResGtLookup:
    """
    H5-backed GT lookup on runtime meshes using precomputed uniform-mesh caches.

    The lookup key is (level, row, col), where row/col are reconstructed from centers.
    """
    def __init__(
        self,
        *,
        level_files: Dict[int, str],
        H: int,
        W: int,
        bbox: Tuple[float, float, float, float],
        refine_ratio: int,
        max_level: int,
        max_cached_feature_steps: int = 16,
    ):
        if h5py is None:
            raise ImportError(
                "Multi-resolution GT lookup requires h5py, but import failed."
            )
        if not level_files:
            raise ValueError("Runtime multi-resolution lookup was enabled, but no level files were provided.")

        self.H = int(H)
        self.W = int(W)
        self.bbox = tuple(float(v) for v in bbox)
        self.refine_ratio = int(refine_ratio)
        self.max_level = int(max(0, max_level))
        self._key_h = int(self.H * (self.refine_ratio ** self.max_level))
        self._key_w = int(self.W * (self.refine_ratio ** self.max_level))

        self._h5_by_file_level: Dict[int, Any] = {}
        self._file_level_index: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}
        self._source_file_for_level: Dict[int, int] = {}
        self._feature_cache: "OrderedDict[Tuple[int, int, str], torch.Tensor]" = OrderedDict()
        self._max_cached_feature_steps = max(1, int(max_cached_feature_steps))
        self.level_files = dict(sorted((int(k), str(v)) for k, v in level_files.items()))

        for file_level, path in self.level_files.items():
            f = h5py.File(path, "r")
            self._h5_by_file_level[file_level] = f
            g_static = self._select_static_group(f)
            if g_static is None:
                raise RuntimeError(f"H5 file has no /static or /tXXXXX groups: {path}")
            if ("pred_centers" not in g_static) or ("pred_levels" not in g_static):
                raise RuntimeError(
                    f"H5 file is missing static geometry datasets in {path} "
                    "(need pred_centers and pred_levels)."
                )
            centers = np.asarray(g_static["pred_centers"][...], dtype=np.float32)
            levels = np.asarray(g_static["pred_levels"][...], dtype=np.int64)
            self._file_level_index[file_level] = self._build_level_index(centers, levels)

        for lv in range(self.max_level + 1):
            src_file = None
            if (lv in self._file_level_index) and (lv in self._file_level_index[lv]):
                src_file = lv
            else:
                for file_level in sorted(self._file_level_index.keys()):
                    if lv in self._file_level_index[file_level]:
                        src_file = file_level
                        break
            if src_file is not None:
                self._source_file_for_level[lv] = src_file

    @staticmethod
    def _select_static_group(f: Any):
        if "static" in f:
            return f["static"]
        if "t00001" in f:
            return f["t00001"]
        t_groups = sorted([k for k in f.keys() if str(k).startswith("t")])
        if not t_groups:
            return None
        return f[t_groups[0]]

    def close(self):
        for f in self._h5_by_file_level.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_by_file_level.clear()
        self._feature_cache.clear()

    def _make_keys(self, levels: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        lv = levels.view(-1).long()
        rr = row.view(-1).long()
        cc = col.view(-1).long()
        return ((lv * int(self._key_h)) + rr) * int(self._key_w) + cc

    def _build_level_index(self, centers: np.ndarray, levels: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        centers_t = torch.from_numpy(centers.astype(np.float32, copy=False))
        levels_t = torch.from_numpy(levels.astype(np.int64, copy=False))
        row_t, col_t = _cell_level_ij_from_centers(
            centers=centers_t,
            levels=levels_t,
            H=self.H,
            W=self.W,
            bbox=self.bbox,
            refine_ratio=self.refine_ratio,
            index_mode="floor",
        )
        keys = self._make_keys(levels_t, row_t, col_t).cpu().numpy().astype(np.int64, copy=False)
        idx_all = np.arange(keys.shape[0], dtype=np.int64)

        out: Dict[int, Dict[str, np.ndarray]] = {}
        for lv in np.unique(levels.astype(np.int64, copy=False)):
            lv_i = int(lv)
            mask = (levels == lv_i)
            k_l = keys[mask]
            i_l = idx_all[mask]
            order = np.argsort(k_l)
            out[lv_i] = {
                "keys_sorted": k_l[order],
                "idx_sorted": i_l[order],
            }
        return out

    def _cache_get(self, key: Tuple[int, int, str]) -> Optional[torch.Tensor]:
        v = self._feature_cache.get(key, None)
        if v is not None:
            self._feature_cache.move_to_end(key)
        return v

    def _cache_put(self, key: Tuple[int, int, str], value: torch.Tensor):
        self._feature_cache[key] = value
        self._feature_cache.move_to_end(key)
        while len(self._feature_cache) > self._max_cached_feature_steps:
            self._feature_cache.popitem(last=False)

    def _read_feature(self, *, file_level: int, t_dst: int, dataset_name: str) -> torch.Tensor:
        ck = (int(file_level), int(t_dst), str(dataset_name))
        cached = self._cache_get(ck)
        if cached is not None:
            return cached

        f = self._h5_by_file_level[int(file_level)]
        gname = f"t{int(t_dst):05d}"
        if gname not in f:
            raise KeyError(f"Missing timestep group '{gname}' in file-level {file_level}.")
        g = f[gname]
        if dataset_name not in g:
            raise KeyError(
                f"Missing dataset '{dataset_name}' in group '{gname}' "
                f"(file-level {file_level})."
            )
        arr = np.asarray(g[dataset_name][...], dtype=np.float32)
        ten = torch.from_numpy(arr)
        self._cache_put(ck, ten)
        return ten

    @torch.no_grad()
    def lookup(
        self,
        *,
        t_dst: int,
        dataset_name: str,
        pred_centers: torch.Tensor,
        pred_levels: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Dict[str, Any]]:
        lv_t = pred_levels.view(-1).long()
        N = int(lv_t.numel())
        matched_cpu = torch.zeros((N,), dtype=torch.bool, device="cpu")
        stats: Dict[str, Any] = {
            "requested": N,
            "matched": 0,
            "missing": N,
            "by_level": {},
        }
        if N == 0:
            return None, matched_cpu.to(device=pred_centers.device), stats

        row_t, col_t = _cell_level_ij_from_centers(
            centers=pred_centers,
            levels=lv_t,
            H=self.H,
            W=self.W,
            bbox=self.bbox,
            refine_ratio=self.refine_ratio,
            index_mode="floor",
        )
        key_np = self._make_keys(lv_t, row_t, col_t).detach().cpu().numpy().astype(np.int64, copy=False)
        lvl_np = lv_t.detach().cpu().numpy().astype(np.int64, copy=False)

        out_cpu: Optional[torch.Tensor] = None
        for lv in sorted(np.unique(lvl_np).tolist()):
            lv_i = int(lv)
            dst_pos = np.nonzero(lvl_np == lv_i)[0]
            if dst_pos.size == 0:
                continue

            src_file_level = self._source_file_for_level.get(lv_i, None)
            if src_file_level is None:
                stats["by_level"][lv_i] = {
                    "requested": int(dst_pos.size),
                    "matched": 0,
                    "source_file_level": None,
                }
                continue

            src_level_map = self._file_level_index.get(src_file_level, {}).get(lv_i, None)
            if src_level_map is None:
                stats["by_level"][lv_i] = {
                    "requested": int(dst_pos.size),
                    "matched": 0,
                    "source_file_level": int(src_file_level),
                }
                continue

            keys_sorted = src_level_map["keys_sorted"]
            idx_sorted = src_level_map["idx_sorted"]
            q_keys = key_np[dst_pos]
            pos = np.searchsorted(keys_sorted, q_keys, side="left")

            valid = pos < keys_sorted.shape[0]
            if np.any(valid):
                eq = np.zeros_like(valid, dtype=np.bool_)
                eq[valid] = (keys_sorted[pos[valid]] == q_keys[valid])
                valid = eq

            n_match = int(np.count_nonzero(valid))
            stats["by_level"][lv_i] = {
                "requested": int(dst_pos.size),
                "matched": n_match,
                "source_file_level": int(src_file_level),
            }
            if n_match <= 0:
                continue

            feat_cpu = self._read_feature(
                file_level=src_file_level,
                t_dst=int(t_dst),
                dataset_name=str(dataset_name),
            )
            if out_cpu is None:
                out_cpu = torch.empty(
                    (N, int(feat_cpu.shape[1])),
                    dtype=torch.float32,
                    device="cpu",
                )

            src_idx_np = idx_sorted[pos[valid]].astype(np.int64, copy=False)
            dst_idx_np = dst_pos[valid].astype(np.int64, copy=False)

            src_idx_t = torch.from_numpy(src_idx_np).to(dtype=torch.long)
            vals = feat_cpu.index_select(0, src_idx_t)
            dst_idx_t = torch.from_numpy(dst_idx_np).to(dtype=torch.long)
            out_cpu.index_copy_(0, dst_idx_t, vals.to(dtype=torch.float32, device="cpu"))
            matched_cpu[dst_idx_t] = True

        matched_n = int(matched_cpu.sum().item())
        stats["matched"] = matched_n
        stats["missing"] = int(N - matched_n)
        out = (
            None
            if out_cpu is None
            else out_cpu.to(device=pred_centers.device, dtype=torch.float32)
        )
        matched = matched_cpu.to(device=pred_centers.device)
        return out, matched, stats


@torch.no_grad()
def _map_gt_to_pred_mesh_subset(
    *,
    src_centers: torch.Tensor,
    src_feats: torch.Tensor,
    pred_centers: torch.Tensor,
    query_idx: torch.Tensor,
    knn_k: int = 8,
    chunk: int = 8192,
    knn_backend: str = "exact",
    knn_backend_kwargs: Dict[str, Any] | None = None,
) -> Tuple[torch.Tensor, str, int]:
    q_idx = query_idx.view(-1).long()
    if q_idx.numel() == 0:
        empty = torch.empty(
            (0, int(src_feats.shape[1])),
            dtype=torch.float32,
            device=pred_centers.device,
        )
        return empty, "", -1

    out_dev = pred_centers.device
    idw_dev, idw_chunk = _select_idw_backend(
        src_n=int(src_centers.shape[0]),
        requested_chunk=int(chunk),
        out_device=out_dev,
    )

    src_c = src_centers.to(idw_dev, dtype=torch.float32)
    src_f = src_feats.to(idw_dev, dtype=torch.float32)
    dst_c = pred_centers.index_select(0, q_idx.to(device=pred_centers.device)).to(idw_dev, dtype=torch.float32)

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
    return mapped.to(out_dev), str(idw_dev.type), int(idw_chunk)


@torch.no_grad()
def _lookup_gt_on_pred_mesh_multires(
    *,
    lookup: _RuntimeMultiResGtLookup,
    t_dst: int,
    dataset_name: str,
    pred_centers: torch.Tensor,
    pred_levels: torch.Tensor,
    fallback_src_centers: torch.Tensor,
    fallback_src_feats: torch.Tensor,
    knn_k: int,
    chunk: int,
    knn_backend: str,
    knn_backend_kwargs: Dict[str, Any] | None,
    allow_fallback_to_idw: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    out: Optional[torch.Tensor] = None
    matched = torch.zeros((pred_levels.view(-1).numel(),), dtype=torch.bool, device=pred_centers.device)
    info: Dict[str, Any] = {
        "requested": int(pred_levels.view(-1).numel()),
        "matched_lookup": 0,
        "missing_lookup": int(pred_levels.view(-1).numel()),
        "fallback_used": False,
        "fallback_idw_dev": "lookup",
        "fallback_idw_chunk": -1,
    }

    try:
        out, matched, st = lookup.lookup(
            t_dst=int(t_dst),
            dataset_name=str(dataset_name),
            pred_centers=pred_centers,
            pred_levels=pred_levels,
        )
        info.update(st)
        info["matched_lookup"] = int(st.get("matched", 0))
        info["missing_lookup"] = int(st.get("missing", 0))
    except Exception as e:
        info["lookup_error"] = repr(e)
        if not allow_fallback_to_idw:
            raise

    missing_idx = (~matched).nonzero(as_tuple=False).view(-1)
    if missing_idx.numel() > 0:
        if not allow_fallback_to_idw:
            req_n = int(pred_levels.view(-1).numel())
            miss_n = int(missing_idx.numel())
            miss_pct = (100.0 * float(miss_n) / max(float(req_n), 1.0))
            by_level = info.get("by_level", {})
            by_level_str = ""
            if isinstance(by_level, dict) and by_level:
                parts = []
                for lv in sorted(by_level.keys()):
                    rec = by_level.get(lv, {}) or {}
                    parts.append(
                        f"L{int(lv)}:{int(rec.get('matched', 0))}/{int(rec.get('requested', 0))}"
                    )
                by_level_str = " | by_level=" + ",".join(parts)
            raise RuntimeError(
                "Multi-resolution lookup missed cells and fallback_to_idw=false. "
                f"dataset={dataset_name} t_dst={int(t_dst)} missing={miss_n}/{req_n} ({miss_pct:.2f}%)."
                f"{by_level_str} "
                "This indicates runtime mesh geometry is not an exact subset of cached uniform meshes."
            )
        mapped_fallback, fb_dev, fb_chunk = _map_gt_to_pred_mesh_subset(
            src_centers=fallback_src_centers,
            src_feats=fallback_src_feats,
            pred_centers=pred_centers,
            query_idx=missing_idx,
            knn_k=int(knn_k),
            chunk=int(chunk),
            knn_backend=str(knn_backend),
            knn_backend_kwargs=knn_backend_kwargs,
        )
        if out is None:
            out_dev = pred_centers.device
            out = torch.empty(
                (int(pred_centers.shape[0]), int(mapped_fallback.shape[1])),
                dtype=torch.float32,
                device=("cpu" if out_dev.type == "mps" else out_dev),
            )
        if out.device.type == "mps":
            out_cpu = out.to("cpu", dtype=torch.float32)
            out_cpu.index_copy_(
                0,
                missing_idx.to(device="cpu", dtype=torch.long),
                mapped_fallback.to(device="cpu", dtype=torch.float32),
            )
            out = out_cpu.to(device=pred_centers.device, dtype=torch.float32)
        else:
            out.index_copy_(
                0,
                missing_idx.to(device=out.device, dtype=torch.long),
                mapped_fallback.to(device=out.device, dtype=torch.float32),
            )
        info["fallback_used"] = True
        info["fallback_idw_dev"] = str(fb_dev)
        info["fallback_idw_chunk"] = int(fb_chunk)

    if out is None:
        raise RuntimeError(
            "Multi-resolution lookup produced no output and fallback did not run."
        )
    if out.device != pred_centers.device:
        out = out.to(device=pred_centers.device, dtype=torch.float32)
    info["matched_total"] = int(pred_centers.shape[0] - int((~matched).sum().item()))
    info["missing_total"] = int((~matched).sum().item())
    return out, info


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
    debug_out: Dict[str, Any] | None = None,
    timing_out: Dict[str, float] | None = None,
) -> Dict[int, torch.Tensor]:
    if runtime_mesh_policy is None:
        raise RuntimeError("runtime_mesh_policy is None for CNN runtime mesh backend.")
    if runtime_mesh_policy.get("backend", None) != "cnn":
        raise RuntimeError("runtime_mesh_policy backend is not 'cnn'.")

    timing: Dict[str, float] = {
        "predict_parent_map_s": 0.0,
        "predict_coarse_agg_s": 0.0,
        "predict_fill_empty_s": 0.0,
        "predict_feature_norm_s": 0.0,
        "predict_gradient_channels_s": 0.0,
        "predict_channel_assemble_s": 0.0,
        "predict_model_forward_s": 0.0,
        "predict_logits_to_masks_s": 0.0,
    }

    pdev = torch.device(runtime_mesh_policy["device"])
    model = runtime_mesh_policy["model"]
    rr = int(runtime_mesh_policy["refine_ratio"])
    Lmax = int(runtime_mesh_policy["max_level"])
    thr_default = float(runtime_mesh_policy["threshold_default"])
    thr_by_level: Dict[int, float] = runtime_mesh_policy["threshold_by_level"]

    t_parent_map_t0 = time.perf_counter()
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
    timing["predict_parent_map_s"] = float(time.perf_counter() - t_parent_map_t0)

    t_coarse_t0 = time.perf_counter()
    coarse = coarse_aggregate_from_dynamic(feat_policy, parents_for_cnn, H, W)
    coarse = coarse.view(H, W, -1).permute(2, 0, 1).contiguous()  # (F,H,W)
    timing["predict_coarse_agg_s"] = float(time.perf_counter() - t_coarse_t0)
    if bool(fill_empty_interior):
        t_fill_t0 = time.perf_counter()
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
        timing["predict_fill_empty_s"] = float(time.perf_counter() - t_fill_t0)

    t_featnorm_t0 = time.perf_counter()
    coarse_raw = coarse
    feature_norm_mode = str(runtime_mesh_policy.get("feature_norm_mode", "none"))
    feature_norm_stats = _parse_feature_norm_stats(
        runtime_mesh_policy.get("feature_norm_stats", None),
        n_channels=int(coarse_raw.shape[0]),
    )
    coarse = _apply_feature_norm_chw(
        coarse_raw,
        mode=feature_norm_mode,
        stats=feature_norm_stats,
        require_stats=True,
    )
    timing["predict_feature_norm_s"] = float(time.perf_counter() - t_featnorm_t0)

    feat_names_cfg = cfg.get("features", {}).get("names", [])
    if not isinstance(feat_names_cfg, (list, tuple)):
        feat_names_cfg = []
    state_names = []
    F_state = int(coarse.shape[0])
    for i in range(F_state):
        if i < len(feat_names_cfg):
            state_names.append(str(feat_names_cfg[i]))
        else:
            state_names.append(f"state[{i}]")

    channels = [coarse]
    channel_names = list(state_names)
    if bool(runtime_mesh_policy.get("include_gradients", False)):
        t_grad_t0 = time.perf_counter()
        gidx = runtime_mesh_policy.get("gradient_feature_indices", None)
        if not isinstance(gidx, dict):
            raise RuntimeError(
                "Runtime CNN include_gradients=true but gradient_feature_indices is missing."
            )
        rho_raw = gidx.get("rho", None)
        e_raw = gidx.get("E", None)
        if rho_raw is None or e_raw is None:
            raise RuntimeError(
                "Runtime CNN include_gradients=true but gradient_feature_indices must contain "
                f"'rho' and 'E'. Got {gidx}"
            )
        rho_idx = int(rho_raw)
        e_idx = int(e_raw)
        Fdim = int(coarse_raw.shape[0])
        if not (0 <= rho_idx < Fdim) or not (0 <= e_idx < Fdim):
            raise RuntimeError(
                "Runtime CNN include_gradients=true but rho/E feature indices are out of bounds: "
                f"rho={rho_idx}, E={e_idx}, F={Fdim}"
            )
        xmin, xmax, ymin, ymax = [float(v) for v in _get_bbox(cfg)]
        dx = max((xmax - xmin) / max(float(W), 1.0), 1e-12)
        dy = max((ymax - ymin) / max(float(H), 1.0), 1e-12)
        grad_norm_mode = str(runtime_mesh_policy.get("gradient_norm_mode", "none"))
        grad_norm_stats = _parse_gradient_norm_stats(
            runtime_mesh_policy.get("gradient_norm_stats", None)
        )
        rho_stats = (None if grad_norm_stats is None else grad_norm_stats.get("rho", None))
        E_stats = (None if grad_norm_stats is None else grad_norm_stats.get("E", None))
        g_rho_flat = _coarse_grad_mag(
            coarse_raw[rho_idx].reshape(-1),
            H,
            W,
            dx,
            dy,
            p=1.0,
        )
        g_E_flat = _coarse_grad_mag(
            coarse_raw[e_idx].reshape(-1),
            H,
            W,
            dx,
            dy,
            p=1.0,
        )
        g_rho = _apply_gradient_norm_flat(
            g_rho_flat,
            mode=grad_norm_mode,
            mean=(None if rho_stats is None else float(rho_stats["mean"])),
            std=(None if rho_stats is None else float(rho_stats["std"])),
            require_stats=True,
        ).view(H, W).unsqueeze(0)
        g_E = _apply_gradient_norm_flat(
            g_E_flat,
            mode=grad_norm_mode,
            mean=(None if E_stats is None else float(E_stats["mean"])),
            std=(None if E_stats is None else float(E_stats["std"])),
            require_stats=True,
        ).view(H, W).unsqueeze(0)
        channels.extend([g_rho, g_E])
        channel_names.extend(["grad_rho", "grad_E"])
        timing["predict_gradient_channels_s"] = float(time.perf_counter() - t_grad_t0)

    t_chan_t0 = time.perf_counter()
    if bool(runtime_mesh_policy.get("include_parent_mask", True)):
        channels.append(mask_for_cnn.to(coarse.device, dtype=torch.float32).unsqueeze(0))
        channel_names.append("parent_mask")
    if bool(runtime_mesh_policy.get("include_coords", True)):
        coord_grid = runtime_mesh_policy.get("coord_grid", None)
        if coord_grid is None:
            raise RuntimeError("CNN runtime mesh policy expects coord channels, but coord_grid is missing.")
        coord_chw = coord_grid.to(coarse.device, dtype=torch.float32)
        channels.append(coord_chw)
        Cc = int(coord_chw.shape[0])
        if Cc >= 2:
            channel_names.extend(["coord_x", "coord_y"])
        else:
            channel_names.extend([f"coord[{i}]" for i in range(Cc)])
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
        channel_names.append("dt_hat" if mode == "dt_hat" else "dt_raw")

    x_in = torch.cat(channels, dim=0).unsqueeze(0)  # (1,C,H,W)
    expected_in = int(runtime_mesh_policy.get("in_channels", -1))
    if expected_in > 0 and int(x_in.shape[1]) != expected_in:
        raise RuntimeError(
            f"CNN runtime mesh input channel mismatch: built={int(x_in.shape[1])}, expected={expected_in}. "
            "Check runtime_mesh.cnn.include_parent_mask/include_coords/include_gradients and feature channel setup."
        )
    timing["predict_channel_assemble_s"] = float(time.perf_counter() - t_chan_t0)

    t_fwd_t0 = time.perf_counter()
    logits_raw = model(x_in.to(pdev, dtype=torch.float32, non_blocking=True))
    if not isinstance(logits_raw, dict):
        raise RuntimeError(f"CNN runtime mesh model forward must return dict[int,Tensor], got {type(logits_raw)}")
    timing["predict_model_forward_s"] = float(time.perf_counter() - t_fwd_t0)

    logits_by_level: Dict[int, torch.Tensor] = {}
    for k, v in logits_raw.items():
        logits_by_level[int(k)] = v

    t_logits_to_masks_t0 = time.perf_counter()
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

    timing["predict_logits_to_masks_s"] = float(time.perf_counter() - t_logits_to_masks_t0)
    if isinstance(debug_out, dict):
        debug_out.clear()
        debug_out["backend"] = "cnn"
        debug_out["x_in_chw"] = x_in[0].detach().to("cpu", dtype=torch.float32).clone()
        debug_out["channel_names"] = list(channel_names)
    if isinstance(timing_out, dict):
        timing_out.clear()
        timing_out.update(timing)
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
    physical_parent_mask: torch.Tensor | None = None,
    coarse_rc_grid: torch.Tensor | None = None,
    use_prev_state_masks: bool = True,
    debug_out: Dict[str, Any] | None = None,
    timing_out: Dict[str, float] | None = None,
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
    timing: Dict[str, float] = {
        "predict_coarse_agg_s": 0.0,
        "predict_fill_empty_s": 0.0,
        "predict_grad_score_s": 0.0,
        "predict_prev_masks_s": 0.0,
        "predict_hierarchy_s": 0.0,
        "predict_threshold_hysteresis_s": 0.0,
        "predict_dilation_s": 0.0,
        "predict_pool_up_s": 0.0,
    }
    pol = cfg.get("policy", {}) or {}
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    gf_cfg = rt_cfg.get("gradient_fast", {}) or {}
    if not isinstance(gf_cfg, dict):
        gf_cfg = {}
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

    def _first_present(src: Dict[str, Any], keys: list[str]):
        if not isinstance(src, dict):
            return None
        for k in keys:
            if k in src:
                return src.get(k)
        return None

    def _resolve_dilate_cells_for_level(L: int) -> float | None:
        raw_cells = _first_present(
            gf_cfg,
            [f"dilate_cells_L{L}", f"dilate_cells_l{L}", f"dilate_cells_level{L}"],
        )
        if raw_cells is None:
            raw_cells = _first_present(
                pol,
                [f"dilate_cells_L{L}", f"dilate_cells_l{L}", f"dilate_cells_level{L}"],
            )
        if raw_cells is None and (L == Lmax):
            raw_cells = _first_present(
                gf_cfg,
                ["dilate_cells_lmax", "dilate_cells_Lmax", "dilate_cells_max"],
            )
        if raw_cells is None and (L == Lmax):
            raw_cells = _first_present(
                pol,
                ["dilate_cells_lmax", "dilate_cells_Lmax", "dilate_cells_max"],
            )
        if raw_cells is None:
            return None
        try:
            return float(raw_cells)
        except Exception:
            return None

    def _resolve_dilate_phys_for_level(L: int) -> float:
        raw_phys = _first_present(
            gf_cfg,
            [f"dilate_phys_L{L}", f"dilate_phys_l{L}", f"dilate_phys_level{L}"],
        )
        if raw_phys is None:
            raw_phys = _first_present(
                pol,
                [f"dilate_phys_L{L}", f"dilate_phys_l{L}", f"dilate_phys_level{L}"],
            )
        try:
            return float(raw_phys) if raw_phys is not None else 0.0
        except Exception:
            return 0.0

    raw_lmax_cells = _first_present(
        gf_cfg,
        ["dilate_cells_lmax", "dilate_cells_Lmax", "dilate_cells_max"],
    )
    if raw_lmax_cells is None:
        raw_lmax_cells = _first_present(
            pol,
            ["dilate_cells_lmax", "dilate_cells_Lmax", "dilate_cells_max"],
        )
    try:
        lmax_cells_val = float(raw_lmax_cells) if raw_lmax_cells is not None else None
    except Exception:
        lmax_cells_val = None
    has_lmax_dilation_cfg = bool((lmax_cells_val is not None) and (lmax_cells_val > 0.0))

    propagate_lmax_to_parents = bool(
        gf_cfg.get(
            "dilate_cells_lmax_propagate_to_parents",
            pol.get("dilate_cells_lmax_propagate_to_parents", True),
        )
    )

    t_coarse_t0 = time.perf_counter()
    coarse = coarse_aggregate_from_dynamic(feat_policy, parents_t, H, W).to(dev, dtype=torch.float32)
    timing["predict_coarse_agg_s"] = float(time.perf_counter() - t_coarse_t0)
    coarse_hwf = coarse.view(H, W, -1).contiguous()
    if bool(gf_cfg.get("fill_empty_interior", True)) and (physical_parent_mask is not None):
        t_fill_t0 = time.perf_counter()
        fill_chunk = int(gf_cfg.get("fill_empty_chunk", 1024))
        if fill_chunk < 1:
            fill_chunk = 1
        coarse_chw = coarse_hwf.permute(2, 0, 1).contiguous()
        coarse_chw = _runtime_fill_empty_interior_coarse_nearest(
            coarse_chw=coarse_chw,
            parents=parents_t,
            H=H,
            W=W,
            physical_parent_mask=torch.as_tensor(physical_parent_mask, dtype=torch.bool, device=dev),
            rc_grid=coarse_rc_grid,
            chunk=fill_chunk,
        )
        coarse_hwf = coarse_chw.permute(1, 2, 0).contiguous()
        timing["predict_fill_empty_s"] = float(time.perf_counter() - t_fill_t0)
    coarse = coarse_hwf.view(H * W, -1).contiguous()
    t_gradscore_t0 = time.perf_counter()
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
    timing["predict_grad_score_s"] = float(time.perf_counter() - t_gradscore_t0)

    xmin, xmax, ymin, ymax = _get_bbox(cfg)
    masks_by_level: Dict[int, torch.Tensor] = {}

    # Reconstruct previous refine masks per level from current runtime mesh state.
    # prev_refine_by_level[L] lives on the parent grid for level L (H*rr^(L-1), W*rr^(L-1)).
    t_prev_t0 = time.perf_counter()
    prev_refine_by_level: Dict[int, torch.Tensor] = {}
    if bool(use_prev_state_masks):
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
    timing["predict_prev_masks_s"] = float(time.perf_counter() - t_prev_t0)

    t_hier_t0 = time.perf_counter()
    dilation_s = 0.0
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

        n_cells = _resolve_dilate_cells_for_level(L)

        if n_cells is not None:
            r_phys = float(max(0.0, n_cells) * dxL_here)
        else:
            r_phys = _resolve_dilate_phys_for_level(L)

        if r_phys > 0:
            t_dilate_t0 = time.perf_counter()
            r_cells = max(1, int(round(r_phys / max(dxL_here, 1e-12))))
            k = 2 * r_cells + 1
            M = F.max_pool2d(
                M.float()[None, None],
                kernel_size=k,
                stride=1,
                padding=r_cells,
            )[0, 0].bool()
            dilation_s += float(time.perf_counter() - t_dilate_t0)

        masks_by_level[L] = M
    thr_total = float(time.perf_counter() - t_hier_t0)

    # If finest-level dilation is requested, ensure hierarchy closure by OR-ing
    # required parent refine masks upward (L -> L-1). Without this, wedge/hierarchy
    # clipping can erase most or all apparent Lmax dilation effect.
    if has_lmax_dilation_cfg and propagate_lmax_to_parents:
        t_pool_t0 = time.perf_counter()
        for L in range(int(Lmax), 1, -1):
            child = masks_by_level.get(L, None)
            parent = masks_by_level.get(L - 1, None)
            if child is None:
                continue
            child = child.to(dev, dtype=torch.bool)
            req_parent = F.max_pool2d(
                child.float()[None, None],
                kernel_size=int(rr),
                stride=int(rr),
            )[0, 0].bool()
            if parent is None:
                parent = torch.zeros_like(req_parent, dtype=torch.bool, device=dev)
            else:
                parent = parent.to(dev, dtype=torch.bool)
                if tuple(parent.shape) != tuple(req_parent.shape):
                    parent = F.interpolate(
                        parent.float()[None, None],
                        size=tuple(req_parent.shape),
                        mode="nearest",
                    )[0, 0].bool()
            masks_by_level[L - 1] = parent | req_parent
        timing["predict_pool_up_s"] = float(time.perf_counter() - t_pool_t0)

    timing["predict_hierarchy_s"] = thr_total
    timing["predict_dilation_s"] = float(dilation_s)
    timing["predict_threshold_hysteresis_s"] = max(0.0, thr_total - float(dilation_s))

    if isinstance(timing_out, dict):
        timing_out.clear()
        timing_out.update(timing)
    if isinstance(debug_out, dict):
        debug_out.clear()
        debug_out["backend"] = "gradient_fast"
        debug_out["selected_idx"] = [int(i) for i in idxs]
        debug_out["coarse_hwf"] = coarse_hwf.detach().to("cpu", dtype=torch.float32).clone()
        debug_out["g_base_hxw"] = g_base.detach().to("cpu", dtype=torch.float32).clone()
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
    predictor_input_out: Dict[str, Any] | None = None,
    step_k: int | None = None,
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
        "predict_parent_map_s": 0.0,
        "predict_coarse_agg_s": 0.0,
        "predict_fill_empty_s": 0.0,
        "predict_feature_norm_s": 0.0,
        "predict_gradient_channels_s": 0.0,
        "predict_channel_assemble_s": 0.0,
        "predict_model_forward_s": 0.0,
        "predict_logits_to_masks_s": 0.0,
        "predict_grad_score_s": 0.0,
        "predict_prev_masks_s": 0.0,
        "predict_hierarchy_s": 0.0,
        "predict_grad_raster_s": 0.0,
        "predict_combine_levels_s": 0.0,
        "predict_pool_up_s": 0.0,
        "predict_threshold_hysteresis_s": 0.0,
        "predict_dilation_s": 0.0,
        "predict_normalize_masks_s": 0.0,
        "predict_debug_out_s": 0.0,
        "predict_legacy_policy_s": 0.0,
        "total_s": 0.0,
    }

    dev = torch.device(device)
    rr = _get_refine_ratio(cfg)
    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    domain_mode = _runtime_mesh_domain_mode(cfg)
    policy_backend = _runtime_mesh_backend(cfg)
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

    parents_for_policy = parents_t
    mask_for_policy = mask_t_parent
    parents_source = "dataset"
    if policy_backend in ("gradient", "gradient_fast"):
        step0_parent_mode_raw = str(
            rt_cfg.get("gradient_step0_parent_mapping_mode", "from_centers")
        ).strip().lower()
        if (step_k is not None) and (int(step_k) == 0):
            if step0_parent_mode_raw in ("from_centers", "centers", "runtime_like", "coarse_from_centers"):
                t_parent_map_t0 = time.perf_counter()
                row_t, col_t = _cell_level_ij_from_centers(
                    centers=centers_t,
                    levels=level_t,
                    H=H,
                    W=W,
                    bbox=_get_bbox(cfg),
                    refine_ratio=rr,
                )
                parents_for_policy = _runtime_parents_from_level_ij(
                    levels=level_t,
                    row=row_t,
                    col=col_t,
                    H=H,
                    W=W,
                    refine_ratio=rr,
                ).view(-1)
                mask_for_policy = _mask_from_parent_indices(parents_for_policy, H, W, dev)
                timing["predict_parent_map_s"] = float(time.perf_counter() - t_parent_map_t0)
                parents_source = "from_centers"
            elif step0_parent_mode_raw in ("dataset", "legacy", "from_dataset"):
                parents_source = "dataset"
            else:
                raise ValueError(
                    "train.runtime_mesh.gradient_step0_parent_mapping_mode must be "
                    f"'dataset' or 'from_centers', got {step0_parent_mode_raw!r}"
                )

    batch_like = {
        "centers_t": centers_t,
        "center_feat_t": feat_policy,
        "dyn_feat_t": feat_policy,
        "dyn_parents": parents_for_policy,
        "mask_t": mask_for_policy.view(-1),
        "level_t": level_t,
    }
    policy_debug_out: Dict[str, Any] | None = {} if isinstance(predictor_input_out, dict) else None
    prev_mode_raw = str((cfg.get("policy", {}) or {}).get("hysteresis_prev_source", "predicted")).strip().lower()
    if prev_mode_raw in ("prediction", "runtime", "state"):
        prev_mode = "predicted"
    elif prev_mode_raw in ("gt", "ground_truth", "ground-truth", "truth"):
        prev_mode = "gt"
    else:
        prev_mode = "predicted"
    use_prev_state_masks = (prev_mode == "gt") or ((step_k is not None) and (int(step_k) > 0))

    t_policy_t0 = time.perf_counter()
    if policy_backend == "cnn":
        policy_timing: Dict[str, float] = {}
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
            debug_out=policy_debug_out,
            timing_out=policy_timing,
        )
        for k, v in policy_timing.items():
            timing[k] = float(v)
    elif policy_backend == "gradient_fast":
        policy_timing = {}
        gf_physical_parent_mask = None
        if (domain_mode == "starting_mesh") and (runtime_base_mesh is not None):
            gf_physical_parent_mask = runtime_base_mesh.get("parent_mask", None)
        elif runtime_wedge_constraints is not None:
            parent_intersect = runtime_wedge_constraints.get("parent_intersect", {})
            if isinstance(parent_intersect, dict):
                gf_physical_parent_mask = parent_intersect.get(1, None)
        masks_pred_by_level = _runtime_predict_masks_from_fast_gradients(
            centers_t=centers_t,
            level_t=level_t,
            feat_policy=feat_policy,
            parents_t=parents_for_policy,
            mask_t_parent=mask_for_policy,
            cfg=cfg,
            H=H,
            W=W,
            dx=dx,
            dy=dy,
            device=dev,
            physical_parent_mask=gf_physical_parent_mask,
            coarse_rc_grid=runtime_mesh_policy.get("coarse_rc_grid", None) if isinstance(runtime_mesh_policy, dict) else None,
            use_prev_state_masks=bool(use_prev_state_masks),
            debug_out=policy_debug_out,
            timing_out=policy_timing,
        )
        for k, v in policy_timing.items():
            timing[k] = float(v)
    else:
        policy_timing = {}
        t_legacy_policy_t0 = time.perf_counter()
        masks_pred_by_level = predict_masks_hierarchical_from_gt_gradients(
            batch_like,
            cfg,
            H,
            W,
            dx,
            dy,
            device=dev,
            debug_out=policy_debug_out,
            timing_out=policy_timing,
        )
        if isinstance(policy_timing, dict) and len(policy_timing) > 0:
            for k, v in policy_timing.items():
                timing[k] = float(v)
        else:
            timing["predict_legacy_policy_s"] = float(time.perf_counter() - t_legacy_policy_t0)
    timing["mesh_predict_s"] = float(time.perf_counter() - t_policy_t0)

    if isinstance(predictor_input_out, dict):
        predictor_input_out.clear()
        predictor_input_out["backend"] = str(policy_backend)
        predictor_input_out["parents_source"] = str(parents_source)
        if (
            (policy_backend in ("gradient", "gradient_fast"))
            and (step_k is not None)
            and (int(step_k) == 0)
            and (parents_source == "from_centers")
            and (parents_t.numel() == parents_for_policy.numel())
        ):
            mismatch = int((parents_t != parents_for_policy).sum().item())
            predictor_input_out["parents_mismatch"] = int(mismatch)
            predictor_input_out["parents_total"] = int(parents_t.numel())
        if isinstance(policy_debug_out, dict):
            if policy_backend == "gradient":
                G = policy_debug_out.get("G", None)
                pooled = policy_debug_out.get("pooled_up", None)
                if isinstance(G, dict):
                    predictor_input_out["G"] = {
                        int(L): v.detach().to("cpu", dtype=torch.float32).clone()
                        for L, v in G.items()
                        if torch.is_tensor(v)
                    }
                if isinstance(pooled, dict):
                    predictor_input_out["pooled_up"] = {
                        int(L): v.detach().to("cpu", dtype=torch.float32).clone()
                        for L, v in pooled.items()
                        if torch.is_tensor(v)
                    }
            else:
                for kk, vv in policy_debug_out.items():
                    predictor_input_out[kk] = vv

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
    # Legacy AMR debug logger intentionally disabled in Cartesian training path.
    return


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


def _step_outputs_are_finite(
    *,
    loss_t: torch.Tensor | None,
    pred_abs: torch.Tensor | None,
    tgt_abs: torch.Tensor | None,
    split: str,
    batch_idx: int,
    step_k: int,
) -> bool:
    ok_loss = _assert_finite(f"{split}.loss_step", loss_t, crash=False)
    ok_pred = _assert_finite(f"{split}.pred_abs", pred_abs, crash=False)
    ok_tgt = _assert_finite(f"{split}.tgt_abs", tgt_abs, crash=False)
    ok = bool(ok_loss and ok_pred and ok_tgt)
    if not ok:
        print(
            f"[WARN][{split}] skipping non-finite step: batch={int(batch_idx)} step_k={int(step_k)} "
            f"(finite: loss={int(ok_loss)} pred={int(ok_pred)} tgt={int(ok_tgt)})",
            flush=True,
        )
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


def _normalize_predict_type_key(predict_type: Any) -> str:
    key = str(predict_type).strip().lower()
    if key == "absolute":
        key = "state"
    if key not in {"state", "delta", "rate"}:
        raise ValueError(
            f"model.predict_type must be one of {{state, delta, rate}}, got {predict_type!r}."
        )
    return key


def _target_for_predict_type(
    *,
    norm_in: torch.Tensor,
    norm_tgt: torch.Tensor,
    dt_hat: torch.Tensor,
    predict_type: str,
) -> torch.Tensor:
    predict_type = _normalize_predict_type_key(predict_type)
    if predict_type == "state":
        return norm_tgt
    delta_target = norm_tgt - norm_in
    if predict_type == "delta":
        return delta_target
    return delta_target / dt_hat.clamp_min(1e-12)


def _state_from_model_output(
    *,
    norm_in: torch.Tensor,
    y_pred: torch.Tensor,
    dt_hat: torch.Tensor,
    predict_type: str,
) -> torch.Tensor:
    predict_type = _normalize_predict_type_key(predict_type)
    if predict_type == "state":
        return y_pred
    if predict_type == "delta":
        return norm_in + y_pred
    return norm_in + y_pred * dt_hat


def _parse_lr_schedule_entries(train_cfg: Dict[str, Any]) -> list[tuple[int, float]]:
    sched_cfg = train_cfg.get("lr_schedule", {}) or {}
    if not isinstance(sched_cfg, dict):
        raise ValueError("train.lr_schedule must be a JSON object when provided.")
    if not bool(sched_cfg.get("enabled", False)):
        return []

    entries_raw = sched_cfg.get("entries", [])
    if entries_raw is None:
        entries_raw = []
    if not isinstance(entries_raw, (list, tuple)):
        raise ValueError("train.lr_schedule.entries must be a list of {epoch, lr} objects.")

    entries: list[tuple[int, float]] = []
    seen: set[int] = set()
    for i, entry in enumerate(entries_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"train.lr_schedule.entries[{i}] must be an object, got {type(entry)}.")
        if "epoch" not in entry or "lr" not in entry:
            raise ValueError(f"train.lr_schedule.entries[{i}] must include both 'epoch' and 'lr'.")
        epoch = int(entry["epoch"])
        lr = float(entry["lr"])
        if epoch < 1:
            raise ValueError(f"train.lr_schedule.entries[{i}].epoch must be >= 1, got {epoch}.")
        if lr <= 0.0:
            raise ValueError(f"train.lr_schedule.entries[{i}].lr must be > 0, got {lr}.")
        if epoch in seen:
            raise ValueError(f"Duplicate train.lr_schedule entry for epoch {epoch}.")
        seen.add(epoch)
        entries.append((epoch, lr))

    entries.sort(key=lambda item: item[0])
    return entries


def _lr_from_schedule_for_epoch(
    entries: list[tuple[int, float]],
    epoch: int,
) -> float | None:
    lr_eff = None
    ep = int(epoch)
    for milestone, lr in entries:
        if int(milestone) <= ep:
            lr_eff = float(lr)
        else:
            break
    return lr_eff


def _set_optimizer_lr(optimizer, lr: float) -> bool:
    lr = float(lr)
    changed = False
    for group in optimizer.param_groups:
        old = float(group.get("lr", lr))
        if abs(old - lr) > max(1e-16, 1e-12 * max(abs(old), abs(lr), 1.0)):
            group["lr"] = lr
            changed = True
    return changed


def _pressure_auxiliary_losses(
    *,
    y_pred_abs: torch.Tensor,
    x_tgt_abs: torch.Tensor | None,
    cfg: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derived ideal-gas pressure supervision.

    Returns:
      pressure_target_loss: compare p(rho,e/E_tot) from prediction to target pressure
      pressure_consistency_loss: compare predicted primitive P channel to p(rho,e)
    """
    zero = y_pred_abs.new_zeros(())
    loss_cfg = cfg.get("loss", {}) or {}
    aux_w = float(loss_cfg.get("pressure_aux_weight", loss_cfg.get("pressure_loss_weight", 0.0)))
    cons_w = float(loss_cfg.get("pressure_consistency_weight", 0.0))
    if aux_w <= 0.0 and cons_w <= 0.0:
        return zero, zero

    Fdim = int(y_pred_abs.size(1))
    idx = dec.infer_feature_indices(cfg, Fdim)
    rep = dec.state_representation_from_cfg(cfg, Fdim)
    gamma = float(dec.gas_gamma_from_cfg(cfg))
    eps = float(loss_cfg.get("pressure_aux_eps", loss_cfg.get("p_floor", 1e-8)))
    eps = max(eps, 1e-12)

    if rep == "primitive_uvrhope":
        rho_pred = y_pred_abs[:, int(idx["rho"])].clamp_min(eps)
        e_pred = y_pred_abs[:, int(idx["E"])].clamp_min(eps)
        p_derived_pred = ((gamma - 1.0) * rho_pred * e_pred).clamp_min(eps)
        p_idx = idx.get("p", None)
        p_channel_pred = None
        if p_idx is not None:
            p_channel_pred = y_pred_abs[:, int(p_idx)].clamp_min(eps)
        if x_tgt_abs is not None:
            if p_idx is not None:
                p_target = x_tgt_abs[:, int(p_idx)].clamp_min(eps)
            else:
                rho_tgt = x_tgt_abs[:, int(idx["rho"])].clamp_min(eps)
                e_tgt = x_tgt_abs[:, int(idx["E"])].clamp_min(eps)
                p_target = ((gamma - 1.0) * rho_tgt * e_tgt).clamp_min(eps)
        else:
            p_target = None
    else:
        p_derived_pred = dec.pressure_from_conservative_state(
            y_pred_abs,
            cfg,
            eps=eps,
            clamp_min=eps,
        )
        p_channel_pred = None
        p_target = (
            None
            if x_tgt_abs is None
            else dec.pressure_from_conservative_state(x_tgt_abs, cfg, eps=eps, clamp_min=eps)
        )

    def _compare_pressure(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        use_log = bool(loss_cfg.get("pressure_aux_log", True))
        use_huber = bool(loss_cfg.get("pressure_aux_huber", True))
        delta = float(loss_cfg.get("pressure_aux_huber_delta", 0.1 if use_log else 1.0))
        if use_log:
            aa = torch.log(a.clamp_min(eps))
            bb = torch.log(b.clamp_min(eps))
        else:
            scale = b.detach().abs().mean().clamp_min(float(loss_cfg.get("pressure_aux_scale_floor", 1.0)))
            aa = a / scale
            bb = b / scale
        return F.huber_loss(aa, bb, delta=delta) if use_huber else F.mse_loss(aa, bb)

    target_loss = zero
    if aux_w > 0.0 and p_target is not None:
        target_loss = _compare_pressure(p_derived_pred, p_target)

    consistency_loss = zero
    if cons_w > 0.0 and p_channel_pred is not None:
        consistency_loss = _compare_pressure(p_channel_pred, p_derived_pred.detach())

    return target_loss, consistency_loss


def _resolve_model_type_key(model_cfg: Dict[str, Any] | None) -> str:
    model_cfg = model_cfg or {}
    raw_model_type = model_cfg.get("type", None)
    model_name = str(model_cfg.get("name", "FeatureNet")).strip().lower().replace("-", "_")
    if raw_model_type is None:
        if model_name in {"sageconvmodel", "sageconv", "advectionsagemodel", "burgerssagemodel"}:
            model_type = "sageconv"
        elif model_name in {
            "meshgraphnet",
            "mesh_graph_net",
            "mgn",
            "meshgraphnetmodel",
            "advectionmeshgraphnetmodel",
            "burgersmeshgraphnetmodel",
        }:
            model_type = "meshgraphnet"
        elif model_name in {
            "fluxgraphnet",
            "flux_graph_net",
            "fluxgnn",
            "flux",
            "fluxgraphnetmodel",
        }:
            model_type = "fluxgraphnet"
        else:
            model_type = "featurenet"
    else:
        model_type = str(raw_model_type).strip().lower().replace("-", "_")

    model_type_aliases = {
        "feature_net": "featurenet",
        "featurenet": "featurenet",
        "graphsage": "sageconv",
        "sage_conv": "sageconv",
        "sageconv": "sageconv",
        "sagemodel": "sageconv",
        "sageconvmodel": "sageconv",
        "advectionsagemodel": "sageconv",
        "burgerssagemodel": "sageconv",
        "mesh_graph_net": "meshgraphnet",
        "meshgraphnet": "meshgraphnet",
        "mgn": "meshgraphnet",
        "meshgraphnetmodel": "meshgraphnet",
        "advectionmeshgraphnetmodel": "meshgraphnet",
        "burgersmeshgraphnetmodel": "meshgraphnet",
        "flux_graph_net": "fluxgraphnet",
        "fluxgraphnet": "fluxgraphnet",
        "fluxgnn": "fluxgraphnet",
        "flux": "fluxgraphnet",
        "fluxgraphnetmodel": "fluxgraphnet",
        "advectionfluxgraphnetmodel": "fluxgraphnet",
        "burgersfluxgraphnetmodel": "fluxgraphnet",
    }
    model_type = model_type_aliases.get(model_type, model_type)
    valid_model_types = {"featurenet", "sageconv", "meshgraphnet", "fluxgraphnet"}
    if model_type not in valid_model_types:
        raise ValueError(
            f"Unsupported model.type '{model_cfg.get('type')}'. "
            "Use model.type='featurenet', 'sageconv', 'meshgraphnet', or 'fluxgraphnet'. "
            "For FeatureNet message passing, set model.conv_type to one of "
            "{sage, gine, nnconv}."
        )
    return model_type


def _model_requires_edge_attr_from_cfg(cfg: Dict[str, Any]) -> bool:
    model_cfg = cfg.get("model", {}) or {}
    model_type = _resolve_model_type_key(model_cfg)
    if model_type in {"meshgraphnet", "fluxgraphnet"}:
        return True
    if model_type != "featurenet":
        return False

    conv_type = str(model_cfg.get("conv_type", "sage")).strip().lower().replace("-", "_")
    if conv_type == "sageconv":
        conv_type = "sage"
    if conv_type in {"gine", "nnconv"}:
        return True

    att_cfg = model_cfg.get("attention", {}) or {}
    use_attention = bool(model_cfg.get("use_attention", att_cfg.get("enabled", False)))
    attention_use_edge_attr = bool(
        model_cfg.get("attention_use_edge_attr", att_cfg.get("use_edge_attr", True))
    )
    return bool(use_attention and attention_use_edge_attr)


def sanitize_state_for_ops(x_abs: torch.Tensor, cfg: dict, rho_floor=1e-6, E_floor=1e-6):
    loss = cfg.get("loss", {}) or {}
    u_clip = float(loss.get("u_clip", 1e3))
    rho_max = float(loss.get("ops_rho_max", loss.get("state_rho_max", 0.0)))
    E_max = float(loss.get("ops_E_max", loss.get("state_E_max", 0.0)))
    m_max = float(loss.get("ops_m_max", loss.get("state_m_max", 0.0)))

    Fdim = int(x_abs.size(1))
    idx = dec.infer_feature_indices(cfg, Fdim)
    rep = dec.state_representation_from_cfg(cfg, Fdim)

    cols = [x_abs[:, j] for j in range(Fdim)]
    rho = cols[idx["rho"]].clamp_min(rho_floor)
    E = cols[idx["E"]].clamp_min(E_floor)
    if rho_max > 0.0:
        rho = rho.clamp_max(rho_max)
    if E_max > 0.0:
        E = E.clamp_max(E_max)

    if rep == "primitive_uvrhope":
        u_idx = int(idx.get("u", idx["mx"]))
        v_idx = int(idx.get("v", idx["my"]))
        u = cols[u_idx]
        v = cols[v_idx]
        if u_clip > 0.0:
            u = u.clamp(min=-u_clip, max=u_clip)
            v = v.clamp(min=-u_clip, max=u_clip)
        u = _sanitize_float_tensor(u, clip_abs=max(u_clip, m_max), fill_value=0.0, nonneg=False)
        v = _sanitize_float_tensor(v, clip_abs=max(u_clip, m_max), fill_value=0.0, nonneg=False)
        cols[u_idx] = u
        cols[v_idx] = v

        p_idx = idx.get("p", None)
        if p_idx is not None:
            p_floor = float(loss.get("p_floor", 0.0))
            p = cols[int(p_idx)].clamp_min(p_floor)
            cols[int(p_idx)] = _sanitize_float_tensor(p, clip_abs=0.0, fill_value=p_floor, nonneg=True)
    else:
        mx = cols[idx["mx"]]
        my = cols[idx["my"]]
        # enforce |u| <= u_clip by clamping momenta given rho
        if u_clip > 0:
            mx = mx.clamp(-u_clip * rho, u_clip * rho)
            my = my.clamp(-u_clip * rho, u_clip * rho)
        if m_max > 0.0:
            mx = mx.clamp(min=-m_max, max=m_max)
            my = my.clamp(min=-m_max, max=m_max)
        cols[idx["mx"]] = _sanitize_float_tensor(mx, clip_abs=m_max, fill_value=0.0, nonneg=False)
        cols[idx["my"]] = _sanitize_float_tensor(my, clip_abs=m_max, fill_value=0.0, nonneg=False)

    rho = _sanitize_float_tensor(rho, clip_abs=rho_max, fill_value=rho_floor, nonneg=True)
    E = _sanitize_float_tensor(E, clip_abs=E_max, fill_value=E_floor, nonneg=True)
    cols[idx["rho"]] = rho
    cols[idx["E"]] = E
    return torch.stack(cols, dim=1)


def _append_abs_samples(
    buckets: list[list[torch.Tensor]],
    block: torch.Tensor,
    *,
    max_samples_per_channel: int,
) -> None:
    if block is None or block.numel() == 0:
        return
    block_cpu = block.detach().abs().to(device="cpu", dtype=torch.float32)
    n_ch = int(block_cpu.size(1))
    if len(buckets) != n_ch:
        raise RuntimeError(f"Internal channel bucket mismatch: {len(buckets)} vs {n_ch}")

    for j in range(n_ch):
        vals = block_cpu[:, j].reshape(-1)
        if vals.numel() == 0:
            continue
        if max_samples_per_channel > 0:
            cur = sum(int(x.numel()) for x in buckets[j])
            remaining = int(max_samples_per_channel) - cur
            if remaining <= 0:
                continue
            if vals.numel() > remaining:
                idx = torch.linspace(
                    0,
                    vals.numel() - 1,
                    steps=remaining,
                    dtype=torch.long,
                )
                vals = vals.index_select(0, idx)
        buckets[j].append(vals)


@torch.no_grad()
def _maybe_calibrate_parc_input_scales(
    *,
    loader,
    cfg: dict,
    device: torch.device,
    dx: float,
    dy: float,
    sigma: torch.Tensor | None,
    predict_type: str,
) -> None:
    loss = cfg.get("loss", {}) or {}
    mode = str(loss.get("parc_input_scale_mode", "none")).strip().lower()
    if mode in ("", "none", "off", "false", "disabled"):
        return
    if mode not in ("robust", "scale_only_robust", "quantile"):
        raise ValueError(
            "loss.parc_input_scale_mode must be one of {none, robust}; "
            f"got {loss.get('parc_input_scale_mode')!r}."
        )
    if not _physics_inputs_active_from_loss_cfg(loss):
        print("[PARC-SCALE] skipped: physics_inputs_enabled=false.", flush=True)
        return

    Fdim = int(cfg.get("features", {}).get("num_features", 0) or 0)
    if Fdim <= 0:
        names = cfg.get("features", {}).get("names", [])
        Fdim = len(names) if isinstance(names, (list, tuple)) else 0
    if Fdim <= 0:
        raise RuntimeError("Could not infer feature count for PARC input scale calibration.")

    include_adv = bool(loss.get("parc_include_adv", True))
    include_diff = bool(loss.get("parc_include_diff", True))
    adv_w = float(loss.get("adv_weight", 1.0))
    diff_w = float(loss.get("diff_weight", 1.0))
    need_adv = include_adv and (adv_w != 0.0)
    need_diff = include_diff and (diff_w != 0.0)
    if not need_adv and not need_diff:
        print(
            "[PARC-SCALE] skipped: neither advection nor diffusion operators are active.",
            flush=True,
        )
        return

    recompute = _cfg_bool_strict(
        loss.get("parc_input_scale_recompute", False),
        key="loss.parc_input_scale_recompute",
    )
    if (not recompute) and (
        ("parc_input_auto_scale_adv" in loss) or ("parc_input_auto_scale_diff" in loss)
    ):
        print("[PARC-SCALE] using existing auto scale values from config.", flush=True)
        return

    sel_adv = dec.parc_select_feature_indices_adv(cfg, Fdim)
    sel_diff = dec.parc_select_feature_indices_diff(cfg, Fdim)
    la = len(sel_adv) if include_adv else 0
    ld = len(sel_diff) if include_diff else 0
    adv_samples: list[list[torch.Tensor]] = [[] for _ in range(la)]
    diff_samples: list[list[torch.Tensor]] = [[] for _ in range(ld)]

    q = float(loss.get("parc_input_scale_quantile", loss.get("parc_input_scale_q", 0.99)))
    q = min(max(q, 0.5), 0.9999)
    target = float(
        loss.get(
            "parc_input_scale_target",
            loss.get("parc_input_scale_target_value", 4.0),
        )
    )
    eps = float(loss.get("parc_input_scale_eps", 1e-12))
    s_min = float(loss.get("parc_input_scale_min", 0.0))
    s_max = float(loss.get("parc_input_scale_max", 0.0))
    max_batches = int(loss.get("parc_input_scale_max_batches", 0))
    max_samples = int(loss.get("parc_input_scale_max_samples_per_channel", 200000))

    backend = str(loss.get("physics_backend", "dec")).strip().lower()
    if backend != "dec":
        print(
            f"[PARC-SCALE] skipped: robust calibration currently supports DEC backend, got {backend!r}.",
            flush=True,
        )
        return

    sigma_f32 = _as_stat_tensor(sigma, device=device, dtype=torch.float32)
    if sigma_f32 is not None:
        sigma_f32 = sigma_f32.clamp_min(1e-12)

    n_batches = 0
    n_steps = 0
    print(
        "[PARC-SCALE] calibrating scale-only operator inputs "
        f"(q={q:.4g}, target={target:.4g}, max_batches={max_batches or 'all'})",
        flush=True,
    )
    for batch in loader:
        n_batches += 1
        if max_batches > 0 and n_batches > max_batches:
            break

        dt_list = batch.get("dt_list", None)
        if dt_list is None:
            raise RuntimeError("PARC scale calibration requires batch['dt_list'].")
        pred_levels_list = _require_list(batch, "pred_levels_list")
        pred_ei_list = _require_list(batch, "pred_ei_list")
        pred_ea_list = batch.get("pred_ea_list", None) or batch.get("pred_edge_attr_list", None)
        if pred_ea_list is None:
            raise RuntimeError("PARC scale calibration requires pred_edge_attr_list.")
        feat_t_on_pred_list = _require_list(batch, "feat_t_on_pred_list")
        K = len(feat_t_on_pred_list)
        dt_ref_scalar = batch.get("dt_ref", None)
        dt_ref_t = (
            torch.tensor(float(dt_ref_scalar), device=device, dtype=torch.float32)
            if dt_ref_scalar is not None
            else None
        )

        for j in range(1, K):
            dt_phys = torch.tensor(float(dt_list[j - 1]), device=device, dtype=torch.float32)
            x_abs = feat_t_on_pred_list[j].to(device=device, dtype=torch.float32)
            x_for_ops = sanitize_state_for_ops(x_abs, cfg, rho_floor=1e-6, E_floor=1e-6)
            pred_levels = pred_levels_list[j].to(device=device, dtype=torch.long)
            pei = pred_ei_list[j].to(device=device, dtype=torch.long)
            pea = pred_ea_list[j].to(device=device, dtype=torch.float32)

            r_adv_abs, r_diff_abs, _area = dec.dec_advdiff_terms_abs(
                x_abs=x_for_ops.float(),
                edge_index=pei,
                pred_ea=pea,
                levels=pred_levels,
                dx0=float(dx),
                dy0=float(dy),
                cfg=cfg,
                compute_adv=need_adv,
                compute_diff=need_diff,
            )
            r_adv_abs = _sanitize_ops_term(r_adv_abs, cfg)
            r_diff_abs = _sanitize_ops_term(r_diff_abs, cfg)
            ref = r_adv_abs if r_adv_abs is not None else r_diff_abs
            if ref is None:
                continue
            r_adv_in = r_adv_abs if r_adv_abs is not None else torch.zeros_like(ref)
            r_diff_in = r_diff_abs if r_diff_abs is not None else torch.zeros_like(ref)

            parc_raw = dec.parc_terms_to_node_inputs(
                r_adv_in,
                r_diff_in,
                dt_phys=dt_phys,
                dt_ref=dt_ref_t,
                sigma=sigma_f32,
                predict_type=predict_type,
                cfg=cfg,
                dtype=torch.float32,
                detach=True,
                apply_input_scales=False,
            )
            off = 0
            if la > 0:
                adv_block = parc_raw[:, off : off + la]
                _append_abs_samples(
                    adv_samples,
                    adv_block,
                    max_samples_per_channel=max_samples,
                )
                off += la
            if ld > 0:
                diff_block = parc_raw[:, off : off + ld]
                _append_abs_samples(
                    diff_samples,
                    diff_block,
                    max_samples_per_channel=max_samples,
                )
            n_steps += 1

    def _finalize(samples: list[list[torch.Tensor]], label: str) -> list[float]:
        scales = []
        qvals = []
        for parts in samples:
            if len(parts) == 0:
                qval = 0.0
                scale = 1.0
            else:
                vals = torch.cat(parts, dim=0)
                qval = float(torch.quantile(vals, q).item())
                scale = float(target / max(qval, eps))
                if s_min > 0.0:
                    scale = max(scale, s_min)
                if s_max > 0.0:
                    scale = min(scale, s_max)
            qvals.append(qval)
            scales.append(scale)
        if len(scales) > 0:
            print(
                f"[PARC-SCALE] {label} q_abs={qvals} scale={scales}",
                flush=True,
            )
        return scales

    if la > 0:
        loss["parc_input_auto_scale_adv"] = _finalize(adv_samples, "adv")
    if ld > 0:
        loss["parc_input_auto_scale_diff"] = _finalize(diff_samples, "diff")
    loss["parc_input_scale_mode"] = mode
    loss["parc_input_scale_quantile"] = q
    loss["parc_input_scale_target"] = target
    cfg["loss"] = loss
    print(
        f"[PARC-SCALE] calibrated from batches={min(n_batches, max_batches or n_batches)} "
        f"steps={n_steps}.",
        flush=True,
    )


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
    rep = dec.state_representation_from_cfg(cfg, Fdim)
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

    cols[idx["rho"]] = _sanitize_float_tensor(rho, clip_abs=rho_max, fill_value=rho_floor, nonneg=True)
    cols[idx["E"]] = _sanitize_float_tensor(E, clip_abs=E_max, fill_value=E_floor, nonneg=True)

    if rep == "primitive_uvrhope":
        u_idx = int(idx.get("u", idx["mx"]))
        v_idx = int(idx.get("v", idx["my"]))
        u = cols[u_idx]
        v = cols[v_idx]
        if u_clip > 0.0:
            u = u.clamp(min=-u_clip, max=u_clip)
            v = v.clamp(min=-u_clip, max=u_clip)
        cols[u_idx] = _sanitize_float_tensor(u, clip_abs=max(u_clip, m_max), fill_value=0.0, nonneg=False)
        cols[v_idx] = _sanitize_float_tensor(v, clip_abs=max(u_clip, m_max), fill_value=0.0, nonneg=False)

        p_idx = idx.get("p", None)
        if p_idx is not None:
            p_floor = float(loss.get("p_floor", 0.0))
            p = cols[int(p_idx)].clamp_min(p_floor)
            cols[int(p_idx)] = _sanitize_float_tensor(p, clip_abs=0.0, fill_value=p_floor, nonneg=True)
    else:
        mx = cols[idx["mx"]]
        my = cols[idx["my"]]
        if u_clip > 0.0:
            mx = mx.clamp(min=-u_clip * rho, max=u_clip * rho)
            my = my.clamp(min=-u_clip * rho, max=u_clip * rho)
        if m_max > 0.0:
            mx = mx.clamp(min=-m_max, max=m_max)
            my = my.clamp(min=-m_max, max=m_max)
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
    chunk_sidecar: ChunkSidecarH5 | None = None,
    ramp_feature_ctx: Dict[str, Any] | None = None,
):
    """
    STRICT multi-step mesh-first training with:
      - Variant-B physics baseline + learned correction
      - Optional PARC operator inputs (advection/diffusion terms)
      - Optional physics residual loss (delta-form), as before

    Supports model.predict_type in {state, delta, rate}; RK4 integration requires rate.
    """

    model.train()
    total_loss_accum = 0.0
    mae_accum = 0.0
    n_steps = 0
    diag_cfg = _resolve_diagnostics_cfg(cfg)
    diagnostics_accum = _new_diagnostics_accumulator(diag_cfg)

    dbg = cfg.get("debug", {})
    nan_watch = False  # legacy AMR NaN debug logging removed
    assert_finite_checks = bool(dbg.get("assert_finite_checks", False))
    ops_danger_watch = bool(dbg.get("ops_danger_watch", False))
    print_batch_time = bool(dbg.get("print_batch_time", False))
    print_runtime_mesh_batch_breakdown = bool(dbg.get("print_runtime_mesh_batch_breakdown", True))
    sync_runtime_timing = bool(dbg.get("sync_runtime_timing", False))
    runtime_mesh_enabled = _runtime_mesh_enabled_from_cfg(cfg)
    progress_every_batches_raw = int(dbg.get("progress_every_batches", 10))
    # 0 (or negative) disables periodic [PROGRESS] logging.
    progress_print_enabled = bool(progress_every_batches_raw > 0)
    progress_every_batches = max(1, int(progress_every_batches_raw))

    if sync_runtime_timing and runtime_mesh_enabled and (device.type in ("cuda", "mps")):
        print(
            f"[RUNTIME-TIMING] sync_runtime_timing=1 on device={device.type}. "
            "Per-step timings include synchronization overhead.",
            flush=True,
        )

    def _rt_t0() -> float:
        if sync_runtime_timing:
            _sync_device(device)
        return time.perf_counter()

    def _rt_dt(t0: float) -> float:
        if sync_runtime_timing:
            _sync_device(device)
        return float(time.perf_counter() - t0)

    if not hasattr(train_one_epoch_multi_step, "_nan_printed"):
        train_one_epoch_multi_step._nan_printed = False

    def _should_print():
        return False

    def _mark_printed():
        return

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
        state = dec.state_views(x, cfg)
        rho = state["rho"]
        mx = state["mx"]
        my = state["my"]
        E = state["E_tot"]
        ux = state["u"]
        uy = state["v"]
        rho_abs = rho.abs()

        print(f"\n[NAN-DBG][{tag}] x stats")
        _tstats("x", x)
        _tstats("rho", rho)
        print("  rho<=0 count:", int((rho <= 0).sum().item()))
        print("  |rho|<1e-6 count:", int((rho_abs < 1e-6).sum().item()))
        _tstats("mx", mx)
        _tstats("my", my)
        _tstats("E", E)

        _tstats("ux", ux)
        _tstats("uy", uy)

    @torch.no_grad()
    def _enforce_physical_state_with_diag(
        x_abs: torch.Tensor,
        *,
        tag: str,
        step_k: int,
        rho_floor: float = 1e-6,
        E_floor: float = 1e-6,
    ) -> torch.Tensor:
        """
        Prints how often rho/E are below floor BEFORE clamp, then applies your existing clamp helper.
        """
        # pre-clamp stats on the tensor you're about to clamp
        idx_here = dec.infer_feature_indices(cfg, int(x_abs.size(1)))
        rho_pre = x_abs[:, int(idx_here["rho"])]
        E_pre = x_abs[:, int(idx_here["E"])]

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
    forward_force_fp32 = bool(speed.get("forward_force_fp32", True))

    huber_delta = float(cfg["loss"].get("huber_delta", 0.05))
    lap_w  = float(cfg["loss"].get("laplacian_weight", 0.0))
    tmp_w  = float(cfg["loss"].get("temporal_weight", 0.0))
    use_huber = bool(cfg["loss"].get("use_huber", True))
    grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0))

    predict_type = _normalize_predict_type_key(
        cfg.get("model", {}).get("predict_type", cfg.get("model", {}).get("target_mode", "rate"))
    )
    cfg.setdefault("model", {})["predict_type"] = predict_type
    time_integrator = _resolve_time_integrator(cfg)
    _time_integrator_eff, rk4_alpha = _resolve_time_integrator_for_epoch(
        cfg, epoch_idx=epoch_idx, base_integrator=time_integrator
    )
    if predict_type != "rate" and rk4_alpha > 0.0:
        raise RuntimeError(
            "RK4/ramped RK4 integration requires model.predict_type='rate'. "
            f"Got predict_type='{predict_type}'."
        )

    # Physics-input controls
    loss_cfg = cfg.get("loss", {}) or {}
    parc_use = _physics_inputs_active_from_loss_cfg(loss_cfg)
    use_adapter = bool(loss_cfg.get("parc_use_adapter", False))

    # Variant-B baseline strength (reuse your existing name)
    dec_blend_w = float(loss_cfg.get("blend_weight", 0.0))        # baseline weight (recommend 1.0 when parc_use)
    dec_resid_w = float(loss_cfg.get("residual_weight", 0.0))     # optional residual loss weight

    # Determine whether we must compute physics operators this step.
    # Physics is now controlled by loss.physics_inputs_enabled.
    need_phy = bool(parc_use)
    model_needs_edge_attr = _model_requires_edge_attr_from_cfg(cfg)

    parc_sel_cache: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {}
    parc_mask_cache: Dict[Tuple[int, str, int], torch.Tensor] = {}

    def _get_parc_selections(Fdim: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        key = int(Fdim)
        cached = parc_sel_cache.get(key, None)
        if cached is not None:
            return cached
        sel_adv = tuple(int(i) for i in dec.parc_select_feature_indices_adv(cfg, key))
        sel_diff = tuple(int(i) for i in dec.parc_select_feature_indices_diff(cfg, key))
        sel_union = tuple(sorted(set(sel_adv + sel_diff)))
        out = (sel_adv, sel_diff, sel_union)
        parc_sel_cache[key] = out
        return out

    def _get_parc_channel_mask(Fdim: int, dev: torch.device) -> torch.Tensor:
        dev_index = int(dev.index) if dev.index is not None else -1
        key = (int(Fdim), str(dev.type), dev_index)
        cached = parc_mask_cache.get(key, None)
        if cached is not None:
            return cached
        _, _, sel_union = _get_parc_selections(int(Fdim))
        mask = torch.zeros((int(Fdim),), device=dev, dtype=torch.float32)
        if len(sel_union) > 0:
            mask[torch.as_tensor(sel_union, device=dev, dtype=torch.long)] = 1.0
        parc_mask_cache[key] = mask
        return mask

    backend = str(loss_cfg.get("physics_backend", "dec")).lower()
    use_mls = (backend in ("mls", "moving_least_squares", "moving-least-squares"))

    if scaler is None and use_amp and device.type == "cuda":
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    amp_ctx = ((lambda: torch.amp.autocast(device_type="cuda", enabled=True))
               if use_amp and device.type == "cuda"
               else (lambda: torch.autocast("cpu", enabled=False)))

    runtime_mesh_enabled = _runtime_mesh_enabled_from_cfg(cfg)
    if runtime_mesh_enabled:
        raise RuntimeError(
            "Runtime mesh is disabled in Physics-Aware-cartesian. "
            "Set train.runtime_mesh.enabled=false."
        )
    runtime_multires_cfg = _runtime_multires_lookup_cfg(cfg)
    runtime_multires_enabled = False
    runtime_multires_fallback_to_idw = bool(runtime_multires_cfg.get("fallback_to_idw", True))
    runtime_multires_lookup: Optional[_RuntimeMultiResGtLookup] = None
    runtime_update_every = 1
    runtime_domain_mode = "disabled"
    runtime_idw_backend = "exact"
    runtime_idw_backend_kwargs: Dict[str, Any] = {}
    runtime_bbox = _get_bbox(cfg)
    runtime_refine_ratio = _get_refine_ratio(cfg)
    runtime_need_edge_attr = bool(model_needs_edge_attr or (need_phy and (not use_mls)))
    runtime_mesh_plot_settings = _runtime_mesh_plot_settings(cfg)
    runtime_mesh_plot_state = {"rebuilds": 0, "saved": 0}
    runtime_mesh_plot_wedge_path = None
    runtime_wedge_path = None
    runtime_wedge_constraints = None
    runtime_base_mesh = None

    chunk_cfg = cfg.get("chunk", {}) or {}
    chunk_enabled = bool(chunk_cfg.get("enabled", False))
    chunk_num_chunks = max(1, int(chunk_cfg.get("num_chunks", 16)))
    chunk_halo_hops = max(0, int(chunk_cfg.get("halo_hops", 2)))
    chunk_partition_mode = str(chunk_cfg.get("partition_mode", "parent_grid")).strip().lower()
    if chunk_enabled and chunk_partition_mode != "parent_grid":
        raise ValueError(
            f"Unsupported chunk.partition_mode='{chunk_partition_mode}'. "
            "Only 'parent_grid' is currently implemented in train loop chunking."
        )
    if chunk_enabled and (chunk_num_chunks > int(H * W)):
        raise ValueError(
            f"chunk.num_chunks={chunk_num_chunks} exceeds H*W={int(H*W)} coarse parent cells."
        )

    chunk_sidecar_ok = bool(chunk_enabled and (chunk_sidecar is not None))
    chunk_warned_missing_sidecar = False
    chunk_warned_sidecar_step_miss = False
    chunk_warned_edge_attr_rebuild = False
    chunk_warned_coverage_gap = False
    chunk_debug_cfg = cfg.get("debug", {}) or {}
    chunk_stats_enabled = bool(chunk_debug_cfg.get("print_chunk_stats", False))
    chunk_stats_every_steps_raw = int(chunk_debug_cfg.get("chunk_stats_every_steps", 1))
    chunk_stats_every_steps = max(1, int(chunk_stats_every_steps_raw))
    chunk_stats_include_rk4 = bool(chunk_debug_cfg.get("chunk_stats_include_rk4", False))

    def _factor_chunk_grid(num_chunks: int, H: int, W: int) -> tuple[int, int]:
        target_aspect = float(W) / max(float(H), 1.0)
        best_rows, best_cols = 1, int(num_chunks)
        best_score = float("inf")
        for rows in range(1, int(math.sqrt(num_chunks)) + 1):
            if num_chunks % rows != 0:
                continue
            cols = num_chunks // rows
            for r, c in ((rows, cols), (cols, rows)):
                aspect = float(c) / max(float(r), 1.0)
                score = abs(math.log(max(aspect, 1e-12)) - math.log(max(target_aspect, 1e-12)))
                if score < best_score:
                    best_score = score
                    best_rows, best_cols = int(r), int(c)
        return best_rows, best_cols

    def _chunk_tile_bounds(num_chunks: int, H: int, W: int) -> list[tuple[int, int, int, int, int]]:
        rows, cols = _factor_chunk_grid(num_chunks, H, W)
        y_edges = np.linspace(0, H, rows + 1, dtype=np.int64)
        x_edges = np.linspace(0, W, cols + 1, dtype=np.int64)
        out: list[tuple[int, int, int, int, int]] = []
        cid = 0
        for ry in range(rows):
            y0, y1 = int(y_edges[ry]), int(y_edges[ry + 1])
            for cx in range(cols):
                x0, x1 = int(x_edges[cx]), int(x_edges[cx + 1])
                out.append((cid, y0, y1, x0, x1))
                cid += 1
        return out

    chunk_tile_spec = _chunk_tile_bounds(chunk_num_chunks, H, W)

    def _expand_halo_mask_np(core_mask: np.ndarray, edge_index_np: np.ndarray, hops: int) -> np.ndarray:
        if hops <= 0 or core_mask.size == 0:
            return core_mask.copy()
        if edge_index_np.size == 0:
            return core_mask.copy()

        src = edge_index_np[0]
        dst = edge_index_np[1]
        full_mask = core_mask.copy()
        frontier = core_mask.copy()
        for _ in range(int(hops)):
            nbr = np.zeros_like(full_mask, dtype=bool)
            if frontier.any():
                nbr[dst[frontier[src]]] = True
                nbr[src[frontier[dst]]] = True
            nbr &= ~full_mask
            if not nbr.any():
                break
            full_mask |= nbr
            frontier = nbr
        return full_mask

    def _build_local_edges_np(
        edge_index_np: np.ndarray,
        full_mask: np.ndarray,
        full_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if edge_index_np.size == 0 or full_idx.size == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)

        src = edge_index_np[0]
        dst = edge_index_np[1]
        keep = full_mask[src] & full_mask[dst]
        if not np.any(keep):
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)

        lut = np.full((full_mask.size,), fill_value=-1, dtype=np.int64)
        lut[full_idx] = np.arange(full_idx.size, dtype=np.int64)
        src_l = lut[src[keep]]
        dst_l = lut[dst[keep]]
        edge_local = np.stack([src_l, dst_l], axis=0).astype(np.int64, copy=False)
        edge_ids = np.nonzero(keep)[0].astype(np.int64, copy=False)
        return edge_local, edge_ids

    def _edge_id_lut_from_global_ei(pei_np: np.ndarray) -> dict[tuple[int, int], int]:
        lut: dict[tuple[int, int], int] = {}
        if pei_np.size == 0:
            return lut
        src = pei_np[0]
        dst = pei_np[1]
        for eid in range(int(pei_np.shape[1])):
            lut[(int(src[eid]), int(dst[eid]))] = int(eid)
        return lut

    def _reindex_core_first(
        full_idx: np.ndarray,
        core_mask_local_u8: np.ndarray,
        edge_index_local: np.ndarray,
        edge_ids_local: np.ndarray | None,
    ) -> tuple[np.ndarray, int, np.ndarray, np.ndarray | None]:
        core_mask = np.asarray(core_mask_local_u8, dtype=np.uint8).reshape(-1) > 0
        if core_mask.size != full_idx.size:
            raise RuntimeError(
                f"Invalid core_mask_local_u8 size ({core_mask.size}) for full_idx size ({full_idx.size})."
            )
        core_local = np.nonzero(core_mask)[0].astype(np.int64, copy=False)
        core_count = int(core_local.size)
        if core_count == 0 or full_idx.size == 0:
            return full_idx, core_count, edge_index_local, edge_ids_local

        if np.array_equal(core_local, np.arange(core_count, dtype=np.int64)):
            return full_idx, core_count, edge_index_local, edge_ids_local

        noncore_local = np.nonzero(~core_mask)[0].astype(np.int64, copy=False)
        order = np.concatenate([core_local, noncore_local], axis=0)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size, dtype=np.int64)
        full_idx_new = full_idx[order]
        if edge_index_local.size > 0:
            edge_index_local = inv[edge_index_local]
        return full_idx_new, core_count, edge_index_local, edge_ids_local

    def _convert_chunk_specs_np(
        specs_np: list[dict[str, Any]],
        pei_np_for_rebuild: np.ndarray,
    ) -> list[dict[str, Any]]:
        nonlocal chunk_warned_edge_attr_rebuild
        out: list[dict[str, Any]] = []
        edge_lut = None

        for spec in specs_np:
            full_idx = np.asarray(spec.get("full_idx", []), dtype=np.int64).reshape(-1)
            if full_idx.size == 0:
                continue

            ei_local = np.asarray(spec.get("edge_index_local", np.zeros((2, 0))), dtype=np.int64)
            if ei_local.ndim != 2:
                ei_local = np.zeros((2, 0), dtype=np.int64)
            if ei_local.shape[0] != 2 and ei_local.shape[1] == 2:
                ei_local = ei_local.T
            if ei_local.shape[0] != 2:
                ei_local = np.zeros((2, 0), dtype=np.int64)

            edge_ids_local = spec.get("edge_ids_local", None)
            if edge_ids_local is None:
                edge_ids_np = None
            else:
                edge_ids_np = np.asarray(edge_ids_local, dtype=np.int64).reshape(-1)

            core_mask_u8 = spec.get("core_mask_local_u8", None)
            if core_mask_u8 is None:
                core_count = int(np.asarray(spec.get("core_idx", []), dtype=np.int64).reshape(-1).size)
                if core_count > full_idx.size:
                    core_count = int(full_idx.size)
                core_mask_u8 = np.zeros((full_idx.size,), dtype=np.uint8)
                core_mask_u8[:core_count] = 1

            full_idx, core_count, ei_local, edge_ids_np = _reindex_core_first(
                full_idx=full_idx,
                core_mask_local_u8=np.asarray(core_mask_u8, dtype=np.uint8).reshape(-1),
                edge_index_local=ei_local,
                edge_ids_local=edge_ids_np,
            )
            if core_count <= 0:
                continue

            # Backward compatibility: older sidecars may lack edge_ids_local.
            if (edge_ids_np is None) and (ei_local.size > 0):
                if edge_lut is None:
                    edge_lut = _edge_id_lut_from_global_ei(pei_np_for_rebuild)
                src_g = full_idx[ei_local[0]]
                dst_g = full_idx[ei_local[1]]
                ids = np.empty((src_g.size,), dtype=np.int64)
                missing = False
                for i in range(int(src_g.size)):
                    key = (int(src_g[i]), int(dst_g[i]))
                    eid = edge_lut.get(key, -1)
                    ids[i] = int(eid)
                    if eid < 0:
                        missing = True
                if missing:
                    raise RuntimeError(
                        "Could not reconstruct local edge_ids from sidecar for edge_attr slicing. "
                        "Please rebuild sidecar with current builder."
                    )
                edge_ids_np = ids
                if not chunk_warned_edge_attr_rebuild:
                    print(
                        "[CHUNK][WARN] sidecar is missing edge_ids_local; reconstructed from global edge_index. "
                        "Rebuild sidecar once to avoid this overhead.",
                        flush=True,
                    )
                    chunk_warned_edge_attr_rebuild = True

            out.append(
                {
                    "full_idx": torch.from_numpy(full_idx.astype(np.int64, copy=False)),
                    "edge_index_local": torch.from_numpy(ei_local.astype(np.int64, copy=False)),
                    "edge_ids_local": (
                        None
                        if edge_ids_np is None
                        else torch.from_numpy(edge_ids_np.astype(np.int64, copy=False))
                    ),
                    "core_count": int(core_count),
                    "is_active": bool(spec.get("is_active", True)),
                    "activity_score": float(spec.get("activity_score", float("nan"))),
                }
            )
        return out

    def _build_chunk_specs_on_the_fly(pred_parents: torch.Tensor, pred_ei: torch.Tensor) -> list[dict[str, Any]]:
        parents_np = pred_parents.detach().to("cpu").long().view(-1).numpy().astype(np.int64, copy=False)
        pei_np = pred_ei.detach().to("cpu").long().numpy().astype(np.int64, copy=False)
        if pei_np.ndim != 2:
            raise RuntimeError(f"chunking expects pred_ei ndim=2, got {pei_np.ndim}")
        if pei_np.shape[0] != 2 and pei_np.shape[1] == 2:
            pei_np = pei_np.T
        if pei_np.shape[0] != 2:
            raise RuntimeError(f"chunking expects pred_ei shape (2,E), got {tuple(pei_np.shape)}")

        out_np: list[dict[str, Any]] = []
        for _cid, y0, y1, x0, x1 in chunk_tile_spec:
            row = parents_np // int(W)
            col = parents_np % int(W)
            core_mask = (row >= int(y0)) & (row < int(y1)) & (col >= int(x0)) & (col < int(x1))
            full_mask = _expand_halo_mask_np(core_mask=core_mask, edge_index_np=pei_np, hops=chunk_halo_hops)
            halo_mask = full_mask & (~core_mask)

            core_idx = np.nonzero(core_mask)[0].astype(np.int64, copy=False)
            if core_idx.size == 0:
                continue
            halo_idx = np.nonzero(halo_mask)[0].astype(np.int64, copy=False)
            full_idx = np.concatenate([core_idx, halo_idx], axis=0).astype(np.int64, copy=False)
            ei_local, edge_ids = _build_local_edges_np(
                edge_index_np=pei_np,
                full_mask=full_mask,
                full_idx=full_idx,
            )
            core_mask_local = np.zeros((full_idx.size,), dtype=np.uint8)
            core_mask_local[: core_idx.size] = 1
            out_np.append(
                {
                    "full_idx": full_idx,
                    "edge_index_local": ei_local,
                    "edge_ids_local": edge_ids,
                    "core_mask_local_u8": core_mask_local,
                    "is_active": True,
                    "activity_score": float("nan"),
                }
            )
        return _convert_chunk_specs_np(out_np, pei_np_for_rebuild=pei_np)

    def _get_chunk_specs_for_step(
        *,
        pred_parents: torch.Tensor,
        pred_ei: torch.Tensor,
        step_t_abs: int | None,
    ) -> list[dict[str, Any]] | None:
        nonlocal chunk_warned_missing_sidecar, chunk_warned_sidecar_step_miss
        if not chunk_enabled:
            return None

        if chunk_sidecar_ok and (step_t_abs is not None):
            chunks_np = chunk_sidecar.get_timestep_chunks(int(step_t_abs))
            if chunks_np is not None:
                pei_np = pred_ei.detach().to("cpu").long().numpy().astype(np.int64, copy=False)
                return _convert_chunk_specs_np(chunks_np, pei_np_for_rebuild=pei_np)
            if not chunk_warned_sidecar_step_miss:
                print(
                    f"[CHUNK][WARN] sidecar has no timestep t={int(step_t_abs)}; falling back to on-the-fly chunking.",
                    flush=True,
                )
                chunk_warned_sidecar_step_miss = True
        elif chunk_enabled and (chunk_sidecar is None) and (not chunk_warned_missing_sidecar):
            print(
                "[CHUNK][INFO] chunk.enabled=true and no sidecar loaded; using on-the-fly chunking.",
                flush=True,
            )
            chunk_warned_missing_sidecar = True

        return _build_chunk_specs_on_the_fly(pred_parents=pred_parents, pred_ei=pred_ei)

    def _forward_main_head_chunked(
        *,
        x_in_all: torch.Tensor,
        edge_index_global: torch.Tensor,
        edge_attr_global: torch.Tensor | None,
        chunk_specs: list[dict[str, Any]] | None,
        step_k: int,
        step_t_abs: int | None,
        stage_tag: str,
    ) -> torch.Tensor:
        nonlocal chunk_warned_coverage_gap
        if (not chunk_enabled) or (not chunk_specs):
            return _forward_main_head_with_edge_attr(
                model,
                x_in_all,
                edge_index_global,
                edge_attr=edge_attr_global,
                force_fp32=forward_force_fp32,
            )

        y_out = None
        covered = torch.zeros((x_in_all.size(0),), device=device, dtype=torch.bool)

        for spec in chunk_specs:
            full_idx = spec["full_idx"]
            core_count = int(spec["core_count"])
            ei_local = spec["edge_index_local"]
            edge_ids_local = spec.get("edge_ids_local", None)

            if core_count <= 0 or full_idx.numel() == 0:
                continue

            full_idx_d = full_idx.to(device=device, dtype=torch.long, non_blocking=True)
            ei_local_d = ei_local.to(device=device, dtype=torch.long, non_blocking=True)
            x_local = x_in_all.index_select(0, full_idx_d)

            ea_local = None
            if edge_attr_global is not None:
                if edge_ids_local is None:
                    raise RuntimeError(
                        "Chunked forward with edge_attr requires edge_ids_local per chunk. "
                        "Rebuild sidecar with current builder."
                    )
                edge_ids_d = edge_ids_local.to(device=device, dtype=torch.long, non_blocking=True)
                if edge_ids_d.numel() != int(ei_local_d.shape[1]):
                    raise RuntimeError(
                        f"edge_ids_local length mismatch in {stage_tag} step_k={step_k}: "
                        f"len(edge_ids_local)={int(edge_ids_d.numel())}, E_local={int(ei_local_d.shape[1])}"
                    )
                ea_local = edge_attr_global.index_select(0, edge_ids_d)

            y_local = _forward_main_head_with_edge_attr(
                model,
                x_local,
                ei_local_d,
                edge_attr=ea_local,
                force_fp32=forward_force_fp32,
            )
            core_global = full_idx_d[:core_count]
            core_local = y_local[:core_count]

            if y_out is None:
                y_out = core_local.new_zeros((x_in_all.size(0), y_local.size(1)))
            # MPS backend does not implement aten::index_copy; direct indexed assignment
            # keeps this path device-native.
            y_out[core_global] = core_local
            covered[core_global] = True

        if y_out is None:
            return _forward_main_head_with_edge_attr(
                model,
                x_in_all,
                edge_index_global,
                edge_attr=edge_attr_global,
                force_fp32=forward_force_fp32,
            )

        if not bool(torch.all(covered).item()):
            y_full = _forward_main_head_with_edge_attr(
                model,
                x_in_all,
                edge_index_global,
                edge_attr=edge_attr_global,
                force_fp32=forward_force_fp32,
            )
            y_out = torch.where(covered.view(-1, 1), y_out, y_full)
            if not chunk_warned_coverage_gap:
                n_missing = int((~covered).sum().item())
                print(
                    f"[CHUNK][WARN] {stage_tag} step_k={step_k}: {n_missing} nodes were not covered by chunk cores; "
                    "filled from full-graph forward.",
                    flush=True,
                )
                chunk_warned_coverage_gap = True

        if chunk_stats_enabled:
            should_print_stage = (stage_tag == "train-main") or bool(chunk_stats_include_rk4)
            if should_print_stage and ((int(step_k) % chunk_stats_every_steps) == 0):
                n_chunks = int(len(chunk_specs))
                n_nodes = int(x_in_all.size(0))
                n_cov = int(covered.sum().item())
                core_counts = [int(s["core_count"]) for s in chunk_specs if int(s["core_count"]) > 0]
                full_counts = [int(s["full_idx"].numel()) for s in chunk_specs if int(s["core_count"]) > 0]
                halo_counts = [max(0, f - c) for c, f in zip(core_counts, full_counts)]
                edge_counts = [int(s["edge_index_local"].shape[1]) for s in chunk_specs if int(s["core_count"]) > 0]
                n_active = int(sum(1 for s in chunk_specs if bool(s.get("is_active", True))))

                core_mean = float(np.mean(core_counts)) if core_counts else 0.0
                halo_mean = float(np.mean(halo_counts)) if halo_counts else 0.0
                full_mean = float(np.mean(full_counts)) if full_counts else 0.0
                edge_mean = float(np.mean(edge_counts)) if edge_counts else 0.0
                cov_frac = (float(n_cov) / float(max(n_nodes, 1))) if n_nodes > 0 else 0.0

                t_abs_str = "n/a" if step_t_abs is None else str(int(step_t_abs))
                print(
                    f"[CHUNK-DBG] stage={stage_tag} step_k={int(step_k)} t_abs={t_abs_str} "
                    f"chunks={n_chunks} active={n_active} covered={n_cov}/{n_nodes} ({cov_frac:.3f}) "
                    f"mean_core={core_mean:.1f} mean_halo={halo_mean:.1f} mean_full={full_mean:.1f} "
                    f"mean_local_edges={edge_mean:.1f}",
                    flush=True,
                )

        return y_out

    train_one_epoch_multi_step._printed_loss_components = False

    epoch_loop_t0 = time.perf_counter()
    iter_wait_anchor_t = float(epoch_loop_t0)
    epoch_body_time_accum_s = 0.0
    try:
        total_batches = int(len(loader))
    except Exception:
        total_batches = -1

    def _maybe_print_batch_progress(
        batch_idx: int,
        batch_wall_s: float,
        *,
        data_wait_s: float,
        avg_body_s: float,
    ) -> None:
        bi = int(batch_idx) + 1
        if print_batch_time:
            if total_batches > 0:
                print(
                    f"[BATCH-TIME] train batch {bi}/{total_batches} wall={batch_wall_s:.3f}s "
                    f"data_wait={float(data_wait_s):.3f}s",
                    flush=True,
                )
            else:
                print(
                    f"[BATCH-TIME] train batch {bi} wall={batch_wall_s:.3f}s "
                    f"data_wait={float(data_wait_s):.3f}s",
                    flush=True,
                )
        if not progress_print_enabled:
            return
        if (bi % progress_every_batches) != 0:
            return
        # Legacy progress logging removed.
        return

    def _maybe_print_runtime_mesh_batch_breakdown(
        batch_idx: int,
        batch_wall_s: float,
        *,
        t_mesh_predict_s: float,
        t_mesh_materialize_s: float,
        t_wedge_clip_s: float,
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
            + float(t_wedge_clip_s)
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
            f"| wedge_clip={float(t_wedge_clip_s):.3f}s ({_pct(t_wedge_clip_s):.1f}%) "
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

    def _maybe_print_runtime_mesh_predict_detail(
        batch_idx: int,
        *,
        t_mesh_predict_s: float,
        n_rebuilds: int,
        predict_parts_s: Dict[str, float],
    ) -> None:
        if (not runtime_mesh_enabled) or (not print_runtime_mesh_batch_breakdown):
            return
        if not isinstance(predict_parts_s, dict):
            return
        parts_raw = {str(k): float(v) for k, v in predict_parts_s.items() if float(v) > 0.0}
        if len(parts_raw) == 0:
            return

        label_map = {
            "predict_parent_map_s": "parent_map",
            "predict_coarse_agg_s": "coarse_agg",
            "predict_fill_empty_s": "fill_empty",
            "predict_feature_norm_s": "feature_norm",
            "predict_gradient_channels_s": "gradient_channels",
            "predict_channel_assemble_s": "assemble_input",
            "predict_model_forward_s": "model_forward",
            "predict_logits_to_masks_s": "logits_to_masks",
            "predict_grad_score_s": "grad_score",
            "predict_prev_masks_s": "prev_masks",
            "predict_hierarchy_s": "hierarchy",
            "predict_grad_raster_s": "grad_raster",
            "predict_combine_levels_s": "combine_levels",
            "predict_pool_up_s": "pool_up",
            "predict_threshold_hysteresis_s": "threshold_hysteresis",
            "predict_dilation_s": "dilation",
            "predict_normalize_masks_s": "normalize_masks",
            "predict_debug_out_s": "debug_out",
            "predict_legacy_policy_s": "legacy_policy",
        }
        parts = sorted(parts_raw.items(), key=lambda kv: kv[1], reverse=True)
        total = float(max(t_mesh_predict_s, 1e-12))
        sum_parts = float(sum(v for _, v in parts))
        unattributed = max(0.0, total - sum_parts)

        segs = []
        for k, v in parts:
            nm = label_map.get(k, k)
            segs.append(f"{nm}={v:.3f}s ({100.0 * v / total:.1f}%)")
        if unattributed > 0.0:
            segs.append(f"unattributed={unattributed:.3f}s ({100.0 * unattributed / total:.1f}%)")
        segs.append(f"per_rebuild={total / float(max(1, int(n_rebuilds))):.3f}s")
        segs.append(f"rebuilds={int(n_rebuilds)}")

        bi = int(batch_idx) + 1
        if total_batches > 0:
            prefix = f"[RUNTIME-MESH-PREDICT-DETAIL] train batch {bi}/{total_batches}"
        else:
            prefix = f"[RUNTIME-MESH-PREDICT-DETAIL] train batch {bi}"
        print(f"{prefix} " + " | ".join(segs), flush=True)

    def _maybe_print_runtime_wedge_detail(
        batch_idx: int,
        *,
        t_wedge_clip_s: float,
        wedge_parts_s: Dict[str, float],
        n_rebuilds: int,
    ) -> None:
        if (not runtime_mesh_enabled) or (not print_runtime_mesh_batch_breakdown):
            return
        total = float(t_wedge_clip_s)
        if total <= 0.0:
            return
        if not isinstance(wedge_parts_s, dict):
            return
        parts_raw = {str(k): float(v) for k, v in wedge_parts_s.items() if float(v) > 0.0}
        if len(parts_raw) == 0:
            return
        label_map = {
            "wedge_lookup_classify_s": "classify",
            "wedge_lookup_refine_s": "refine",
            "wedge_edge_build_s": "edge_build",
            "wedge_legacy_geom_s": "legacy_geom",
        }
        parts = sorted(parts_raw.items(), key=lambda kv: kv[1], reverse=True)
        sum_parts = float(sum(v for _, v in parts))
        unattributed = max(0.0, total - sum_parts)

        bi = int(batch_idx) + 1
        if total_batches > 0:
            prefix = f"[RUNTIME-WEDGE-DETAIL] train batch {bi}/{total_batches}"
        else:
            prefix = f"[RUNTIME-WEDGE-DETAIL] train batch {bi}"
        segs = [f"wedge_clip={total:.3f}s"]
        for k, v in parts:
            nm = label_map.get(k, k)
            segs.append(f"{nm}={v:.3f}s ({100.0 * v / max(total, 1e-12):.1f}%)")
        if unattributed > 0.0:
            segs.append(f"unattributed={unattributed:.3f}s ({100.0 * unattributed / max(total, 1e-12):.1f}%)")
        segs.append(f"per_rebuild={total / float(max(1, int(n_rebuilds))):.3f}s")
        segs.append(f"rebuilds={int(n_rebuilds)}")
        print(f"{prefix} " + " | ".join(segs), flush=True)

    def _maybe_print_runtime_outside_detail(
        batch_idx: int,
        *,
        data_wait_s: float,
    ) -> None:
        if (not runtime_mesh_enabled) or (not print_runtime_mesh_batch_breakdown):
            return
        if float(data_wait_s) <= 0.0:
            return
        bi = int(batch_idx) + 1
        if total_batches > 0:
            prefix = f"[RUNTIME-OUTSIDE-DETAIL] train batch {bi}/{total_batches}"
        else:
            prefix = f"[RUNTIME-OUTSIDE-DETAIL] train batch {bi}"
        print(
            f"{prefix} dataloader_wait={float(data_wait_s):.3f}s (outside batch_wall)",
            flush=True,
        )

    def _maybe_print_runtime_other_detail(
        batch_idx: int,
        batch_wall_s: float,
        *,
        t_mesh_predict_s: float,
        t_mesh_materialize_s: float,
        t_wedge_clip_s: float,
        t_edge_attr_s: float,
        t_idw_remap_s: float,
        other_parts_s: Dict[str, float],
    ) -> None:
        if (not runtime_mesh_enabled) or (not print_runtime_mesh_batch_breakdown):
            return
        total = float(max(batch_wall_s, 1e-12))
        accounted = (
            float(t_mesh_predict_s)
            + float(t_mesh_materialize_s)
            + float(t_wedge_clip_s)
            + float(t_edge_attr_s)
            + float(t_idw_remap_s)
        )
        other = max(0.0, total - accounted)
        if other <= 0.0:
            return
        if not isinstance(other_parts_s, dict):
            return
        parts_raw = {str(k): float(v) for k, v in other_parts_s.items() if float(v) > 0.0}
        if len(parts_raw) == 0:
            return
        label_map = {
            "model_step_s": "model_step",
            "metrics_sync_s": "metrics_sync",
            "state_diag_s": "state_diag",
            "mem_snapshot_s": "mem_snapshot",
            "step_log_io_s": "step_log_io",
            "mesh_plot_s": "mesh_plot",
            "backward_s": "backward",
            "unscale_clip_s": "unscale_clip",
            "optimizer_s": "optimizer",
            "zero_grad_s": "zero_grad",
        }
        parts = sorted(parts_raw.items(), key=lambda kv: kv[1], reverse=True)
        sum_parts = float(sum(v for _, v in parts))
        unattributed = max(0.0, float(other - sum_parts))

        bi = int(batch_idx) + 1
        if total_batches > 0:
            prefix = f"[RUNTIME-OTHER-DETAIL] train batch {bi}/{total_batches}"
        else:
            prefix = f"[RUNTIME-OTHER-DETAIL] train batch {bi}"

        segs = [f"other={other:.3f}s"]
        denom = float(max(other, 1e-12))
        for k, v in parts:
            nm = label_map.get(k, k)
            segs.append(f"{nm}={v:.3f}s ({100.0 * v / denom:.1f}%)")
        segs.append(f"unattributed={unattributed:.3f}s ({100.0 * unattributed / denom:.1f}%)")
        print(f"{prefix} " + " | ".join(segs), flush=True)

    def _maybe_print_runtime_model_step_detail(
        batch_idx: int,
        *,
        t_model_step_s: float,
        model_calls: int,
        model_parts_s: Dict[str, float],
    ) -> None:
        if (not runtime_mesh_enabled) or (not print_runtime_mesh_batch_breakdown):
            return
        if float(t_model_step_s) <= 0.0:
            return
        if not isinstance(model_parts_s, dict):
            return
        parts_raw = {
            str(k): float(v)
            for k, v in model_parts_s.items()
            if (str(k) != "model_step_total_s") and (float(v) > 0.0)
        }
        if len(parts_raw) == 0:
            return

        label_map = {
            "prep_norm_dt_s": "prep_norm_dt",
            "chunk_specs_s": "chunk_specs",
            "physics_ops_s": "physics_ops",
            "input_assemble_s": "input_assemble",
            "gnn_forward_s": "gnn_forward",
            "baseline_blend_s": "baseline_blend",
            "rk4_s": "rk4",
            "target_loss_s": "target_loss",
        }
        parts = sorted(parts_raw.items(), key=lambda kv: kv[1], reverse=True)
        total = float(max(t_model_step_s, 1e-12))
        sum_parts = float(sum(v for _, v in parts))
        unattributed = max(0.0, total - sum_parts)

        bi = int(batch_idx) + 1
        if total_batches > 0:
            prefix = f"[RUNTIME-MODEL-STEP-DETAIL] train batch {bi}/{total_batches}"
        else:
            prefix = f"[RUNTIME-MODEL-STEP-DETAIL] train batch {bi}"

        segs = [f"model_step={total:.3f}s"]
        for k, v in parts:
            nm = label_map.get(k, k)
            segs.append(f"{nm}={v:.3f}s ({100.0 * v / total:.1f}%)")
        segs.append(f"unattributed={unattributed:.3f}s ({100.0 * unattributed / total:.1f}%)")
        segs.append(f"per_call={total / float(max(1, int(model_calls))):.3f}s")
        segs.append(f"calls={int(model_calls)}")
        print(f"{prefix} " + " | ".join(segs), flush=True)

    def _maybe_print_runtime_multires_batch_mismatch(
        batch_idx: int,
        *,
        lookup_calls: int,
        requested_total: int,
        missing_total: int,
        fallback_calls: int,
        xin_requested: int,
        xin_missing: int,
        xtgt_requested: int,
        xtgt_missing: int,
    ) -> None:
        if (not runtime_mesh_enabled) or (not runtime_multires_enabled):
            return
        bi = int(batch_idx) + 1
        miss_pct = 100.0 * float(max(0, missing_total)) / float(max(1, requested_total))
        if total_batches > 0:
            prefix = f"[RUNTIME-MULTIRES-MISMATCH] train batch {bi}/{total_batches}"
        else:
            prefix = f"[RUNTIME-MULTIRES-MISMATCH] train batch {bi}"
        print(
            f"{prefix} calls={int(lookup_calls)} "
            f"| missing={int(missing_total)}/{int(requested_total)} ({miss_pct:.2f}%) "
            f"| feat_t_on_pred={int(xin_missing)}/{int(xin_requested)} "
            f"| feat_tp1_on_pred={int(xtgt_missing)}/{int(xtgt_requested)} "
            f"| fallback_calls={int(fallback_calls)}",
            flush=True,
        )

    for batch_idx, batch in enumerate(loader):
        iter_enter_t = time.perf_counter()
        data_wait_s = max(0.0, float(iter_enter_t - iter_wait_anchor_t))
        batch_t0 = float(iter_enter_t)

        batch_rt_mesh_predict_s = 0.0
        batch_rt_mesh_materialize_s = 0.0
        batch_rt_wedge_clip_s = 0.0
        batch_rt_wedge_parts_s: Dict[str, float] = {}
        batch_rt_edge_attr_s = 0.0
        batch_rt_idw_remap_s = 0.0
        batch_rt_rebuilds = 0
        batch_rt_mesh_predict_parts_s: Dict[str, float] = {}
        batch_rt_other_model_s = 0.0
        batch_rt_other_metrics_sync_s = 0.0
        batch_rt_other_state_diag_s = 0.0
        batch_rt_other_mem_snapshot_s = 0.0
        batch_rt_other_step_log_io_s = 0.0
        batch_rt_other_mesh_plot_s = 0.0
        batch_rt_other_backward_s = 0.0
        batch_rt_other_unscale_clip_s = 0.0
        batch_rt_other_optimizer_s = 0.0
        batch_rt_other_zero_grad_s = 0.0
        batch_rt_model_calls = 0
        batch_rt_model_parts_s: Dict[str, float] = {}
        batch_rt_multires_lookup_calls = 0
        batch_rt_multires_requested_total = 0
        batch_rt_multires_missing_total = 0
        batch_rt_multires_fallback_calls = 0
        batch_rt_multires_xin_requested = 0
        batch_rt_multires_xin_missing = 0
        batch_rt_multires_xtgt_requested = 0
        batch_rt_multires_xtgt_missing = 0

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

        pred_centers_list = _require_list(batch, "pred_centers_list")
        pred_levels_list = _require_list(batch, "pred_levels_list")
        pred_parents_list = _require_list(batch, "pred_parents_list")
        pred_ei_list = _require_list(batch, "pred_ei_list")
        mask_pred_list = _require_list(batch, "mask_pred_list")
        feat_t_on_pred_list = _require_list(batch, "feat_t_on_pred_list")
        feat_tp1_on_pred_list = _require_list(batch, "feat_tp1_on_pred_list")

        if (model_needs_edge_attr or (need_phy and (not use_mls))) and pred_ea_list is None:
            raise RuntimeError(
                "Model/DEC physics requires edge attributes, but batch is missing "
                "pred_ea_list / pred_edge_attr_list. "
                "Update H5 loader + CollateWithPrecompute to attach edge_attr per step."
            )

        pred_lists = (pred_centers_list, pred_levels_list, pred_parents_list, pred_ei_list, mask_pred_list)

        window_loss = 0.0
        window_mae = 0.0
        window_loss_graph = None

        t_zero_grad_t0 = _rt_t0()
        opt.zero_grad(set_to_none=True)
        batch_rt_other_zero_grad_s += _rt_dt(t_zero_grad_t0)

        # Only DEC physics and edge-aware model families need edge_attr. MLS does not.
        if (model_needs_edge_attr or (need_phy and (not use_mls))) and (pred_ea_list is None):
            raise RuntimeError(
                "Model/DEC physics requires edge attributes, but batch is missing "
                "pred_ea_list / pred_edge_attr_list. "
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

        def _assert_finite_if(name: str, t: torch.Tensor, crash: bool = True):
            if not assert_finite_checks:
                return True
            return _assert_finite(name, t, crash=crash)


        # ==========================================================
        # Helper closure: one step computation (k -> k+1 on pred mesh)
        # ==========================================================
        def _run_step(
            *,
            step_k: int,
            step_t_abs: int | None,
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
            chunk_specs_step: list[dict[str, Any]] | None = None,
            timing_out: Dict[str, float] | None = None,
        ):
            step_timing: Dict[str, float] = {
                "prep_norm_dt_s": 0.0,
                "chunk_specs_s": 0.0,
                "physics_ops_s": 0.0,
                "input_assemble_s": 0.0,
                "gnn_forward_s": 0.0,
                "baseline_blend_s": 0.0,
                "rk4_s": 0.0,
                "target_loss_s": 0.0,
            }
            def _step_t0() -> float:
                if sync_runtime_timing:
                    _sync_device(device)
                return time.perf_counter()

            def _step_dt(t0: float) -> float:
                if sync_runtime_timing:
                    _sync_device(device)
                return float(time.perf_counter() - t0)

            t_prep_t0 = _step_t0()
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

                state_dbg = dec.state_views(x_in_abs, cfg)
                rho = state_dbg["rho"]
                ux = state_dbg["u"]
                uy = state_dbg["v"]

                _tstats("rho", rho)
                print(f"[NAN-debug] |rho|<1e-6 count: {(rho.abs() < 1e-6).sum().item()} / {rho.numel()}")
                _tstats("ux", ux)
                _tstats("uy", uy)

                _mark_printed()

            _assert_finite_if("norm_in", norm_in)
            _assert_finite_if("norm_tgt", norm_tgt)

            dt_ref_t = (torch.tensor(float(dt_ref_scalar), device=device, dtype=norm_in.dtype)
                        if dt_ref_scalar is not None else None)
            dt_hat = (dt_phys / dt_ref_t) if dt_ref_t is not None else dt_phys

            if _should_print() and nan_watch:
                print(f"[NAN-DBG] step={step_k} ---- dt checks ----")
                _tstats("dt_phys", dt_phys)
                _tstats("dt_ref_t", dt_ref_t if dt_ref_t is not None else None)
                _tstats("dt_hat", dt_hat)
                _mark_printed()

            _assert_finite_if("dt_hat", dt_hat)

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

            _assert_finite_if("dt_hat", dt_hat)

            if torch.is_tensor(dt_hat):
                if (dt_hat <= 0).any():
                    print("[NAN-DBG][WARN] dt_hat has non-positive values")

            sigma_f32 = _as_stat_tensor(sigma, device=device, dtype=torch.float32)

            if sigma_f32 is not None:
                sigma_f32 = sigma_f32.clamp_min(1e-12)

            dt_phys_f32 = dt_phys.to(device=device, dtype=torch.float32)
            dt_ref_f32  = (dt_ref_t.to(device=device, dtype=torch.float32) if dt_ref_t is not None else None)
            step_timing["prep_norm_dt_s"] += _step_dt(t_prep_t0)

            with amp_ctx():
                pei = pred_ei.to(device) if torch.is_tensor(pred_ei) else pred_ei
                pea = pred_ea.to(device) if (pred_ea is not None and torch.is_tensor(pred_ea)) else pred_ea
                if model_needs_edge_attr and pea is None:
                    raise RuntimeError(
                        f"model.type='{cfg.get('model', {}).get('type', cfg.get('model', {}).get('name', ''))}' "
                        f"requires edge_attr, but step_k={step_k} has pred_ea=None."
                    )
                chunk_specs_eff = chunk_specs_step
                if chunk_enabled and (chunk_specs_eff is None):
                    t_chunk_t0 = _step_t0()
                    chunk_specs_eff = _get_chunk_specs_for_step(
                        pred_parents=pred_parents,
                        pred_ei=pei,
                        step_t_abs=step_t_abs,
                    )
                    step_timing["chunk_specs_s"] += _step_dt(t_chunk_t0)

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

                t_phy_t0 = _step_t0()
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
                        if ops_danger_watch and torch.is_tensor(x_in_abs):
                            danger = (
                                (~torch.isfinite(x_in_abs)).any()
                                or (x_in_abs.abs().max() > 1e6)          # tune threshold
                            )
                            if danger:
                                print(f"\n[NAN-DBG] step={step_k} TRIGGER: x_in_abs looks dangerous before ops")
                                _dump_state("x_in_abs", x_in_abs)

                        # after enforce_physical_state
                        if ops_danger_watch and torch.is_tensor(x_for_ops):
                            danger2 = (
                                (~torch.isfinite(x_for_ops)).any()
                                or (x_for_ops.abs().max() > 1e6)
                            )
                            if danger2:
                                print(f"\n[NAN-DBG] step={step_k} TRIGGER: x_for_ops looks dangerous after enforce")
                                _dump_state("x_for_ops", x_for_ops)

                        with torch.no_grad():
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

                        Fdim = int(x_in_abs.size(1))
                        ch_mask = _get_parc_channel_mask(Fdim, device)

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

                    _assert_finite_if("r_adv_abs", r_adv_abs, crash=False)
                    _assert_finite_if("r_diff_abs", r_diff_abs, crash=False)
                    _assert_finite_if("area", area)
                    #_assert_finite("r_adv_abs", r_adv_abs)
                    if r_phy_abs is not None:
                        _assert_finite_if("r_phy_abs", r_phy_abs)
                step_timing["physics_ops_s"] += _step_dt(t_phy_t0)

                # ----- build node input X (PARC appends operator inputs) -----
                t_input_t0 = _step_t0()
                x_in = _build_X(
                    norm_in,
                    pred_centers,
                    pred_levels,
                    cfg,
                    dt_hat=dt_hat,
                    ramp_feature_ctx=ramp_feature_ctx,
                    step_t_abs=step_t_abs,
                )

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
                    if cfg.get("debug", {}).get("parc_scale_watch", False) and (step_k in watch_steps):
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
                        _assert_finite_if("parc_extra", parc_extra)
                    _assert_finite_if("x_in (final)", x_in)
                step_timing["input_assemble_s"] += _step_dt(t_input_t0)

                # one-time print guard
                if not hasattr(train_one_epoch_multi_step, "_printed_input_contract"):
                    train_one_epoch_multi_step._printed_input_contract = False

                if (not train_one_epoch_multi_step._printed_input_contract):
                    Fdim = x_in_abs.size(1)

                    # Base X is built from norm_in + geometry (pos, level, etc.)
                    base_X = _build_X(
                        norm_in,
                        pred_centers,
                        pred_levels,
                        cfg,
                        dt_hat=dt_hat,
                        ramp_feature_ctx=ramp_feature_ctx,
                        step_t_abs=step_t_abs,
                    )

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
                    expected_parc = (
                        (len(sel_adv) if inc_adv else 0) + (len(sel_diff) if inc_diff else 0)
                        if parc_use else 0
                    )
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
                t_gnn_t0 = _step_t0()
                y_corr = _forward_main_head_chunked(
                    x_in_all=x_in,
                    edge_index_global=pei,
                    edge_attr_global=pea,
                    chunk_specs=chunk_specs_eff,
                    step_k=step_k,
                    step_t_abs=step_t_abs,
                    stage_tag="train-main",
                )
                step_timing["gnn_forward_s"] += _step_dt(t_gnn_t0)

                # ---------------------------
                # NAN DEBUG: NN forward output
                # ---------------------------
                if _should_print():
                    print(f"[NAN-DBG] step={step_k} ---- y_corr checks ----")
                    _tstats("y_corr", y_corr)
                    _mark_printed()

                _assert_finite_if("y_corr", y_corr)

                # ----- add Variant-B baseline in model units -----
                t_blend_t0 = _step_t0()
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

                    _assert_finite_if("phy_units", phy_units)
                    _assert_finite_if("y_pred", y_pred)
                else:
                    if _should_print():
                        print(f"[NAN-DBG] step={step_k} ---- y_pred (no baseline) ----")
                        _tstats("y_pred", y_pred)
                        _mark_printed()
                    _assert_finite_if("y_pred", y_pred)
                step_timing["baseline_blend_s"] += _step_dt(t_blend_t0)

                r_phy_for_loss = r_phy_abs

                # Optional higher-order integration in normalized-rate space.
                t_rk4_t0 = _step_t0()
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

                        x_stage_in = _build_X(
                            norm_state,
                            pred_centers,
                            pred_levels,
                            cfg,
                            dt_hat=dt_hat,
                            ramp_feature_ctx=ramp_feature_ctx,
                            step_t_abs=step_t_abs,
                        )
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

                        y_corr_stage = _forward_main_head_chunked(
                            x_in_all=x_stage_in,
                            edge_index_global=pei,
                            edge_attr_global=pea,
                            chunk_specs=chunk_specs_eff,
                            step_k=step_k,
                            step_t_abs=step_t_abs,
                            stage_tag="train-rk4",
                        )
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

                    _assert_finite_if("y_pred_rk_sched", y_pred)
                step_timing["rk4_s"] += _step_dt(t_rk4_t0)

                # ----- supervision target -----
                t_loss_t0 = _step_t0()
                delta_target = norm_tgt - norm_in
                rate_target = delta_target / dt_hat.clamp_min(1e-12)
                model_target = _target_for_predict_type(
                    norm_in=norm_in,
                    norm_tgt=norm_tgt,
                    dt_hat=dt_hat,
                    predict_type=predict_type,
                )

                if _should_print():
                    print(f"[NAN-DBG] step={step_k} ---- target checks ----")
                    _tstats("delta_target", delta_target)
                    _tstats("rate_target", rate_target)
                    _tstats("model_target", model_target)
                    _mark_printed()

                _assert_finite_if("model_target", model_target)

                watch_steps = set(cfg.get("debug", {}).get("parc_scale_watch_steps", [0, 3, 6, 9]))
                if cfg.get("debug", {}).get("parc_scale_watch", False) and (step_k in watch_steps):
                    with torch.no_grad():
                        _tstats("model_target", model_target)
                        _tstats("y_pred", y_pred)

                if predict_type == "rate":
                    y_pred = _apply_rate_guardrails(y_pred, dt_hat, cfg)


                # ==========================================================
                # FAST NEXT DIAGNOSTIC (RUN ONCE, STEP 0 ONLY)
                #   - norm-space call: best for dt scale + channel permutation
                #   - phys-space call: best to detect normalization mismatch
                # NOTE: do NOT pass the AMP GradScaler "scaler" here.
                # ==========================================================
                center_loss = (F.huber_loss(y_pred, model_target, delta=huber_delta)
                               if use_huber else F.mse_loss(y_pred, model_target))

                y_pred_norm = _state_from_model_output(
                    norm_in=norm_in,
                    y_pred=y_pred,
                    dt_hat=dt_hat,
                    predict_type=predict_type,
                )
                y_pred_abs = _maybe_denorm(y_pred_norm, mu, sigma)

                # ---------------- DEBUG: oracle check ----------------
                if cfg.get("debug", {}).get("oracle_watch", False) and predict_type == "rate":
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

                pressure_aux_w = float(loss_cfg.get("pressure_aux_weight", loss_cfg.get("pressure_loss_weight", 0.0)))
                pressure_consistency_w = float(loss_cfg.get("pressure_consistency_weight", 0.0))
                pressure_aux_loss = y_pred.new_zeros(())
                pressure_consistency_loss = y_pred.new_zeros(())
                if pressure_aux_w > 0.0 or pressure_consistency_w > 0.0:
                    with torch.autocast(device_type=device.type, enabled=False):
                        p_aux, p_cons = _pressure_auxiliary_losses(
                            y_pred_abs=y_pred_abs.float(),
                            x_tgt_abs=x_tgt_abs.float(),
                            cfg=cfg,
                        )
                    pressure_aux_loss = p_aux.to(dtype=y_pred.dtype)
                    pressure_consistency_loss = p_cons.to(dtype=y_pred.dtype)

                loss_step = (
                    center_loss
                    + lap_w * lap_loss
                    + tmp_w * tmp_loss
                    + dec_resid_w * phy_loss
                    + pressure_aux_w * pressure_aux_loss
                    + pressure_consistency_w * pressure_consistency_loss
                )

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
                        pressure_aux_w=pressure_aux_w,
                        pressure_aux_loss=pressure_aux_loss,
                        pressure_consistency_w=pressure_consistency_w,
                        pressure_consistency_loss=pressure_consistency_loss,
                        total_loss=loss_step,
                        y_nn=y_corr,
                        y_pred=y_pred,
                    )
                    train_one_epoch_multi_step._printed_loss_components = True
                step_timing["target_loss_s"] += _step_dt(t_loss_t0)

            if isinstance(timing_out, dict):
                timing_out.clear()
                timing_out.update(step_timing)
                timing_out["model_step_total_s"] = float(sum(step_timing.values()))

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
                and (not runtime_multires_enabled)
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
                t_src_abs = (
                    int(t_indices[k].item())
                    if (t_indices is not None and torch.is_tensor(t_indices))
                    else (int(t_indices[k]) if t_indices is not None else int(k))
                )
                t_dst_abs = (
                    int(t_indices[k + 1].item())
                    if (t_indices is not None and torch.is_tensor(t_indices))
                    else (int(t_indices[k + 1]) if t_indices is not None else int(k + 1))
                )

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
                        t_rebuild_t0 = _rt_t0()
                        rebuild_timing: Dict[str, float] = {}
                        predictor_input_dbg: Dict[str, Any] = {}
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
                            predictor_input_out=predictor_input_dbg,
                            step_k=k,
                            cnn_parent_mapping_mode=("from_centers" if (k == 0) else "dataset"),
                            cnn_fill_empty_interior=bool(k == 0),
                            dt_phys=dtk,
                            dt_ref=dt_ref_scalar,
                            timing_out=rebuild_timing,
                        )
                        t_rebuild_s = _rt_dt(t_rebuild_t0)
                        batch_rt_rebuilds += 1
                        batch_rt_mesh_predict_s += float(rebuild_timing.get("mesh_predict_s", 0.0))
                        batch_rt_mesh_materialize_s += float(rebuild_timing.get("mesh_materialize_s", 0.0))
                        batch_rt_wedge_clip_s += float(rebuild_timing.get("wedge_clip_s", 0.0))
                        batch_rt_edge_attr_s += float(rebuild_timing.get("edge_attr_s", 0.0))
                        for tk, tv in rebuild_timing.items():
                            if str(tk).startswith("predict_"):
                                batch_rt_mesh_predict_parts_s[tk] = (
                                    float(batch_rt_mesh_predict_parts_s.get(tk, 0.0)) + float(tv)
                                )
                            elif str(tk).startswith("wedge_"):
                                batch_rt_wedge_parts_s[tk] = (
                                    float(batch_rt_wedge_parts_s.get(tk, 0.0)) + float(tv)
                                )
                        t_plot_t0 = time.perf_counter()
                        _runtime_mesh_plot_maybe_save(
                            settings=runtime_mesh_plot_settings,
                            state=runtime_mesh_plot_state,
                            split="train",
                            epoch_idx=epoch_idx,
                            batch_idx=batch_idx,
                            step_k=k,
                            t_abs=int(t_dst_abs),
                            pred_centers=active_pred_centers,
                            pred_levels=active_pred_levels,
                            H=H,
                            W=W,
                            bbox=runtime_bbox,
                            refine_ratio=runtime_refine_ratio,
                            wedge_path=runtime_mesh_plot_wedge_path,
                            predictor_input=predictor_input_dbg,
                            gt_snapshots=[
                                {
                                    "tag": "gt_t",
                                    "t_abs": int(t_src_abs),
                                    "centers": centers_list[k],
                                    "levels": level_list[k],
                                    "feat": feat_list[k],
                                },
                                {
                                    "tag": "gt_tp1",
                                    "t_abs": int(t_dst_abs),
                                    "centers": centers_list[k + 1],
                                    "levels": level_list[k + 1],
                                    "feat": feat_list[k + 1],
                                },
                            ],
                        )
                        batch_rt_other_mesh_plot_s += float(time.perf_counter() - t_plot_t0)

                        if k == 0:
                            n_src_xin = int(state_centers.shape[0])
                            n_dst_xin = int(active_pred_centers.shape[0])
                            if runtime_multires_enabled:
                                if runtime_multires_lookup is None:
                                    raise RuntimeError(
                                        "Runtime multires GT lookup is enabled but lookup object is not initialized."
                                    )
                                t_xin_t0 = _rt_t0()
                                x_in_abs, xin_lookup_info = _lookup_gt_on_pred_mesh_multires(
                                    lookup=runtime_multires_lookup,
                                    t_dst=int(t_dst_abs),
                                    dataset_name="feat_t_on_pred",
                                    pred_centers=active_pred_centers,
                                    pred_levels=active_pred_levels,
                                    fallback_src_centers=state_centers,
                                    fallback_src_feats=state_feat,
                                    knn_k=runtime_knn_k,
                                    chunk=runtime_chunk,
                                    knn_backend=runtime_idw_backend,
                                    knn_backend_kwargs=runtime_idw_backend_kwargs,
                                    allow_fallback_to_idw=runtime_multires_fallback_to_idw,
                                )
                                xin_requested = int(xin_lookup_info.get("requested", n_dst_xin))
                                xin_missing = int(
                                    xin_lookup_info.get(
                                        "missing_lookup",
                                        xin_lookup_info.get("missing_total", 0),
                                    )
                                )
                                batch_rt_multires_lookup_calls += 1
                                batch_rt_multires_requested_total += xin_requested
                                batch_rt_multires_missing_total += xin_missing
                                batch_rt_multires_xin_requested += xin_requested
                                batch_rt_multires_xin_missing += xin_missing
                                if bool(xin_lookup_info.get("fallback_used", False)):
                                    batch_rt_multires_fallback_calls += 1
                                t_xin_map_s = _rt_dt(t_xin_t0)
                                idw_dev_xin = str(xin_lookup_info.get("fallback_idw_dev", "lookup"))
                                idw_chunk_xin = int(xin_lookup_info.get("fallback_idw_chunk", -1))
                            else:
                                idw_dev_xin_t, idw_chunk_xin = _select_idw_backend(
                                    src_n=n_src_xin,
                                    requested_chunk=int(runtime_chunk),
                                    out_device=device,
                                )
                                idw_dev_xin = idw_dev_xin_t.type
                                t_xin_t0 = _rt_t0()
                                x_in_abs = _map_gt_to_pred_mesh_once(
                                    src_centers=state_centers,
                                    src_feats=state_feat,
                                    pred_centers=active_pred_centers,
                                    knn_k=runtime_knn_k,
                                    chunk=runtime_chunk,
                                    knn_backend=runtime_idw_backend,
                                    knn_backend_kwargs=runtime_idw_backend_kwargs,
                                )
                                t_xin_map_s = _rt_dt(t_xin_t0)
                        else:
                            n_src_xin = int(state_centers.shape[0])
                            n_dst_xin = int(active_pred_centers.shape[0])
                            idw_dev_xin_t, idw_chunk_xin = _select_idw_backend(
                                src_n=n_src_xin,
                                requested_chunk=int(runtime_chunk),
                                out_device=device,
                            )
                            idw_dev_xin = idw_dev_xin_t.type
                            t_xin_t0 = _rt_t0()
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
                            t_xin_map_s = _rt_dt(t_xin_t0)
                    else:
                        if active_pred_centers is None:
                            raise RuntimeError("Runtime mesh state is not initialized before a reuse step.")
                        x_in_abs = state_feat

                    gt_centers_tp1 = _to_dev_nb(centers_list[k + 1], dtype=torch.float32)
                    gt_feat_tp1 = _to_dev_nb(feat_list[k + 1], dtype=torch.float32)
                    n_src_tgt = int(gt_centers_tp1.shape[0])
                    n_dst_tgt = int(active_pred_centers.shape[0])
                    if runtime_multires_enabled:
                        if runtime_multires_lookup is None:
                            raise RuntimeError(
                                "Runtime multires GT lookup is enabled but lookup object is not initialized."
                            )
                        t_xtgt_t0 = _rt_t0()
                        x_tgt_abs, xtgt_lookup_info = _lookup_gt_on_pred_mesh_multires(
                            lookup=runtime_multires_lookup,
                            t_dst=int(t_dst_abs),
                            dataset_name="feat_tp1_on_pred",
                            pred_centers=active_pred_centers,
                            pred_levels=active_pred_levels,
                            fallback_src_centers=gt_centers_tp1,
                            fallback_src_feats=gt_feat_tp1,
                            knn_k=runtime_knn_k,
                            chunk=runtime_chunk,
                            knn_backend=runtime_idw_backend,
                            knn_backend_kwargs=runtime_idw_backend_kwargs,
                            allow_fallback_to_idw=runtime_multires_fallback_to_idw,
                        )
                        xtgt_requested = int(xtgt_lookup_info.get("requested", n_dst_tgt))
                        xtgt_missing = int(
                            xtgt_lookup_info.get(
                                "missing_lookup",
                                xtgt_lookup_info.get("missing_total", 0),
                            )
                        )
                        batch_rt_multires_lookup_calls += 1
                        batch_rt_multires_requested_total += xtgt_requested
                        batch_rt_multires_missing_total += xtgt_missing
                        batch_rt_multires_xtgt_requested += xtgt_requested
                        batch_rt_multires_xtgt_missing += xtgt_missing
                        if bool(xtgt_lookup_info.get("fallback_used", False)):
                            batch_rt_multires_fallback_calls += 1
                        t_xtgt_map_s = _rt_dt(t_xtgt_t0)
                        idw_dev_tgt = str(xtgt_lookup_info.get("fallback_idw_dev", "lookup"))
                        idw_chunk_tgt = int(xtgt_lookup_info.get("fallback_idw_chunk", -1))
                    else:
                        idw_dev_tgt_t, idw_chunk_tgt = _select_idw_backend(
                            src_n=n_src_tgt,
                            requested_chunk=int(runtime_chunk),
                            out_device=device,
                        )
                        idw_dev_tgt = idw_dev_tgt_t.type
                        t_xtgt_t0 = _rt_t0()
                        x_tgt_abs = _map_gt_to_pred_mesh_once(
                            src_centers=gt_centers_tp1,
                            src_feats=gt_feat_tp1,
                            pred_centers=active_pred_centers,
                            knn_k=runtime_knn_k,
                            chunk=runtime_chunk,
                            knn_backend=runtime_idw_backend,
                            knn_backend_kwargs=runtime_idw_backend_kwargs,
                        )
                        t_xtgt_map_s = _rt_dt(t_xtgt_t0)

                batch_rt_idw_remap_s += float(t_xin_map_s + t_xtgt_map_s)

                dtk = dtk.to(device=device, dtype=x_in_abs.dtype, non_blocking=True)
                step_t_abs_for_chunk = None
                if t_indices is not None:
                    step_t_abs_for_chunk = (
                        int(t_indices[k + 1].item())
                        if torch.is_tensor(t_indices)
                        else int(t_indices[k + 1])
                    )

                t_model_t0 = _rt_t0()
                step_model_timing: Dict[str, float] = {}
                loss_k, y_pred_abs_k, _ = _run_step(
                    step_k=k,
                    step_t_abs=step_t_abs_for_chunk,
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
                    timing_out=step_model_timing,
                )
                t_model_s = _rt_dt(t_model_t0)
                batch_rt_other_model_s += float(t_model_s)
                batch_rt_model_calls += 1
                for tk, tv in step_model_timing.items():
                    batch_rt_model_parts_s[tk] = float(batch_rt_model_parts_s.get(tk, 0.0)) + float(tv)

                if not _step_outputs_are_finite(
                    loss_t=loss_k,
                    pred_abs=y_pred_abs_k,
                    tgt_abs=x_tgt_abs,
                    split="train",
                    batch_idx=batch_idx,
                    step_k=k,
                ):
                    continue

                t_metrics_t0 = time.perf_counter()
                window_loss += float(loss_k.detach().cpu())
                step_mae = float(torch.mean(torch.abs(y_pred_abs_k.detach() - x_tgt_abs)).cpu())
                window_mae += step_mae
                n_steps += 1
                window_loss_graph = loss_k if window_loss_graph is None else (window_loss_graph + loss_k)
                batch_rt_other_metrics_sync_s += float(time.perf_counter() - t_metrics_t0)

                t_state_diag_t0 = time.perf_counter()
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
                batch_rt_other_state_diag_s += float(time.perf_counter() - t_state_diag_t0)

                step_t_abs = -1
                if t_indices is not None:
                    step_t_abs = int(t_indices[k + 1].item()) if torch.is_tensor(t_indices) else int(t_indices[k + 1])
                t_mem_snap_t0 = time.perf_counter()
                mem_alloc_mb, mem_reserved_mb = _runtime_memory_snapshot_mb(device)
                batch_rt_other_mem_snapshot_s += float(time.perf_counter() - t_mem_snap_t0)
                t_steplog_t0 = time.perf_counter()
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
                batch_rt_other_step_log_io_s += float(time.perf_counter() - t_steplog_t0)

            # ===== backward + step once per window =====
            if window_loss_graph is not None:
                if scaler is not None:
                    t_bw_real_t0 = _rt_t0()
                    scaler.scale(window_loss_graph).backward()
                    batch_rt_other_backward_s += _rt_dt(t_bw_real_t0)
                    t_clip_real_t0 = _rt_t0()
                    scaler.unscale_(opt)  # <-- required before clipping
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    batch_rt_other_unscale_clip_s += _rt_dt(t_clip_real_t0)
                    t_step_real_t0 = _rt_t0()
                    scaler.step(opt)
                    scaler.update()
                    batch_rt_other_optimizer_s += _rt_dt(t_step_real_t0)
                else:
                    t_bw_real_t0 = _rt_t0()
                    window_loss_graph.backward()
                    batch_rt_other_backward_s += _rt_dt(t_bw_real_t0)
                    t_clip_real_t0 = _rt_t0()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    batch_rt_other_unscale_clip_s += _rt_dt(t_clip_real_t0)
                    t_step_real_t0 = _rt_t0()
                    opt.step()
                    batch_rt_other_optimizer_s += _rt_dt(t_step_real_t0)
            else:
                t_step_real_t0 = _rt_t0()
                opt.step()
                batch_rt_other_optimizer_s += _rt_dt(t_step_real_t0)

            total_loss_accum += window_loss
            mae_accum += window_mae
            batch_wall = _rt_dt(batch_t0)
            epoch_body_time_accum_s += float(batch_wall)
            avg_body_s = float(epoch_body_time_accum_s / float(max(int(batch_idx) + 1, 1)))
            _maybe_print_batch_progress(
                batch_idx,
                batch_wall,
                data_wait_s=float(data_wait_s),
                avg_body_s=avg_body_s,
            )
            _maybe_print_runtime_mesh_batch_breakdown(
                batch_idx,
                batch_wall,
                t_mesh_predict_s=batch_rt_mesh_predict_s,
                t_mesh_materialize_s=batch_rt_mesh_materialize_s,
                t_wedge_clip_s=batch_rt_wedge_clip_s,
                t_edge_attr_s=batch_rt_edge_attr_s,
                t_idw_remap_s=batch_rt_idw_remap_s,
                n_rebuilds=batch_rt_rebuilds,
            )
            _maybe_print_runtime_mesh_predict_detail(
                batch_idx,
                t_mesh_predict_s=batch_rt_mesh_predict_s,
                n_rebuilds=batch_rt_rebuilds,
                predict_parts_s=batch_rt_mesh_predict_parts_s,
            )
            _maybe_print_runtime_wedge_detail(
                batch_idx,
                t_wedge_clip_s=batch_rt_wedge_clip_s,
                wedge_parts_s=batch_rt_wedge_parts_s,
                n_rebuilds=batch_rt_rebuilds,
            )
            _maybe_print_runtime_other_detail(
                batch_idx,
                batch_wall,
                t_mesh_predict_s=batch_rt_mesh_predict_s,
                t_mesh_materialize_s=batch_rt_mesh_materialize_s,
                t_wedge_clip_s=batch_rt_wedge_clip_s,
                t_edge_attr_s=batch_rt_edge_attr_s,
                t_idw_remap_s=batch_rt_idw_remap_s,
                other_parts_s={
                    "model_step_s": float(batch_rt_other_model_s),
                    "metrics_sync_s": float(batch_rt_other_metrics_sync_s),
                    "state_diag_s": float(batch_rt_other_state_diag_s),
                    "mem_snapshot_s": float(batch_rt_other_mem_snapshot_s),
                    "step_log_io_s": float(batch_rt_other_step_log_io_s),
                    "mesh_plot_s": float(batch_rt_other_mesh_plot_s),
                    "backward_s": float(batch_rt_other_backward_s),
                    "unscale_clip_s": float(batch_rt_other_unscale_clip_s),
                    "optimizer_s": float(batch_rt_other_optimizer_s),
                    "zero_grad_s": float(batch_rt_other_zero_grad_s),
                },
            )
            _maybe_print_runtime_outside_detail(
                batch_idx,
                data_wait_s=float(data_wait_s),
            )
            _maybe_print_runtime_model_step_detail(
                batch_idx,
                t_model_step_s=float(batch_rt_other_model_s),
                model_calls=int(batch_rt_model_calls),
                model_parts_s=batch_rt_model_parts_s,
            )
            _maybe_print_runtime_multires_batch_mismatch(
                batch_idx,
                lookup_calls=batch_rt_multires_lookup_calls,
                requested_total=batch_rt_multires_requested_total,
                missing_total=batch_rt_multires_missing_total,
                fallback_calls=batch_rt_multires_fallback_calls,
                xin_requested=batch_rt_multires_xin_requested,
                xin_missing=batch_rt_multires_xin_missing,
                xtgt_requested=batch_rt_multires_xtgt_requested,
                xtgt_missing=batch_rt_multires_xtgt_missing,
            )
            iter_wait_anchor_t = time.perf_counter()
            continue

        # ======================
        # STEP 0 (k=0)
        # ======================
        abort_window_nonfinite = False
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
        step0_t_abs = None
        if t_indices is not None:
            step0_t_abs = int(t_indices[1].item()) if torch.is_tensor(t_indices) else int(t_indices[1])

        loss0, y_pred_abs0, _x_tgt_abs0 = _run_step(
            step_k=0,
            step_t_abs=step0_t_abs,
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

        if not _step_outputs_are_finite(
            loss_t=loss0,
            pred_abs=y_pred_abs0,
            tgt_abs=x_tgt_abs0,
            split="train",
            batch_idx=batch_idx,
            step_k=0,
        ):
            abort_window_nonfinite = True

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

        if not abort_window_nonfinite:
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
            if abort_window_nonfinite:
                break
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

                        print(f"    drift={d.tolist()}")
                        print(f"    roll ={xs.tolist()}")
                        print(f"    teach={xt.tolist()}")

                        # velocity diagnostic from the configured state representation
                        s_roll = dec.state_views(xs.view(1, -1).to(dtype=torch.float32), cfg)
                        s_teach = dec.state_views(xt.view(1, -1).to(dtype=torch.float32), cfg)
                        urx = float(s_roll["u"][0].item())
                        ury = float(s_roll["v"][0].item())
                        utx = float(s_teach["u"][0].item())
                        uty = float(s_teach["v"][0].item())

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
            stepk_t_abs = None
            if t_indices is not None:
                stepk_t_abs = int(t_indices[k + 1].item()) if torch.is_tensor(t_indices) else int(t_indices[k + 1])

            loss_k, y_pred_abs_k, _ = _run_step(
                step_k=k,
                step_t_abs=stepk_t_abs,
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

            if not _step_outputs_are_finite(
                loss_t=loss_k,
                pred_abs=y_pred_abs_k,
                tgt_abs=x_tgt_abs,
                split="train",
                batch_idx=batch_idx,
                step_k=k,
            ):
                abort_window_nonfinite = True
                break

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
        batch_wall = _rt_dt(batch_t0)
        epoch_body_time_accum_s += float(batch_wall)
        avg_body_s = float(epoch_body_time_accum_s / float(max(int(batch_idx) + 1, 1)))
        _maybe_print_batch_progress(
            batch_idx,
            batch_wall,
            data_wait_s=float(data_wait_s),
            avg_body_s=avg_body_s,
        )
        iter_wait_anchor_t = time.perf_counter()

    denom = max(n_steps, 1)
    stats = {
        "num_windows": len(loader),
        "num_steps": n_steps,
        **_finalize_diagnostics_accumulator(diagnostics_accum, diag_cfg),
    }
    if runtime_multires_lookup is not None:
        runtime_multires_lookup.close()
        runtime_multires_lookup = None
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
    ramp_feature_ctx: Dict[str, Any] | None = None,
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
    use_huber = bool(cfg["loss"].get("use_huber", True))

    predict_type = _normalize_predict_type_key(
        cfg.get("model", {}).get("predict_type", cfg.get("model", {}).get("target_mode", "rate"))
    )
    cfg.setdefault("model", {})["predict_type"] = predict_type
    time_integrator = _resolve_time_integrator(cfg)
    _time_integrator_eff, rk4_alpha = _resolve_time_integrator_for_epoch(
        cfg, epoch_idx=epoch_idx, base_integrator=time_integrator
    )
    if predict_type != "rate" and rk4_alpha > 0.0:
        raise RuntimeError(
            "RK4/ramped RK4 integration requires model.predict_type='rate'. "
            f"Got predict_type='{predict_type}'."
        )

    loss_cfg = cfg.get("loss", {}) or {}
    parc_use = _physics_inputs_active_from_loss_cfg(loss_cfg)
    use_adapter = bool(loss_cfg.get("parc_use_adapter", False))

    dec_blend_w = float(loss_cfg.get("blend_weight", 0.0))
    dec_resid_w = float(loss_cfg.get("residual_weight", 0.0))

    need_phy = bool(parc_use)
    model_needs_edge_attr = _model_requires_edge_attr_from_cfg(cfg)

    backend = str(loss_cfg.get("physics_backend", "dec")).lower()
    use_mls = (backend in ("mls", "moving_least_squares", "moving-least-squares"))

    amp_ctx = ((lambda: torch.amp.autocast(device_type="cuda", enabled=True))
               if use_amp and device.type == "cuda"
               else (lambda: torch.autocast("cpu", enabled=False)))

    runtime_mesh_enabled = _runtime_mesh_enabled_from_cfg(cfg)
    if runtime_mesh_enabled:
        raise RuntimeError(
            "Runtime mesh is disabled in Physics-Aware-cartesian. "
            "Set train.runtime_mesh.enabled=false."
        )
    runtime_multires_cfg = _runtime_multires_lookup_cfg(cfg)
    runtime_multires_enabled = False
    runtime_multires_fallback_to_idw = bool(runtime_multires_cfg.get("fallback_to_idw", True))
    runtime_multires_lookup: Optional[_RuntimeMultiResGtLookup] = None
    runtime_infer_only = False
    runtime_update_every = 1
    runtime_domain_mode = "disabled"
    runtime_idw_backend = "exact"
    runtime_idw_backend_kwargs: Dict[str, Any] = {}
    runtime_bbox = _get_bbox(cfg)
    runtime_refine_ratio = _get_refine_ratio(cfg)
    runtime_need_edge_attr = bool(model_needs_edge_attr or (need_phy and (not use_mls)))
    runtime_mesh_plot_settings = _runtime_mesh_plot_settings(cfg)
    runtime_mesh_plot_state = {"rebuilds": 0, "saved": 0}
    runtime_mesh_plot_wedge_path = None
    runtime_wedge_path = None
    runtime_wedge_constraints = None
    runtime_base_mesh = None

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
        if x_abs.ndim != 2:
            raise RuntimeError(f"x_abs must be [N,F], got {tuple(x_abs.shape)}")

        # area weights on pred mesh
        w = dec.cell_area_from_levels(
            pred_levels, dx0=float(dx), dy0=float(dy),
            dtype=x_abs.dtype, device=x_abs.device,
            refine_ratio=_get_refine_ratio(cfg),
        )  # [N]

        state = dec.state_views(x_abs, cfg)
        rho = state["rho"]
        mx = state["mx"]
        my = state["my"]
        E = state["E_tot"]

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

            pred_centers_list = _require_list(batch, "pred_centers_list")
            pred_levels_list = _require_list(batch, "pred_levels_list")
            pred_parents_list = _require_list(batch, "pred_parents_list")
            pred_ei_list = _require_list(batch, "pred_ei_list")
            mask_pred_list = _require_list(batch, "mask_pred_list")
            feat_t_on_pred_list = _require_list(batch, "feat_t_on_pred_list")
            feat_tp1_on_pred_list = _require_list(batch, "feat_tp1_on_pred_list")

            if (model_needs_edge_attr or (need_phy and (not use_mls))) and pred_ea_list is None:
                raise RuntimeError(
                    "Model/DEC physics requires edge attributes, but batch is missing "
                    "pred_ea_list / pred_edge_attr_list."
                )

            pred_lists = (pred_centers_list, pred_levels_list, pred_parents_list, pred_ei_list, mask_pred_list)

            # helper closure for one step (mirrors train)
            def _run_step_eval(
                *,
                step_k: int,
                step_t_abs: int | None,
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
                    if model_needs_edge_attr and pea is None:
                        raise RuntimeError(
                            f"model.type='{cfg.get('model', {}).get('type', cfg.get('model', {}).get('name', ''))}' "
                            f"requires edge_attr, but eval step_k={step_k} has pred_ea=None."
                        )

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
                    x_in = _build_X(
                        norm_in,
                        pred_centers,
                        pred_levels,
                        cfg,
                        dt_hat=dt_hat,
                        ramp_feature_ctx=ramp_feature_ctx,
                        step_t_abs=step_t_abs,
                    )

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

                            x_stage_in = _build_X(
                                norm_state,
                                pred_centers,
                                pred_levels,
                                cfg,
                                dt_hat=dt_hat,
                                ramp_feature_ctx=ramp_feature_ctx,
                                step_t_abs=step_t_abs,
                            )
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

                    if predict_type == "rate":
                        y_pred = _apply_rate_guardrails(y_pred, dt_hat, cfg)

                    # Absolute prediction update in fp32 for stability.
                    with torch.autocast(device_type=device.type, enabled=False):
                        dt_hat_f32 = dt_hat.to(dtype=torch.float32)
                        y_pred_f32 = y_pred.to(dtype=torch.float32)
                        y_pred_norm_f32 = _state_from_model_output(
                            norm_in=norm_in.to(dtype=torch.float32),
                            y_pred=y_pred_f32,
                            dt_hat=dt_hat_f32,
                            predict_type=predict_type,
                        )
                        y_pred_abs = _maybe_denorm(
                            y_pred_norm_f32,
                            mu, sigma
                        )

                    center_loss = y_pred.new_zeros(())
                    tmp_loss = y_pred.new_zeros(())
                    if norm_tgt is not None:
                        with torch.autocast(device_type=device.type, enabled=False):
                            dt_hat_f32 = dt_hat.to(dtype=torch.float32)
                            model_target_f32 = _target_for_predict_type(
                                norm_in=norm_in.to(dtype=torch.float32),
                                norm_tgt=norm_tgt.to(dtype=torch.float32),
                                dt_hat=dt_hat_f32,
                                predict_type=predict_type,
                            )
                            center_loss = (
                                F.huber_loss(y_pred_f32, model_target_f32, delta=huber_delta)
                                if use_huber else F.mse_loss(y_pred_f32, model_target_f32)
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

                    pressure_aux_w = float(loss_cfg.get("pressure_aux_weight", loss_cfg.get("pressure_loss_weight", 0.0)))
                    pressure_consistency_w = float(loss_cfg.get("pressure_consistency_weight", 0.0))
                    pressure_aux_loss = y_pred.new_zeros(())
                    pressure_consistency_loss = y_pred.new_zeros(())
                    if (
                        norm_tgt is not None
                        and (pressure_aux_w > 0.0 or pressure_consistency_w > 0.0)
                    ):
                        with torch.autocast(device_type=device.type, enabled=False):
                            p_aux, p_cons = _pressure_auxiliary_losses(
                                y_pred_abs=y_pred_abs.float(),
                                x_tgt_abs=x_tgt_abs.float() if x_tgt_abs is not None else None,
                                cfg=cfg,
                            )
                        pressure_aux_loss = p_aux.to(dtype=y_pred.dtype)
                        pressure_consistency_loss = p_cons.to(dtype=y_pred.dtype)

                    loss_step = (
                        center_loss
                        + lap_w * lap_loss
                        + tmp_w * tmp_loss
                        + dec_resid_w * phy_loss
                        + pressure_aux_w * pressure_aux_loss
                        + pressure_consistency_w * pressure_consistency_loss
                    )
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
                    and (not runtime_multires_enabled)
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
                    t_src_abs = (
                        int(t_indices[k].item())
                        if (t_indices is not None and torch.is_tensor(t_indices))
                        else (int(t_indices[k]) if t_indices is not None else int(k))
                    )
                    t_dst_abs = (
                        int(t_indices[k + 1].item())
                        if (t_indices is not None and torch.is_tensor(t_indices))
                        else (int(t_indices[k + 1]) if t_indices is not None else int(k + 1))
                    )

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
                            predictor_input_dbg: Dict[str, Any] = {}
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
                                predictor_input_out=predictor_input_dbg,
                                step_k=k,
                                cnn_parent_mapping_mode=("from_centers" if (k == 0) else "dataset"),
                                cnn_fill_empty_interior=bool(k == 0),
                                dt_phys=dtk,
                                dt_ref=dt_ref_scalar,
                            )
                            t_rebuild_s = time.perf_counter() - t_rebuild_t0
                            _runtime_mesh_plot_maybe_save(
                                settings=runtime_mesh_plot_settings,
                                state=runtime_mesh_plot_state,
                                split="eval",
                                epoch_idx=epoch_idx,
                                batch_idx=batch_idx,
                                step_k=k,
                                t_abs=int(t_dst_abs),
                                pred_centers=active_pred_centers,
                                pred_levels=active_pred_levels,
                                H=H,
                                W=W,
                                bbox=runtime_bbox,
                                refine_ratio=runtime_refine_ratio,
                                wedge_path=runtime_mesh_plot_wedge_path,
                                predictor_input=predictor_input_dbg,
                                gt_snapshots=[
                                    {
                                        "tag": "gt_t",
                                        "t_abs": int(t_src_abs),
                                        "centers": centers_list[k],
                                        "levels": level_list[k],
                                        "feat": feat_list[k],
                                    },
                                    {
                                        "tag": "gt_tp1",
                                        "t_abs": int(t_dst_abs),
                                        "centers": centers_list[k + 1],
                                        "levels": level_list[k + 1],
                                        "feat": feat_list[k + 1],
                                    },
                                ],
                            )

                            if k == 0:
                                n_src_xin = int(state_centers.shape[0])
                                n_dst_xin = int(active_pred_centers.shape[0])
                                if runtime_multires_enabled:
                                    if runtime_multires_lookup is None:
                                        raise RuntimeError(
                                            "Runtime multires GT lookup is enabled but lookup object is not initialized."
                                        )
                                    t_xin_t0 = time.perf_counter()
                                    x_in_abs, xin_lookup_info = _lookup_gt_on_pred_mesh_multires(
                                        lookup=runtime_multires_lookup,
                                        t_dst=int(t_dst_abs),
                                        dataset_name="feat_t_on_pred",
                                        pred_centers=active_pred_centers,
                                        pred_levels=active_pred_levels,
                                        fallback_src_centers=state_centers,
                                        fallback_src_feats=state_feat,
                                        knn_k=runtime_knn_k,
                                        chunk=runtime_chunk,
                                        knn_backend=runtime_idw_backend,
                                        knn_backend_kwargs=runtime_idw_backend_kwargs,
                                        allow_fallback_to_idw=runtime_multires_fallback_to_idw,
                                    )
                                    t_xin_map_s = time.perf_counter() - t_xin_t0
                                    idw_dev_xin = str(xin_lookup_info.get("fallback_idw_dev", "lookup"))
                                    idw_chunk_xin = int(xin_lookup_info.get("fallback_idw_chunk", -1))
                                else:
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
                            if runtime_multires_enabled:
                                if runtime_multires_lookup is None:
                                    raise RuntimeError(
                                        "Runtime multires GT lookup is enabled but lookup object is not initialized."
                                    )
                                t_xtgt_t0 = time.perf_counter()
                                x_tgt_abs, xtgt_lookup_info = _lookup_gt_on_pred_mesh_multires(
                                    lookup=runtime_multires_lookup,
                                    t_dst=int(t_dst_abs),
                                    dataset_name="feat_tp1_on_pred",
                                    pred_centers=active_pred_centers,
                                    pred_levels=active_pred_levels,
                                    fallback_src_centers=gt_centers_tp1,
                                    fallback_src_feats=gt_feat_tp1,
                                    knn_k=runtime_knn_k,
                                    chunk=runtime_chunk,
                                    knn_backend=runtime_idw_backend,
                                    knn_backend_kwargs=runtime_idw_backend_kwargs,
                                    allow_fallback_to_idw=runtime_multires_fallback_to_idw,
                                )
                                t_xtgt_map_s = time.perf_counter() - t_xtgt_t0
                                idw_dev_tgt = str(xtgt_lookup_info.get("fallback_idw_dev", "lookup"))
                                idw_chunk_tgt = int(xtgt_lookup_info.get("fallback_idw_chunk", -1))
                            else:
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
                    step_t_abs_eval = None
                    if t_indices is not None:
                        step_t_abs_eval = (
                            int(t_indices[k + 1].item())
                            if torch.is_tensor(t_indices)
                            else int(t_indices[k + 1])
                        )

                    t_model_t0 = time.perf_counter()
                    loss_k, y_pred_abs_k = _run_step_eval(
                        step_k=k,
                        step_t_abs=step_t_abs_eval,
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

                    if not _step_outputs_are_finite(
                        loss_t=loss_k,
                        pred_abs=y_pred_abs_k,
                        tgt_abs=x_tgt_abs,
                        split="eval",
                        batch_idx=batch_idx,
                        step_k=k,
                    ):
                        continue

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
            step0_t_abs = None
            if t_indices is not None:
                step0_t_abs = (
                    int(t_indices[1].item())
                    if torch.is_tensor(t_indices)
                    else int(t_indices[1])
                )

            loss0, y_pred_abs0 = _run_step_eval(
                step_k=0,
                step_t_abs=step0_t_abs,
                pred_centers=pred_centers_1,
                pred_levels=pred_levels_1,
                pred_parents=pred_parents_1,
                pred_ei=pred_ei_1,
                pred_ea=pred_ea_1,
                x_in_abs=x_in_abs0,
                x_tgt_abs=x_tgt_abs0,
                dt_phys=dt0,
            )

            if not _step_outputs_are_finite(
                loss_t=loss0,
                pred_abs=y_pred_abs0,
                tgt_abs=x_tgt_abs0,
                split="eval",
                batch_idx=batch_idx,
                step_k=0,
            ):
                continue

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
                stepk_t_abs = None
                if t_indices is not None:
                    stepk_t_abs = (
                        int(t_indices[k + 1].item())
                        if torch.is_tensor(t_indices)
                        else int(t_indices[k + 1])
                    )

                loss_k, y_pred_abs_k = _run_step_eval(
                    step_k=k,
                    step_t_abs=stepk_t_abs,
                    pred_centers=pred_centers_next,
                    pred_levels=pred_levels_next,
                    pred_parents=pred_parents_next,
                    pred_ei=pred_ei_next,
                    pred_ea=pred_ea_next,
                    x_in_abs=x_in_abs,
                    x_tgt_abs=x_tgt_abs,
                    dt_phys=dtk,
                )

                if not _step_outputs_are_finite(
                    loss_t=loss_k,
                    pred_abs=y_pred_abs_k,
                    tgt_abs=x_tgt_abs,
                    split="eval",
                    batch_idx=batch_idx,
                    step_k=k,
                ):
                    continue

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
            if runtime_multires_lookup is not None:
                runtime_multires_lookup.close()
                runtime_multires_lookup = None
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
    if runtime_multires_lookup is not None:
        runtime_multires_lookup.close()
        runtime_multires_lookup = None
    return avg_loss, stats

def build_model_from_cfg(cfg, device, ramp_feature_ctx: Dict[str, Any] | None = None):
    feats_cfg = cfg.get("features", {}) or {}
    use_cols = feats_cfg.get("use_columns", None)
    if isinstance(use_cols, (list, tuple)) and len(use_cols) > 0:
        use_cols = list(use_cols)
    else:
        names = feats_cfg.get("names", None)
        if isinstance(names, (list, tuple)) and len(names) > 0:
            use_cols = list(range(len(names)))
        else:
            use_cols = [0, 1, 2, 3, 4]
    Fdim = int(feats_cfg.get("num_features", len(use_cols)))
    model_cfg = cfg.get("model", {}) or {}

    b = cfg.get("features", {}).get("build", {})
    pos_dim = _position_feature_dim(cfg)
    pos_mode = _position_feature_mode(cfg)
    boundary_distance_start_col = Fdim if pos_mode == "boundary_distances" else None
    in_ch = Fdim + pos_dim + (1 if b.get("use_level", True) else 0)
    layout_cursor = Fdim + pos_dim
    if b.get("use_level", True):
        layout_cursor += 1
    if _include_dt_hat_from_build_cfg(b):
        in_ch += 1
        layout_cursor += 1
    if bool(b.get("include_ramp_angle", False)):
        in_ch += 1
        layout_cursor += 1
    ramp_signed_distance_col = None
    if bool(b.get("include_signed_distance_to_ramp", False)):
        ramp_signed_distance_col = layout_cursor
        in_ch += 1
        layout_cursor += 1

    loss = cfg.get("loss", {}) or {}
    parc_on = _physics_inputs_active_from_loss_cfg(loss)
    if parc_on:
        in_ch += dec.parc_extra_in_channels(cfg, Fdim)

    out_ch = Fdim

    predict_type = _normalize_predict_type_key(
        model_cfg.get("predict_type", model_cfg.get("target_mode", "rate"))
    )
    model_cfg["predict_type"] = predict_type

    raw_conv_type = model_cfg.get("conv_type", None)
    model_type = _resolve_model_type_key(model_cfg)

    conv_type = str(raw_conv_type if raw_conv_type is not None else "sage").strip().lower().replace("-", "_")
    if conv_type == "sageconv":
        conv_type = "sage"
    if model_type == "sageconv":
        conv_type = "sage"
    elif model_type in {"meshgraphnet", "fluxgraphnet"}:
        conv_type = "sage"
    elif conv_type not in {"sage", "gine", "nnconv"}:
        raise ValueError(
            f"Unsupported model.conv_type '{raw_conv_type}'. "
            "Use one of: sage, gine, nnconv."
        )

    edge_dim = int(model_cfg.get("edge_dim", 5)) if conv_type in ("gine", "nnconv") else None
    edge_attr_channels_raw = model_cfg.get("edge_attr_channels", model_cfg.get("edge_dim", 5))
    if edge_attr_channels_raw is None:
        edge_attr_channels_raw = 5
    edge_attr_channels = int(edge_attr_channels_raw)
    nnconv_hidden = int(model_cfg.get("nnconv_hidden", model_cfg.get("hidden", 128)))
    att_cfg = model_cfg.get("attention", {}) or {}
    use_attention = bool(model_cfg.get("use_attention", att_cfg.get("enabled", False)))
    attention_heads = int(model_cfg.get("attention_heads", att_cfg.get("heads", 4)))
    attention_dropout = float(model_cfg.get("attention_dropout", att_cfg.get("dropout", 0.0)))
    attention_replace_last = bool(
        model_cfg.get("attention_replace_last", att_cfg.get("replace_last_layer", True))
    )
    attention_use_edge_attr = bool(
        model_cfg.get("attention_use_edge_attr", att_cfg.get("use_edge_attr", True))
    )
    attention_edge_dim = model_cfg.get(
        "attention_edge_dim",
        att_cfg.get("edge_dim", model_cfg.get("edge_dim", 5)),
    )
    attention_edge_dim = None if attention_edge_dim is None else int(attention_edge_dim)
    activation = str(model_cfg.get("activation", model_cfg.get("nonlinearity", "relu")))
    activation_negative_slope = float(
        model_cfg.get(
            "activation_negative_slope",
            model_cfg.get("leaky_relu_negative_slope", 0.01),
        )
    )
    activation_elu_alpha = float(
        model_cfg.get("activation_elu_alpha", model_cfg.get("elu_alpha", 1.0))
    )

    if model_type == "sageconv":
        model = SAGEConvModel(
            in_channels=in_ch,
            out_channels=out_ch,
            state_channel=int(model_cfg.get("state_channel", 0)),
            hidden=int(model_cfg.get("hidden", 128)),
            layers=int(model_cfg.get("layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            use_skip=bool(model_cfg.get("use_skip", True)),
            use_layernorm=bool(model_cfg.get("use_layernorm", False)),
            layernorm_eps=float(model_cfg.get("layernorm_eps", 1e-6)),
            predict_type=predict_type,
        ).to(device)
    elif model_type == "meshgraphnet":
        model = MeshGraphNetModel(
            in_channels=in_ch,
            out_channels=out_ch,
            state_channel=int(model_cfg.get("state_channel", 0)),
            edge_attr_channels=edge_attr_channels,
            hidden=int(model_cfg.get("hidden", 128)),
            layers=int(model_cfg.get("layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            use_layernorm=bool(model_cfg.get("use_layernorm", False)),
            layernorm_eps=float(model_cfg.get("layernorm_eps", 1e-6)),
            predict_type=predict_type,
        ).to(device)
    elif model_type == "fluxgraphnet":
        idx = dec.infer_feature_indices(cfg, Fdim)
        state_rep = dec.state_representation_from_cfg(cfg, Fdim)
        open_boundary_modes_by_side = model_cfg.get("open_boundary_modes_by_side", None)
        open_boundary_flux_sides_default = (
            None if open_boundary_modes_by_side is not None else ["left", "right", "top"]
        )

        ramp_normal = model_cfg.get("ramp_normal", None)
        if ramp_normal is None and isinstance(ramp_feature_ctx, dict):
            ramp_normal = ramp_feature_ctx.get("distance_normal", None)
        if ramp_normal is None:
            angle_deg = _to_scalar_float(
                b.get("ramp_angle_deg", b.get("ramp_angle_degrees", None))
            )
            if angle_deg is not None:
                theta = math.radians(float(angle_deg))
                ramp_normal = [-math.sin(theta), math.cos(theta)]

        model = FluxGraphNetModel(
            in_channels=in_ch,
            out_channels=out_ch,
            state_channel=int(model_cfg.get("state_channel", 0)),
            edge_attr_channels=edge_attr_channels,
            hidden=int(model_cfg.get("hidden", 128)),
            layers=int(model_cfg.get("layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
            use_layernorm=bool(model_cfg.get("use_layernorm", False)),
            layernorm_eps=float(model_cfg.get("layernorm_eps", 1e-6)),
            predict_type=predict_type,
            state_representation=str(model_cfg.get("state_representation", state_rep)),
            u_index=int(idx.get("u", idx.get("mx", 0))),
            v_index=int(idx.get("v", idx.get("my", 1))),
            rho_index=int(idx.get("rho", 2 if Fdim >= 5 else 0)),
            p_index=idx.get("p", None),
            energy_index=int(idx.get("E", min(Fdim - 1, 4))),
            pressure_prediction_mode=str(model_cfg.get("pressure_prediction_mode", "eos")),
            gamma=float(model_cfg.get("gamma", dec.gas_gamma_from_cfg(cfg))),
            rho_floor=float(model_cfg.get("rho_floor", loss.get("rho_eps", 1e-8))),
            e_floor=float(model_cfg.get("e_floor", loss.get("E_floor", 1e-8))),
            p_floor=float(model_cfg.get("p_floor", loss.get("p_floor", 1e-8))),
            velocity_clip=float(model_cfg.get("velocity_clip", loss.get("u_clip", 0.0))),
            signed_edge_channels=model_cfg.get("signed_edge_channels", [0, 1]),
            fallback_directed_flux=bool(model_cfg.get("fallback_directed_flux", False)),
            use_open_boundary_source=bool(model_cfg.get("use_open_boundary_source", True)),
            open_boundary_mode=str(model_cfg.get("open_boundary_mode", "learned_source")),
            open_boundary_modes_by_side=open_boundary_modes_by_side,
            use_ramp_boundary_source=bool(model_cfg.get("use_ramp_boundary_source", True)),
            open_boundary_source_channels=model_cfg.get(
                "open_boundary_source_channels",
                [0, 1, 2, 3],
            ),
            open_boundary_flux_sides=model_cfg.get(
                "open_boundary_flux_sides",
                open_boundary_flux_sides_default,
            ),
            open_boundary_flux_scale=float(model_cfg.get("open_boundary_flux_scale", 0.05)),
            open_boundary_flux_outflow_only=bool(model_cfg.get("open_boundary_flux_outflow_only", True)),
            open_boundary_flux_include_pressure=bool(model_cfg.get("open_boundary_flux_include_pressure", False)),
            ramp_boundary_source_channels=model_cfg.get(
                "ramp_boundary_source_channels",
                [1, 2],
            ),
            boundary_distance_start_col=boundary_distance_start_col,
            boundary_distance_dim=4,
            boundary_width=float(model_cfg.get("boundary_width", 0.02)),
            ramp_signed_distance_col=ramp_signed_distance_col,
            ramp_boundary_width=(
                None
                if model_cfg.get("ramp_boundary_width", None) is None
                else float(model_cfg.get("ramp_boundary_width"))
            ),
            ramp_normal=ramp_normal,
            ramp_pressure_source_weight=float(model_cfg.get("ramp_pressure_source_weight", 0.0)),
        ).to(device)
    else:
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
            use_attention=use_attention,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            attention_edge_dim=attention_edge_dim,
            attention_use_edge_attr=attention_use_edge_attr,
            attention_replace_last=attention_replace_last,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            activation_elu_alpha=activation_elu_alpha,
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

def main(
    config_path: str | None = None,
    out_dir: str | None = None,
    resume_from: str | None = None,
):
    # -------- load & normalize config --------
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config_feature_first.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    if out_dir is not None:
        out_dir_abs = os.path.abspath(os.path.expanduser(str(out_dir)))
        cfg.setdefault("train", {})["save_dir"] = out_dir_abs
        print(f"[CLI] Overriding train.save_dir -> {out_dir_abs}")

    # Cartesian project contract:
    # - fixed uniform mesh (no AMR hierarchy)
    # - level depth is forced to zero regardless of config defaults
    data_cfg = cfg.setdefault("data", {})
    policy_cfg = cfg.setdefault("policy", {})
    eval_cfg = cfg.setdefault("eval", {})
    features_cfg = cfg.setdefault("features", {})
    norm_cfg = features_cfg.setdefault("normalization", {})
    data_cfg["L_max"] = 0
    policy_cfg["max_level"] = 0
    policy_cfg["starting_refine_to_level"] = 0
    eval_cfg["fine_L"] = 0
    print(
        "[CFG] fixed-mesh contract active: data.L_max=0, policy.max_level=0, eval.fine_L=0",
        flush=True,
    )

    momentum_sigma_mode = _resolve_momentum_sigma_mode(cfg)
    norm_cfg.setdefault(
        "_comment_momentum_sigma_mode",
        "independent (per-channel std) or shared_rms (tie x/y momentum sigmas by RMS).",
    )
    norm_cfg["momentum_sigma_mode"] = momentum_sigma_mode
    print(f"[NORM-CFG] momentum_sigma_mode={momentum_sigma_mode}")

    # Feature-builder defaults
    features_cfg = cfg.setdefault("features", {})
    build_cfg = features_cfg.setdefault("build", {})
    if not isinstance(build_cfg, dict):
        raise ValueError("features.build must be a JSON object when provided.")
    model_type_for_defaults = _resolve_model_type_key(cfg.get("model", {}) or {})
    fluxgraphnet_defaults = model_type_for_defaults == "fluxgraphnet"
    build_cfg.setdefault("use_pos", True)
    if "position_mode" not in build_cfg:
        build_cfg["position_mode"] = "boundary_distances" if fluxgraphnet_defaults else "xy"
    build_cfg.setdefault("boundary_distance_normalize", True)
    build_cfg.setdefault("boundary_distance_clip", None)
    build_cfg.setdefault("boundary_distance_clip_normalize", True)
    # Cartesian project contract: no AMR-level feature channel.
    build_cfg["use_level"] = False
    if "include_dt_hat" not in build_cfg:
        build_cfg["include_dt_hat"] = _include_dt_hat_from_build_cfg(build_cfg)
    build_cfg.pop("use_dt_hat_fixed", None)
    # Optional geometry channels (default-off to preserve current behavior)
    build_cfg.setdefault("include_ramp_angle", False)
    build_cfg.setdefault("ramp_angle_deg", None)
    build_cfg.setdefault("ramp_angle_units", "radians")
    if "include_signed_distance_to_ramp" not in build_cfg:
        build_cfg["include_signed_distance_to_ramp"] = bool(fluxgraphnet_defaults)
    build_cfg.setdefault("signed_distance_low_y_quantile", 0.12)
    build_cfg.setdefault("signed_distance_min_points", 32)
    build_cfg.setdefault("signed_distance_normalize", True)

    if fluxgraphnet_defaults:
        model_defaults = cfg.setdefault("model", {})
        model_defaults.setdefault("predict_type", "rate")
        model_defaults.setdefault("use_open_boundary_source", True)
        model_defaults.setdefault("open_boundary_mode", "learned_source")
        model_defaults.setdefault("open_boundary_modes_by_side", None)
        model_defaults.setdefault("use_ramp_boundary_source", True)
        model_defaults.setdefault("open_boundary_source_channels", [0, 1, 2, 3])
        if "open_boundary_flux_sides" not in model_defaults:
            model_defaults["open_boundary_flux_sides"] = (
                None
                if model_defaults.get("open_boundary_modes_by_side", None) is not None
                else ["left", "right", "top"]
            )
        model_defaults.setdefault("open_boundary_flux_scale", 0.05)
        model_defaults.setdefault("open_boundary_flux_outflow_only", True)
        model_defaults.setdefault("open_boundary_flux_include_pressure", False)
        model_defaults.setdefault("ramp_boundary_source_channels", [1, 2])
        model_defaults.setdefault("boundary_width", 0.02)
        model_defaults.setdefault("pressure_prediction_mode", "eos")

    model_cfg = cfg.setdefault("model", {})
    predict_type = _normalize_predict_type_key(
        model_cfg.get("predict_type", model_cfg.get("target_mode", "rate"))
    )
    model_cfg["predict_type"] = predict_type

    use_cols_cfg = features_cfg.get("use_columns", None)
    if isinstance(use_cols_cfg, (list, tuple)) and len(use_cols_cfg) > 0:
        f_guess = int(len(use_cols_cfg))
    else:
        f_guess = int(features_cfg.get("num_features", 4))
    idx = dec.infer_feature_indices(cfg, f_guess)
    rep = dec.state_representation_from_cfg(cfg, f_guess)
    print(f"[IDX] rep={rep} map={idx}")

    loss = cfg.setdefault("loss", {})
    if fluxgraphnet_defaults:
        loss.setdefault("advection_type", "euler")
        loss.setdefault("pressure_aux_weight", 0.05)
        loss.setdefault("pressure_aux_log", True)
        loss.setdefault("pressure_aux_huber", True)
        loss.setdefault("pressure_aux_huber_delta", 0.1)
        loss.setdefault("pressure_consistency_weight", 0.01)

    loss.setdefault("mode", "diffusion")           # "diffusion" | "advection" | "advdiff"
    loss.setdefault("blend_weight", 0.0)
    loss.setdefault("residual_weight", 0.0)

    loss.setdefault("nu", 0.0)
    loss.setdefault("advection_scheme", "upwind")
    loss.setdefault("advection_type", "scalar")
    loss.setdefault(
        "euler_flux_pressure_source",
        "channel" if rep == "primitive_uvrhope" and idx.get("p", None) is not None else "eos",
    )
    loss.setdefault("rho_eps", 1e-8)

    # Physics-input controls (authoritative):
    #   - physics_inputs_enabled: master on/off for physics-derived GNN input channels
    #   - physics_backend: operator backend used when physics inputs are enabled
    loss.setdefault("physics_inputs_enabled", True)
    loss.setdefault("physics_backend", "dec")
    loss.setdefault("parc_input_form", "rate")     # "rate" (dt_ref*r/sigma) or "delta" (dt*r/sigma)
    loss.setdefault("parc_input_time_scale", "model")  # "model" preserves legacy dt_ref/dt scaling; "unit" uses r/sigma
    loss.setdefault("parc_input_scale_mode", "none")   # "robust" calibrates scale-only PARC factors on the train loader
    loss.setdefault("parc_include_adv", True)
    loss.setdefault("parc_include_diff", True)
    loss.setdefault("parc_input_weighted", False)  # usually False
    loss.setdefault("parc_detach_inputs", True)

    if rep == "primitive_uvrhope":
        loss.setdefault("channels", ["U", "V", "RHO", "P", "E"])
        loss.setdefault("adv_channels", ["U", "V", "RHO", "P", "E"])
        loss.setdefault("diff_channels", ["RHO", "P", "E"])
    else:
        loss.setdefault("channels", ["rho", "x_momentum", "y_momentum", "E"])

    backend_mode = str(loss.get("physics_backend", "dec")).strip().lower()
    backend_alias = {
        "dec": "dec",
        "mls": "mls",
        "moving_least_squares": "mls",
        "moving-least-squares": "mls",
    }
    if backend_mode not in backend_alias:
        raise ValueError(
            "loss.physics_backend must be one of {dec, mls, moving_least_squares, moving-least-squares}. "
            f"Got: {loss.get('physics_backend')!r}"
        )
    loss["physics_backend"] = backend_alias[backend_mode]

    advection_type = str(loss.get("advection_type", "scalar")).strip().lower()
    if advection_type not in ("scalar", "euler"):
        raise ValueError(
            "loss.advection_type must be one of {scalar, euler}. "
            f"Got: {loss.get('advection_type')!r}"
        )
    loss["advection_type"] = advection_type

    if (not _physics_inputs_active_from_loss_cfg(loss)) and (
        float(loss.get("blend_weight", 0.0)) != 0.0 or float(loss.get("residual_weight", 0.0)) != 0.0
    ):
        print(
            "[PHYSICS-CFG][WARN] physics_inputs_enabled=false disables physics operators; "
            "forcing blend_weight=0 and residual_weight=0.",
            flush=True,
        )
        loss["blend_weight"] = 0.0
        loss["residual_weight"] = 0.0

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

    print("[PHYSICS-CFG] backend=", loss.get("physics_backend"),
      "inputs_enabled=", int(_physics_inputs_active_from_loss_cfg(loss)),
      "advection_type=", loss.get("advection_type"),
      "euler_p=", loss.get("euler_flux_pressure_source"),
      "mode=", loss.get("mode"),
      "adv_w=", loss.get("adv_weight"),
      "diff_w=", loss.get("diff_weight"),
      "nu=", loss.get("nu"),
      "resid_w=", loss.get("residual_weight"),
      "blend_w=", loss.get("blend_weight"),
      "channels=", loss.get("channels"))
    print(
        "[PHYSICS-INPUTS]",
        "enabled=",
        int(_physics_inputs_active_from_loss_cfg(loss)),
        "include_adv=",
        int(bool(loss.get("parc_include_adv", True))),
        "include_diff=",
        int(bool(loss.get("parc_include_diff", True))),
        "time_scale=",
        loss.get("parc_input_time_scale"),
        "scale_mode=",
        loss.get("parc_input_scale_mode"),
    )

    # Speed defaults (non‑breaking):
    cfg.setdefault("speed", {}).setdefault("amp", True)
    cfg.setdefault("speed", {}).setdefault("interp_chunk", 8192)
    cfg.setdefault("speed", {}).setdefault("knn_k", 8)
    cfg.setdefault("speed", {}).setdefault("cache_interps", True)

    train_cfg = cfg.setdefault("train", {})
    train_cfg.setdefault("validation_every_epochs", 1)
    train_cfg.setdefault("checkpoint_every_epochs", 1)
    lr_schedule_cfg = train_cfg.setdefault("lr_schedule", {})
    if not isinstance(lr_schedule_cfg, dict):
        raise ValueError("train.lr_schedule must be a JSON object when provided.")
    lr_schedule_cfg.setdefault("enabled", False)
    lr_schedule_cfg.setdefault("entries", [])
    train_cfg.setdefault("mesh_mode", "uniform")
    train_cfg.setdefault("use_precompute", False)
    mesh_mode = str(train_cfg.get("mesh_mode", "uniform")).strip().lower()
    train_cfg["mesh_mode"] = mesh_mode

    # Runtime mesh is intentionally disabled in this Cartesian project.
    runtime_mesh_cfg = train_cfg.get("runtime_mesh", {})
    if isinstance(runtime_mesh_cfg, bool):
        runtime_mesh_cfg = {"enabled": bool(runtime_mesh_cfg)}
    if not isinstance(runtime_mesh_cfg, dict):
        raise ValueError("train.runtime_mesh must be a JSON object or bool when provided.")
    runtime_mesh_cfg["enabled"] = False
    train_cfg["runtime_mesh"] = runtime_mesh_cfg
    chunk_cfg = cfg.setdefault("chunk", {})
    if not isinstance(chunk_cfg, dict):
        raise ValueError("chunk must be a JSON object when provided.")
    chunk_cfg.setdefault("enabled", False)
    chunk_cfg.setdefault("num_chunks", 16)
    chunk_cfg.setdefault("halo_hops", 2)
    chunk_cfg.setdefault("partition_mode", "parent_grid")
    chunk_cfg.setdefault("activity_threshold", 0.0)
    chunk_builder_cfg = chunk_cfg.setdefault("builder", {})
    if not isinstance(chunk_builder_cfg, dict):
        raise ValueError("chunk.builder must be a JSON object when provided.")
    chunk_builder_cfg.setdefault("enabled", False)
    chunk_builder_cfg.setdefault("sidecar_path", "")
    chunk_builder_cfg.setdefault("overwrite", False)
    chunk_builder_cfg.setdefault("progress", True)
    chunk_builder_cfg.setdefault("compute_activity", True)
    chunk_builder_cfg.setdefault("activity_feature_idx", [])
    chunk_builder_cfg.setdefault("max_timesteps", None)
    debug_cfg = cfg.setdefault("debug", {})
    if not isinstance(debug_cfg, dict):
        raise ValueError("debug must be a JSON object when provided.")
    debug_cfg.setdefault("print_chunk_stats", False)
    debug_cfg.setdefault("chunk_stats_every_steps", 1)
    debug_cfg.setdefault("chunk_stats_include_rk4", False)
    debug_cfg.setdefault("sync_runtime_timing", False)
    diag_cfg_main = _resolve_diagnostics_cfg(cfg)

    _enforce_cartesian_project_contract(cfg)
    runtime_mesh_enabled = _runtime_mesh_enabled_from_cfg(cfg)

    # Ensure dataset sees the intended columns
    if cfg.get("data", {}).get("feature_idx") and not cfg.get("features", {}).get("use_columns"):
        cfg.setdefault("features", {}).setdefault("use_columns", cfg["data"]["feature_idx"])

    #device = pick_device(cfg.get("train", {}).get("device", "auto"))
    raw_dev = cfg.get("device", "cpu")
    device = torch.device(raw_dev)
    _print_training_device_report(cfg, device)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    H = int(cfg["data"].get("H", 64)); W = int(cfg["data"].get("W", 64))
    xmin, xmax, ymin, ymax = _get_bbox(cfg)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H

    runtime_mesh_policy = None

    # -------- dataset sources --------
    pt_paths = _resolve_pt_path_list(cfg)
    cfg.setdefault("data", {})["pt_paths"] = pt_paths
    if len(pt_paths) == 1:
        cfg["data"]["pt_path"] = pt_paths[0]
    else:
        cfg["data"]["pt_path"] = pt_paths

    mesh_spec_root_cfg = cfg.get("mesh", {}).get("starting_mesh_path", None)
    mesh_spec_root_abs = (
        os.path.abspath(os.path.expanduser(str(mesh_spec_root_cfg)))
        if mesh_spec_root_cfg is not None
        else None
    )

    source_records: List[Dict[str, Any]] = []
    data_list: List[Any] = []
    step_source_ids: List[int] = []
    step_mesh_spec_paths: List[str | None] = []

    for sid, p_in in enumerate(pt_paths):
        p_abs, seq = _load_series_from_pt_path(p_in)
        mesh_spec_for_src = _resolve_mesh_spec_path_for_input(
            mesh_spec_root_abs,
            input_pt_path=p_abs,
        ) if (mesh_spec_root_abs is not None) else None

        angle_src = _extract_angle_from_data_obj(seq)
        if angle_src is None:
            angle_src = _extract_angle_from_path(p_abs)
        if angle_src is None and mesh_spec_for_src is not None:
            angle_src = _extract_angle_from_path(mesh_spec_for_src)
        pressure_src = _extract_pressure_from_path(p_abs)

        source_records.append(
            {
                "source_id": int(sid),
                "pt_path": p_abs,
                "series": seq,
                "mesh_spec_path": mesh_spec_for_src,
                "angle_deg": angle_src,
                "pressure": pressure_src,
            }
        )

        data_list.extend(seq)
        step_source_ids.extend([int(sid)] * len(seq))
        step_mesh_spec_paths.extend([mesh_spec_for_src] * len(seq))

    if len(data_list) < 2:
        raise RuntimeError("Need at least 2 snapshots total across input files.")

    from_h5_inputs = any(str(p).lower().endswith((".h5", ".hdf5")) for p in pt_paths)
    if from_h5_inputs and isinstance(data_list[0], dict):
        first_snap = data_list[0]

        # Auto-wire channel names for primitive-state semantics.
        ch_names = first_snap.get("channel_names", None)
        if isinstance(ch_names, (list, tuple)) and len(ch_names) > 0:
            cfg.setdefault("features", {})["names"] = [str(v) for v in ch_names]
            if not cfg.get("features", {}).get("use_columns"):
                cfg["features"]["use_columns"] = list(range(len(ch_names)))
            cfg["features"]["num_features"] = int(len(ch_names))
            print(f"[DATA] HDF5 channels -> features.names: {cfg['features']['names']}")

        # Auto-sync H/W with the file payload when present.
        h_file = first_snap.get("H", None)
        w_file = first_snap.get("W", None)
        if h_file is not None and w_file is not None:
            h_file = int(h_file); w_file = int(w_file)
            if h_file != H or w_file != W:
                print(
                    f"[DATA][WARN] Overriding cfg data.H/W ({H},{W}) with HDF5 payload ({h_file},{w_file})."
                )
            H, W = h_file, w_file
            cfg.setdefault("data", {})["H"] = int(H)
            cfg.setdefault("data", {})["W"] = int(W)

        # Auto-sync bbox from xy/pos payload.
        pos0 = first_snap.get("pos", None)
        if torch.is_tensor(pos0) and pos0.ndim == 2 and pos0.size(1) >= 2 and pos0.numel() > 0:
            xmin, xmax, ymin, ymax, dx_center, dy_center = _cell_edge_bbox_from_centers(
                pos0,
                H=int(H),
                W=int(W),
            )
            cfg.setdefault("data", {})["bbox"] = [xmin, xmax, ymin, ymax]
            dx = (xmax - xmin) / float(W)
            dy = (ymax - ymin) / float(H)
            print(
                "[DATA] HDF5 center coordinates -> cell-edge "
                f"data.bbox=[{xmin:.6g}, {xmax:.6g}, {ymin:.6g}, {ymax:.6g}] "
                f"(center_spacing=({dx_center:.6g},{dy_center:.6g}), dx/dy=({dx:.6g},{dy:.6g}))"
            )

    if len(source_records) > 1:
        print(f"[DATA] Loaded {len(source_records)} input simulation files (total snapshots={len(data_list)}).")
    for rec in source_records:
        print(
            f"[DATA] source_id={rec['source_id']} snapshots={len(rec['series'])} "
            f"pt={rec['pt_path']} mesh_spec={rec['mesh_spec_path']}"
        )

    unique_mesh_specs = sorted({p for p in step_mesh_spec_paths if p})
    if len(unique_mesh_specs) == 1:
        cfg.setdefault("mesh", {})["starting_mesh_path"] = unique_mesh_specs[0]
    if (mesh_mode == "uniform") and (len(unique_mesh_specs) > 1):
        raise RuntimeError(
            "train.mesh_mode='uniform' with multiple input files currently requires one shared mesh spec file. "
            f"Resolved multiple specs: {unique_mesh_specs}"
        )

    if runtime_mesh_enabled:
        raise RuntimeError("Internal error: runtime mesh must remain disabled in Cartesian mode.")

    ramp_feature_ctx = _build_ramp_feature_context(
        cfg,
        data_obj=data_list,
        pt_path=source_records[0]["pt_path"],
        mesh_spec_path=(
            unique_mesh_specs[0]
            if len(unique_mesh_specs) == 1
            else mesh_spec_root_abs
        ),
    )

    include_ramp_angle = bool(ramp_feature_ctx.get("include_ramp_angle", False))
    include_signed_dist = bool(ramp_feature_ctx.get("include_signed_distance_to_ramp", False))
    if len(source_records) > 1 and (include_ramp_angle or include_signed_dist):
        angle_by_source: Dict[int, float] = {}
        dist_by_source: Dict[int, Dict[str, Any]] = {}
        angle_units = str(ramp_feature_ctx.get("angle_units", "radians")).lower()
        for rec in source_records:
            sid = int(rec["source_id"])
            ctx_i = _build_ramp_feature_context(
                cfg,
                data_obj=rec["series"],
                pt_path=rec["pt_path"],
                mesh_spec_path=rec["mesh_spec_path"],
            )
            if include_ramp_angle:
                aval = ctx_i.get("angle_feature_value", None)
                if aval is None:
                    raise RuntimeError(
                        f"Could not build per-source ramp angle feature for source_id={sid} ({rec['pt_path']})."
                    )
                angle_by_source[sid] = float(aval)
                if "angle_deg" in ctx_i:
                    angle_deg_by_source = ramp_feature_ctx.setdefault("angle_deg_by_source", {})
                    angle_deg_by_source[sid] = float(ctx_i["angle_deg"])
            if include_signed_dist:
                p0 = ctx_i.get("distance_point", None)
                n = ctx_i.get("distance_normal", None)
                sc = float(ctx_i.get("distance_scale", 1.0))
                if (p0 is None) or (n is None):
                    raise RuntimeError(
                        f"Could not build per-source signed-distance feature for source_id={sid} ({rec['pt_path']})."
                    )
                dist_by_source[sid] = {"point": p0, "normal": n, "scale": sc}

        ramp_feature_ctx["source_id_by_t"] = [int(v) for v in step_source_ids]
        if include_ramp_angle:
            ramp_feature_ctx["angle_feature_by_source"] = angle_by_source
            ramp_feature_ctx["angle_units"] = angle_units
        if include_signed_dist:
            ramp_feature_ctx["distance_by_source"] = dist_by_source

    if bool(ramp_feature_ctx.get("include_ramp_angle", False)):
        angle_units = str(ramp_feature_ctx.get("angle_units", "radians"))
        angle_feature_by_source = ramp_feature_ctx.get("angle_feature_by_source", None)
        angle_deg_by_source = ramp_feature_ctx.get("angle_deg_by_source", None)
        if isinstance(angle_feature_by_source, dict) and len(angle_feature_by_source) > 0:
            source_parts = []
            for sid in sorted(int(k) for k in angle_feature_by_source.keys())[:12]:
                feat_val = float(angle_feature_by_source.get(sid, angle_feature_by_source.get(str(sid))))
                deg_val = None
                if isinstance(angle_deg_by_source, dict):
                    deg_val = angle_deg_by_source.get(sid, angle_deg_by_source.get(str(sid), None))
                if deg_val is None:
                    deg_val = float(np.rad2deg(feat_val)) if angle_units == "radians" else feat_val
                source_parts.append(f"{sid}:deg={float(deg_val):.6g},feature={feat_val:.6g}")
            suffix = ""
            if len(angle_feature_by_source) > 12:
                suffix = f", ... ({len(angle_feature_by_source)} sources total)"
            print(
                "[GEOM-FEAT] include_ramp_angle=1 "
                f"units={angle_units} per-source [{'; '.join(source_parts)}{suffix}]",
                flush=True,
            )
        else:
            feature_val = float(ramp_feature_ctx.get("angle_feature_value"))
            deg_val = float(ramp_feature_ctx.get("angle_deg"))
            print(
                "[GEOM-FEAT] include_ramp_angle=1 "
                f"deg={deg_val:.6g}, feature={feature_val:.6g}, units={angle_units}",
                flush=True,
            )
    else:
        print(
            "[GEOM-FEAT] include_ramp_angle=0 "
            "(ramp angle will not be appended to node inputs)",
            flush=True,
        )
    if bool(ramp_feature_ctx.get("include_signed_distance_to_ramp", False)):
        print(
            "[GEOM-FEAT] include_signed_distance_to_ramp=1 "
            f"(distance_scale={float(ramp_feature_ctx.get('distance_scale', 1.0)):.6g})"
        )

    model = build_model_from_cfg(cfg, device, ramp_feature_ctx=ramp_feature_ctx)

    # After model is created, before optimizer is created:
    loss_cfg = cfg.get("loss", {}) or {}
    parc_use = _physics_inputs_active_from_loss_cfg(loss_cfg)
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
    lr_schedule_entries = _parse_lr_schedule_entries(cfg.get("train", {}) or {})
    if lr_schedule_entries:
        parts = ", ".join(f"epoch {ep}->lr {lr:.3e}" for ep, lr in lr_schedule_entries)
        print(f"[LR-SCHEDULE] enabled: {parts}", flush=True)

    # 3) Now create the scheduler (safe: param_groups size is final)
    sch_cfg   = cfg.get("scheduler", {})
    use_sched = bool(sch_cfg.get("use", True))
    if lr_schedule_entries and use_sched:
        print(
            "[LR-SCHEDULE] train.lr_schedule is enabled; disabling ReduceLROnPlateau "
            "from the top-level scheduler block for this run.",
            flush=True,
        )
        use_sched = False
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

    resume_ckpt = None
    resume_epoch = 0
    resume_norm_stats = None
    resume_active = False
    resume_path = None
    if resume_from is not None and str(resume_from).strip() != "":
        resume_active = True
        resume_raw = str(resume_from).strip()
        if resume_raw.lower() in ("auto", "last"):
            resume_path = os.path.join(cfg["train"]["save_dir"], "last_model.pt")
        else:
            resume_path = os.path.abspath(os.path.expanduser(resume_raw))
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"[RESUME] Loading checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(resume_ckpt, dict) and ("model" in resume_ckpt):
            model.load_state_dict(resume_ckpt["model"], strict=True)
            resume_epoch = int(resume_ckpt.get("epoch", 0))

            opt_state = resume_ckpt.get("optimizer", None)
            if isinstance(opt_state, dict):
                try:
                    opt.load_state_dict(opt_state)
                    print("[RESUME] Restored optimizer state.")
                except Exception as e:
                    print(f"[RESUME][WARN] Could not restore optimizer state: {e!r}")

            sch_state = resume_ckpt.get("scheduler", None)
            if (scheduler is not None) and isinstance(sch_state, dict):
                try:
                    scheduler.load_state_dict(sch_state)
                    print("[RESUME] Restored scheduler state.")
                except Exception as e:
                    print(f"[RESUME][WARN] Could not restore scheduler state: {e!r}")

            ns = resume_ckpt.get("norm_stats", None)
            if isinstance(ns, dict):
                resume_norm_stats = ns
            print(f"[RESUME] Loaded model at epoch={resume_epoch}.")
        elif isinstance(resume_ckpt, dict):
            # fallback: plain state_dict checkpoint
            model.load_state_dict(resume_ckpt, strict=True)
            print("[RESUME][WARN] Checkpoint has no training-state payload; restored model weights only.")
        else:
            raise RuntimeError(
                f"Unsupported resume checkpoint payload type: {type(resume_ckpt)}"
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
        step_source_ids = step_source_ids[start : end + 1]
        step_mesh_spec_paths = step_mesh_spec_paths[start : end + 1]
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
        if "source_id_by_t" in ramp_feature_ctx:
            src_map = ramp_feature_ctx.get("source_id_by_t", [])
            if isinstance(src_map, (list, tuple)) and len(src_map) >= (end + 1):
                ramp_feature_ctx["source_id_by_t"] = [int(v) for v in src_map[start : end + 1]]

    dt_transitions, dt_ref = _build_dt_transitions_from_cfg(
        cfg,
        data_list,
        step_source_ids=step_source_ids,
        source_records=source_records,
    )

    full_ds = CellRefineWindowDataset(
        series=data_list,       # <-- pass the list directly
        cfg=cfg,
        window_size=K,
        stride=stride,
        H=H, W=W, device=str(device)
        # is_processed_file can be left None; passing raw list triggers in-memory preprocess
    )

    if len(full_ds.steps) != len(step_source_ids):
        raise RuntimeError(
            f"Internal mismatch: full_ds.steps={len(full_ds.steps)} vs source-id map={len(step_source_ids)}."
        )
    src_ids_np = np.asarray(step_source_ids, dtype=np.int64)
    for ii, step in enumerate(full_ds.steps):
        if isinstance(step, dict):
            step["__source_id"] = int(step_source_ids[ii])
            step["__mesh_spec_path"] = step_mesh_spec_paths[ii]
            step["__source_pt_path"] = source_records[int(step_source_ids[ii])]["pt_path"]

    # Build contiguous source segments on the current (possibly sliced) timeline.
    source_segments: List[Dict[str, Any]] = []
    if len(step_source_ids) > 0:
        seg_start = 0
        seg_sid = int(step_source_ids[0])
        for i in range(1, len(step_source_ids)):
            sid_i = int(step_source_ids[i])
            if sid_i != seg_sid:
                source_segments.append(
                    {
                        "source_id": int(seg_sid),
                        "t_start": int(seg_start),
                        "t_end": int(i - 1),
                    }
                )
                seg_start = i
                seg_sid = sid_i
        source_segments.append(
            {
                "source_id": int(seg_sid),
                "t_start": int(seg_start),
                "t_end": int(len(step_source_ids) - 1),
            }
        )

    if len(source_segments) > 1:
        print(
            "[DATA] source segments:",
            [
                (int(s["source_id"]), int(s["t_start"]), int(s["t_end"]))
                for s in source_segments
            ],
        )

    # Filter out windows that cross source boundaries.
    n_windows_all = len(full_ds)
    all_win_idx = np.arange(n_windows_all, dtype=np.int64)
    t0_all = all_win_idx * int(stride)
    tlast_all = t0_all + (int(K) - 1)
    valid_mask = (src_ids_np[t0_all] == src_ids_np[tlast_all])
    valid_window_idx = all_win_idx[valid_mask]
    if valid_window_idx.size == 0:
        raise RuntimeError(
            "No valid windows remain after filtering cross-source boundaries. "
            "Check data.window_size/stride and input sequence lengths."
        )
    T = int(valid_window_idx.size)
    split_cfg = cfg.get("split", {}) or {}
    seed = int(cfg.get("seed", 1337))
    rng = np.random.default_rng(seed)

    by_file_raw = split_cfg.get("by_file", None)
    by_file = (
        ("val_files" in split_cfg) or ("test_files" in split_cfg)
        if by_file_raw is None
        else _cfg_bool_strict(by_file_raw, key="split.by_file")
    )

    if by_file:
        n_sources = int(len(source_records))
        val_files = int(split_cfg.get("val_files", 0))
        test_files = int(split_cfg.get("test_files", 0))
        if val_files < 0 or test_files < 0:
            raise ValueError(
                f"split.val_files and split.test_files must be >= 0, got "
                f"val_files={val_files}, test_files={test_files}"
            )
        if (val_files + test_files) >= n_sources:
            raise ValueError(
                f"split.val_files + split.test_files must be < number of files ({n_sources}) "
                "so at least one file remains for training."
            )

        source_ids = np.arange(n_sources, dtype=np.int64)
        source_ids = source_ids[rng.permutation(n_sources)]
        val_source_ids = source_ids[:val_files]
        test_source_ids = source_ids[val_files : (val_files + test_files)]
        train_source_ids = source_ids[(val_files + test_files) :]

        valid_t0 = (valid_window_idx * int(stride)).astype(np.int64)
        valid_win_src = src_ids_np[valid_t0]

        train_mask = np.isin(valid_win_src, train_source_ids)
        val_mask = np.isin(valid_win_src, val_source_ids)
        test_mask = np.isin(valid_win_src, test_source_ids)

        train_idx = valid_window_idx[train_mask]
        val_idx = valid_window_idx[val_mask]
        test_idx = valid_window_idx[test_mask]

        if train_idx.size == 0:
            raise RuntimeError(
                "File-level split produced zero training windows. "
                "Reduce split.val_files/split.test_files or use longer files."
            )

        if train_idx.size > 1:
            train_idx = train_idx[rng.permutation(train_idx.size)]
        if val_idx.size > 1:
            val_idx = val_idx[rng.permutation(val_idx.size)]
        if test_idx.size > 1:
            test_idx = test_idx[rng.permutation(test_idx.size)]

        print(
            f"[SPLIT] by_file=true seed={seed} "
            f"files(total/train/val/test)={n_sources}/{len(train_source_ids)}/{len(val_source_ids)}/{len(test_source_ids)} "
            f"windows(train/val/test)={int(train_idx.size)}/{int(val_idx.size)}/{int(test_idx.size)}"
        )
        print(
            f"[SPLIT] source_ids train={train_source_ids.tolist()} "
            f"val={val_source_ids.tolist()} test={test_source_ids.tolist()}"
        )
    else:
        idxs = valid_window_idx.copy()
        idxs = idxs[rng.permutation(T)]

        train_frac = float(split_cfg.get("train", 0.8))
        val_frac = float(split_cfg.get("val", 0.1))
        if train_frac < 0.0 or val_frac < 0.0 or (train_frac + val_frac) > 1.0:
            raise ValueError(
                f"split.train and split.val must be non-negative with train+val<=1. "
                f"Got train={train_frac}, val={val_frac}."
            )

        n_train = int(round(train_frac * T))
        n_val = int(round(val_frac * T))
        n_train = max(0, min(n_train, T))
        n_val = max(0, min(n_val, T - n_train))

        train_idx = idxs[:n_train]
        val_idx = idxs[n_train : n_train + n_val]
        test_idx = idxs[n_train + n_val :]

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    test_ds  = Subset(full_ds, test_idx.tolist())

    if len(source_records) > 1:
        counts_by_src = {}
        for wi in valid_window_idx.tolist():
            s0 = int(src_ids_np[int(wi) * int(stride)])
            counts_by_src[s0] = int(counts_by_src.get(s0, 0) + 1)
        print(
            f"[DATA] valid windows after boundary filtering: {int(T)} "
            f"(per source: {counts_by_src})"
        )

    precomp = None
    precomp_path_for_sidecar = ""
    use_precomp_collate = bool((cfg.get("train", {}) or {}).get("use_precompute", False))

    if use_precomp_collate:
        mesh_mode = str(cfg.get("train", {}).get("mesh_mode", "uniform")).strip().lower()
        if mesh_mode != "uniform":
            raise RuntimeError(
                "Cartesian project contract violation: train.mesh_mode must be 'uniform'. "
                f"Got {mesh_mode!r}."
            )
        force_recompute = bool(cfg["train"].get("precomp_force_recompute", False))
        cache_dir = cfg.get("train", {}).get("cache_dir", cfg.get("train", {}).get("save_dir", "."))

        def _build_precomp_for_steps(step_list_slice, *, cache_path_slice, progress_slice):
            return precompute_uniform_mesh_in_memory(
                step_list_slice,
                cfg,
                H,
                W,
                dx,
                dy,
                device=device,
                progress=progress_slice,
                cache_path=cache_path_slice,
                force_recompute=force_recompute,
                require_existing_cache_when_not_forced=False,
            )

        used_cache_paths: List[str] = []
        if len(source_segments) <= 1:
            cache_path = os.path.join(cache_dir, "precomp_uniform.h5")

            if mesh_mode == "uniform":
                print(f"[PRECOMP] train.mesh_mode=uniform: static-mesh precompute with cache_path={cache_path}")
            precomp = _build_precomp_for_steps(
                full_ds.steps,
                cache_path_slice=cache_path,
                progress_slice=True,
            )
            precomp = move_precomp_to_device(precomp, device="cpu")
            if cache_path:
                used_cache_paths.append(str(cache_path))
        else:
            print(
                f"[PRECOMP] Multi-source run detected ({len(source_segments)} segments); "
                "building/loading precompute per source-segment."
            )
            seg_views: List[Dict[str, Any]] = []
            for seg in source_segments:
                sid = int(seg["source_id"])
                s0 = int(seg["t_start"])
                s1 = int(seg["t_end"])
                cache_seg = None
                steps_seg = full_ds.steps[s0 : s1 + 1]
                if cache_seg is None:
                    # Keep deterministic, segment-specific default names to avoid collisions.
                    cache_seg = os.path.join(
                        cache_dir,
                        f"precomp_uniform_source-{sid}_t{s0:05d}-{s1:05d}.h5",
                    )

                print(
                    f"[PRECOMP][SEG] source_id={sid} t=[{s0},{s1}] "
                    f"len={len(steps_seg)} cache_path={cache_seg}"
                )
                pre_seg = _build_precomp_for_steps(
                    steps_seg,
                    cache_path_slice=cache_seg,
                    progress_slice=True,
                )
                pre_seg = move_precomp_to_device(pre_seg, device="cpu")
                seg_views.append(
                    {
                        "source_id": sid,
                        "t_start": s0,
                        "t_end": s1,
                        "precomp": pre_seg,
                    }
                )
                if cache_seg:
                    used_cache_paths.append(str(cache_seg))

            precomp = SegmentedPrecompView(
                segments=seg_views,
                T=len(full_ds.steps),
                H=H,
                W=W,
            )

        used_cache_paths = sorted({p for p in used_cache_paths if str(p).strip() != ""})
        if len(used_cache_paths) == 1:
            precomp_path_for_sidecar = used_cache_paths[0]
        elif len(used_cache_paths) > 1:
            precomp_path_for_sidecar = ""
            print(
                "[CHUNK][INFO] Multiple precomp cache files are active in this run; "
                "sidecar auto-load is disabled."
            )

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
    else:
        mesh_mode = str(cfg.get("train", {}).get("mesh_mode", "uniform")).strip().lower()
        if mesh_mode != "uniform":
            raise RuntimeError(
                "Cartesian project contract violation: train.mesh_mode must be 'uniform'. "
                f"Got {mesh_mode!r}."
            )
        print("[PRECOMP] disabled (train.use_precompute=false): using on-the-fly uniform static collate.")
        collate = CollateWithUniformStaticNoPrecomp(
            cfg=cfg,
            H=H,
            W=W,
            dx=dx,
            dy=dy,
            dt_transitions=dt_transitions,
            dt_ref=dt_ref,
            device="cpu",
        )

    data_cfg = cfg.get("data", {}) or {}
    loader_num_workers = int(data_cfg.get("num_workers", 0))
    if loader_num_workers < 0:
        raise ValueError(f"data.num_workers must be >= 0, got {loader_num_workers}.")
    loader_pin_memory = _cfg_bool_strict(
        data_cfg.get("pin_memory", (device.type == "cuda")),
        key="data.pin_memory",
    )
    loader_persistent_workers = _cfg_bool_strict(
        data_cfg.get("persistent_workers", (loader_num_workers > 0)),
        key="data.persistent_workers",
    ) if loader_num_workers > 0 else False
    print(
        "[DATA] DataLoader settings: "
        f"num_workers={loader_num_workers} "
        f"pin_memory={loader_pin_memory} "
        f"persistent_workers={loader_persistent_workers}",
        flush=True,
    )

    loader_common_kwargs: Dict[str, Any] = {
        "batch_size": 1,
        "num_workers": loader_num_workers,
        "pin_memory": loader_pin_memory,
        "collate_fn": collate,
    }
    if loader_num_workers > 0:
        loader_common_kwargs["persistent_workers"] = loader_persistent_workers

    train_loader = DataLoader(
        train_ds,
        sampler=RandomSampler(train_ds),
        **loader_common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        sampler=RandomSampler(val_ds),
        **loader_common_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        sampler=RandomSampler(test_ds),
        **loader_common_kwargs,
    )

    chunk_sidecar_reader = None
    chunk_cfg_main = cfg.get("chunk", {}) or {}
    if bool(chunk_cfg_main.get("enabled", False)):
        chunk_builder_cfg_main = chunk_cfg_main.get("builder", {}) or {}
        sidecar_path_cfg = str(chunk_builder_cfg_main.get("sidecar_path", "")).strip()
        if precomp_path_for_sidecar:
            sidecar_path = derive_sidecar_path(precomp_path_for_sidecar, sidecar_path_cfg)
            if os.path.exists(sidecar_path):
                try:
                    chunk_sidecar_reader = ChunkSidecarH5(sidecar_path)
                    print(f"[CHUNK] loaded sidecar: {sidecar_path}")
                except Exception as e:
                    print(f"[CHUNK][WARN] failed to open sidecar '{sidecar_path}': {e!r}. Falling back to on-the-fly chunking.")
            else:
                print(
                    f"[CHUNK][INFO] sidecar not found at '{sidecar_path}'. "
                    "Falling back to on-the-fly chunking."
                )
        else:
            print(
                "[CHUNK][INFO] precompute cache path was not resolved for this run; "
                "chunking will use on-the-fly partitioning."
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
    if do_norm and isinstance(resume_norm_stats, dict):
        mu_raw = resume_norm_stats.get("mu", None)
        sigma_raw = resume_norm_stats.get("sigma", None)
        if (mu_raw is not None) and (sigma_raw is not None):
            try:
                mu = torch.as_tensor(mu_raw, dtype=torch.float32, device=device).view(-1)
                sigma = torch.as_tensor(sigma_raw, dtype=torch.float32, device=device).view(-1)
                if mu.numel() != sigma.numel():
                    raise ValueError(
                        f"mu/sigma length mismatch ({mu.numel()} vs {sigma.numel()})"
                    )
                print(f"[RESUME] Reusing normalization stats from checkpoint (F={mu.numel()}).")
            except Exception as e:
                print(
                    "[RESUME][WARN] Could not use checkpoint norm_stats; will recompute from train loader: "
                    f"{e!r}"
                )
                mu = sigma = None

    if do_norm and (mu is None or sigma is None):
        cfg_norm = (feats_cfg.get("norm_stats", None) if isinstance(feats_cfg, dict) else None)
        if isinstance(cfg_norm, dict):
            mu_raw = cfg_norm.get("mu", None)
            sigma_raw = cfg_norm.get("sigma", None)
            if (mu_raw is not None) and (sigma_raw is not None):
                try:
                    mu_cfg = torch.as_tensor(mu_raw, dtype=torch.float32, device=device).view(-1)
                    sigma_cfg = torch.as_tensor(sigma_raw, dtype=torch.float32, device=device).view(-1)
                    if mu_cfg.numel() != sigma_cfg.numel():
                        raise ValueError(
                            f"mu/sigma length mismatch ({mu_cfg.numel()} vs {sigma_cfg.numel()})"
                        )
                    if (not torch.isfinite(mu_cfg).all().item()) or (not torch.isfinite(sigma_cfg).all().item()):
                        raise ValueError("mu/sigma contain non-finite values.")
                    sigma_cfg = sigma_cfg.clamp_min(1e-12)
                    mu, sigma = mu_cfg, sigma_cfg
                    print(f"[NORM] Reusing normalization stats from config (F={mu.numel()}).")
                except Exception as e:
                    print(
                        "[NORM][WARN] Could not use features.norm_stats from config; "
                        f"will recompute from train loader: {e!r}"
                    )
                    mu = sigma = None

    if do_norm and (mu is None or sigma is None):
        norm_progress_enabled = _cfg_bool_strict(
            cfg.get("train", {}).get("norm_stats_progress", True),
            key="train.norm_stats_progress",
        )
        norm_progress_every = max(
            1,
            int(cfg.get("train", {}).get("norm_stats_progress_every", 25)),
        )
        # compute from training loader on CPU/GPU (device already set)
        mu, sigma = _compute_norm_stats_from_loader(
            train_loader,
            device,
            momentum_sigma_mode=momentum_sigma_mode,
            momentum_x_idx=int(idx["mx"]),
            momentum_y_idx=int(idx["my"]),
            print_progress=norm_progress_enabled,
            progress_every=norm_progress_every,
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

    norm_setter = getattr(model, "set_normalization_stats", None)
    if callable(norm_setter):
        norm_setter(mu, sigma)
        print(
            f"[MODEL] synced normalization stats into {model.__class__.__name__} "
            f"(enabled={int(mu is not None and sigma is not None)})",
            flush=True,
        )

    # --- ABS really means ABS check (one batch) ---
    if cfg.get("debug", {}).get("abs_means_abs_check", True):
        batch0 = next(iter(train_loader))
        abs_means_abs_check(cfg=cfg, batch=batch0, mu=mu, sigma=sigma, device=device, tag="ABS-MEANS-ABS/TRAIN")

    if cfg.get("debug", {}).get("abs_means_abs_check_val", False):
        batchv = next(iter(val_loader))
        abs_means_abs_check(cfg=cfg, batch=batchv, mu=mu, sigma=sigma, device=device, tag="ABS-MEANS-ABS/VAL")

    _maybe_calibrate_parc_input_scales(
        loader=train_loader,
        cfg=cfg,
        device=device,
        dx=float(dx),
        dy=float(dy),
        sigma=sigma,
        predict_type=predict_type,
    )
    
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

    def _best_val_from_existing_log(path: str) -> float | None:
        if not os.path.exists(path):
            return None
        best = None
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("split", "")).strip().lower() != "val":
                        continue
                    raw = row.get("loss", None)
                    if raw in (None, ""):
                        continue
                    v = float(raw)
                    if not np.isfinite(v):
                        continue
                    best = v if (best is None or v < best) else best
        except Exception as e:
            print(f"[RESUME][WARN] Could not parse existing log for best val: {e!r}")
            return None
        return best

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

    log_exists = os.path.exists(log_csv)
    if resume_active and log_exists:
        log_mode = "a"
        write_header = (os.path.getsize(log_csv) == 0)
        print(f"[RESUME] Appending logs to existing file: {log_csv}")
    else:
        log_mode = "w"
        write_header = True
    with open(log_csv, log_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        if write_header:
            writer.writeheader()

    print("[INFO] Starting training...")
    total_epochs = int(cfg["train"]["epochs"])
    val_every = int(cfg.get("train", {}).get("validation_every_epochs", 1))
    if val_every < 1:
        raise ValueError("train.validation_every_epochs must be >= 1.")
    print(f"[INFO] Validation cadence: every {val_every} epoch(s) + final epoch.")

    start_epoch = int(resume_epoch) + 1 if resume_active else 1
    if start_epoch < 1:
        start_epoch = 1
    if start_epoch > total_epochs:
        print(
            f"[RESUME][WARN] start_epoch={start_epoch} is beyond total_epochs={total_epochs}. "
            "Skipping training loop and running final evaluation only."
        )
    if resume_active:
        prior_best = _best_val_from_existing_log(log_csv)
        best_val = float(prior_best) if prior_best is not None else float("inf")
        if prior_best is not None:
            print(f"[RESUME] Continuing with best prior val loss={best_val:.6f}.")
        else:
            print("[RESUME] No prior val history found; best-model tracking restarts from +inf.")
    else:
        best_val = float("inf")

    TR, VL = [], []
    last_completed_epoch = int(resume_epoch) if resume_active else 0
    for epoch in range(start_epoch, total_epochs + 1):
        lr_scheduled = _lr_from_schedule_for_epoch(lr_schedule_entries, epoch)
        if lr_scheduled is not None:
            changed_lr = _set_optimizer_lr(opt, lr_scheduled)
            if changed_lr or any(int(ep) == int(epoch) for ep, _ in lr_schedule_entries):
                print(f"[LR-SCHEDULE] epoch {epoch:03d}: lr={lr_scheduled:.3e}", flush=True)

        t0 = time.time()
        #batch = next(iter(train_loader))

        # unpack: loss, mae, stats
        tr_loss, tr_mae, tr_stats = train_one_epoch_multi_step(
            model, train_loader, opt, cfg, device,
            H=H, W=W, dx=dx, dy=dy, mu=mu, sigma=sigma, epoch_idx=epoch,
            runtime_mesh_policy=runtime_mesh_policy,
            chunk_sidecar=chunk_sidecar_reader,
            ramp_feature_ctx=ramp_feature_ctx,
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
                ramp_feature_ctx=ramp_feature_ctx,
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

        do_periodic_ckpt = (checkpoint_every > 0) and (
            (epoch % checkpoint_every) == 0 or (epoch == total_epochs)
        )
        if do_periodic_ckpt:
            torch.save(_build_last_checkpoint_payload(epoch), last_ckpt_path)
            print(f"[INFO] Saved last checkpoint at epoch {epoch:03d}: {last_ckpt_path}")
        last_completed_epoch = int(epoch)

    # Always write final last-model checkpoint when training loop completes.
    torch.save(_build_last_checkpoint_payload(last_completed_epoch), last_ckpt_path)
    print(
        f"[INFO] Saved final last checkpoint at epoch {last_completed_epoch:03d}: {last_ckpt_path}"
    )

    if len(TR) > 0:
        plot_loss_curves(
            os.path.join(save_dir, "loss_curves.png"),
            list(range(start_epoch, start_epoch + len(TR))),
            TR, VL
        )
    else:
        print("[INFO] No training epochs were run in this invocation; skipping loss curve plot update.")

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
        ramp_feature_ctx=ramp_feature_ctx,
    )

    print(f"[TEST] loss={test_loss:.4e}")

    step_log_enabled, step_log_path, _step_log_append = _runtime_step_log_settings(cfg)
    if step_log_enabled:
        summary_path = os.path.splitext(step_log_path)[0] + "_summary.txt"
        try:
            _runtime_step_log_write_summary(step_log_path, summary_path)
            print(f"[RUNTIME-MESH] step log summary written: {summary_path}")
        except Exception as e:
            print(f"[RUNTIME-MESH][WARN] failed to write step log summary: {e!r}")

    if chunk_sidecar_reader is not None:
        try:
            chunk_sidecar_reader.close()
        except Exception:
            pass

                
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
    ap.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path to resume training state from (expects last_model.pt payload). "
            "Use 'auto' to load <train.save_dir>/last_model.pt."
        ),
    )
    args = ap.parse_args()
    print("Running main...", flush=True)
    main(args.config, args.out_dir, args.resume_from)
