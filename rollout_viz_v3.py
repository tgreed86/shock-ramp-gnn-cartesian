#!/usr/bin/env python
"""
rollout_viz.py (replacement)

Long-rollout visualization script for trained models (memory-safe + faster).

What it produces:
- One GIF per feature, named: rollout_<feature_name>.gif
- Default frame layout is 1x2: [Pred(t+1), GT(t+1)].
- Optional (--include-deltas) frame layout is 2x3:
    Top:    [GT(t), Pred(t+1) on Pred-mesh, GT(t+1)]
    Bottom: [GT(t+1)-GT(t) on GT(t+1), Pred(t+1)-GT(t+1) on Pred, Pred(t+1)-GT(t) on Pred]

Key performance fixes:
1) Only evaluates the single requested window (start_t .. start_t+horizon),
   instead of evaluating ALL windows and then filtering.
2) Computes plot deltas once per timestep (not once per feature per timestep).
3) Streams GIF writing (doesn't keep all frames in RAM).
4) Reuses ONE Matplotlib figure per timestep and updates the plotted arrays per feature.
   This removes the biggest overhead in the old approach.

Typical usage:
  python rollout_viz.py --checkpoint path/to/last_model.pt --horizon 50 --start-t 48

Optional speed knobs:
  --no-edges              Disable mesh edges in plots (often a big speedup).
  --colorbars shared      Use one colorbar per row instead of per-panel (speedup).
  --clim-sample 20000     Approximate percentile limits using a subsample (speedup).
  --runtime-knn-k 4               Smaller kNN for runtime remap interpolation (faster, less robust).
  --runtime-interp-chunk 4096     Chunk size for runtime remap interpolation.
  --runtime-update-every-steps 2  Rebuild runtime mesh every 2 steps instead of every step.
  --runtime-idw-backend faiss_ivf Use approximate FAISS backend for runtime remap on CUDA.

Notes:
- Uses Agg backend to avoid GUI/memory issues on macOS.
- Assumes these repo imports exist:
    dataset.CellRefineWindowDataset
    pretrain.CollateWithUniformStaticNoPrecomp
    plots.compute_plot_deltas, plots._recover_parent_mask, plots._unwrap_delta, plots._fidx_or_none, plots._draw_amr_cells
    train.build_model_from_cfg, train.evaluate_one_epoch_multi_step, train._get_bbox

Example command for pre-caluclated mesh rollout:
    python rollout_viz_v3.py --checkpoint runs_mesh_first/advection_diffusion/euler_adv/resid_weight-0/mse_loss/adv_at_all_steps/1000ep_5ts_window_bw-0_no-loss-sched/last_model.pt \                                                                                                                                                    
        --start-t 50 \
        --out-dir runs_mesh_first/advection_diffusion/euler_adv/resid_weight-0/mse_loss/adv_at_all_steps/1000ep_5ts_window_bw-0_no-loss-sched \
        --horizon 100 \
        --device cpu \
        --progress-every 1 \
        --unify-clims \
        --raster-mode block --precomp-path cache/first_simulation/all/shock_ramp_precomp_pLow85_pHigh95_faceAdj_DEC.h5 \
        --raster-lmax 3 --pt-path /Users/trevorreed/UVA_postdoc_stuff/gparc/data/shock_ramp/shock_ramp_amr.pt

Example command for CNN policy rollout:
    python rollout_viz_v3.py \
        --checkpoint ./runs_mesh_first/CNN_mesh_predictor/10ep_test/best_model.pt \
        --start-t 20 \
        --horizon 40 \
        --out-dir ./runs_mesh_first/CNN_mesh_predictor/10ep_test/runtime_mesh_cnn_t20_h40 \
        --fps 4 --device cpu \
        --pt-path /Users/trevorreed/UVA_postdoc_stuff/gparc/data/shock_ramp/shock_ramp_amr.pt \
        --unify-clims \
        --config ./runs_mesh_first/CNN_mesh_predictor/10ep_test/config.json \
        --runtime-mesh-cnn-ckpt ./runs_mesh_policy_cnn/without_parent_masks/unet_hierarchical/500ep_lr-scheduler_3-6-26/mesh_policy_cnn_best.pt \
        --runtime-mesh-spec-path utils/wedge_mesh_spec.pt

    NOTE: May need to use --runtime-idw-backend exact if faiss_ivf or faiss_flat used during training

============ OTHER OPTIONS ============
To make to 2 x 1 plot (showing just pred(t+1) and GT(t+1) without deltas), add:
        --truth-mode
To change figure sizes from command line:
        --two-panel-fig-w 16, --two-panel-fig-h 8

        (16 and and 8 are just examples)
Use:
        --truth-zoom-features density, energy, or both
    to produce additional zoomed-in GIFs for the specified features, showing a tight view of the region around the shock.
Use 
        --truth-gt-render-levels 0,1,2,3
    to change the maximum level of refinement shown in the GT(t+1) panel of the main GIF. By default, all levels are shown.
=======================================
"""

import os
import io
import gc
import json
import time
import argparse
import zipfile
import re
from typing import Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # critical on macOS for headless/stability
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection, PolyCollection
try:
    from scipy.spatial import cKDTree as _SciPyKDTree
except Exception:
    _SciPyKDTree = None

import imageio.v2 as imageio
from torch.utils.data import DataLoader, SequentialSampler, Subset

# Project imports – must exist in your repo.
from dataset import CellRefineWindowDataset
from pretrain import (
    CollateWithUniformStaticNoPrecomp,
)
from plots import (
    compute_plot_deltas,
    _recover_parent_mask,
    _unwrap_delta,
    _fidx_or_none,
    _draw_amr_cells,
)
from train import (
    build_model_from_cfg,
    evaluate_one_epoch_multi_step,
    _get_bbox,
    _physics_inputs_active_from_loss_cfg,
    _load_series_from_pt_path,
)
from models import ParcFeatureAdapter
from utils_geom import build_idw_map, apply_idw_map
from utils.degree_error_diag import make_degree_error_bar_plots
import utils.dec_ops as dec


# ----------------- Logging helpers ----------------- #

def _now() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)

class Timer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        log(f"{self.name}...")
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.perf_counter()
        log(f"{self.name}...done in {t1 - self.t0:.2f}s")


# ----------------- Small helpers ----------------- #

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cfg_bool_strict(value, *, key: str) -> bool:
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

def _load_pt_series(pt_path: str):
    """Load raw time series from .pt/.pth/.zip or uniform-grid .h5/.hdf5."""
    _resolved_path, data_list = _load_series_from_pt_path(pt_path)
    return data_list

def _get_refine_ratio(cfg: dict) -> int:
    pol = cfg.get("policy", {}) or {}
    mesh = cfg.get("mesh", {}) or {}
    rr = pol.get("refine_ratio", mesh.get("refine_ratio", 2))
    rr = int(rr)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {rr}")
    return rr

def _extract_time(step_obj) -> Optional[float]:
    """Try to pull a float time from a PyG Data-like object or dict."""
    if isinstance(step_obj, dict):
        for k in ("time", "t", "sim_time"):
            if k in step_obj:
                try:
                    return float(step_obj[k])
                except Exception:
                    pass
        return None

    for k in ("time", "t", "sim_time"):
        if hasattr(step_obj, k):
            try:
                return float(getattr(step_obj, k))
            except Exception:
                pass
        try:
            v = step_obj[k]
            try:
                return float(v)
            except Exception:
                pass
        except Exception:
            pass
    return None

def _compute_dt_transitions(series):
    """dt_transitions[i] = time[i+1]-time[i] if metadata exists, else empty list."""
    times = [_extract_time(s) for s in series]
    if any(t is None for t in times):
        return [], None
    dts = [float(times[i + 1] - times[i]) for i in range(len(times) - 1)]
    if not dts:
        return [], None
    dt_ref = float(np.median(np.asarray(dts, dtype=np.float64)))
    return dts, dt_ref

def _maybe_empty_device_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _extract_xy_from_snapshot(step_obj):
    if isinstance(step_obj, dict):
        c = step_obj.get("pos", step_obj.get("xy", step_obj.get("centers", None)))
        p = step_obj.get("pos_phys", step_obj.get("xy_phys", None))
        return c, p
    c = getattr(step_obj, "pos", getattr(step_obj, "xy", getattr(step_obj, "centers", None)))
    p = getattr(step_obj, "pos_phys", getattr(step_obj, "xy_phys", None))
    return c, p


def _extract_hw_from_snapshot(step_obj):
    if isinstance(step_obj, dict):
        H = step_obj.get("H", None)
        W = step_obj.get("W", None)
    else:
        H = getattr(step_obj, "H", None)
        W = getattr(step_obj, "W", None)
    try:
        H = int(H) if H is not None else None
    except Exception:
        H = None
    try:
        W = int(W) if W is not None else None
    except Exception:
        W = None
    return H, W


def _extract_ij_level_from_snapshot(step_obj):
    if isinstance(step_obj, dict):
        ij = step_obj.get("ij", None)
        level = step_obj.get("level", None)
    else:
        ij = getattr(step_obj, "ij", None)
        level = getattr(step_obj, "level", None)
    return ij, level


def _extract_features_from_snapshot(step_obj):
    if isinstance(step_obj, dict):
        feat = step_obj.get("features", step_obj.get("x", None))
        names = step_obj.get("component_names", None)
    else:
        feat = getattr(step_obj, "features", getattr(step_obj, "x", None))
        names = getattr(step_obj, "component_names", None)
    return feat, names


def _align_snapshot_features_to_targets(
    feat_raw,
    comp_names_raw,
    target_feature_names: list[str] | None,
    F_target: int,
) -> torch.Tensor | None:
    try:
        feat = torch.as_tensor(feat_raw, dtype=torch.float32, device="cpu")
    except Exception:
        return None
    if feat.ndim != 2 or feat.size(0) <= 0 or feat.size(1) <= 0:
        return None

    if target_feature_names is not None and comp_names_raw is not None:
        try:
            raw_names = [str(n).strip().lower() for n in list(comp_names_raw)]
            idxs = []
            for tn in list(target_feature_names)[:F_target]:
                t = str(tn).strip().lower()
                if t in raw_names:
                    idxs.append(raw_names.index(t))
                    continue
                # basic alias fallbacks
                aliases = []
                if t in ("rho", "density"):
                    aliases = ["density", "rho"]
                elif t in ("energy", "e"):
                    aliases = ["energy", "e"]
                elif "x_momentum" in t or "mom_x" in t:
                    aliases = ["x_momentum", "mom_x"]
                elif "y_momentum" in t or "mom_y" in t:
                    aliases = ["y_momentum", "mom_y"]
                found = None
                for a in aliases:
                    if a in raw_names:
                        found = raw_names.index(a)
                        break
                if found is None:
                    # substring fallback
                    for i, rn in enumerate(raw_names):
                        if t in rn:
                            found = i
                            break
                if found is None:
                    idxs = []
                    break
                idxs.append(found)
            if len(idxs) == F_target:
                return feat[:, torch.as_tensor(idxs, dtype=torch.long, device="cpu")]
        except Exception:
            pass

    if feat.size(1) >= int(F_target):
        return feat[:, : int(F_target)]
    return None


def _align_snapshot_features_to_targets_strict(
    feat_raw,
    comp_names_raw,
    target_feature_names: list[str] | None,
    F_target: int,
) -> torch.Tensor:
    """
    Strict variant: no column-order fallback.
    Requires component names and semantic name matches (alias-aware), but never
    falls back to positional channel order.
    """
    try:
        feat = torch.as_tensor(feat_raw, dtype=torch.float32, device="cpu")
    except Exception as e:
        raise RuntimeError(f"GT strict feature parse failed: {e}") from e
    if feat.ndim != 2 or feat.size(0) <= 0 or feat.size(1) <= 0:
        raise RuntimeError(f"GT strict feature tensor must be [N,F], got {tuple(feat.shape)}")

    if target_feature_names is None:
        if feat.size(1) < int(F_target):
            raise RuntimeError(
                f"GT strict feature tensor has {feat.size(1)} channels but expected >= {int(F_target)}"
            )
        return feat[:, : int(F_target)]

    if comp_names_raw is None:
        raise RuntimeError(
            "GT strict mode requires raw snapshot 'component_names' to align features."
        )

    def _canon(name: str) -> str:
        s = str(name).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    raw_names = [str(n) for n in list(comp_names_raw)]
    raw_canon = [_canon(n) for n in raw_names]
    raw_index = {k: i for i, k in enumerate(raw_canon)}

    alias = {
        "density": ["density", "rho"],
        "rho": ["rho", "density"],
        "x_momentum": ["x_momentum", "x_mom", "mom_x", "momentum_x", "xmomentum", "x-momentum"],
        "y_momentum": ["y_momentum", "y_mom", "mom_y", "momentum_y", "ymomentum", "y-momentum"],
        "energy": ["energy", "energy_density", "energy-density", "total_energy", "e"],
        "energy_density": ["energy_density", "energy-density", "energy", "total_energy", "e"],
    }

    idxs = []
    missing = []
    for tn in list(target_feature_names)[:F_target]:
        t_raw = str(tn)
        t = _canon(t_raw)
        cands = [_canon(c) for c in alias.get(t, [t_raw])]
        found = None
        for c in cands:
            if c in raw_index:
                found = int(raw_index[c])
                break
        if found is None:
            missing.append(str(tn))
        else:
            idxs.append(int(found))
    if missing:
        raise RuntimeError(
            "GT strict mode could not align required feature names. "
            f"Missing={missing}, raw_component_names={raw_names}"
        )
    return feat[:, torch.as_tensor(idxs, dtype=torch.long, device="cpu")]


def _extract_truth_snapshot_strict(
    step_obj,
    *,
    feature_names: list[str] | None,
    F_target: int,
) -> dict:
    """
    Parse a raw GT snapshot with strict requirements and no fallback to non-raw sources.
    Required fields: pos, pos_phys, ij, level, H, W, features, component_names.
    """
    cxy_raw, pxy_raw = _extract_xy_from_snapshot(step_obj)
    ij_raw, lev_raw = _extract_ij_level_from_snapshot(step_obj)
    H_raw, W_raw = _extract_hw_from_snapshot(step_obj)
    feat_raw, comp_names_raw = _extract_features_from_snapshot(step_obj)

    missing = []
    if cxy_raw is None:
        missing.append("pos")
    if pxy_raw is None:
        missing.append("pos_phys")
    if ij_raw is None:
        missing.append("ij")
    if lev_raw is None:
        missing.append("level")
    if H_raw is None:
        missing.append("H")
    if W_raw is None:
        missing.append("W")
    if feat_raw is None:
        missing.append("features/x")
    if comp_names_raw is None:
        missing.append("component_names")
    if missing:
        raise RuntimeError(f"GT strict snapshot is missing required fields: {missing}")

    cxy = torch.as_tensor(cxy_raw, dtype=torch.float32, device="cpu")
    pxy = torch.as_tensor(pxy_raw, dtype=torch.float32, device="cpu")
    ij = torch.as_tensor(ij_raw, dtype=torch.long, device="cpu")
    lev = torch.as_tensor(lev_raw, dtype=torch.long, device="cpu").view(-1)
    H = int(H_raw)
    W = int(W_raw)
    if H <= 0 or W <= 0:
        raise RuntimeError(f"GT strict snapshot has invalid H/W: H={H} W={W}")
    if cxy.ndim != 2 or cxy.size(1) < 2:
        raise RuntimeError(f"GT strict pos must be [N,2+], got {tuple(cxy.shape)}")
    if pxy.ndim != 2 or pxy.size(1) < 2:
        raise RuntimeError(f"GT strict pos_phys must be [N,2+], got {tuple(pxy.shape)}")
    if ij.ndim != 2 or ij.size(1) < 2:
        raise RuntimeError(f"GT strict ij must be [N,2+], got {tuple(ij.shape)}")
    n = int(cxy.size(0))
    if int(pxy.size(0)) != n or int(ij.size(0)) != n or int(lev.numel()) != n:
        raise RuntimeError(
            "GT strict snapshot length mismatch: "
            f"pos={n} pos_phys={int(pxy.size(0))} ij={int(ij.size(0))} level={int(lev.numel())}"
        )

    feat = _align_snapshot_features_to_targets_strict(
        feat_raw,
        comp_names_raw,
        feature_names,
        F_target,
    )
    if int(feat.size(0)) != n:
        raise RuntimeError(
            f"GT strict features count {int(feat.size(0))} does not match pos count {n}"
        )

    rr_hint = _infer_ref_ratio_from_snapshot_ij(step_obj, candidates=(2.0, 4.0, 8.0))
    return {
        "comp": cxy[:, :2].contiguous(),
        "phys": pxy[:, :2].contiguous(),
        "ij": ij[:, :2].contiguous(),
        "level": lev.contiguous(),
        "features": feat.contiguous(),
        "component_names": list(comp_names_raw),
        "H_gt": int(H),
        "W_gt": int(W),
        "rr_hint": (None if rr_hint is None else float(rr_hint)),
    }


@torch.no_grad()
def _map_phys_points_to_comp_idw(
    q_phys_xy: torch.Tensor,
    src_comp_xy: torch.Tensor,
    src_phys_xy: torch.Tensor,
    *,
    k: int,
    chunk: int,
    backend: str = "cdist",
    kdtree=None,
) -> torch.Tensor:
    """
    Inverse map approximation (phys->comp) via IDW on raw snapshot points.
    """
    backend = str(backend).lower()
    if backend == "kdtree":
        if _SciPyKDTree is None:
            raise RuntimeError("Requested truth-idw backend='kdtree' but scipy is unavailable.")
        src_phys_np = src_phys_xy.detach().cpu().numpy().astype(np.float32, copy=False)
        src_comp_np = src_comp_xy.detach().cpu().numpy().astype(np.float32, copy=False)
        q_np = q_phys_xy.detach().cpu().numpy().astype(np.float32, copy=False)
        if src_phys_np.shape[0] == 0:
            raise RuntimeError("Cannot inverse-map points: source phys centers are empty.")
        k_eff = max(1, min(int(k), int(src_phys_np.shape[0])))
        tree = kdtree if kdtree is not None else _SciPyKDTree(src_phys_np)
        out = np.empty((q_np.shape[0], 2), dtype=np.float32)
        eps = 1e-8
        for s in range(0, q_np.shape[0], int(chunk)):
            e = min(q_np.shape[0], s + int(chunk))
            q_chunk = q_np[s:e]
            try:
                d, idx = tree.query(q_chunk, k=k_eff, workers=-1)
            except TypeError:
                d, idx = tree.query(q_chunk, k=k_eff)
            if k_eff == 1:
                d = d[:, None]
                idx = idx[:, None]
            d = np.asarray(d, dtype=np.float32)
            idx = np.asarray(idx, dtype=np.int64)
            exact = d[:, 0] <= eps
            inv = 1.0 / np.maximum(d, eps)
            inv[exact, :] = 0.0
            inv_sum = inv.sum(axis=1, keepdims=True)
            w = np.divide(inv, inv_sum, out=np.zeros_like(inv), where=(inv_sum > 0.0))
            if np.any(exact):
                w[exact, :] = 0.0
                w[exact, 0] = 1.0
            nbr_comp = src_comp_np[idx]  # [Q,k,2]
            out[s:e] = (w[..., None] * nbr_comp).sum(axis=1)
        return torch.from_numpy(out).to(dtype=torch.float32, device="cpu")

    if backend != "cdist":
        raise ValueError(f"Unsupported truth-idw backend: {backend}")
    idx, w = build_idw_map(
        dst_xy=q_phys_xy.to(dtype=torch.float32, device="cpu"),
        src_xy=src_phys_xy.to(dtype=torch.float32, device="cpu"),
        k=int(k),
        chunk=int(chunk),
    )
    return apply_idw_map(idx, w, src_comp_xy.to(dtype=torch.float32, device="cpu"))


@torch.no_grad()
def _paint_amr_ij_to_uniform_grid(
    *,
    ij: torch.Tensor,          # [N,2] (i,j) at each cell's own level
    levels: torch.Tensor,      # [N]
    values: torch.Tensor,      # [N,F]
    H0: int,
    W0: int,
    rr: int,
    render_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact AMR paint to a uniform computational grid at render_level.
    Returns:
      img: [H_render, W_render, F] float32 (NaN outside domain)
      occ: [H_render, W_render] bool
    """
    if int(rr) < 2:
        raise ValueError(f"rr must be >=2, got {rr}")
    if int(render_level) < 0:
        raise ValueError(f"render_level must be >=0, got {render_level}")

    ij = ij.to(dtype=torch.long, device="cpu")
    levels = levels.to(dtype=torch.long, device="cpu").view(-1)
    vals = values.to(dtype=torch.float32, device="cpu")
    if vals.ndim == 1:
        vals = vals[:, None]
    if ij.ndim != 2 or ij.size(1) < 2:
        raise RuntimeError(f"ij must be [N,2+], got {tuple(ij.shape)}")
    if ij.size(0) != levels.numel() or ij.size(0) != vals.size(0):
        raise RuntimeError(
            f"paint input size mismatch: ij={ij.size(0)} levels={levels.numel()} values={vals.size(0)}"
        )

    F = int(vals.size(1))
    Lmax_step = int(levels.max().item()) if levels.numel() > 0 else 0

    def _paint_direct(level_target: int, value_src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = int(rr) ** int(level_target)
        HH = int(H0 * scale)
        WW = int(W0 * scale)
        img = torch.full((HH * WW, F), torch.nan, dtype=torch.float32, device="cpu")
        occ = torch.zeros((HH * WW,), dtype=torch.bool, device="cpu")

        for l in range(0, int(level_target) + 1):
            m = (levels == int(l))
            if not torch.any(m):
                continue
            b = int(rr) ** int(level_target - l)
            i0 = ij[m, 0] * b
            j0 = ij[m, 1] * b
            off = torch.arange(b, dtype=torch.long, device="cpu")
            rr2 = (j0[:, None] + off[None, :])[:, :, None].expand(-1, b, b)
            cc2 = (i0[:, None] + off[None, :])[:, None, :].expand(-1, b, b)
            flat_idx = (rr2 * WW + cc2).reshape(-1)
            vrep = value_src[m].repeat_interleave(b * b, dim=0)
            img[flat_idx] = vrep
            occ[flat_idx] = True
        return img.view(HH, WW, F), occ.view(HH, WW)

    if int(render_level) >= int(Lmax_step):
        img, occ = _paint_direct(int(render_level), vals)
        return (
            img.detach().cpu().numpy().astype(np.float32, copy=False),
            occ.detach().cpu().numpy().astype(bool, copy=False),
        )

    # Downsample from step Lmax to requested coarser level using occupancy-weighted averaging.
    img_hi, occ_hi = _paint_direct(int(Lmax_step), vals)
    factor = int(rr) ** int(Lmax_step - int(render_level))
    Hr = int(H0 * (int(rr) ** int(render_level)))
    Wr = int(W0 * (int(rr) ** int(render_level)))

    img_hi = img_hi.view(Hr, factor, Wr, factor, F)
    occ_hi_f = occ_hi.to(torch.float32).view(Hr, factor, Wr, factor)
    sum_v = (img_hi * occ_hi_f[..., None]).sum(dim=(1, 3))
    cnt = occ_hi_f.sum(dim=(1, 3))

    img = torch.full((Hr, Wr, F), torch.nan, dtype=torch.float32, device="cpu")
    valid = cnt > 0.0
    if torch.any(valid):
        img[valid] = sum_v[valid] / cnt[valid, None]
    occ = valid
    return (
        img.detach().cpu().numpy().astype(np.float32, copy=False),
        occ.detach().cpu().numpy().astype(bool, copy=False),
    )


@torch.no_grad()
def _sample_comp_raster_on_phys_grid(
    *,
    img_comp: np.ndarray,     # [HH,WW,F]
    occ_comp: np.ndarray,     # [HH,WW]
    q_grid_xy: np.ndarray,    # [Q,2] physical
    ny: int,
    nx: int,
    bbox_comp: tuple[float, float, float, float],
    H0: int,
    W0: int,
    rr: int,
    render_level: int,
    map_mode_eff: str,
    affine_sol: torch.Tensor,
    src_comp_xy: torch.Tensor,
    src_phys_xy: torch.Tensor,
    idw_backend: str,
    idw_k: int,
    idw_chunk: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Raster sample from computational uniform grid to a physical uniform query grid.
    No GT value interpolation across scattered points; values come from exact painted comp raster.
    """
    HH, WW, F = int(img_comp.shape[0]), int(img_comp.shape[1]), int(img_comp.shape[2])
    if occ_comp.shape != (HH, WW):
        raise RuntimeError(f"occ_comp shape {occ_comp.shape} does not match img_comp {(HH, WW)}")

    q_phys = torch.as_tensor(q_grid_xy, dtype=torch.float32, device="cpu")
    mode = str(map_mode_eff).lower().strip()
    if mode == "affine":
        q_comp = _map_phys_points_affine_inverse(q_phys, affine_sol).cpu().numpy()
    elif mode == "idw":
        q_comp = _map_phys_points_to_comp_idw(
            q_phys,
            src_comp_xy=src_comp_xy,
            src_phys_xy=src_phys_xy,
            k=int(idw_k),
            chunk=int(idw_chunk),
            backend=idw_backend,
        ).cpu().numpy()
    else:
        raise RuntimeError(f"Unsupported map mode in strict GT sampler: {map_mode_eff}")

    xmin, xmax, ymin, ymax = [float(v) for v in bbox_comp]
    dx0 = (xmax - xmin) / float(W0)
    dy0 = (ymax - ymin) / float(H0)
    scale = int(rr) ** int(render_level)
    x_units = (q_comp[:, 0] - xmin) / max(dx0, 1.0e-12)
    y_units = (q_comp[:, 1] - ymin) / max(dy0, 1.0e-12)
    cc = np.floor(x_units * scale).astype(np.int64, copy=False)
    rr_idx = np.floor(y_units * scale).astype(np.int64, copy=False)
    inb = (cc >= 0) & (cc < WW) & (rr_idx >= 0) & (rr_idx < HH)

    flat_img = img_comp.reshape(HH * WW, F)
    flat_occ = occ_comp.reshape(HH * WW)
    out_flat = np.full((q_grid_xy.shape[0], F), np.nan, dtype=np.float32)
    valid = np.zeros((q_grid_xy.shape[0],), dtype=bool)
    if np.any(inb):
        lin = rr_idx[inb] * WW + cc[inb]
        occ_here = flat_occ[lin]
        if np.any(occ_here):
            dst_idx = np.nonzero(inb)[0][occ_here]
            src_idx = lin[occ_here]
            out_flat[dst_idx, :] = flat_img[src_idx, :]
            valid[dst_idx] = True

    out = out_flat.reshape(ny, nx, F)
    valid_m = valid.reshape(ny, nx)
    meta = {
        "status": "ok",
        "map_mode": mode,
        "valid_frac": float(np.mean(valid_m)) if valid_m.size > 0 else 0.0,
        "mask_mode": "strict_comp_occupancy",
        "comp_grid": f"{WW}x{HH}",
    }
    return out, valid_m, meta


def _build_domain_mask_from_points_tri(
    pts_xy: np.ndarray,
    q_grid_xy: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    """
    Build query mask from triangulation of raw physical points.
    Strict/raw-only helper used by truth physical default path.
    """
    meta = {"status": "init"}
    pts = np.asarray(pts_xy, dtype=np.float32)
    q = np.asarray(q_grid_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or q.ndim != 2 or q.shape[1] != 2:
        meta["status"] = "bad_shape"
        return None, meta
    if pts.shape[0] < 3:
        meta["status"] = "too_few_points"
        return None, meta
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 3:
        meta["status"] = "too_few_finite_points"
        return None, meta
    try:
        tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
        finder = tri.get_trifinder()
        mask = np.asarray(finder(q[:, 0], q[:, 1]) >= 0, dtype=bool)
        meta.update(
            {
                "status": "ok",
                "n_points": int(pts.shape[0]),
                "valid_frac": float(mask.mean()) if mask.size > 0 else 0.0,
            }
        )
        return mask, meta
    except Exception as e:
        meta.update({"status": "triangulation_invalid", "error": str(e)})
        return None, meta


def _resolve_truth_timestep(raw_series, t_hint: int, n_gt_cells: int) -> int:
    """
    Resolve which raw-series timestep should provide comp->phys mapping for ex['centers_tp1'].
    We prefer t_hint, but also test +/-1 to tolerate off-by-one labeling in examples.
    """
    cand_times = []
    for tt in (int(t_hint), int(t_hint) + 1, int(t_hint) - 1):
        if 0 <= tt < len(raw_series):
            cand_times.append(tt)
    cand_times = list(dict.fromkeys(cand_times))
    if not cand_times:
        raise RuntimeError(
            f"Could not resolve truth timestep for t_hint={t_hint}; raw series length={len(raw_series)}."
        )

    best_t = None
    best_score = None
    for tt in cand_times:
        cxy, _pxy = _extract_xy_from_snapshot(raw_series[tt])
        if cxy is None:
            continue
        try:
            n_here = int(torch.as_tensor(cxy).shape[0])
        except Exception:
            continue
        score = abs(int(n_here) - int(n_gt_cells))
        if (best_score is None) or (score < best_score):
            best_score = score
            best_t = tt
        if score == 0 and tt == int(t_hint):
            break

    if best_t is None:
        raise RuntimeError(
            f"Could not resolve truth timestep for t_hint={t_hint}: "
            "raw snapshots are missing usable comp centers."
        )
    return int(best_t)


def _corners_from_centers_levels(
    centers: torch.Tensor,
    levels: torch.Tensor,
    H: int,
    W: int,
    bbox_comp: tuple[float, float, float, float],
    *,
    ref_ratio: float = 2.0,
) -> torch.Tensor:
    xmin, xmax, ymin, ymax = bbox_comp
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    c = centers.to(torch.float32)
    l = levels.to(torch.int64).view(-1)
    rr = max(float(ref_ratio), 1.0 + 1e-12)
    scale = torch.pow(torch.tensor(1.0 / rr, dtype=torch.float32), l.to(torch.float32))
    hx = 0.5 * dx0 * scale
    hy = 0.5 * dy0 * scale

    x = c[:, 0]
    y = c[:, 1]
    x0 = x - hx
    x1 = x + hx
    y0 = y - hy
    y1 = y + hy

    corners = torch.stack(
        [
            torch.stack([x0, y0], dim=1),
            torch.stack([x1, y0], dim=1),
            torch.stack([x1, y1], dim=1),
            torch.stack([x0, y1], dim=1),
        ],
        dim=1,
    )  # [N,4,2]
    return corners


@torch.no_grad()
def _fit_affine_comp_to_phys(src_comp_xy: torch.Tensor, src_phys_xy: torch.Tensor):
    comp = src_comp_xy.to(dtype=torch.float32, device="cpu")
    phys = src_phys_xy.to(dtype=torch.float32, device="cpu")
    if comp.ndim != 2 or phys.ndim != 2 or comp.size(0) != phys.size(0) or comp.size(1) != 2 or phys.size(1) != 2:
        raise RuntimeError(
            f"Bad shapes for affine fit: comp={tuple(comp.shape)} phys={tuple(phys.shape)}"
        )
    A = torch.cat([comp, torch.ones((comp.size(0), 1), dtype=comp.dtype)], dim=1)
    try:
        sol = torch.linalg.lstsq(A, phys).solution
    except Exception:
        sol = torch.linalg.pinv(A) @ phys
    pred = A @ sol
    rmse = float(torch.sqrt(torch.mean(torch.sum((pred - phys) ** 2, dim=1))).item())
    return sol.to(dtype=torch.float32, device="cpu"), rmse


@torch.no_grad()
def _map_comp_points_affine(q_xy: torch.Tensor, affine_sol: torch.Tensor) -> torch.Tensor:
    q = q_xy.to(dtype=torch.float32, device="cpu")
    A = torch.cat([q, torch.ones((q.size(0), 1), dtype=q.dtype)], dim=1)
    return (A @ affine_sol.to(dtype=torch.float32, device="cpu")).to(dtype=torch.float32, device="cpu")


@torch.no_grad()
def _map_phys_points_affine_inverse(q_phys_xy: torch.Tensor, affine_sol: torch.Tensor) -> torch.Tensor:
    """
    Invert affine comp->phys map used by _map_comp_points_affine.
    """
    q = q_phys_xy.to(dtype=torch.float32, device="cpu")
    sol = affine_sol.to(dtype=torch.float32, device="cpu")
    A = sol[:2, :]   # comp @ A + b = phys  (row-vector form)
    b = sol[2, :]
    try:
        A_inv = torch.linalg.inv(A)
    except Exception:
        A_inv = torch.linalg.pinv(A)
    comp = (q - b) @ A_inv
    return comp.to(dtype=torch.float32, device="cpu")


@torch.no_grad()
def _infer_ref_ratio_from_levels(levels: torch.Tensor, H: int, W: int, candidates=(2.0, 4.0, 8.0)) -> float:
    lev = levels.to(torch.int64).view(-1).detach().cpu()
    if lev.numel() == 0 or H <= 0 or W <= 0:
        return 2.0
    lmax = int(lev.max().item())
    counts = torch.bincount(lev, minlength=lmax + 1).to(torch.float64)
    base = float(H * W)
    best_r = 2.0
    best_err = float("inf")
    for r in candidates:
        rr2 = float(r) * float(r)
        denom = torch.pow(torch.tensor(rr2, dtype=torch.float64), torch.arange(lmax + 1, dtype=torch.float64))
        area_sum = float((counts / denom).sum().item() / base)
        err = abs(area_sum - 1.0)
        if err < best_err:
            best_err = err
            best_r = float(r)
    return best_r


@torch.no_grad()
def _infer_ref_ratio_from_snapshot_ij(step_obj, *, candidates=(2.0, 4.0, 8.0)) -> float | None:
    """
    Infer refinement ratio from snapshot indexing metadata (ij, level, H, W).
    Returns None when required fields are unavailable/invalid.
    """
    H, W = _extract_hw_from_snapshot(step_obj)
    if H is None or W is None or H <= 0 or W <= 0:
        return None

    ij_raw, lev_raw = _extract_ij_level_from_snapshot(step_obj)
    if ij_raw is None or lev_raw is None:
        return None

    try:
        ij = torch.as_tensor(ij_raw, dtype=torch.long, device="cpu")
        lev = torch.as_tensor(lev_raw, dtype=torch.long, device="cpu").view(-1)
    except Exception:
        return None

    if ij.ndim != 2 or ij.size(1) < 2 or lev.numel() == 0 or ij.size(0) != lev.numel():
        return None

    lmax = int(lev.max().item())
    if lmax <= 0:
        return None

    i_max = float(ij[:, 0].max().item())
    j_max = float(ij[:, 1].max().item())
    sx = (i_max + 1.0) / float(W)
    sy = (j_max + 1.0) / float(H)
    if (not np.isfinite(sx)) or (not np.isfinite(sy)):
        return None
    target_scale = max(sx, sy)
    if target_scale <= 1.0:
        return None

    target_r = float(target_scale) ** (1.0 / float(lmax))
    best = min((float(r) for r in candidates), key=lambda r: abs(r - target_r))
    return float(best)


@torch.no_grad()
def _infer_hw_rr_from_snapshot_ij(step_obj, *, candidates=(2.0, 4.0, 8.0)) -> tuple[int | None, int | None, float | None]:
    """
    Infer base H/W and refinement ratio from snapshot ij/level metadata when H/W are absent.
    Uses:
      i_max+1 ~= W0 * rr**Lmax
      j_max+1 ~= H0 * rr**Lmax
    """
    ij_raw, lev_raw = _extract_ij_level_from_snapshot(step_obj)
    if ij_raw is None or lev_raw is None:
        return None, None, None
    try:
        ij = torch.as_tensor(ij_raw, dtype=torch.long, device="cpu")
        lev = torch.as_tensor(lev_raw, dtype=torch.long, device="cpu").view(-1)
    except Exception:
        return None, None, None
    if ij.ndim != 2 or ij.size(1) < 2 or lev.numel() == 0 or ij.size(0) != lev.numel():
        return None, None, None

    i_span = float(ij[:, 0].max().item()) + 1.0
    j_span = float(ij[:, 1].max().item()) + 1.0
    if (not np.isfinite(i_span)) or (not np.isfinite(j_span)) or i_span <= 0.0 or j_span <= 0.0:
        return None, None, None

    Lmax = int(max(0, int(lev.max().item())))
    if Lmax == 0:
        return int(round(j_span)), int(round(i_span)), None

    best = None  # (err, -rr, H0, W0, rr)
    for rr in (float(r) for r in candidates):
        if rr <= 1.0:
            continue
        scale = float(rr) ** float(Lmax)
        if (not np.isfinite(scale)) or scale <= 0.0:
            continue
        W0 = max(1, int(round(i_span / scale)))
        H0 = max(1, int(round(j_span / scale)))
        i_back = float(W0) * scale
        j_back = float(H0) * scale
        err = abs(i_span - i_back) + abs(j_span - j_back)
        cand = (float(err), -float(rr), int(H0), int(W0), float(rr))
        if (best is None) or (cand < best):
            best = cand

    if best is None:
        return None, None, None
    _err, _neg_rr, H0, W0, rr_best = best
    return int(H0), int(W0), float(rr_best)


@torch.no_grad()
def _map_comp_points_to_phys_idw(
    q_xy: torch.Tensor,
    src_comp_xy: torch.Tensor,
    src_phys_xy: torch.Tensor,
    *,
    k: int,
    chunk: int,
    backend: str = "cdist",
    kdtree=None,
) -> torch.Tensor:
    backend = str(backend).lower()
    if backend == "kdtree":
        if _SciPyKDTree is None:
            raise RuntimeError("Requested truth-idw backend='kdtree' but scipy is unavailable.")

        src_comp_np = src_comp_xy.detach().cpu().numpy().astype(np.float32, copy=False)
        src_phys_np = src_phys_xy.detach().cpu().numpy().astype(np.float32, copy=False)
        q_np = q_xy.detach().cpu().numpy().astype(np.float32, copy=False)
        if src_comp_np.shape[0] == 0:
            raise RuntimeError("Cannot map points: source comp centers are empty.")

        k_eff = max(1, min(int(k), int(src_comp_np.shape[0])))
        tree = kdtree if kdtree is not None else _SciPyKDTree(src_comp_np)
        out = np.empty((q_np.shape[0], 2), dtype=np.float32)
        eps = 1e-8
        for s in range(0, q_np.shape[0], int(chunk)):
            e = min(q_np.shape[0], s + int(chunk))
            q_chunk = q_np[s:e]
            try:
                d, idx = tree.query(q_chunk, k=k_eff, workers=-1)
            except TypeError:
                d, idx = tree.query(q_chunk, k=k_eff)

            if k_eff == 1:
                d = d[:, None]
                idx = idx[:, None]

            d = np.asarray(d, dtype=np.float32)
            idx = np.asarray(idx, dtype=np.int64)
            exact = d[:, 0] <= eps

            inv = 1.0 / np.maximum(d, eps)
            inv[exact, :] = 0.0
            inv_sum = inv.sum(axis=1, keepdims=True)
            w = np.divide(inv, inv_sum, out=np.zeros_like(inv), where=(inv_sum > 0.0))
            if np.any(exact):
                w[exact, :] = 0.0
                w[exact, 0] = 1.0

            nbr_phys = src_phys_np[idx]  # [Q,k,2]
            out[s:e] = (w[..., None] * nbr_phys).sum(axis=1)

        return torch.from_numpy(out).to(dtype=torch.float32, device="cpu")

    if backend != "cdist":
        raise ValueError(f"Unsupported truth-idw backend: {backend}")

    idx, w = build_idw_map(
        dst_xy=q_xy.to(dtype=torch.float32, device="cpu"),
        src_xy=src_comp_xy.to(dtype=torch.float32, device="cpu"),
        k=int(k),
        chunk=int(chunk),
    )
    return apply_idw_map(idx, w, src_phys_xy.to(dtype=torch.float32, device="cpu"))


@torch.no_grad()
def _map_comp_points_to_phys_unique_vertices(
    q_xy: torch.Tensor,
    *,
    bbox_comp: tuple[float, float, float, float],
    map_mode: str,
    affine_sol: torch.Tensor,
    src_comp_xy: torch.Tensor,
    src_phys_xy: torch.Tensor,
    idw_backend: str,
    idw_k: int,
    idw_chunk: int,
    kdtree=None,
    quantize_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, float]:
    """
    Map points by first deduplicating (quantized) vertices, then expanding back.
    This enforces shared physical vertices for adjacent cells and reduces IDW calls.
    """
    q = q_xy.to(dtype=torch.float32, device="cpu").contiguous()
    n_all = int(q.shape[0])
    if n_all == 0:
        empty = torch.empty((0, 2), dtype=torch.float32, device="cpu")
        inv_empty = torch.empty((0,), dtype=torch.long, device="cpu")
        return empty, empty, inv_empty, 0, 0, 0.0

    xmin, xmax, ymin, ymax = [float(v) for v in bbox_comp]
    span = max(abs(xmax - xmin), abs(ymax - ymin), 1.0)
    eps = float(quantize_eps) if quantize_eps is not None else (span * 1.0e-9)
    if not np.isfinite(eps) or eps <= 0.0:
        eps = max(span * 1.0e-9, 1.0e-12)

    q_np = q.detach().cpu().numpy().astype(np.float64, copy=False)
    key_x = np.rint((q_np[:, 0] - xmin) / eps).astype(np.int64, copy=False)
    key_y = np.rint((q_np[:, 1] - ymin) / eps).astype(np.int64, copy=False)
    keys = np.stack([key_x, key_y], axis=1)

    _, first_idx, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    first_t = torch.from_numpy(first_idx.astype(np.int64, copy=False))
    inv_t = torch.from_numpy(inv.astype(np.int64, copy=False))
    q_unique = q.index_select(0, first_t)

    mode = str(map_mode).lower().strip()
    if mode == "affine":
        q_unique_phys = _map_comp_points_affine(q_unique, affine_sol)
    elif mode == "idw":
        q_unique_phys = _map_comp_points_to_phys_idw(
            q_unique,
            src_comp_xy,
            src_phys_xy,
            k=int(idw_k),
            chunk=int(idw_chunk),
            backend=idw_backend,
            kdtree=kdtree,
        )
    else:
        raise ValueError(f"Unsupported map mode for unique-vertex mapping: {map_mode}")

    q_phys = q_unique_phys.index_select(0, inv_t)
    return q_phys, q_unique_phys, inv_t.to(torch.long), int(q_unique.shape[0]), int(n_all), float(eps)


def _smooth_idw_raster_from_points(
    pts_xy: np.ndarray,       # [N,2] in physical space
    vals_nf: np.ndarray,      # [N,F]
    q_xy: np.ndarray,         # [Q,2] query grid points (physical space)
    *,
    ny: int,
    nx: int,
    k: int = 8,
    valid_radius_factor: float = 1.5,
    eps: float = 1e-8,
    valid_mask_override: np.ndarray | None = None,
    query_chunk: int = 262144,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Scattered-data IDW interpolation on a regular grid for visualization.
    Returns:
      img: [ny,nx,F] float32
      valid_mask: [ny,nx] bool
      meta: diagnostics dict
    """
    pts = np.asarray(pts_xy, dtype=np.float32)
    vals = np.asarray(vals_nf, dtype=np.float32)
    q = np.asarray(q_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pts_xy must be [N,2], got {pts.shape}")
    if vals.ndim != 2 or vals.shape[0] != pts.shape[0]:
        raise ValueError(f"vals_nf must be [N,F] with N={pts.shape[0]}, got {vals.shape}")
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError(f"q_xy must be [Q,2], got {q.shape}")

    n = int(pts.shape[0])
    fdim = int(vals.shape[1])
    qn = int(q.shape[0])
    if n == 0 or qn == 0:
        out = np.full((ny, nx, fdim), np.nan, dtype=np.float32)
        vm = np.zeros((ny, nx), dtype=bool)
        return out, vm, {"nn_scale_p95": float("nan"), "valid_frac": 0.0}

    k_eff = max(1, min(int(k), n))

    # Fast path: scipy KDTree
    if _SciPyKDTree is not None:
        tree = _SciPyKDTree(pts)
        q_chunk = max(1024, int(query_chunk))
        out_flat = np.empty((qn, fdim), dtype=np.float32)

        # Compute a coarse spacing scale for diagnostics/fallback masking.
        try:
            dnn, _ = tree.query(pts, k=min(2, n), workers=-1)
        except TypeError:
            dnn, _ = tree.query(pts, k=min(2, n))
        if n == 1:
            nn_scale = 1.0e-6
        else:
            nn1 = np.asarray(dnn[:, 1], dtype=np.float32)
            nn1 = nn1[np.isfinite(nn1) & (nn1 > 0.0)]
            if nn1.size == 0:
                nn_scale = 1.0e-6
            else:
                nn_scale = float(np.nanpercentile(nn1, 95.0))
        if (not np.isfinite(nn_scale)) or nn_scale <= 0.0:
            try:
                d0, _ = tree.query(q[: min(qn, q_chunk)], k=1, workers=-1)
            except TypeError:
                d0, _ = tree.query(q[: min(qn, q_chunk)], k=1)
            d0 = np.asarray(d0, dtype=np.float32).reshape(-1)
            d0 = d0[np.isfinite(d0) & (d0 > 0.0)]
            if d0.size > 0:
                nn_scale = float(np.nanmedian(d0))
            if (not np.isfinite(nn_scale)) or nn_scale <= 0.0:
                nn_scale = 1.0e-6

        if valid_mask_override is not None:
            valid = np.asarray(valid_mask_override, dtype=bool).reshape(-1)
            if valid.shape[0] != qn:
                raise ValueError(
                    f"valid_mask_override has wrong size: got {valid.shape[0]}, expected {qn}"
                )
            mask_mode = "provided"
        else:
            mask_mode = "distance_threshold"
            valid_thr = float(valid_radius_factor) * float(nn_scale)
            valid = np.empty((qn,), dtype=bool)

        for s in range(0, qn, q_chunk):
            e = min(s + q_chunk, qn)
            try:
                d, idx = tree.query(q[s:e], k=k_eff, workers=-1)
            except TypeError:
                d, idx = tree.query(q[s:e], k=k_eff)

            if k_eff == 1:
                d = d[:, None]
                idx = idx[:, None]

            d = np.asarray(d, dtype=np.float32)
            idx = np.asarray(idx, dtype=np.int64)
            exact = d[:, 0] <= float(eps)

            invd = 1.0 / np.maximum(d, float(eps))
            invd[exact, :] = 0.0
            invd_sum = invd.sum(axis=1, keepdims=True)
            w = np.divide(invd, invd_sum, out=np.zeros_like(invd), where=(invd_sum > 0.0))
            if np.any(exact):
                w[exact, :] = 0.0
                w[exact, 0] = 1.0

            nbr_vals = vals[idx]  # [chunk,k,F]
            out_flat[s:e, :] = (w[..., None] * nbr_vals).sum(axis=1).astype(np.float32, copy=False)

            if valid_mask_override is None:
                valid[s:e] = np.asarray(d[:, 0], dtype=np.float32) <= valid_thr

    else:
        # Fallback (slower): torch IDW map from existing helper.
        dst_t = torch.from_numpy(q).to(dtype=torch.float32, device="cpu")
        src_t = torch.from_numpy(pts).to(dtype=torch.float32, device="cpu")
        val_t = torch.from_numpy(vals).to(dtype=torch.float32, device="cpu")
        idx_t, w_t = build_idw_map(dst_xy=dst_t, src_xy=src_t, k=k_eff, chunk=32768)
        out_flat = apply_idw_map(idx_t, w_t, val_t).detach().cpu().numpy().astype(np.float32, copy=False)
        nn_scale = float("nan")
        if valid_mask_override is not None:
            valid = np.asarray(valid_mask_override, dtype=bool).reshape(-1)
            if valid.shape[0] != qn:
                raise ValueError(
                    f"valid_mask_override has wrong size: got {valid.shape[0]}, expected {qn}"
                )
            mask_mode = "provided"
        else:
            valid = np.ones((qn,), dtype=bool)
            mask_mode = "none"

    out_flat[~valid, :] = np.nan
    out = out_flat.reshape(ny, nx, fdim)
    valid_mask = valid.reshape(ny, nx)
    meta = {
        "nn_scale_p95": float(nn_scale),
        "valid_frac": float(np.mean(valid_mask)),
        "mask_mode": str(mask_mode),
    }
    return out, valid_mask, meta


def _make_poly_collection(
    ax,
    verts: np.ndarray,  # [N,4,2]
    vals: np.ndarray,   # [N]
    *,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
) -> PolyCollection:
    pc = PolyCollection(
        verts,
        cmap=cmap,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
    )
    pc.set_rasterized(True)
    pc.set_array(vals.astype(np.float32, copy=False))
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_aspect("equal", adjustable="box")
    return pc


def _resolve_truth_zoom_feature_indices(
    feature_names: list[str],
    spec: str,
) -> list[int]:
    """
    Parse user spec and resolve indices for truth moving-zoom GIFs.
    Supported tokens:
      - density (aliases: rho)
      - energy  (aliases: e)
      - both / all
    """
    raw = str(spec or "").strip().lower()
    if raw in ("", "none", "off", "false", "0"):
        return []

    toks = [t.strip().lower() for t in raw.replace(";", ",").split(",") if t.strip()]
    if any(t in ("both", "all", "true", "1") for t in toks):
        want = {"density", "energy"}
    else:
        want = set()
        for t in toks:
            if t in ("density", "rho"):
                want.add("density")
            elif t in ("energy", "e"):
                want.add("energy")
            else:
                raise ValueError(
                    f"Invalid truth zoom feature token '{t}'. Use density, energy, both, or comma-separated list."
                )

    if not feature_names:
        return []

    by_name = {str(n).strip().lower(): i for i, n in enumerate(feature_names)}

    def _find_idx(cands: tuple[str, ...]) -> int | None:
        for c in cands:
            if c in by_name:
                return int(by_name[c])
        # fallback: substring match
        for k, i in by_name.items():
            if any(c in k for c in cands):
                return int(i)
        return None

    out = []
    if "density" in want:
        i = _find_idx(("density", "rho"))
        if i is not None:
            out.append(i)
    if "energy" in want:
        i = _find_idx(("energy",))
        if i is not None and i not in out:
            out.append(i)
    return out


def _fallback_center_from_mask(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return (0.0, 0.0)
    return (float(np.mean(xx)), float(np.mean(yy)))


def _track_moving_zoom_center(
    field_2d: np.ndarray,
    prev_center_xy: tuple[float, float] | None,
    *,
    pct: float = 96.0,
    ema_alpha: float = 0.35,
    search_radius_px: float | None = None,
    max_step_px: float | None = None,
) -> tuple[tuple[float, float], dict]:
    """
    Track a moving ROI center from shock indicator S=|grad(field)| on GT raster.
    Returns center in pixel coordinates (x,y).
    """
    arr = np.asarray(field_2d, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"field_2d must be 2D, got shape={arr.shape}")
    ny, nx = arr.shape
    valid = np.isfinite(arr)
    if not np.any(valid):
        c0 = prev_center_xy if prev_center_xy is not None else (0.5 * (nx - 1), 0.5 * (ny - 1))
        return (float(c0[0]), float(c0[1])), {"status": "no_valid"}

    # Fill NaNs with in-domain median before gradient.
    med = float(np.nanmedian(arr[valid])) if np.any(valid) else 0.0
    a = np.where(valid, arr, med).astype(np.float32, copy=False)

    gy, gx = np.gradient(a)
    s = np.hypot(gx, gy).astype(np.float32, copy=False)
    s[~valid] = 0.0

    if prev_center_xy is not None and (search_radius_px is not None) and (search_radius_px > 0.0):
        yy, xx = np.ogrid[:ny, :nx]
        cxp, cyp = float(prev_center_xy[0]), float(prev_center_xy[1])
        local = ((xx - cxp) ** 2 + (yy - cyp) ** 2) <= float(search_radius_px) ** 2
        if np.any(s[local] > 0.0):
            s = np.where(local, s, 0.0)

    pos = s[s > 0.0]
    if pos.size == 0:
        cfb = _fallback_center_from_mask(valid)
        c = prev_center_xy if prev_center_xy is not None else cfb
        return (float(c[0]), float(c[1])), {"status": "zero_grad"}

    thr = float(np.percentile(pos, float(np.clip(pct, 50.0, 99.9))))
    m = (s >= thr) & valid
    if int(np.count_nonzero(m)) < 8:
        k = min(1024, int(s.size))
        flat = s.reshape(-1)
        idx = np.argpartition(flat, -k)[-k:]
        m = np.zeros_like(flat, dtype=bool)
        m[idx] = True
        m = m.reshape(s.shape) & valid

    w = np.where(m, s, 0.0).astype(np.float64, copy=False)
    sw = float(np.sum(w))
    if sw <= 0.0 or (not np.isfinite(sw)):
        cfb = _fallback_center_from_mask(valid)
        c = prev_center_xy if prev_center_xy is not None else cfb
        return (float(c[0]), float(c[1])), {"status": "bad_weights"}

    xg = np.arange(nx, dtype=np.float64)[None, :]
    yg = np.arange(ny, dtype=np.float64)[:, None]
    cx = float(np.sum(w * xg) / sw)
    cy = float(np.sum(w * yg) / sw)

    if prev_center_xy is not None:
        px, py = float(prev_center_xy[0]), float(prev_center_xy[1])
        dx = cx - px
        dy = cy - py
        d = float(np.hypot(dx, dy))
        if (max_step_px is not None) and (max_step_px > 0.0) and np.isfinite(d) and (d > float(max_step_px)):
            r = float(max_step_px) / max(d, 1.0e-12)
            cx = px + dx * r
            cy = py + dy * r
        a_ema = float(np.clip(ema_alpha, 0.0, 1.0))
        cx = (1.0 - a_ema) * px + a_ema * cx
        cy = (1.0 - a_ema) * py + a_ema * cy

    cx = float(np.clip(cx, 0.0, max(0.0, nx - 1.0)))
    cy = float(np.clip(cy, 0.0, max(0.0, ny - 1.0)))
    return (cx, cy), {
        "status": "ok",
        "thr": float(thr),
        "active_frac": float(np.mean(m)),
    }


def _build_domain_mask_from_tri(
    q_unique_phys_np: np.ndarray,
    quad_vidx_np: np.ndarray,
    q_grid_xy: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    """
    Build a boolean in-domain mask for query points using a triangle finder.
    Returns (mask_or_none, meta). Falls back to None when triangulation is invalid.
    """
    meta = {"status": "init"}
    qv = np.asarray(q_unique_phys_np, dtype=np.float32)
    quads = np.asarray(quad_vidx_np, dtype=np.int64)
    qg = np.asarray(q_grid_xy, dtype=np.float32)

    if qv.ndim != 2 or qv.shape[1] != 2 or quads.ndim != 2 or quads.shape[1] != 4:
        meta["status"] = "bad_shape"
        return None, meta

    tri0 = quads[:, [0, 1, 2]]
    tri1 = quads[:, [0, 2, 3]]
    tri = np.concatenate([tri0, tri1], axis=0)
    n_tri_raw = int(tri.shape[0])
    n_vert = int(qv.shape[0])

    if n_tri_raw == 0 or n_vert == 0:
        meta.update({"status": "empty", "n_tri_raw": n_tri_raw, "n_vert": n_vert})
        return None, meta

    # Index bounds + non-repeated vertex IDs.
    good = (tri >= 0).all(axis=1) & (tri < n_vert).all(axis=1)
    tri = tri[good]
    if tri.shape[0] == 0:
        meta.update({"status": "no_inbounds_tri", "n_tri_raw": n_tri_raw})
        return None, meta
    good = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri = tri[good]
    if tri.shape[0] == 0:
        meta.update({"status": "all_repeated_vidx", "n_tri_raw": n_tri_raw})
        return None, meta

    v0 = qv[tri[:, 0]]
    v1 = qv[tri[:, 1]]
    v2 = qv[tri[:, 2]]
    finite_tri = np.isfinite(v0).all(axis=1) & np.isfinite(v1).all(axis=1) & np.isfinite(v2).all(axis=1)
    tri = tri[finite_tri]
    if tri.shape[0] == 0:
        meta.update({"status": "all_nonfinite", "n_tri_raw": n_tri_raw})
        return None, meta

    v0 = qv[tri[:, 0]]
    v1 = qv[tri[:, 1]]
    v2 = qv[tri[:, 2]]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    # Relative area tolerance against domain scale.
    xspan = float(np.nanmax(qv[:, 0]) - np.nanmin(qv[:, 0])) if np.isfinite(qv[:, 0]).any() else 1.0
    yspan = float(np.nanmax(qv[:, 1]) - np.nanmin(qv[:, 1])) if np.isfinite(qv[:, 1]).any() else 1.0
    area_eps = (max(xspan, 1.0e-6) * max(yspan, 1.0e-6)) * 1.0e-12
    tri = tri[np.abs(cross) > area_eps]
    if tri.shape[0] == 0:
        meta.update({"status": "all_degenerate", "n_tri_raw": n_tri_raw, "area_eps": area_eps})
        return None, meta

    # Dedup triangles up to orientation.
    tri_key = np.sort(tri, axis=1)
    _, uniq_idx = np.unique(tri_key, axis=0, return_index=True)
    tri = tri[np.sort(uniq_idx)]

    try:
        triang = mtri.Triangulation(qv[:, 0], qv[:, 1], triangles=tri)
        finder = triang.get_trifinder()
        mask = np.asarray(finder(qg[:, 0], qg[:, 1]) >= 0, dtype=bool)
        meta.update(
            {
                "status": "ok",
                "n_tri_raw": n_tri_raw,
                "n_tri_used": int(tri.shape[0]),
                "valid_frac": float(mask.mean()) if mask.size > 0 else 0.0,
            }
        )
        return mask, meta
    except Exception as e:
        meta.update(
            {
                "status": "triangulation_invalid",
                "n_tri_raw": n_tri_raw,
                "n_tri_used": int(tri.shape[0]),
                "error": str(e),
            }
        )
        return None, meta


def _dedup_quad_vertices_np(
    corners_phys_np: np.ndarray,
    *,
    quantize_eps: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given quadrilateral corners [N,4,2] in physical space, return:
      unique vertices [M,2], and quad vertex indices [N,4] into unique vertices.
    """
    corners = np.asarray(corners_phys_np, dtype=np.float32)
    if corners.ndim != 3 or corners.shape[1] != 4 or corners.shape[2] != 2:
        raise ValueError(f"corners_phys_np must be [N,4,2], got {corners.shape}")
    flat = corners.reshape(-1, 2).astype(np.float64, copy=False)
    if flat.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 4), dtype=np.int64)

    xmin = float(np.nanmin(flat[:, 0])); xmax = float(np.nanmax(flat[:, 0]))
    ymin = float(np.nanmin(flat[:, 1])); ymax = float(np.nanmax(flat[:, 1]))
    span = max(abs(xmax - xmin), abs(ymax - ymin), 1.0)
    eps = float(quantize_eps) if quantize_eps is not None else (span * 1.0e-9)
    if not np.isfinite(eps) or eps <= 0.0:
        eps = max(span * 1.0e-9, 1.0e-12)

    key_x = np.rint((flat[:, 0] - xmin) / eps).astype(np.int64, copy=False)
    key_y = np.rint((flat[:, 1] - ymin) / eps).astype(np.int64, copy=False)
    keys = np.stack([key_x, key_y], axis=1)
    _, first_idx, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    unique = flat[first_idx].astype(np.float32, copy=False)
    quad_vidx = inv.reshape(corners.shape[0], 4).astype(np.int64, copy=False)
    return unique, quad_vidx


@torch.no_grad()
def _build_gt_domain_mask_from_cell_polygons(
    *,
    centers_comp: torch.Tensor,
    levels: torch.Tensor,
    H: int,
    W: int,
    bbox_comp: tuple[float, float, float, float],
    ref_ratio: float,
    q_grid_xy: np.ndarray,
    src_comp_xy: torch.Tensor,
    src_phys_xy: torch.Tensor,
    affine_sol: torch.Tensor,
    map_mode: str = "idw",
    idw_backend: str = "cdist",
    idw_k: int = 8,
    idw_chunk: int = 32768,
    kdtree=None,
) -> tuple[np.ndarray | None, dict]:
    """
    Build a GT domain mask from GT cell polygons:
      cell centers/levels -> comp corners -> mapped phys corners -> triangle finder mask.
    """
    meta = {"status": "init"}
    try:
        c = torch.as_tensor(centers_comp, dtype=torch.float32, device="cpu")
        l = torch.as_tensor(levels, dtype=torch.long, device="cpu").view(-1)
        if c.ndim != 2 or c.size(1) < 2:
            meta["status"] = "bad_centers_shape"
            return None, meta
        if c.size(0) != l.numel() or l.numel() == 0:
            meta["status"] = "bad_levels_shape"
            return None, meta

        corners_comp = _corners_from_centers_levels(
            c[:, :2].contiguous(),
            l,
            int(H),
            int(W),
            bbox_comp,
            ref_ratio=float(ref_ratio),
        )
        q_comp = corners_comp.view(-1, 2).contiguous()
        mode = str(map_mode).lower().strip()
        if mode not in ("affine", "idw"):
            mode = "idw"

        q_all_phys, q_unique_phys, q_inv, n_unique, n_total, quant_eps = _map_comp_points_to_phys_unique_vertices(
            q_comp,
            bbox_comp=bbox_comp,
            map_mode=mode,
            affine_sol=affine_sol,
            src_comp_xy=src_comp_xy,
            src_phys_xy=src_phys_xy,
            idw_backend=idw_backend,
            idw_k=int(idw_k),
            idw_chunk=int(idw_chunk),
            kdtree=kdtree,
        )
        _ = q_all_phys  # explicit: mapped all corners currently only needed for side effects/shape checks

        quad_vidx = q_inv.view(-1, 4).detach().cpu().numpy().astype(np.int64, copy=False)
        q_unique_np = q_unique_phys.detach().cpu().numpy().astype(np.float32, copy=False)
        mask, tri_meta = _build_domain_mask_from_tri(
            q_unique_np,
            quad_vidx,
            q_grid_xy,
        )

        meta.update(
            {
                "status": str(tri_meta.get("status", "tri_unknown")),
                "mask_mode": "gt_cell_polygon_tri",
                "map_mode": mode,
                "n_unique_corner": int(n_unique),
                "n_total_corner": int(n_total),
                "quant_eps": float(quant_eps),
            }
        )
        for k in ("n_tri_raw", "n_tri_used", "valid_frac", "error"):
            if k in tri_meta:
                meta[k] = tri_meta[k]
        return mask, meta
    except Exception as e:
        meta.update({"status": "exception", "error": str(e)})
        return None, meta


def _build_gt_domain_mask_from_snapshot_affine(
    *,
    raw_step_obj,
    q_grid_xy: np.ndarray,
    bbox_comp: tuple[float, float, float, float],
    affine_sol: torch.Tensor,
    gt_rr_render: float,
    H_gt_override: int | None = None,
    W_gt_override: int | None = None,
) -> tuple[np.ndarray | None, dict]:
    """
    Build GT domain mask directly from raw snapshot AMR geometry:
      (raw pos + raw levels) -> comp occupancy raster -> affine inverse sampling at q_grid.
    This is independent of predicted mesh geometry.
    """
    meta = {"status": "init"}
    try:
        cxy_raw, _pxy_raw = _extract_xy_from_snapshot(raw_step_obj)
        ij_raw, lev_raw = _extract_ij_level_from_snapshot(raw_step_obj)
        H_gt, W_gt = _extract_hw_from_snapshot(raw_step_obj)
        if (H_gt is None or W_gt is None) and (H_gt_override is not None) and (W_gt_override is not None):
            H_gt = int(H_gt_override)
            W_gt = int(W_gt_override)
        if cxy_raw is None or lev_raw is None or H_gt is None or W_gt is None:
            meta["status"] = "missing_raw_fields"
            return None, meta

        cxy = torch.as_tensor(cxy_raw, dtype=torch.float32, device="cpu")
        lev = torch.as_tensor(lev_raw, dtype=torch.long, device="cpu").view(-1)
        if cxy.ndim != 2 or cxy.size(1) < 2 or lev.numel() == 0 or cxy.size(0) != lev.numel():
            meta["status"] = "bad_raw_shapes"
            return None, meta

        Lmax = int(lev.max().item())
        rr = max(2, int(round(float(gt_rr_render))))
        occ_flat, HH, WW = _rasterize_block_common_global(
            centers=cxy[:, :2].contiguous(),
            levels=lev,
            values=torch.ones((lev.numel(), 1), dtype=torch.float32, device="cpu"),
            H=int(H_gt),
            W=int(W_gt),
            bbox=bbox_comp,
            Lmax=Lmax,
            refine_ratio=rr,
        )
        occ = torch.isfinite(occ_flat[:, 0]).view(HH, WW).cpu().numpy()

        q_phys = torch.as_tensor(q_grid_xy, dtype=torch.float32, device="cpu")
        q_comp = _map_phys_points_affine_inverse(q_phys, affine_sol).cpu().numpy()

        xmin, xmax, ymin, ymax = [float(v) for v in bbox_comp]
        dx0 = (xmax - xmin) / float(W_gt)
        dy0 = (ymax - ymin) / float(H_gt)
        scale = rr ** int(Lmax)
        x_units = (q_comp[:, 0] - xmin) / max(dx0, 1.0e-12)
        y_units = (q_comp[:, 1] - ymin) / max(dy0, 1.0e-12)
        cc = np.floor(x_units * scale).astype(np.int64, copy=False)
        rr_idx = np.floor(y_units * scale).astype(np.int64, copy=False)
        inb = (cc >= 0) & (cc < WW) & (rr_idx >= 0) & (rr_idx < HH)

        mask = np.zeros((q_comp.shape[0],), dtype=bool)
        if np.any(inb):
            mask[inb] = occ[rr_idx[inb], cc[inb]]
        meta.update(
            {
                "status": "snapshot_affine",
                "Lmax": int(Lmax),
                "rr": int(rr),
                "grid": f"{WW}x{HH}",
                "valid_frac": float(mask.mean()) if mask.size > 0 else 0.0,
            }
        )
        return mask, meta
    except Exception as e:
        meta.update({"status": "snapshot_affine_exception", "error": str(e)})
        return None, meta


def _set_axis_to_phys_extent(ax, phys_xy: torch.Tensor, pad_frac: float = 0.02):
    x = phys_xy[:, 0].detach().cpu().numpy()
    y = phys_xy[:, 1].detach().cpu().numpy()
    x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
    y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
    dx = max(1e-12, x1 - x0)
    dy = max(1e-12, y1 - y0)
    ax.set_xlim(x0 - pad_frac * dx, x1 + pad_frac * dx)
    ax.set_ylim(y0 - pad_frac * dy, y1 + pad_frac * dy)


def _parse_degree_diag_steps(raw: Optional[str], n_steps: int) -> Optional[list[int]]:
    """
    Parse comma-separated rollout-step selectors into sorted unique 0-based indices.
    Supported tokens: integers and 'last'.
    Returns None when raw is None (feature disabled).
    """
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt == "":
        return []
    if n_steps <= 0:
        return []

    out: set[int] = set()
    for tok in txt.split(","):
        t = tok.strip()
        if not t:
            continue
        if t.lower() == "last":
            idx = n_steps - 1
        else:
            try:
                idx = int(t)
            except ValueError as e:
                raise ValueError(
                    f"Invalid --degree-diag-steps token '{t}'. Use comma-separated 0-based integers and/or 'last'."
                ) from e
        if idx < 0 or idx >= n_steps:
            raise ValueError(
                f"--degree-diag-steps index {idx} is out of range for rollout length {n_steps}."
            )
        out.add(int(idx))
    return sorted(out)


# ----------------- Percentile helpers (optional sampling) ----------------- #

def _as_1d(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1)

def _sample_view_1d(a: np.ndarray, max_n: int) -> np.ndarray:
    """Deterministic stride-based subsample view for quantiles (fast, reproducible)."""
    x = _as_1d(a)
    n = x.size
    if max_n is None or max_n <= 0 or n <= max_n:
        return x
    step = max(1, n // max_n)
    return x[::step]

def _nanpercentile_sample(a: np.ndarray, q: float, max_n: Optional[int]) -> float:
    x = _sample_view_1d(a, max_n)
    # np.nanpercentile handles NaNs; q is [0,100]
    return float(np.nanpercentile(x, q))

def _sym_lims_from_abs(a: np.ndarray, abs_q: float, max_n: Optional[int]) -> Tuple[float, float]:
    x = _sample_view_1d(np.abs(a), max_n)
    lim = float(np.nanpercentile(x, abs_q))
    if not np.isfinite(lim) or lim == 0.0:
        lim = 1e-12
    return -lim, lim


# ----------------- GIF creation (fast + streaming) ----------------- #

@torch.inference_mode()
def make_rollout_gifs(
    examples,
    cfg,
    out_dir: str,
    feature_names=None,
    unify_clims: bool = False,
    fps: int = 4,
    dpi: int = 100,
    draw_edges: bool = True,
    colorbars: str = "per-panel",   # per-panel | shared | none
    clim_sample: Optional[int] = None,
    interp_k: Optional[int] = None,
    interp_chunk: Optional[int] = None,
    progress_every: int = 1,
):
    """
    Streaming writer: frames are written as rendered.
    Major speedup: create ONE figure per timestep, create the PolyCollections once,
    then update arrays/clims per feature and write frames to each feature's GIF.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not examples:
        log("[WARN] make_rollout_gifs: no examples provided; nothing to do.")
        return

    # Ensure chronological
    examples = sorted(examples, key=lambda e: int(e.get("t", 0)))

    speed = cfg.get("speed", {})
    knn_k = int(interp_k if interp_k is not None else cfg.get("loss", {}).get("interp_k", speed.get("knn_k", 8)))
    chunk = int(interp_chunk if interp_chunk is not None else speed.get("interp_chunk", 8192))

    # Feature names
    F = int(examples[0]["gt_t"].shape[1])
    if feature_names is None:
        feature_names = [f"Feat_{i}" for i in range(F)]
    else:
        if len(feature_names) < F:
            feature_names = list(feature_names) + [f"Feat_{i}" for i in range(len(feature_names), F)]

    # Edge styling
    edgecolor = "k" if draw_edges else "none"
    linewidth = 0.15 if draw_edges else 0.0

    # Writers (one per feature)
    duration = 1.0 / max(1, int(fps))
    writers = []
    gif_paths = []
    for f in range(F):
        safe_name = str(feature_names[f]).replace(" ", "_")
        gif_path = os.path.join(out_dir, f"rollout_{safe_name}.gif")
        gif_paths.append(gif_path)
        writers.append(imageio.get_writer(gif_path, mode="I", duration=duration))

    # Optionally compute global clims (still computed via sampling if clim_sample is set)
    global_top_min = None
    global_top_max = None
    global_d_abs = None

    if unify_clims:
        log("[GIF] Computing global color limits (may take a bit)...")
        top_mins, top_maxs, d_abs = [], [], []
        for ex in examples:
            gt_t = ex["gt_t"].detach().cpu().numpy().astype(np.float32, copy=False)
            pred = ex["pred_tp1"].detach().cpu().numpy().astype(np.float32, copy=False)
            gt_tp1 = ex["gt_tp1"].detach().cpu().numpy().astype(np.float32, copy=False)

            # sample from all values without concatenating huge arrays
            top_mins.append(min(
                _nanpercentile_sample(gt_t, 1.0, clim_sample),
                _nanpercentile_sample(pred, 1.0, clim_sample),
                _nanpercentile_sample(gt_tp1, 1.0, clim_sample),
            ))
            top_maxs.append(max(
                _nanpercentile_sample(gt_t, 99.0, clim_sample),
                _nanpercentile_sample(pred, 99.0, clim_sample),
                _nanpercentile_sample(gt_tp1, 99.0, clim_sample),
            ))

            # deltas (once)
            H, W = int(ex["H"]), int(ex["W"])
            mask_pred_parent = _recover_parent_mask(ex, H, W)
            deltas = compute_plot_deltas(
                gt_t_centers=ex["centers_t"].to(torch.float32),
                gt_t_feats=ex["gt_t"].to(torch.float32),
                gt_tp1_centers=ex["centers_tp1"].to(torch.float32),
                gt_tp1_feats=ex["gt_tp1"].to(torch.float32),
                pred_centers=ex["pred_centers"].to(torch.float32),
                pred_levels=ex["pred_levels"].to(torch.int64),
                pred_parents=ex["pred_parents"].to(torch.int64),
                mask_pred=mask_pred_parent,
                pred_feats=ex["pred_tp1"].to(torch.float32),
                H=H, W=W,
                knn_k=knn_k, chunk=chunk,
            )
            # Use abs 99th percentile across all features (sampled)
            for k in ("delta_gt", "delta_pred_gt", "delta_pred_t"):
                arr = deltas[k].detach().cpu().numpy().astype(np.float32, copy=False)
                d_abs.append(_nanpercentile_sample(np.abs(arr), 99.0, clim_sample))

        global_top_min = float(np.min(top_mins)) if top_mins else None
        global_top_max = float(np.max(top_maxs)) if top_maxs else None
        global_d_abs = float(np.max(d_abs)) if d_abs else None
        log(f"[GIF] Global clims: top=[{global_top_min:.3g}, {global_top_max:.3g}], |delta|~{global_d_abs:.3g}")

    # Main rendering loop
    try:
        for step_i, ex in enumerate(examples):
            t_abs = int(ex.get("t", step_i))
            if (progress_every > 0) and (step_i % progress_every == 0):
                log(f"[GIF] timestep {step_i+1}/{len(examples)} (t={t_abs})")

            H, W = int(ex["H"]), int(ex["W"])
            bbox = tuple(ex["bbox"])

            # CPU numpy for plotting updates
            gt_t_centers = ex["centers_t"].detach().cpu().to(torch.float32)
            gt_tp1_centers = ex["centers_tp1"].detach().cpu().to(torch.float32)
            pred_centers = ex["pred_centers"].detach().cpu().to(torch.float32)
            pred_levels = ex["pred_levels"].detach().cpu().to(torch.int64)
            pred_parents = ex["pred_parents"].detach().cpu().to(torch.int64)

            gt_t = ex["gt_t"].detach().cpu().numpy().astype(np.float32, copy=False)          # [Nt, F]
            gt_tp1 = ex["gt_tp1"].detach().cpu().numpy().astype(np.float32, copy=False)      # [Ntp1, F]
            pred = ex["pred_tp1"].detach().cpu().numpy().astype(np.float32, copy=False)      # [Npred, F]

            # deltas ONCE per timestep
            mask_pred_parent = _recover_parent_mask(ex, H, W)
            deltas = compute_plot_deltas(
                gt_t_centers=gt_t_centers,
                gt_t_feats=torch.from_numpy(gt_t),
                gt_tp1_centers=gt_tp1_centers,
                gt_tp1_feats=torch.from_numpy(gt_tp1),
                pred_centers=pred_centers,
                pred_levels=pred_levels,
                pred_parents=pred_parents,
                mask_pred=mask_pred_parent,
                pred_feats=torch.from_numpy(pred),
                H=H, W=W,
                knn_k=knn_k, chunk=chunk,
            )

            # unwrap delta tensors once
            vals_gt, centers_gt, levels_gt = _unwrap_delta(
                deltas.get("delta_gt"),
                fallback_centers=gt_tp1_centers,
                fallback_levels=ex.get("level_tp1", None),
            )
            vals_pgt, centers_pgt, levels_pgt = _unwrap_delta(
                deltas.get("delta_pred_gt"),
                fallback_centers=pred_centers,
                fallback_levels=pred_levels,
            )
            vals_pt, centers_pt, levels_pt = _unwrap_delta(
                deltas.get("delta_pred_t"),
                fallback_centers=pred_centers,
                fallback_levels=pred_levels,
            )

            # Convert unwrapped deltas to numpy once
            vals_gt_np = vals_gt.detach().cpu().numpy().astype(np.float32, copy=False)
            vals_pgt_np = vals_pgt.detach().cpu().numpy().astype(np.float32, copy=False)
            vals_pt_np = vals_pt.detach().cpu().numpy().astype(np.float32, copy=False)

            # --- Create ONE figure per timestep; build collections once using feature 0 ---
            fig, ax = plt.subplots(2, 3, figsize=(12, 8), dpi=int(dpi))
            fig.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.92, wspace=0.25, hspace=0.25)

            supt = fig.suptitle(f"t={t_abs}", fontsize=12)

            # titles (static)
            ax[0, 0].set_title("GT(t)")
            ax[0, 1].set_title("Pred(t+1) | M_pred")
            ax[0, 2].set_title("GT(t+1)")
            ax[1, 0].set_title("Δ₁ = GT(t+1) − GT(t)  (on GT(t+1))")
            ax[1, 1].set_title("Δ₂ = Pred(t+1) − GT(t+1)  (on Pred)")
            ax[1, 2].set_title("Δ₃ = Pred(t+1) − GT(t)  (on Pred)")

            # Build collections once with f=0
            pc_gt_t = _draw_amr_cells(
                ax[0, 0],
                centers=gt_t_centers, levels=ex.get("level_t", None),
                values=torch.from_numpy(gt_t), f_idx=0,
                H=H, W=W, bbox=bbox,
                vmin=0.0, vmax=1.0, cmap="viridis",
                edgecolor=edgecolor, linewidth=linewidth,
            )
            pc_pred = _draw_amr_cells(
                ax[0, 1],
                centers=pred_centers, levels=pred_levels,
                values=torch.from_numpy(pred), f_idx=0,
                H=H, W=W, bbox=bbox,
                vmin=0.0, vmax=1.0, cmap="viridis",
                edgecolor=edgecolor, linewidth=linewidth,
            )
            pc_gt_tp1 = _draw_amr_cells(
                ax[0, 2],
                centers=gt_tp1_centers, levels=ex.get("level_tp1", None),
                values=torch.from_numpy(gt_tp1), f_idx=0,
                H=H, W=W, bbox=bbox,
                vmin=0.0, vmax=1.0, cmap="viridis",
                edgecolor=edgecolor, linewidth=linewidth,
            )

            pc_d1 = _draw_amr_cells(
                ax[1, 0],
                centers=centers_gt, levels=levels_gt,
                values=torch.from_numpy(vals_gt_np), f_idx=_fidx_or_none(torch.from_numpy(vals_gt_np), 0),
                H=H, W=W, bbox=bbox,
                vmin=-1.0, vmax=1.0, cmap="coolwarm",
                edgecolor=edgecolor, linewidth=linewidth,
            )
            pc_d2 = _draw_amr_cells(
                ax[1, 1],
                centers=centers_pgt, levels=levels_pgt,
                values=torch.from_numpy(vals_pgt_np), f_idx=_fidx_or_none(torch.from_numpy(vals_pgt_np), 0),
                H=H, W=W, bbox=bbox,
                vmin=-1.0, vmax=1.0, cmap="coolwarm",
                edgecolor=edgecolor, linewidth=linewidth,
            )
            pc_d3 = _draw_amr_cells(
                ax[1, 2],
                centers=centers_pt, levels=levels_pt,
                values=torch.from_numpy(vals_pt_np), f_idx=_fidx_or_none(torch.from_numpy(vals_pt_np), 0),
                H=H, W=W, bbox=bbox,
                vmin=-1.0, vmax=1.0, cmap="coolwarm",
                edgecolor=edgecolor, linewidth=linewidth,
            )

            # Colorbars (created once per timestep; updated per feature)
            cb_top = cb_d = None
            cb_panels = []
            if colorbars == "per-panel":
                cb_panels = [
                    fig.colorbar(pc_gt_t, ax=ax[0, 0]),
                    fig.colorbar(pc_pred, ax=ax[0, 1]),
                    fig.colorbar(pc_gt_tp1, ax=ax[0, 2]),
                    fig.colorbar(pc_d1, ax=ax[1, 0]),
                    fig.colorbar(pc_d2, ax=ax[1, 1]),
                    fig.colorbar(pc_d3, ax=ax[1, 2]),
                ]
            elif colorbars == "shared":
                cb_top = fig.colorbar(pc_gt_t, ax=ax[0, :].ravel().tolist(), shrink=0.95)
                cb_d = fig.colorbar(pc_d1, ax=ax[1, :].ravel().tolist(), shrink=0.95)
            elif colorbars == "none":
                pass
            else:
                raise ValueError("--colorbars must be per-panel|shared|none")

            # --- Per feature: update arrays/clims and write frame to that feature's GIF ---
            for f in range(F):
                supt.set_text(f"t={t_abs} — {feature_names[f]}")

                # top row limits
                if unify_clims and (global_top_min is not None):
                    tmin, tmax = global_top_min, global_top_max
                else:
                    # 1st/99th percentiles of (GT(t), Pred(t+1), GT(t+1)) for this feature
                    a = gt_t[:, f]
                    b = pred[:, f]
                    c = gt_tp1[:, f]
                    tmin = min(
                        _nanpercentile_sample(a, 1.0, clim_sample),
                        _nanpercentile_sample(b, 1.0, clim_sample),
                        _nanpercentile_sample(c, 1.0, clim_sample),
                    )
                    tmax = max(
                        _nanpercentile_sample(a, 99.0, clim_sample),
                        _nanpercentile_sample(b, 99.0, clim_sample),
                        _nanpercentile_sample(c, 99.0, clim_sample),
                    )
                    if not np.isfinite(tmin) or not np.isfinite(tmax) or (tmin == tmax):
                        tmin, tmax = float(np.nanmin([np.nanmin(a), np.nanmin(b), np.nanmin(c)])), float(np.nanmax([np.nanmax(a), np.nanmax(b), np.nanmax(c)]))

                # delta limits anchored to delta_gt (as in your prior logic)
                if unify_clims and (global_d_abs is not None):
                    dmin, dmax = -global_d_abs, global_d_abs
                else:
                    dmin, dmax = _sym_lims_from_abs(vals_gt_np[:, f], 99.0, clim_sample)

                # update arrays (PolyCollection mappables)
                pc_gt_t.set_array(gt_t[:, f])
                pc_pred.set_array(pred[:, f])
                pc_gt_tp1.set_array(gt_tp1[:, f])

                pc_d1.set_array(vals_gt_np[:, f])
                pc_d2.set_array(vals_pgt_np[:, f])
                pc_d3.set_array(vals_pt_np[:, f])

                # update clims
                pc_gt_t.set_clim(tmin, tmax)
                pc_pred.set_clim(tmin, tmax)
                pc_gt_tp1.set_clim(tmin, tmax)

                pc_d1.set_clim(dmin, dmax)
                pc_d2.set_clim(dmin, dmax)
                pc_d3.set_clim(dmin, dmax)

                # update colorbars (cheap compared to recreating)
                if colorbars == "per-panel":
                    for cb in cb_panels:
                        cb.update_normal(cb.mappable)
                elif colorbars == "shared":
                    if cb_top is not None:
                        cb_top.update_normal(pc_gt_t)
                    if cb_d is not None:
                        cb_d.update_normal(pc_d1)

                # render -> frame
                fig.canvas.draw()
                buf, (w, h) = fig.canvas.print_to_buffer()  # RGBA bytes
                rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                frame = rgba[..., :3].copy()

                writers[f].append_data(frame)

            plt.close(fig)

            # free memory aggressively per timestep
            del deltas, vals_gt, vals_pgt, vals_pt
            if step_i % 5 == 0:
                gc.collect()

        # Close writers
        for w in writers:
            w.close()

    except Exception:
        for w in writers:
            try:
                w.close()
            except Exception:
                pass
        raise

    for p in gif_paths:
        log(f"[INFO] wrote {p}")

def plot_rollout_metrics_maew_rell2w(
    test_stats: dict,
    *,
    feature_names=None,
    start_t: int | None = None,
    save_dir: str | None = None,
    prefix: str = "rollout_metrics",
    dpi: int = 150,
    show: bool = False,
):
    """
    Plot:
      (1) RelL2w for all features on one plot
      (2) MAEw as 2x2 subplot grid (one feature per panel)

    Inputs:
      test_stats: dict returned by evaluate_one_epoch_multi_step (new version),
                  expected keys:
                    - 't_values' (list[int]) optional
                    - 'rell2w_feat_by_t' (torch.Tensor [T,F]) optional
                    - 'maew_feat_by_t' (torch.Tensor [T,F]) optional
                  fallback keys:
                    - 'rell2w_feat_by_rollout_step' (torch.Tensor [S,F])
                    - 'maew_feat_by_rollout_step' (torch.Tensor [S,F])

      feature_names: list[str] length F (optional)
      start_t: optional integer used only for nicer labeling when t_values missing
      save_dir: if provided, saves PNGs there
      prefix: filename prefix
      dpi: save dpi
      show: if True, calls plt.show()

    Returns:
      dict with:
        - 'fig_rell2w', 'ax_rell2w'
        - 'fig_maew', 'axes_maew'
        - 'x', 'x_label'
        - 'paths' (dict of saved file paths, if save_dir provided)
    """

    def _to_numpy(x):
        # torch.Tensor -> numpy
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # ---- Choose x-axis + arrays (prefer absolute time indexing if present) ----
    t_values = test_stats.get("t_values", None)
    rell2_t = test_stats.get("rell2w_feat_by_t", None)
    maew_t  = test_stats.get("maew_feat_by_t", None)

    if t_values is not None and len(t_values) > 0 and rell2_t is not None and maew_t is not None:
        x = np.asarray(t_values, dtype=int)
        rell2 = _to_numpy(rell2_t)  # [T,F]
        maew  = _to_numpy(maew_t)   # [T,F]
        x_label = "Time step (absolute index)"
    else:
        # Fallback: rollout-step axis
        rell2 = _to_numpy(test_stats["rell2w_feat_by_rollout_step"])
        maew  = _to_numpy(test_stats["maew_feat_by_rollout_step"])

        # allow both [S] overall or [S,F] per-feature; enforce [S,F]
        if rell2.ndim == 1:
            rell2 = rell2[:, None]
        if maew.ndim == 1:
            maew = maew[:, None]

        S = rell2.shape[0]
        if start_t is None:
            x = np.arange(1, S + 1, dtype=int)
            x_label = "Rollout step (k=1 is first predicted step)"
        else:
            # interpret step k=1 as predicting t=start_t+1 (common convention)
            x = start_t + np.arange(1, S + 1, dtype=int)
            x_label = "Time step (approx; start_t provided, but no t_values in stats)"

    # ---- Basic dims / labels ----
    if rell2.ndim != 2 or maew.ndim != 2:
        raise ValueError(f"Expected per-feature arrays [N,F]. Got rell2 shape {rell2.shape}, maew shape {maew.shape}")

    N, Fdim = rell2.shape
    if maew.shape[1] != Fdim:
        raise ValueError(f"Feature dim mismatch: rell2 has F={Fdim}, maew has F={maew.shape[1]}")

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(Fdim)]
    else:
        if len(feature_names) != Fdim:
            raise ValueError(f"feature_names length {len(feature_names)} != number of features {Fdim}")

    # =========================
    # (1) RelL2w: single plot
    # =========================
    fig_r, ax_r = plt.subplots(figsize=(8.0, 4.5))
    for f in range(Fdim):
        ax_r.plot(x, rell2[:, f], label=str(feature_names[f]))
    ax_r.set_xlabel(x_label)
    ax_r.set_ylabel(r"$RelL2_w$ (area-weighted relative L$^2$ error)")
    ax_r.set_title("RelL2w vs time step")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="best", frameon=True)

    # =========================
    # (2) MAEw: 2x2 subplots
    # =========================
    fig_m, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    axes_flat = axes.ravel()

    for f in range(4):
        ax = axes_flat[f]
        if f < Fdim:
            ax.plot(x, maew[:, f])
            ax.set_title(str(feature_names[f]))
            ax.set_ylabel("MAEw (area-weighted MAE)")
            ax.grid(True, alpha=0.3)
        else:
            # if Fdim < 4, hide unused panels
            ax.axis("off")

    # x-labels on bottom row
    axes_flat[2].set_xlabel(x_label)
    axes_flat[3].set_xlabel(x_label)
    fig_m.suptitle("MAEw vs time step", y=0.98)

    fig_m.tight_layout(rect=[0, 0, 1, 0.96])

    # ---- Save (optional) ----
    paths = {}
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        p1 = os.path.join(save_dir, f"{prefix}_rell2w.png")
        p2 = os.path.join(save_dir, f"{prefix}_maew_2x2.png")
        fig_r.savefig(p1, dpi=dpi, bbox_inches="tight")
        fig_m.savefig(p2, dpi=dpi, bbox_inches="tight")
        paths["rell2w"] = p1
        paths["maew"] = p2

    if show:
        plt.show()

    return {
        "fig_rell2w": fig_r,
        "ax_rell2w": ax_r,
        "fig_maew": fig_m,
        "axes_maew": axes,
        "x": x,
        "x_label": x_label,
        "paths": paths,
    }

def _fill_internal_nans_nearest(img_flat: torch.Tensor, HH: int, WW: int) -> torch.Tensor:
    """
    Fill only NaN 'holes' that are enclosed by valid pixels (i.e., interior gaps),
    leaving the exterior/background (connected to boundary) untouched.
    """

    try:
        from scipy import ndimage as ndi
    except ImportError as e:
        raise ImportError("This hole-filling helper requires scipy (scipy.ndimage).") from e

    arr = img_flat.view(HH, WW, -1).cpu().numpy()  # (HH,WW,C)
    out = arr.copy()

    # A pixel is "valid" if it has finite values (use any channel; or require all channels)
    valid = np.isfinite(arr[..., 0])

    # Fill holes *inside* the valid region; does NOT fill exterior/background.
    filled = ndi.binary_fill_holes(valid)
    holes = filled & (~valid)
    if not holes.any():
        return img_flat  # nothing to do

    # Nearest-neighbor indices to the closest valid pixel for every location
    # distance_transform_edt works on True=background; we want background = invalid
    _, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)

    # Copy nearest valid pixel values into hole pixels (for every channel)
    for c in range(out.shape[2]):
        chan = out[..., c]
        chan[holes] = chan[iy[holes], ix[holes]]
        out[..., c] = chan

    return torch.from_numpy(out.reshape(-1, out.shape[2])).to(dtype=img_flat.dtype)


def _rasterize_block_common_global(
    centers: torch.Tensor,
    levels: torch.Tensor,
    values: torch.Tensor,
    H: int,
    W: int,
    bbox: tuple[float, float, float, float],
    Lmax: int,
    *,
    refine_ratio: int = 2,
):
    """
    Module-level AMR block replication onto a common fine grid sized
    (H*refine_ratio^Lmax, W*refine_ratio^Lmax).
    """
    device_cpu = torch.device("cpu")
    centers = centers.to(device=device_cpu, dtype=torch.float32)
    levels = levels.to(device=device_cpu, dtype=torch.int64).view(-1)
    values = values.to(device=device_cpu, dtype=torch.float32)

    N = centers.shape[0]
    if values.dim() == 1:
        values = values.view(N, 1)
    C = int(values.shape[1])

    xmin, xmax, ymin, ymax = map(float, bbox)
    dx = (xmax - xmin) / float(W)
    dy = (ymax - ymin) / float(H)

    rr = int(max(1, int(refine_ratio)))
    scale = rr ** int(Lmax)
    HH, WW = int(H * scale), int(W * scale)
    img = torch.full((HH * WW, C), torch.nan, dtype=torch.float32, device=device_cpu)

    x_units = (centers[:, 0] - xmin) / dx
    y_units = (centers[:, 1] - ymin) / dy
    col_fine = torch.clamp(torch.floor(x_units * scale).to(torch.int64), 0, WW - 1)
    row_fine = torch.clamp(torch.floor(y_units * scale).to(torch.int64), 0, HH - 1)

    for l in range(Lmax + 1):
        m = (levels == l)
        if not torch.any(m):
            continue
        b = rr ** int(Lmax - l)
        r0 = (row_fine[m] // b) * b
        c0 = (col_fine[m] // b) * b

        off = torch.arange(b, device=device_cpu, dtype=torch.int64)
        rr2 = (r0[:, None] + off[None, :])[:, :, None].expand(-1, b, b)
        cc2 = (c0[:, None] + off[None, :])[:, None, :].expand(-1, b, b)
        flat_idx = (rr2 * WW + cc2).reshape(-1)
        vrep = values[m].repeat_interleave(b * b, dim=0)
        img[flat_idx] = vrep

    return img, HH, WW


@torch.inference_mode()
def make_rollout_gifs_raster(
    examples,
    cfg,
    out_dir: str,
    feature_names=None,
    unify_clims: bool = False,
    fps: int = 4,
    dpi: int = 100,
    *,
    raster_mode: str = "block",          # "block" (fast) or "idw" (slow for large N)
    raster_bins: int = 256,              # only used for raster_mode="idw"
    raster_k: int = 8,                   # only used for raster_mode="idw"
    raster_chunk: int = 32768,           # only used for raster_mode="idw"
    raster_lmax: int | None = None,      # if None, infer from levels per-step; else clamp to this
    delta_scale: str = "gt",             # "gt" or "each" (when unify_clims=True, we always use "gt")
    progress_every: int = 1,             # print progress every N steps
    include_deltas: bool = False,        # if False, only render Pred(t+1) vs GT(t+1)
    two_panel_fig_w: float = 14.0,       # figure width (inches) for 2-panel mode
    two_panel_fig_h: float = 7.0,        # figure height (inches) for 2-panel mode
):
    """
    Raster/imshow rollout GIF writer.

    Layout:
      - include_deltas=False (default): 1x2 [Pred(t+1), GT(t+1)]
      - include_deltas=True:            2x3
            Top:    [GT(t), Pred(t+1), GT(t+1)]
            Bottom: [GT(t+1)-GT(t), Pred(t+1)-GT(t+1), Pred(t+1)-GT(t)]

    Fixed-scale behavior (when unify_clims=True):
      - Top-row scale is fixed across all timesteps PER FEATURE, determined ONLY from GT(t+1) plots.
      - If include_deltas=True, delta-row scale is fixed across all timesteps PER FEATURE,
        determined ONLY from GT(t+1)-GT(t), and shared across all delta panels.

    Notes:
      - raster_mode="block" uses AMR block replication to a fine uniform grid; fast and memory-safe.
      - raster_mode="idw" is expensive for large N (uses cdist to all centers); generally avoid for big meshes.
    """

    import imageio
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    if not examples:
        print("[WARN] make_rollout_gifs_raster: no examples provided; nothing to do.")
        return

    # Ensure chronological
    examples = sorted(examples, key=lambda e: int(e.get("t", 0)))
    refine_ratio = _get_refine_ratio(cfg)

    # Feature dimension
    F = int(examples[0]["gt_t"].shape[1])
    if feature_names is None:
        feature_names = [f"Feat_{i}" for i in range(F)]
    elif len(feature_names) < F:
        feature_names = list(feature_names) + [f"Feat_{i}" for i in range(len(feature_names), F)]

    # ----------------- helpers -----------------
    def _nanmin_per_feature(x: torch.Tensor) -> torch.Tensor:
        # x: (M,F)
        return torch.nan_to_num(x, nan=float("inf")).amin(dim=0)

    def _nanmax_per_feature(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=float("-inf")).amax(dim=0)

    def _nanmaxabs_per_feature(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x.abs(), nan=0.0).amax(dim=0)

    def _imshow_flat(ax, flat: torch.Tensor, HH: int, WW: int, title: str, vmin=None, vmax=None, cmap="viridis"):
        arr = flat.view(HH, WW).detach().cpu().numpy()
        arr = np.ma.masked_invalid(arr)
        im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return im

    def _infer_lmax(ex) -> int:
        cands = []
        for k in ("pred_levels", "level_t", "level_tp1"):
            if k in ex and ex[k] is not None:
                t = ex[k]
                if torch.is_tensor(t) and t.numel() > 0:
                    cands.append(int(t.max().item()))
        return max(cands) if cands else 0

    def _rasterize_block_common(
        centers: torch.Tensor,
        levels: torch.Tensor,
        values: torch.Tensor,
        H: int,
        W: int,
        bbox: tuple[float, float, float, float],
        Lmax: int,
    ):
        """
        AMR block replication onto a common fine grid sized
        (H*refine_ratio^Lmax, W*refine_ratio^Lmax).
        Fine levels overwrite coarse levels naturally by iterating l=0..Lmax.
        """
        device_cpu = torch.device("cpu")
        centers = centers.to(device=device_cpu, dtype=torch.float32)
        levels = levels.to(device=device_cpu, dtype=torch.int64).view(-1)
        values = values.to(device=device_cpu, dtype=torch.float32)

        N = centers.shape[0]
        if values.dim() == 1:
            values = values.view(N, 1)
        C = int(values.shape[1])

        xmin, xmax, ymin, ymax = map(float, bbox)
        dx = (xmax - xmin) / float(W)
        dy = (ymax - ymin) / float(H)

        scale = int(refine_ratio) ** int(Lmax)
        HH, WW = int(H * scale), int(W * scale)

        img = torch.full((HH * WW, C), torch.nan, dtype=torch.float32, device=device_cpu)

        # map centers to fine-grid indices
        x_units = (centers[:, 0] - xmin) / dx
        y_units = (centers[:, 1] - ymin) / dy
        col_fine = torch.clamp(torch.floor(x_units * scale).to(torch.int64), 0, WW - 1)
        row_fine = torch.clamp(torch.floor(y_units * scale).to(torch.int64), 0, HH - 1)

        for l in range(Lmax + 1):
            m = (levels == l)
            if not torch.any(m):
                continue
            b = int(refine_ratio) ** int(Lmax - l)  # block size in fine pixels
            r0 = (row_fine[m] // b) * b
            c0 = (col_fine[m] // b) * b

            off = torch.arange(b, device=device_cpu, dtype=torch.int64)
            rr = r0[:, None] + off[None, :]          # (n,b)
            cc = c0[:, None] + off[None, :]          # (n,b)

            rr2 = rr[:, :, None].expand(-1, b, b)    # (n,b,b)
            cc2 = cc[:, None, :].expand(-1, b, b)    # (n,b,b)
            flat_idx = (rr2 * WW + cc2).reshape(-1)  # (n*b*b,)

            vrep = values[m].repeat_interleave(b * b, dim=0)  # (n*b*b,C)
            img[flat_idx] = vrep

        return img, HH, WW

    def _rasterize_idw(
        centers: torch.Tensor,
        values: torch.Tensor,
        bins: int,
        k: int,
        bbox: tuple[float, float, float, float],
        chunk: int,
    ):
        """
        IDW from cell centers to a uniform bins×bins grid.
        WARNING: expensive for large N (uses cdist).
        """
        device_cpu = torch.device("cpu")
        centers = centers.to(device=device_cpu, dtype=torch.float32)
        values = values.to(device=device_cpu, dtype=torch.float32)
        if values.dim() == 1:
            values = values[:, None]
        F_ = int(values.shape[1])

        xmin, xmax, ymin, ymax = map(float, bbox)
        xs = torch.linspace(xmin, xmax, bins, device=device_cpu)
        ys = torch.linspace(ymin, ymax, bins, device=device_cpu)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (M,2)
        M = grid.size(0)

        out = torch.full((M, F_), torch.nan, device=device_cpu, dtype=torch.float32)

        N = centers.size(0)
        kk = min(int(k), int(N))
        for s in range(0, M, int(chunk)):
            e = min(s + int(chunk), M)
            q = grid[s:e]                    # (m,2)
            d = torch.cdist(q, centers)      # (m,N)
            d, idx = torch.topk(d, k=kk, dim=1, largest=False)
            w = 1.0 / (d + 1e-8)
            w = w / w.sum(dim=1, keepdim=True)
            vals = values.index_select(0, idx.reshape(-1)).view(idx.size(0), idx.size(1), F_)
            out[s:e] = (w.unsqueeze(-1) * vals).sum(dim=1)
        return out, bins, bins

    # ----------------- PASS 1: compute fixed clims (if requested) -----------------
    global_top_min_f = None  # torch.Tensor (F,)
    global_top_max_f = None  # torch.Tensor (F,)
    global_d_abs_f   = None  # torch.Tensor (F,)

    if unify_clims:
        print("[CLIM] Computing fixed clims across rollout (per feature).")
        global_top_min_f = torch.full((F,), float("inf"))
        global_top_max_f = torch.full((F,), float("-inf"))
        if include_deltas:
            global_d_abs_f = torch.zeros((F,), dtype=torch.float32)

        for step, ex in enumerate(examples):
            H = int(ex["H"]); W = int(ex["W"])
            bbox = tuple(ex["bbox"])

            step_lmax = _infer_lmax(ex)
            if raster_lmax is not None:
                step_lmax = min(int(step_lmax), int(raster_lmax))

            if not hasattr(make_rollout_gifs_raster, "_printed_keys"):
                make_rollout_gifs_raster._printed_keys = True
                print("[ROLLOUT-CHK] ex keys:", sorted(list(ex.keys())))

            # Pull GT(t+1) for top-row clims (always needed)
            B_cent = ex["centers_tp1"]

            # Prefer level_tp1 if present; block needs it
            B_lev = ex.get("level_tp1", None)

            B_val = ex["gt_tp1"]

            if raster_mode.lower() == "idw":
                B_img, HH, WW = _rasterize_idw(B_cent, B_val, raster_bins, raster_k, bbox, raster_chunk)
            else:
                if B_lev is None:
                    raise KeyError(
                        "unify_clims=True with raster_mode='block' requires ex['level_tp1'] "
                        "in every example."
                    )
                HH, WW = (int(H * (refine_ratio ** step_lmax)), int(W * (refine_ratio ** step_lmax)))
                B_img, _,  _  = _rasterize_block_common(B_cent, B_lev, B_val, H, W, bbox, step_lmax)

            B_img = _fill_internal_nans_nearest(B_img, HH, WW)

            # TOP clims from GT(t+1) ONLY
            global_top_min_f = torch.minimum(global_top_min_f, _nanmin_per_feature(B_img))
            global_top_max_f = torch.maximum(global_top_max_f, _nanmax_per_feature(B_img))

            # DELTA clims from GTΔ ONLY (when requested)
            if include_deltas:
                A_cent = ex["centers_t"]
                A_lev = ex.get("level_t", None)
                A_val = ex["gt_t"]
                if raster_mode.lower() == "idw":
                    A_img, _, _ = _rasterize_idw(A_cent, A_val, raster_bins, raster_k, bbox, raster_chunk)
                else:
                    if A_lev is None:
                        raise KeyError(
                            "unify_clims=True with include_deltas=True and raster_mode='block' "
                            "requires ex['level_t'] in every example."
                        )
                    A_img, _, _ = _rasterize_block_common(A_cent, A_lev, A_val, H, W, bbox, step_lmax)
                A_img = _fill_internal_nans_nearest(A_img, HH, WW)
                Dgt = (B_img - A_img)
                global_d_abs_f = torch.maximum(global_d_abs_f, _nanmaxabs_per_feature(Dgt))

            if (step + 1) % max(1, int(progress_every)) == 0:
                t = int(ex.get("t", 0))
                print(f"[CLIM] step={step+1}/{len(examples)} t={t} (grid={HH}x{WW}, Lmax={step_lmax})")

        # Guard degenerate/non-finite
        bad = ~torch.isfinite(global_top_min_f) | ~torch.isfinite(global_top_max_f)
        if torch.any(bad):
            global_top_min_f[bad] = 0.0
            global_top_max_f[bad] = 1.0

        eq = (global_top_min_f == global_top_max_f)
        if torch.any(eq):
            eps = torch.where(global_top_min_f == 0.0, torch.tensor(1e-12), global_top_min_f.abs() * 1e-6)
            global_top_min_f = torch.where(eq, global_top_min_f - eps, global_top_min_f)
            global_top_max_f = torch.where(eq, global_top_max_f + eps, global_top_max_f)

        if include_deltas:
            global_d_abs_f = torch.where((~torch.isfinite(global_d_abs_f)) | (global_d_abs_f <= 0.0),
                                         torch.tensor(1e-12),
                                         global_d_abs_f)

        print("[CLIM] Done.")
        print("[CLIM] Example (first 3 feats):")
        for f in range(min(3, F)):
            if include_deltas:
                print(f"       f={f}: top=[{float(global_top_min_f[f]):.6e}, {float(global_top_max_f[f]):.6e}] "
                      f"delta=±{float(global_d_abs_f[f]):.6e}")
            else:
                print(f"       f={f}: top=[{float(global_top_min_f[f]):.6e}, {float(global_top_max_f[f]):.6e}]")

    # ----------------- GIF writers -----------------
    duration = 1.0 / max(1, int(fps))
    writers = []
    gif_paths = []
    try:
        for f in range(F):
            safe = str(feature_names[f]).replace(" ", "_")
            path = os.path.join(out_dir, f"rollout_{safe}.gif")
            gif_paths.append(path)
            writers.append(imageio.get_writer(path, mode="I", duration=duration))

        # ----------------- PASS 2: render frames -----------------
        prev_n = None
        prev_s = None

        for step, ex in enumerate(examples):
            H = int(ex["H"]); W = int(ex["W"])
            t = int(ex.get("t", 0))
            bbox = tuple(ex["bbox"])

            step_lmax = _infer_lmax(ex)
            if raster_lmax is not None:
                step_lmax = min(int(step_lmax), int(raster_lmax))

            # pull meshes/values
            B_cent = ex["centers_tp1"]
            P_cent = ex["pred_centers"]

            B_lev = ex.get("level_tp1", ex.get("pred_levels", None))
            P_lev = ex["pred_levels"]

            B_val = ex["gt_tp1"]
            P_val = ex["pred_tp1"]

            # per-step geometry debug (optional)
            n = int(P_cent.shape[0])
            s10 = float(P_cent[:10].sum().item()) if n >= 10 else float(P_cent.sum().item())
            if prev_n is None:
                print(f"[MESH] step={step} t={t} N={n} sum10={s10:.6e}")
            else:
                print(f"[MESH] step={step} t={t} N={n} sum10={s10:.6e} (ΔN={n-prev_n:+d}, Δsum10={s10-prev_s:+.3e})")
            prev_n, prev_s = n, s10
            
            # Rasterize shared top-row fields.
            if raster_mode.lower() == "idw":
                B_img, HH, WW = _rasterize_idw(B_cent, B_val, raster_bins, raster_k, bbox, raster_chunk)
                P_img, _,  _  = _rasterize_idw(P_cent, P_val, raster_bins, raster_k, bbox, raster_chunk)
                mesh_label = f"(idw {raster_bins}×{raster_bins}, k={raster_k})"
            else:
                if B_lev is None:
                    raise KeyError("Block raster requires level_tp1 (or compatible) in each example.")
                B_img, HH, WW = _rasterize_block_common(B_cent, B_lev, B_val, H, W, bbox, step_lmax)
                P_img, _,  _  = _rasterize_block_common(P_cent, P_lev, P_val, H, W, bbox, step_lmax)
                mesh_label = f"(block @ {HH}×{WW}, Lmax={step_lmax})"

            B_img = _fill_internal_nans_nearest(B_img, HH, WW)
            P_img = _fill_internal_nans_nearest(P_img, HH, WW)

            # Optional delta terms (expensive, including an extra IDW mapping in block mode).
            Dgt = Dpg = Dpt = None
            if include_deltas:
                A_cent = ex["centers_t"]
                A_lev = ex.get("level_t", ex.get("pred_levels", None))
                A_val = ex["gt_t"]
                if raster_mode.lower() == "idw":
                    A_img, _, _ = _rasterize_idw(A_cent, A_val, raster_bins, raster_k, bbox, raster_chunk)
                else:
                    if A_lev is None:
                        raise KeyError("Block raster with include_deltas=True requires level_t in each example.")
                    A_img, _, _ = _rasterize_block_common(A_cent, A_lev, A_val, H, W, bbox, step_lmax)

                A_img = _fill_internal_nans_nearest(A_img, HH, WW)
                Dgt = (B_img - A_img)  # GT(t+1)-GT(t)

                if raster_mode.lower() == "block":
                    # Map GT(t+1) features onto Pred(t+1) centers (IDW on centers).
                    idx_tp1, w_tp1 = build_idw_map(
                        dst_xy=P_cent.to(dtype=torch.float32, device="cpu"),
                        src_xy=B_cent.to(dtype=torch.float32, device="cpu"),
                        k=raster_k,
                        chunk=raster_chunk,
                    )
                    gt_tp1_on_pred = apply_idw_map(
                        idx_tp1, w_tp1, B_val.to(dtype=torch.float32, device="cpu")
                    )  # (N_pred, F)

                    # Rasterize mapped GT(t+1) on pred mesh geometry.
                    B_on_pred_img, _, _ = _rasterize_block_common(
                        centers=P_cent,
                        levels=P_lev,
                        values=gt_tp1_on_pred,
                        H=H, W=W, bbox=bbox, Lmax=step_lmax,
                    )
                    B_on_pred_img = _fill_internal_nans_nearest(B_on_pred_img, HH, WW)
                    Dpg = (P_img - B_on_pred_img)
                else:
                    Dpg = (P_img - B_img)

                Dpt = (P_img - A_img)  # Pred(t+1)-GT(t)

            # per-feature frames
            for f in range(F):
                # TOP scale: determined by GT(t+1) ONLY (B_img)
                if unify_clims:
                    tmin = float(global_top_min_f[f].item())
                    tmax = float(global_top_max_f[f].item())
                else:
                    # per-step, still anchored to GT(t+1) only
                    bmin = float(torch.nan_to_num(B_img[:, f], nan=float("inf")).min().item())
                    bmax = float(torch.nan_to_num(B_img[:, f], nan=float("-inf")).max().item())
                    if not np.isfinite(bmin) or not np.isfinite(bmax) or bmin == bmax:
                        bmin, bmax = 0.0, 1.0
                    tmin, tmax = bmin, bmax

                if include_deltas:
                    # DELTA scale: determined by GTΔ ONLY (Dgt), shared by all 3 delta panels.
                    if unify_clims:
                        m = float(global_d_abs_f[f].item())
                    else:
                        m = float(torch.nan_to_num(Dgt[:, f].abs(), nan=0.0).max().item())

                    # If user insists on "each" and unify_clims is off, allow it; otherwise force "gt".
                    if (not unify_clims) and (delta_scale.lower() == "each"):
                        m1 = float(torch.nan_to_num(Dgt[:, f].abs(), nan=0.0).max().item())
                        m2 = float(torch.nan_to_num(Dpg[:, f].abs(), nan=0.0).max().item())
                        m3 = float(torch.nan_to_num(Dpt[:, f].abs(), nan=0.0).max().item())
                        lims = [(-m1, +m1), (-m2, +m2), (-m3, +m3)]
                    else:
                        lims = [(-m, +m), (-m, +m), (-m, +m)]

                    fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=int(dpi), constrained_layout=True)
                    fig.suptitle(f"t={t} — {feature_names[f]} {mesh_label}", fontsize=12)

                    # Top row: GT(t), Pred(t+1), GT(t+1)
                    im0 = _imshow_flat(axs[0, 0], A_img[:, f], HH, WW, "GT(t)",     vmin=tmin, vmax=tmax, cmap="viridis")
                    im1 = _imshow_flat(axs[0, 1], P_img[:, f], HH, WW, "Pred(t+1)", vmin=tmin, vmax=tmax, cmap="viridis")
                    im2 = _imshow_flat(axs[0, 2], B_img[:, f], HH, WW, "GT(t+1)",   vmin=tmin, vmax=tmax, cmap="viridis")

                    fig.colorbar(im0, ax=axs[0, 0], shrink=0.75)
                    fig.colorbar(im1, ax=axs[0, 1], shrink=0.75)
                    fig.colorbar(im2, ax=axs[0, 2], shrink=0.75)

                    # Bottom row: deltas.
                    im3 = _imshow_flat(axs[1, 0], Dgt[:, f], HH, WW, "GT(t+1) − GT(t)",
                                       vmin=lims[0][0], vmax=lims[0][1], cmap="coolwarm")
                    im4 = _imshow_flat(axs[1, 1], Dpg[:, f], HH, WW, "Pred(t+1) − GT(t+1)",
                                       vmin=lims[1][0], vmax=lims[1][1], cmap="coolwarm")
                    im5 = _imshow_flat(axs[1, 2], Dpt[:, f], HH, WW, "Pred(t+1) − GT(t)",
                                       vmin=lims[2][0], vmax=lims[2][1], cmap="coolwarm")

                    fig.colorbar(im3, ax=axs[1, 0], shrink=0.75)
                    fig.colorbar(im4, ax=axs[1, 1], shrink=0.75)
                    fig.colorbar(im5, ax=axs[1, 2], shrink=0.75)
                else:
                    fig, axs = plt.subplots(
                        1,
                        2,
                        figsize=(float(two_panel_fig_w), float(two_panel_fig_h)),
                        dpi=int(dpi),
                        constrained_layout=True,
                    )
                    fig.suptitle(f"t={t} — {feature_names[f]} {mesh_label}", fontsize=12)
                    im_pred = _imshow_flat(axs[0], P_img[:, f], HH, WW, "Pred(t+1)", vmin=tmin, vmax=tmax, cmap="viridis")
                    im_gt = _imshow_flat(axs[1], B_img[:, f], HH, WW, "GT(t+1)", vmin=tmin, vmax=tmax, cmap="viridis")
                    fig.colorbar(im_pred, ax=axs[0], shrink=0.75)
                    fig.colorbar(im_gt, ax=axs[1], shrink=0.75)

                # convert to frame
                fig.canvas.draw()
                buf, (w, h) = fig.canvas.print_to_buffer()
                rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                frame = rgba[..., :3].copy()
                plt.close(fig)

                writers[f].append_data(frame)

            if (step + 1) % max(1, int(progress_every)) == 0:
                print(f"[GIF] wrote frames for step {step+1}/{len(examples)} (t={t})")

        for w in writers:
            w.close()

    except Exception:
        for w in writers:
            try:
                w.close()
            except Exception:
                pass
        raise

    for p in gif_paths:
        print(f"[INFO] wrote {p}")


@torch.inference_mode()
def make_rollout_gifs_truth(
    examples,
    cfg,
    raw_series,
    out_dir: str,
    feature_names=None,
    unify_clims: bool = False,
    fps: int = 4,
    dpi: int = 100,
    *,
    progress_every: int = 1,
    two_panel_fig_w: float = 14.0,
    two_panel_fig_h: float = 7.0,
    truth_map_mode: str = "auto",      # auto|idw|affine
    truth_map_affine_tol: float = 1e-5,
    truth_idw_backend: str = "auto",   # auto|cdist|kdtree
    truth_idw_k: int = 8,
    truth_idw_chunk: int = 32768,
    truth_gt_ref_ratio: str = "auto",  # auto|2|4|...
    truth_gt_render_level: str = "max",  # max|0|1|...
    truth_zoom_features: str = "",  # off|density|energy|both|density,energy
    truth_zoom_w_frac: float = 0.28,
    truth_zoom_h_frac: float = 0.28,
    truth_gt_comp_view: bool = False,
    truth_pred_ref_ratio: float | None = None,
):
    """
    Physical-space truth renderer for 2-panel rollout GIFs:
      [Pred(t+1), GT(t+1)] rendered as warped AMR polygons in physical coordinates.

    This mode uses per-timestep comp->phys mapping from raw_series snapshot fields
    (pos -> pos_phys), and supports:
      - explicit GT refinement ratio (e.g. 4), and
      - GT render level control relative to simulation base mesh HxW.
      - optional moving zoom GIFs for selected features (density/energy).
    """
    os.makedirs(out_dir, exist_ok=True)
    if not examples:
        print("[WARN] make_rollout_gifs_truth: no examples provided; nothing to do.")
        return
    if raw_series is None:
        raise RuntimeError("make_rollout_gifs_truth requires raw_series with pos/pos_phys data.")

    examples = sorted(examples, key=lambda e: int(e.get("t", 0)))
    bbox_comp = tuple(float(v) for v in _get_bbox(cfg))

    F = int(examples[0]["gt_t"].shape[1])
    if feature_names is None:
        feature_names = [f"Feat_{i}" for i in range(F)]
    elif len(feature_names) < F:
        feature_names = list(feature_names) + [f"Feat_{i}" for i in range(len(feature_names), F)]

    zoom_idx = _resolve_truth_zoom_feature_indices(feature_names, truth_zoom_features)
    zoom_idx_set = set(int(i) for i in zoom_idx)
    if zoom_idx:
        selected = [str(feature_names[i]) for i in zoom_idx]
        log(
            "[TRUTH-ZOOM] enabled moving ROI zoom for features: "
            f"{selected} (w_frac={float(truth_zoom_w_frac):.3f}, h_frac={float(truth_zoom_h_frac):.3f})"
        )

    rr_cfg = float(_get_refine_ratio(cfg))
    pred_ref_ratio = float(rr_cfg if truth_pred_ref_ratio is None else truth_pred_ref_ratio)

    gt_rr_arg = str(truth_gt_ref_ratio).strip().lower()
    if gt_rr_arg != "auto":
        try:
            gt_rr_fixed = float(gt_rr_arg)
            if gt_rr_fixed <= 1.0:
                raise ValueError
        except Exception as e:
            raise ValueError(
                f"Invalid --truth-gt-ref-ratio='{truth_gt_ref_ratio}'. Use 'auto' or a numeric value > 1."
            ) from e
    else:
        gt_rr_fixed = None

    idw_backend = str(truth_idw_backend).lower().strip()
    if idw_backend not in ("auto", "cdist", "kdtree"):
        raise ValueError(
            f"Invalid --truth-idw-backend='{truth_idw_backend}'. Use auto|cdist|kdtree."
        )
    if idw_backend == "auto":
        idw_backend_eff = "kdtree" if (_SciPyKDTree is not None) else "cdist"
    else:
        idw_backend_eff = idw_backend
    if (idw_backend_eff == "kdtree") and (_SciPyKDTree is None):
        raise RuntimeError("truth-idw backend 'kdtree' requested but scipy is unavailable.")

    map_mode = str(truth_map_mode).lower().strip()
    if map_mode not in ("auto", "idw", "affine"):
        raise ValueError(f"Invalid --truth-map-mode='{truth_map_mode}'. Use auto|idw|affine.")

    # Build strict per-step GT cache from raw snapshots (no t-shift fallback).
    # Contract: each example step i stores transition t -> t+1 and GT tensors are on (t+1).
    comp_phys_cache = {}
    resolved_t = []
    for ex in examples:
        t_hint = int(ex.get("t", -1))
        t_abs = int(t_hint) + 1
        if t_abs < 0 or t_abs >= len(raw_series):
            raise RuntimeError(
                "GT strict mode requires raw snapshot at t+1 for each transition. "
                f"Got transition t={t_hint}, required t+1={t_abs}, series_length={len(raw_series)}"
            )
        resolved_t.append(t_abs)
        if t_abs in comp_phys_cache:
            continue

        rec = _extract_truth_snapshot_strict(
            raw_series[t_abs],
            feature_names=feature_names,
            F_target=F,
        )
        aff_sol, aff_rmse = _fit_affine_comp_to_phys(
            rec["comp"].contiguous(),
            rec["phys"].contiguous(),
        )
        rec["affine_sol"] = aff_sol
        rec["affine_rmse"] = float(aff_rmse)
        comp_phys_cache[t_abs] = rec

    # Global physical extent for consistent axes.
    phys_extent_points = torch.cat([rec["phys"] for rec in comp_phys_cache.values()], dim=0)
    x_all = phys_extent_points[:, 0].detach().cpu().numpy()
    y_all = phys_extent_points[:, 1].detach().cpu().numpy()
    x0 = float(np.nanmin(x_all)); x1 = float(np.nanmax(x_all))
    y0 = float(np.nanmin(y_all)); y1 = float(np.nanmax(y_all))
    dx = max(1e-12, x1 - x0); dy = max(1e-12, y1 - y0)
    pad = 0.02
    x0p, x1p = x0 - pad * dx, x1 + pad * dx
    y0p, y1p = y0 - pad * dy, y1 + pad * dy

    # Resolve GT base mesh resolution from raw simulation snapshots.
    hw_hist = {}
    rr_hist = {}
    for rec in comp_phys_cache.values():
        key = (int(rec["H_gt"]), int(rec["W_gt"]))
        hw_hist[key] = int(hw_hist.get(key, 0)) + 1
        rrh = rec.get("rr_hint", None)
        if rrh is not None and np.isfinite(float(rrh)) and float(rrh) > 1.0:
            rr_key = float(rrh)
            rr_hist[rr_key] = int(rr_hist.get(rr_key, 0)) + 1
    if not hw_hist:
        raise RuntimeError("GT strict mode: no valid raw snapshot geometry records were found.")
    (gt_base_h, gt_base_w), _hw_count = max(hw_hist.items(), key=lambda kv: kv[1])
    if len(hw_hist) > 1:
        raise RuntimeError(
            "GT strict mode requires consistent base H/W across steps; "
            f"found={sorted(hw_hist.items())}"
        )

    # Resolve GT refinement ratio used for truth-grid construction.
    if gt_rr_fixed is not None:
        gt_rr_render = float(gt_rr_fixed)
        gt_rr_src = "arg"
    else:
        if not rr_hist:
            raise RuntimeError(
                "GT strict mode could not infer refinement ratio from raw snapshot ij/level/H/W metadata."
            )
        gt_rr_render = float(max(rr_hist.items(), key=lambda kv: kv[1])[0])
        gt_rr_src = "snapshot_ij"

    # Resolve GT render level (0..Lmax or max).
    gt_lmax_global = 0
    for rec in comp_phys_cache.values():
        lev_here = rec["level"].view(-1)
        if lev_here.numel() > 0:
            gt_lmax_global = max(gt_lmax_global, int(lev_here.max().item()))

    lvl_arg = str(truth_gt_render_level).strip().lower()
    if lvl_arg in ("max", "auto", "-1"):
        gt_render_level = int(gt_lmax_global)
    else:
        try:
            gt_render_level = int(lvl_arg)
        except Exception as e:
            raise ValueError(
                f"Invalid --truth-gt-render-level='{truth_gt_render_level}'. Use 'max' or an integer >= 0."
            ) from e
        if gt_render_level < 0:
            raise ValueError(
                f"Invalid --truth-gt-render-level='{truth_gt_render_level}'. Must be >= 0 or 'max'."
            )
        if gt_render_level > int(gt_lmax_global):
            raise ValueError(
                f"--truth-gt-render-level={gt_render_level} exceeds GT max level {gt_lmax_global} for this rollout."
            )

    gt_scale = float(gt_rr_render) ** int(gt_render_level)
    gt_grid_nx = max(2, int(round(float(gt_base_w) * gt_scale)))
    gt_grid_ny = max(2, int(round(float(gt_base_h) * gt_scale)))
    gx = np.linspace(x0p, x1p, gt_grid_nx, dtype=np.float32)
    gy = np.linspace(y0p, y1p, gt_grid_ny, dtype=np.float32)
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    q_grid_xy = np.stack([GX.reshape(-1), GY.reshape(-1)], axis=1).astype(np.float32, copy=False)

    if int(gt_grid_nx) * int(gt_grid_ny) > 8_000_000:
        log(
            f"[TRUTH] warning: very large GT interpolation grid {gt_grid_nx}x{gt_grid_ny}; "
            "consider lowering --truth-gt-render-level for faster plotting."
        )

    log(
        "[TRUTH] physical render mode: "
        f"map_mode={map_mode} idw_backend={idw_backend_eff} k={int(truth_idw_k)} "
        f"chunk={int(truth_idw_chunk)} pred_r={pred_ref_ratio:g} "
        f"gt_base={gt_base_w}x{gt_base_h} gt_r={gt_rr_render:g}({gt_rr_src}) "
        f"gt_level={gt_render_level} affine_tol={float(truth_map_affine_tol):.2e} "
        f"gt_interp_grid={gt_grid_nx}x{gt_grid_ny}"
    )
    if bool(truth_gt_comp_view):
        log("[TRUTH] GT view mode: comp-like (uses truth-map-mode mapping, e.g. affine/auto).")
    else:
        log("[TRUTH] GT view mode: physical (default; forces IDW phys->comp sampling for GT raster).")
    log("[TRUTH] GT renderer: strict_raw_amr_paint (no GT fallbacks, no scattered GT interpolation).")
    log("[TRUTH] pred renderer: legacy block-raster imshow (reverted from polygon fill).")

    static_gt_interp_mask = None
    static_gt_interp_mask_meta = {"status": "disabled"}
    if (not bool(truth_gt_comp_view)) and len(examples) > 0:
        try:
            ex0 = examples[0]
            H0_pred = int(ex0.get("H", cfg.get("data", {}).get("H", 64)))
            W0_pred = int(ex0.get("W", cfg.get("data", {}).get("W", 64)))
            p0_cent = torch.as_tensor(ex0["pred_centers"], dtype=torch.float32, device="cpu")
            p0_lev = torch.as_tensor(ex0["pred_levels"], dtype=torch.long, device="cpu").view(-1)
            if p0_cent.ndim == 2 and p0_cent.size(0) > 0:
                p0_lmax = int(p0_lev.max().item()) if p0_lev.numel() > 0 else 0
                ones = torch.ones((int(p0_cent.size(0)), 1), dtype=torch.float32, device="cpu")
                occ_flat, occ_h, occ_w = _rasterize_block_common_global(
                    p0_cent,
                    p0_lev,
                    ones,
                    H0_pred,
                    W0_pred,
                    bbox_comp,
                    p0_lmax,
                    refine_ratio=int(round(float(pred_ref_ratio))),
                )
                occ0 = torch.isfinite(occ_flat[:, 0]).view(occ_h, occ_w).detach().cpu().numpy()
                yy = np.linspace(0, max(0, occ_h - 1), gt_grid_ny, dtype=np.float64)
                xx = np.linspace(0, max(0, occ_w - 1), gt_grid_nx, dtype=np.float64)
                yi = np.clip(np.rint(yy).astype(np.int64), 0, max(0, occ_h - 1))
                xi = np.clip(np.rint(xx).astype(np.int64), 0, max(0, occ_w - 1))
                static_gt_interp_mask = np.asarray(occ0[np.ix_(yi, xi)], dtype=bool)
                static_gt_interp_mask_meta = {
                    "status": "ok",
                    "source": "pred_domain_step0",
                    "pred_grid": f"{occ_w}x{occ_h}",
                    "gt_grid": f"{gt_grid_nx}x{gt_grid_ny}",
                    "valid_frac": float(np.mean(static_gt_interp_mask)) if static_gt_interp_mask.size > 0 else 0.0,
                }
                log(
                    "[TRUTH] static GT interpolation mask: "
                    f"source={static_gt_interp_mask_meta['source']} "
                    f"pred_grid={static_gt_interp_mask_meta['pred_grid']} "
                    f"gt_grid={static_gt_interp_mask_meta['gt_grid']} "
                    f"valid={100.0*static_gt_interp_mask_meta['valid_frac']:.1f}%"
                )
        except Exception as e:
            static_gt_interp_mask = None
            static_gt_interp_mask_meta = {"status": "build_failed", "error": str(e)}
            log(f"[TRUTH] static GT interpolation mask disabled (build failed): {e}")

    # Optional fixed top-row clims (GT only, per feature).
    top_min_f = top_max_f = None
    if unify_clims:
        top_min_f = torch.full((F,), float("inf"), dtype=torch.float32)
        top_max_f = torch.full((F,), float("-inf"), dtype=torch.float32)
        for t_abs_i in sorted(set(int(t) for t in resolved_t)):
            gt_vals = comp_phys_cache[int(t_abs_i)]["features"]
            vmin = torch.nan_to_num(gt_vals, nan=float("inf")).amin(dim=0)
            vmax = torch.nan_to_num(gt_vals, nan=float("-inf")).amax(dim=0)
            top_min_f = torch.minimum(top_min_f, vmin)
            top_max_f = torch.maximum(top_max_f, vmax)

        bad = (~torch.isfinite(top_min_f)) | (~torch.isfinite(top_max_f)) | (top_min_f == top_max_f)
        if torch.any(bad):
            top_min_f = torch.where(bad, torch.zeros_like(top_min_f), top_min_f)
            top_max_f = torch.where(bad, torch.ones_like(top_max_f), top_max_f)

    # GIF writers
    duration = 1.0 / max(1, int(fps))
    writers = []
    gif_paths = []
    for f in range(F):
        safe = str(feature_names[f]).replace(" ", "_")
        path = os.path.join(out_dir, f"rollout_{safe}.gif")
        gif_paths.append(path)
        writers.append(imageio.get_writer(path, mode="I", duration=duration))

    zoom_writers = {}
    zoom_paths = {}
    zoom_center_by_feat = {}
    if zoom_idx:
        for f in zoom_idx:
            safe = str(feature_names[f]).replace(" ", "_")
            zpath = os.path.join(out_dir, f"rollout_{safe}_zoom.gif")
            zoom_paths[int(f)] = zpath
            zoom_writers[int(f)] = imageio.get_writer(zpath, mode="I", duration=duration)
            zoom_center_by_feat[int(f)] = None

    last_tree_t = None
    last_tree = None
    unique_corner_logged = False
    try:
        for step, ex in enumerate(examples):
            t_abs = int(resolved_t[step])
            rec = comp_phys_cache[t_abs]
            src_comp = rec["comp"]
            src_phys = rec["phys"]
            H_gt = int(rec["H_gt"])
            W_gt = int(rec["W_gt"])
            affine_sol = rec["affine_sol"]
            affine_rmse = float(rec["affine_rmse"])

            pred_cent = torch.as_tensor(ex["pred_centers"], dtype=torch.float32, device="cpu")
            pred_lev = torch.as_tensor(ex["pred_levels"], dtype=torch.long, device="cpu").view(-1)
            pred_vals_all = torch.as_tensor(ex["pred_tp1"], dtype=torch.float32, device="cpu")

            # Strict contract: model-side GT tensors in examples must match raw snapshot counts.
            gt_cent_ex = torch.as_tensor(ex["centers_tp1"], dtype=torch.float32, device="cpu")
            gt_lev_ex = torch.as_tensor(ex["level_tp1"], dtype=torch.long, device="cpu").view(-1)
            gt_vals_ex = torch.as_tensor(ex["gt_tp1"], dtype=torch.float32, device="cpu")
            n_raw = int(rec["features"].size(0))
            if int(gt_cent_ex.size(0)) != n_raw or int(gt_lev_ex.numel()) != n_raw or int(gt_vals_ex.size(0)) != n_raw:
                raise RuntimeError(
                    "GT strict mode mismatch between model/eval GT tensors and raw snapshot tensors at "
                    f"t={t_abs}: ex_centers={int(gt_cent_ex.size(0))} ex_levels={int(gt_lev_ex.numel())} "
                    f"ex_feats={int(gt_vals_ex.size(0))} raw={n_raw}"
                )
            gt_plot_np = rec["features"].detach().cpu().numpy().astype(np.float32, copy=False)
            gt_plot_src = "strict_raw_snapshot"

            H_pred = int(ex.get("H", cfg.get("data", {}).get("H", 64)))
            W_pred = int(ex.get("W", cfg.get("data", {}).get("W", 64)))

            pred_corners_comp = _corners_from_centers_levels(
                pred_cent, pred_lev, H_pred, W_pred, bbox_comp, ref_ratio=float(pred_ref_ratio)
            )

            t_map0 = time.perf_counter()
            q_pred = pred_corners_comp.view(-1, 2)
            q_all = q_pred

            if map_mode == "auto":
                map_mode_eff = "affine" if (affine_rmse <= float(truth_map_affine_tol)) else "idw"
            else:
                map_mode_eff = map_mode

            step_tree = None
            if map_mode_eff == "affine":
                q_all_phys, q_unique_phys, q_inv, n_unique_corner, n_total_corner, quant_eps = _map_comp_points_to_phys_unique_vertices(
                    q_all,
                    bbox_comp=bbox_comp,
                    map_mode="affine",
                    affine_sol=affine_sol,
                    src_comp_xy=src_comp,
                    src_phys_xy=src_phys,
                    idw_backend=idw_backend_eff,
                    idw_k=int(truth_idw_k),
                    idw_chunk=int(truth_idw_chunk),
                    kdtree=None,
                )
            else:
                if idw_backend_eff == "kdtree":
                    if (last_tree_t != t_abs) or (last_tree is None):
                        last_tree = _SciPyKDTree(src_comp.detach().cpu().numpy().astype(np.float32, copy=False))
                        last_tree_t = t_abs
                    step_tree = last_tree
                q_all_phys, q_unique_phys, q_inv, n_unique_corner, n_total_corner, quant_eps = _map_comp_points_to_phys_unique_vertices(
                    q_all,
                    bbox_comp=bbox_comp,
                    map_mode="idw",
                    affine_sol=affine_sol,
                    src_comp_xy=src_comp,
                    src_phys_xy=src_phys,
                    idw_backend=idw_backend_eff,
                    idw_k=int(truth_idw_k),
                    idw_chunk=int(truth_idw_chunk),
                    kdtree=step_tree,
                )
            if not unique_corner_logged:
                frac = (100.0 * float(n_unique_corner) / max(1, int(n_total_corner)))
                log(
                    "[TRUTH] unique-corner mapping enabled: "
                    f"{n_unique_corner}/{n_total_corner} ({frac:.1f}%) vertices mapped "
                    f"(quant_eps={quant_eps:.3e})."
                )
                unique_corner_logged = True

            t_map_s = time.perf_counter() - t_map0

            pred_np = pred_vals_all.detach().cpu().numpy().astype(np.float32, copy=False)
            gt_ij = rec["ij"]
            gt_levels = rec["level"]
            if bool(truth_gt_comp_view):
                gt_comp_img, gt_comp_occ = _paint_amr_ij_to_uniform_grid(
                    ij=gt_ij,
                    levels=gt_levels,
                    values=rec["features"],
                    H0=int(H_gt),
                    W0=int(W_gt),
                    rr=int(round(float(gt_rr_render))),
                    render_level=int(gt_render_level),
                )
                gt_map_mode_eff = str(map_mode_eff)
                gt_img_all, _gt_valid_mask, gt_interp_meta = _sample_comp_raster_on_phys_grid(
                    img_comp=gt_comp_img,
                    occ_comp=gt_comp_occ,
                    q_grid_xy=q_grid_xy,
                    ny=gt_grid_ny,
                    nx=gt_grid_nx,
                    bbox_comp=bbox_comp,
                    H0=int(H_gt),
                    W0=int(W_gt),
                    rr=int(round(float(gt_rr_render))),
                    render_level=int(gt_render_level),
                    map_mode_eff=gt_map_mode_eff,
                    affine_sol=affine_sol,
                    src_comp_xy=src_comp,
                    src_phys_xy=src_phys,
                    idw_backend=idw_backend_eff,
                    idw_k=int(truth_idw_k),
                    idw_chunk=int(truth_idw_chunk),
                )
            else:
                gt_map_mode_eff = "physical_idw"
                gt_plot_phys_np = src_phys.detach().cpu().numpy().astype(np.float32, copy=False)
                gt_domain_mask = None
                _gt_domain_meta = {}
                if static_gt_interp_mask is not None:
                    gt_domain_mask = np.asarray(static_gt_interp_mask, dtype=bool).copy()
                    if gt_plot_phys_np.ndim == 2 and gt_plot_phys_np.shape[1] >= 2:
                        vx = np.asarray(gt_plot_phys_np[:, 0], dtype=np.float32)
                        vy = np.asarray(gt_plot_phys_np[:, 1], dtype=np.float32)
                        ok = np.isfinite(vx) & np.isfinite(vy)
                        if np.any(ok):
                            gx_f = ((vx[ok] - float(x0p)) / max(1.0e-12, float(x1p - x0p))) * float(gt_grid_nx)
                            gy_f = ((vy[ok] - float(y0p)) / max(1.0e-12, float(y1p - y0p))) * float(gt_grid_ny)
                            gx_i = np.clip(np.floor(gx_f).astype(np.int64), 0, max(0, int(gt_grid_nx) - 1))
                            gy_i = np.clip(np.floor(gy_f).astype(np.int64), 0, max(0, int(gt_grid_ny) - 1))
                            occ = np.zeros((int(gt_grid_ny), int(gt_grid_nx)), dtype=bool)
                            occ[gy_i, gx_i] = True
                            p = np.pad(occ, ((1, 1), (1, 1)), mode="constant", constant_values=False)
                            occ_dil = (
                                p[0:-2, 0:-2] | p[0:-2, 1:-1] | p[0:-2, 2:] |
                                p[1:-1, 0:-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
                                p[2:, 0:-2] | p[2:, 1:-1] | p[2:, 2:]
                            )
                            gt_domain_mask |= occ_dil
                    _gt_domain_meta = {
                        "status": str(static_gt_interp_mask_meta.get("status", "ok")),
                        "mask_mode": "static_pred_domain_plus_points",
                        "valid_frac": float(np.mean(gt_domain_mask)) if gt_domain_mask.size > 0 else 0.0,
                    }
                else:
                    gt_domain_mask_pts, gt_domain_meta_pts = _build_domain_mask_from_points_tri(
                        gt_plot_phys_np,
                        q_grid_xy=q_grid_xy,
                    )
                    gt_domain_mask = gt_domain_mask_pts
                    _gt_domain_meta = dict(gt_domain_meta_pts or {})
                    _gt_domain_meta.setdefault("mask_mode", "center_triangulation_fallback")

                gt_img_all, _gt_valid_mask, gt_interp_meta = _smooth_idw_raster_from_points(
                    gt_plot_phys_np,
                    gt_plot_np,
                    q_grid_xy,
                    ny=gt_grid_ny,
                    nx=gt_grid_nx,
                    k=max(4, int(truth_idw_k)),
                    valid_radius_factor=3.5,
                    valid_mask_override=gt_domain_mask,
                )
                gt_interp_meta["gt_mask_source"] = str(_gt_domain_meta.get("mask_mode", "unknown"))
                gt_interp_meta["gt_mask_status"] = str(_gt_domain_meta.get("status", "unknown"))

            # Legacy pred visualization path (same drawing method as raster rollout):
            # rasterize Pred(t+1) on a uniform comp-space grid, then display via imshow.
            pred_lmax_step = int(pred_lev.max().item()) if pred_lev.numel() > 0 else 0
            pred_img_flat, pred_hh, pred_ww = _rasterize_block_common_global(
                pred_cent,
                pred_lev,
                pred_vals_all,
                H_pred,
                W_pred,
                bbox_comp,
                pred_lmax_step,
                refine_ratio=int(round(float(pred_ref_ratio))),
            )
            pred_img_flat = _fill_internal_nans_nearest(pred_img_flat, pred_hh, pred_ww)
            pred_img_all = (
                pred_img_flat.view(pred_hh, pred_ww, -1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

            # Update moving zoom centers (in GT raster pixel coords) for selected features.
            if zoom_idx:
                sr_px = 0.20 * float(max(gt_grid_nx, gt_grid_ny))
                ms_px = 0.09 * float(max(gt_grid_nx, gt_grid_ny))
                for zf in zoom_idx:
                    prev_c = zoom_center_by_feat.get(int(zf), None)
                    c_new, _cmeta = _track_moving_zoom_center(
                        gt_img_all[:, :, int(zf)],
                        prev_c,
                        pct=96.0,
                        ema_alpha=0.35,
                        search_radius_px=sr_px,
                        max_step_px=ms_px,
                    )
                    zoom_center_by_feat[int(zf)] = c_new

            for f in range(F):
                if unify_clims:
                    tmin = float(top_min_f[f].item())
                    tmax = float(top_max_f[f].item())
                else:
                    g = gt_plot_np[:, f]
                    tmin = float(np.nanmin(g))
                    tmax = float(np.nanmax(g))
                    if (not np.isfinite(tmin)) or (not np.isfinite(tmax)) or (tmin == tmax):
                        tmin, tmax = 0.0, 1.0

                fig, axs = plt.subplots(
                    1,
                    2,
                    figsize=(float(two_panel_fig_w), float(two_panel_fig_h)),
                    dpi=int(dpi),
                    constrained_layout=True,
                )
                fig.suptitle(
                    f"t={t_abs} — {feature_names[f]} (truth mode; gt_map={gt_map_mode_eff}, gt_r={gt_rr_render:g}, lvl={gt_render_level})",
                    fontsize=12,
                )

                im_pred = axs[0].imshow(
                    pred_img_all[:, :, f],
                    origin="lower",
                    extent=(x0p, x1p, y0p, y1p),
                    cmap="viridis",
                    vmin=tmin,
                    vmax=tmax,
                    aspect="equal",
                )
                axs[0].set_title("Pred(t+1)")
                axs[0].set_xlabel("x (physical)")
                axs[0].set_ylabel("y (physical)")
                axs[0].set_xlim(x0p, x1p)
                axs[0].set_ylim(y0p, y1p)

                pc_gt = axs[1].imshow(
                    gt_img_all[:, :, f],
                    origin="lower",
                    extent=(x0p, x1p, y0p, y1p),
                    cmap="viridis",
                    vmin=tmin,
                    vmax=tmax,
                    interpolation="bilinear",
                    aspect="equal",
                )
                if (not bool(truth_gt_comp_view)) and (gt_plot_phys_np is not None):
                    gg = gt_plot_np[:, f]
                    gm = (
                        np.isfinite(gt_plot_phys_np[:, 0])
                        & np.isfinite(gt_plot_phys_np[:, 1])
                        & np.isfinite(gg)
                    )
                    if np.any(gm):
                        axs[1].scatter(
                            gt_plot_phys_np[gm, 0],
                            gt_plot_phys_np[gm, 1],
                            c=gg[gm],
                            s=0.45,
                            marker="o",
                            linewidths=0.0,
                            edgecolors="none",
                            cmap="viridis",
                            vmin=tmin,
                            vmax=tmax,
                            alpha=0.22,
                        )
                axs[1].set_title("GT(t+1)")
                axs[1].set_xlabel("x (physical)")
                axs[1].set_ylabel("y (physical)")
                axs[1].set_xlim(x0p, x1p)
                axs[1].set_ylim(y0p, y1p)

                fig.colorbar(im_pred, ax=axs[0], shrink=0.75)
                fig.colorbar(pc_gt, ax=axs[1], shrink=0.75)

                fig.canvas.draw()
                buf, (w, h) = fig.canvas.print_to_buffer()
                rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                frame = rgba[..., :3].copy()

                plt.close(fig)
                writers[f].append_data(frame)

                if f in zoom_idx_set:
                    cxy = zoom_center_by_feat.get(int(f), None)
                    if cxy is None:
                        cxy = (0.5 * (gt_grid_nx - 1.0), 0.5 * (gt_grid_ny - 1.0))
                    cxp, cyp = float(cxy[0]), float(cxy[1])

                    xc = x0p + ((cxp + 0.5) / max(1.0, float(gt_grid_nx))) * (x1p - x0p)
                    yc = y0p + ((cyp + 0.5) / max(1.0, float(gt_grid_ny))) * (y1p - y0p)
                    hx = 0.5 * float(np.clip(truth_zoom_w_frac, 0.02, 1.0)) * (x1p - x0p)
                    hy = 0.5 * float(np.clip(truth_zoom_h_frac, 0.02, 1.0)) * (y1p - y0p)

                    xlo, xhi = xc - hx, xc + hx
                    ylo, yhi = yc - hy, yc + hy
                    if xlo < x0p:
                        xhi += (x0p - xlo)
                        xlo = x0p
                    if xhi > x1p:
                        xlo -= (xhi - x1p)
                        xhi = x1p
                    if ylo < y0p:
                        yhi += (y0p - ylo)
                        ylo = y0p
                    if yhi > y1p:
                        ylo -= (yhi - y1p)
                        yhi = y1p
                    xlo = max(x0p, xlo); xhi = min(x1p, xhi)
                    ylo = max(y0p, ylo); yhi = min(y1p, yhi)

                    figz, axz = plt.subplots(
                        1,
                        2,
                        figsize=(float(two_panel_fig_w), float(two_panel_fig_h)),
                        dpi=int(dpi),
                        constrained_layout=True,
                    )
                    figz.suptitle(
                        f"t={t_abs} — {feature_names[f]} (truth zoom; moving ROI)",
                        fontsize=12,
                    )

                    imz_pred = axz[0].imshow(
                        pred_img_all[:, :, f],
                        origin="lower",
                        extent=(x0p, x1p, y0p, y1p),
                        cmap="viridis",
                        vmin=tmin,
                        vmax=tmax,
                        aspect="equal",
                    )
                    axz[0].set_title("Pred(t+1) [zoom]")
                    axz[0].set_xlabel("x (physical)")
                    axz[0].set_ylabel("y (physical)")
                    axz[0].set_xlim(xlo, xhi)
                    axz[0].set_ylim(ylo, yhi)

                    imz_gt = axz[1].imshow(
                        gt_img_all[:, :, f],
                        origin="lower",
                        extent=(x0p, x1p, y0p, y1p),
                        cmap="viridis",
                        vmin=tmin,
                        vmax=tmax,
                        interpolation="bilinear",
                        aspect="equal",
                    )
                    if (not bool(truth_gt_comp_view)) and (gt_plot_phys_np is not None):
                        gg = gt_plot_np[:, f]
                        gm = (
                            np.isfinite(gt_plot_phys_np[:, 0])
                            & np.isfinite(gt_plot_phys_np[:, 1])
                            & np.isfinite(gg)
                            & (gt_plot_phys_np[:, 0] >= xlo)
                            & (gt_plot_phys_np[:, 0] <= xhi)
                            & (gt_plot_phys_np[:, 1] >= ylo)
                            & (gt_plot_phys_np[:, 1] <= yhi)
                        )
                        if np.any(gm):
                            axz[1].scatter(
                                gt_plot_phys_np[gm, 0],
                                gt_plot_phys_np[gm, 1],
                                c=gg[gm],
                                s=2.0,
                                marker="o",
                                linewidths=0.0,
                                edgecolors="none",
                                cmap="viridis",
                                vmin=tmin,
                                vmax=tmax,
                                alpha=0.30,
                            )
                    axz[1].set_title("GT(t+1) [zoom]")
                    axz[1].set_xlabel("x (physical)")
                    axz[1].set_ylabel("y (physical)")
                    axz[1].set_xlim(xlo, xhi)
                    axz[1].set_ylim(ylo, yhi)

                    figz.colorbar(imz_pred, ax=axz[0], shrink=0.75)
                    figz.colorbar(imz_gt, ax=axz[1], shrink=0.75)

                    figz.canvas.draw()
                    zbuf, (zw, zh) = figz.canvas.print_to_buffer()
                    zrgba = np.frombuffer(zbuf, dtype=np.uint8).reshape(zh, zw, 4)
                    zframe = zrgba[..., :3].copy()
                    plt.close(figz)
                    zoom_writers[int(f)].append_data(zframe)

            if (step + 1) % max(1, int(progress_every)) == 0:
                log(
                    f"[TRUTH-GIF] wrote step {step+1}/{len(examples)} t={t_abs} "
                    f"(gt_r={gt_rr_render:g} lvl={gt_render_level} map={t_map_s:.3f}s pred_mode={map_mode_eff} gt_mode={gt_map_mode_eff} "
                    f"aff_rmse={affine_rmse:.2e} gt_valid={100.0*float(gt_interp_meta.get('valid_frac',0.0)):.1f}% "
                    f"mask={gt_interp_meta.get('mask_mode','?')} "
                    f"mask_src={gt_interp_meta.get('gt_mask_source','?')} "
                    f"mask_status={gt_interp_meta.get('gt_mask_status','?')} "
                    f"src={gt_plot_src})"
                )
    finally:
        for w in writers:
            try:
                w.close()
            except Exception:
                pass
        for w in zoom_writers.values():
            try:
                w.close()
            except Exception:
                pass

    for p in gif_paths:
        print(f"[INFO] wrote {p}")
    for _f, p in sorted(zoom_paths.items(), key=lambda kv: kv[0]):
        print(f"[INFO] wrote {p}")


# ----------------- Fast drift diagnostics ----------------- #

def _as_cpu_f32(x):
    import torch
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().to("cpu", dtype=torch.float32)
    # numpy / list
    return torch.as_tensor(x, dtype=torch.float32, device="cpu")


def _as_cpu_i64(x):
    import torch
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().to("cpu", dtype=torch.int64)
    return torch.as_tensor(x, dtype=torch.int64, device="cpu")


def _fmt_vec(names, v, fmt="{:.3e}"):
    return ", ".join([f"{n}={fmt.format(float(v[i]))}" for i, n in enumerate(names)])


@torch.no_grad()
def run_fast_drift_checks(
    test_stats: dict,
    examples: list[dict],
    *,
    feat_names: list[str] | None,
    bbox: tuple[float, float, float, float],
    knn_k: int = 8,
    chunk: int = 8192,
    n_probe: int = 5,
    eps: float = 1e-12,
):
    """
    Four decisive checks:
      (1) Does error grow with rollout step k? (uses *_by_rollout_step from test_stats)
      (2) Is k=1 already bad on the *pred mesh*? (pred(t+1) vs GT(t+1) mapped to pred mesh)
      (3) Is your update magnitude wrong (rate-vs-delta style)? (Δ_pred vs Δ_GT on pred mesh)
      (4) Are mapping/mesh basics sane? (IDW weight sums, out-of-bounds centers, NaNs, bad levels)
    """
    import numpy as np
    import torch

    if not examples:
        print("[FASTCHECK] No examples found in test_stats['examples']; nothing to probe.")
        return

    # ----------------- (1) rollout-step growth -----------------
    r = test_stats.get("rell2w_feat_by_rollout_step", None)
    m = test_stats.get("maew_feat_by_rollout_step", None)

    print("\n" + "=" * 90)
    print("[FASTCHECK-1] Rollout-step growth (if this explodes with k, that IS state drift).")

    if r is None and m is None:
        print("[FASTCHECK-1] Missing keys in test_stats: 'rell2w_feat_by_rollout_step' and 'maew_feat_by_rollout_step'.")
    else:
        if r is not None:
            r_np = np.asarray(r)
            if r_np.ndim == 1:
                r_np = r_np[:, None]
            K, F = r_np.shape
            names = feat_names if (feat_names and len(feat_names) == F) else [f"f{i}" for i in range(F)]
            kshow = min(K, 10)
            print(f"[FASTCHECK-1] rell2w by rollout step (showing first {kshow}/{K}):")
            for k in range(kshow):
                print(f"  k={k+1:02d}: " + _fmt_vec(names, r_np[k], "{:.3e}"))
            growth = r_np[-1] / (r_np[0] + eps)
            print("[FASTCHECK-1] growth factor (last / k=1): " + _fmt_vec(names, growth, "{:.3e}"))

        if m is not None:
            m_np = np.asarray(m)
            if m_np.ndim == 1:
                m_np = m_np[:, None]
            K, F = m_np.shape
            names = feat_names if (feat_names and len(feat_names) == F) else [f"f{i}" for i in range(F)]
            kshow = min(K, 10)
            print(f"[FASTCHECK-1] maew by rollout step (showing first {kshow}/{K}):")
            for k in range(kshow):
                print(f"  k={k+1:02d}: " + _fmt_vec(names, m_np[k], "{:.3e}"))
            growth = m_np[-1] / (m_np[0] + eps)
            print("[FASTCHECK-1] growth factor (last / k=1): " + _fmt_vec(names, growth, "{:.3e}"))

    # ----------------- probe examples -----------------
    n_probe = max(1, min(int(n_probe), len(examples)))
    x0, x1, y0, y1 = bbox

    print("\n" + "-" * 90)
    print(f"[FASTCHECK-2/3/4] Probing first {n_probe} transitions using IDW maps onto pred mesh.")

    # aggregate summaries
    agg_rel = None
    agg_ratio = None
    agg_cos = None
    agg_wdev = []
    agg_oob = []
    agg_nan = []
    agg_badlvl = []

    for j in range(n_probe):
        ex = examples[j]

        centers_t   = _as_cpu_f32(ex.get("centers_t"))
        centers_tp1 = _as_cpu_f32(ex.get("centers_tp1"))
        gt_t        = _as_cpu_f32(ex.get("gt_t"))
        gt_tp1      = _as_cpu_f32(ex.get("gt_tp1"))

        pred_centers = _as_cpu_f32(ex.get("pred_centers"))
        pred_levels  = _as_cpu_i64(ex.get("pred_levels"))
        pred_tp1     = _as_cpu_f32(ex.get("pred_tp1"))

        if any(v is None for v in [centers_t, centers_tp1, gt_t, gt_tp1, pred_centers, pred_tp1]):
            print(f"[FASTCHECK] ex[{j}] missing required keys; skipping.")
            continue

        F = int(gt_t.shape[1])
        names = feat_names if (feat_names and len(feat_names) == F) else [f"f{i}" for i in range(F)]

        # (4a) mesh sanity
        nan_ct = int(torch.isnan(pred_centers).any().item()) + int(torch.isnan(pred_tp1).any().item())
        agg_nan.append(nan_ct)

        oob = ((pred_centers[:, 0] < x0) | (pred_centers[:, 0] > x1) |
               (pred_centers[:, 1] < y0) | (pred_centers[:, 1] > y1))
        oob_frac = float(oob.float().mean().item())
        agg_oob.append(oob_frac)

        badlvl_frac = 0.0
        if pred_levels is not None:
            bad = (pred_levels < 0) | (pred_levels > 10)  # loose sanity bound
            badlvl_frac = float(bad.float().mean().item())
        agg_badlvl.append(badlvl_frac)

        # Build maps: GT(t)->predmesh, GT(t+1)->predmesh
        # NOTE: build_idw_map/apply_idw_map are already imported in your script (utils_geom).
        idx_t, w_t = build_idw_map(
            dst_xy=pred_centers,
            src_xy=centers_t,
            k=int(knn_k),
            chunk=int(chunk),
        )
        idx_t = _as_cpu_i64(idx_t)
        w_t = _as_cpu_f32(w_t)

        idx_tp1, w_tp1 = build_idw_map(
            dst_xy=pred_centers,
            src_xy=centers_tp1,
            k=int(knn_k),
            chunk=int(chunk),
        )
        idx_tp1 = _as_cpu_i64(idx_tp1)
        w_tp1 = _as_cpu_f32(w_tp1)

        # (4b) IDW weight sum sanity
        wsum_dev = float((w_t.sum(dim=1) - 1.0).abs().max().item())
        agg_wdev.append(wsum_dev)

        #gt_t_on_pred   = apply_idw_map(gt_t, idx_t, w_t)
        #gt_tp1_on_pred = apply_idw_map(gt_tp1, idx_tp1, w_tp1)
        gt_t_on_pred = apply_idw_map(idx_t, w_t, gt_t)
        gt_tp1_on_pred = apply_idw_map(idx_tp1, w_tp1, gt_tp1)

        # ----------------- (2) k=1 correctness on pred mesh -----------------
        err = pred_tp1 - gt_tp1_on_pred
        rel_l2 = err.pow(2).sum(dim=0).sqrt() / (gt_tp1_on_pred.pow(2).sum(dim=0).sqrt() + eps)
        mae = err.abs().mean(dim=0)

        # ranges (unit/normalization mismatch pops here immediately)
        p_min, p_max = pred_tp1.amin(dim=0), pred_tp1.amax(dim=0)
        g_min, g_max = gt_tp1_on_pred.amin(dim=0), gt_tp1_on_pred.amax(dim=0)

        # ----------------- (3) update magnitude / rate-vs-delta smell test -----------------
        d_pred = pred_tp1 - gt_t_on_pred
        d_gt   = gt_tp1_on_pred - gt_t_on_pred

        rms_pred = d_pred.pow(2).mean(dim=0).sqrt()
        rms_gt   = d_gt.pow(2).mean(dim=0).sqrt()
        ratio    = rms_pred / (rms_gt + eps)

        # cosine similarity per feature (directional agreement)
        cos = (d_pred * d_gt).mean(dim=0) / (rms_pred * rms_gt + eps)

        # accumulate
        agg_rel = rel_l2 if agg_rel is None else (agg_rel + rel_l2)
        agg_ratio = ratio if agg_ratio is None else (agg_ratio + ratio)
        agg_cos = cos if agg_cos is None else (agg_cos + cos)

        print("\n" + "-" * 70)
        print(f"[FASTCHECK ex[{j}]] pred vs GT(t+1) on pred mesh:")
        print("  relL2: " + _fmt_vec(names, rel_l2, "{:.3e}"))
        print("  MAE  : " + _fmt_vec(names, mae, "{:.3e}"))

        print(f"[FASTCHECK ex[{j}]] value ranges on pred mesh (watch for normalization/unit mismatch):")
        print("  pred(t+1) min/max: " + _fmt_vec(names, p_min, "{:.3e}") + "  |  " + _fmt_vec(names, p_max, "{:.3e}"))
        print("  GT(t+1)  min/max: " + _fmt_vec(names, g_min, "{:.3e}") + "  |  " + _fmt_vec(names, g_max, "{:.3e}"))

        print(f"[FASTCHECK ex[{j}]] Δ magnitude check (Δ_pred vs Δ_GT on pred mesh):")
        print("  RMS ratio (pred/GT): " + _fmt_vec(names, ratio, "{:.3e}"))
        print("  cosine similarity  : " + _fmt_vec(names, cos, "{:.3e}"))

        print(f"[FASTCHECK ex[{j}]] mapping/mesh sanity:")
        print(f"  IDW max |sum(w)-1| = {wsum_dev:.3e}")
        print(f"  pred_centers OOB frac = {oob_frac:.3%}, NaN flag = {bool(nan_ct)}, badlevel frac = {badlvl_frac:.3%}")

    # finalize aggregates
    if agg_rel is not None:
        agg_rel = agg_rel / float(n_probe)
        agg_ratio = agg_ratio / float(n_probe)
        agg_cos = agg_cos / float(n_probe)

        F = int(agg_rel.numel())
        names = feat_names if (feat_names and len(feat_names) == F) else [f"f{i}" for i in range(F)]

        print("\n" + "=" * 90)
        print("[FASTCHECK SUMMARY] Averages over probed transitions:")
        print("  mean relL2(pred vs GT(t+1) on pred mesh): " + _fmt_vec(names, agg_rel, "{:.3e}"))
        print("  mean RMS ratio (Δ_pred/Δ_GT):            " + _fmt_vec(names, agg_ratio, "{:.3e}"))
        print("  mean cosine(Δ_pred, Δ_GT):              " + _fmt_vec(names, agg_cos, "{:.3e}"))
        print(f"  IDW max |sum(w)-1| (max over probes):   {max(agg_wdev) if agg_wdev else float('nan'):.3e}")
        print(f"  pred_centers OOB frac (max over probes): {max(agg_oob) if agg_oob else float('nan'):.3%}")
        print(f"  any NaN flags (sum over probes):        {sum(agg_nan) if agg_nan else 0}")
        print(f"  badlevel frac (max over probes):        {max(agg_badlvl) if agg_badlvl else float('nan'):.3%}")

        # quick interpretation hints
        print("\n[FASTCHECK INTERPRETATION HINTS]")
        print("  - If relL2 is huge at k=1 AND pred ranges look 'normalized' (~O(1)) while GT ranges are physical: scaling mismatch.")
        print("  - If RMS ratio (Δ_pred/Δ_GT) is consistently ~dt or ~1/dt-ish across features: rate-vs-delta misuse.")
        print("  - If cosine is ~-1: sign flip bug (e.g., dt sign, swapped t/t+1, or wrong delta direction).")
        print("  - If IDW |sum(w)-1| is large or OOB is non-trivial: mapping/mesh bug contaminating comparisons.")
        print("=" * 90 + "\n")

# ----------------- Main rollout script ----------------- #

def main():
    ap = argparse.ArgumentParser(description="Long-rollout visualization with GIFs (raster version + fixed clims).")

    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint .pt (e.g. best_model.pt or last_model.pt).")
    ap.add_argument("--config", type=str, default=None,
                    help="Optional JSON config; if omitted, uses cfg from checkpoint.")
    ap.add_argument("--pt-path", type=str, default=None,
                    help="Optional override for cfg['data']['pt_path'].")

    ap.add_argument("--horizon", type=int, default=27,
                    help="Number of predicted steps to visualize (rollout length).")
    ap.add_argument("--start-t", type=int, default=0,
                    help="Absolute starting time index t0 for the rollout.")

    ap.add_argument("--device", type=str, default=None,
                    help="Device string (e.g. 'cpu', 'cuda', 'mps'); overrides cfg if set.")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Directory to write GIFs to; defaults to save_dir/rollout_t{start_t}_H{horizon}.")

    ap.add_argument("--unify-clims", action="store_true",
                    help="If set, use fixed global color limits across the rollout (per feature).")
    ap.add_argument("--fps", type=int, default=4,
                    help="Frames per second for GIFs.")
    ap.add_argument("--dpi", type=int, default=100,
                    help="Matplotlib DPI for frames.")
    ap.add_argument("--progress-every", type=int, default=1,
                    help="Print progress every N rollout steps.")
    ap.add_argument(
        "--include-deltas",
        action="store_true",
        help="If set, include delta panels in rollout plots (slower). Default: off (Pred(t+1) vs GT(t+1) only).",
    )
    ap.add_argument(
        "--two-panel-fig-w",
        type=float,
        default=14.0,
        help="Figure width in inches for 2-panel mode (used when --include-deltas is not set).",
    )
    ap.add_argument(
        "--two-panel-fig-h",
        type=float,
        default=7.0,
        help="Figure height in inches for 2-panel mode (used when --include-deltas is not set).",
    )
    ap.add_argument(
        "--truth-mode",
        action="store_true",
        help=(
            "Render Pred/GT panels in physical space using comp->phys mapping from raw snapshots "
            "(pos -> pos_phys). Keeps default raster mode unchanged when not set."
        ),
    )
    ap.add_argument(
        "--truth-map-mode",
        type=str,
        default="auto",
        choices=["auto", "idw", "affine"],
        help="Truth-mode comp->phys mapping backend selection.",
    )
    ap.add_argument(
        "--truth-map-affine-tol",
        type=float,
        default=1e-5,
        help="Truth-mode affine RMSE threshold for auto map mode (auto chooses affine below this, else IDW).",
    )
    ap.add_argument(
        "--truth-idw-backend",
        type=str,
        default="auto",
        choices=["auto", "cdist", "kdtree"],
        help="Truth-mode IDW backend for comp->phys warping.",
    )
    ap.add_argument(
        "--truth-idw-k",
        type=int,
        default=8,
        help="Truth-mode IDW k for comp->phys warping.",
    )
    ap.add_argument(
        "--truth-idw-chunk",
        type=int,
        default=32768,
        help="Truth-mode IDW chunk size for comp->phys warping.",
    )
    ap.add_argument(
        "--truth-gt-ref-ratio",
        type=str,
        default="auto",
        help="Truth-mode GT refinement ratio in computational space (auto, 2, 4, ...).",
    )
    ap.add_argument(
        "--truth-gt-render-level",
        type=str,
        default="max",
        help=(
            "Truth-mode GT render level relative to simulation base mesh HxW "
            "(0..Lmax or 'max'). Default: max."
        ),
    )
    ap.add_argument(
        "--truth-zoom-features",
        type=str,
        default="",
        help=(
            "Optional moving-ROI zoom GIFs in truth mode. "
            "Use: density, energy, both, or comma-separated list. Empty disables."
        ),
    )
    ap.add_argument(
        "--truth-zoom-w-frac",
        type=float,
        default=0.28,
        help="Truth-mode moving zoom window width as fraction of full physical domain.",
    )
    ap.add_argument(
        "--truth-zoom-h-frac",
        type=float,
        default=0.28,
        help="Truth-mode moving zoom window height as fraction of full physical domain.",
    )
    ap.add_argument(
        "--truth-gt-comp-view",
        action="store_true",
        help=(
            "Optional: render GT in comp-like view using truth-map-mode warping "
            "(typically affine/auto). Default is physical-space GT rendering."
        ),
    )
    # DataLoader
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Number of workers for DataLoader.")

    # Precomp controls
    ap.add_argument("--precomp-path", type=str, default=None,
                    help="Legacy AMR flag (ignored in cartesian static rollout).")
    ap.add_argument("--save-precomp", type=str, default=None,
                    help="Legacy AMR flag (ignored in cartesian static rollout).")
    ap.add_argument("--recompute-precomp", action="store_true",
                    help="Legacy AMR flag (ignored in cartesian static rollout).")
    ap.add_argument("--precompute-scope", type=str, default="all", choices=["window", "all"],
                    help="Legacy AMR flag (ignored in cartesian static rollout).")
    ap.add_argument("--precompute-device", type=str, default="cpu",
                    help="Legacy AMR flag (ignored in cartesian static rollout).")
    ap.add_argument(
        "--runtime-mesh-cnn-ckpt",
        type=str,
        default=None,
        help="Optional override for train.runtime_mesh.cnn.checkpoint_path at rollout time.",
    )
    ap.add_argument(
        "--runtime-mesh-spec-path",
        type=str,
        default=None,
        help="Optional override for mesh.starting_mesh_path at rollout time.",
    )
    ap.add_argument(
        "--runtime-idw-backend",
        type=str,
        default=None,
        choices=["exact", "faiss_flat", "faiss_ivf", "cdist", "flat", "faiss", "ivf", "ann", "approx"],
        help=(
            "Optional override for train.runtime_mesh.idw.backend used by runtime remap "
            "(state/GT mapping on changing meshes)."
        ),
    )
    ap.add_argument(
        "--runtime-knn-k",
        type=int,
        default=None,
        help=(
            "Optional override for train.knn_k used by runtime remap interpolation. "
            "Lower values are faster but less robust."
        ),
    )
    ap.add_argument(
        "--runtime-interp-chunk",
        type=int,
        default=None,
        help="Optional override for speed.interp_chunk used by runtime remap interpolation.",
    )
    ap.add_argument(
        "--runtime-update-every-steps",
        type=int,
        default=None,
        help=(
            "Optional override for train.runtime_mesh.update_every_steps in rollout evaluation. "
            "Use >1 to rebuild runtime mesh less often for speed."
        ),
    )
    ap.add_argument(
        "--runtime-multires-dir",
        type=str,
        default=None,
        help=(
            "Optional override for train.runtime_mesh.multires_gt_lookup.directory "
            "used by runtime multires GT lookup."
        ),
    )
    ap.add_argument(
        "--infer-only",
        action="store_true",
        help=(
            "Run rollout in inference-only mode (runtime mesh path): skip GT->pred target remap and "
            "target-based eval loss/metrics for faster timing."
        ),
    )

    # Raster controls
    ap.add_argument("--raster-mode", type=str, default="block", choices=["block", "idw"],
                    help="Rasterization mode for plots.")
    ap.add_argument("--raster-lmax", type=int, default=None,
                    help="Clamp raster Lmax (e.g. 3). If omitted, infer per step.")
    ap.add_argument("--delta-scale", type=str, default="gt", choices=["gt", "each"],
                    help="Delta scaling when unify_clims is OFF. When unify_clims is ON, we force 'gt' behavior.")

    ap.add_argument("--raster-bins", type=int, default=256, help="IDW grid bins (only for raster-mode=idw).")
    ap.add_argument(
        "--viz-raster-k",
        "--raster-k",
        dest="viz_raster_k",
        type=int,
        default=8,
        help=(
            "Visualization raster IDW kNN k (only for raster-mode=idw). "
            "Deprecated alias: --raster-k."
        ),
    )
    ap.add_argument(
        "--viz-raster-chunk",
        "--raster-chunk",
        dest="viz_raster_chunk",
        type=int,
        default=32768,
        help=(
            "Visualization raster IDW chunk (only for raster-mode=idw). "
            "Deprecated alias: --raster-chunk."
        ),
    )
    ap.add_argument(
        "--degree-diag-steps",
        type=str,
        default=None,
        help="Comma-separated rollout-step indices (0-based) and/or 'last' for node-degree error bar plots. If omitted, disabled.",
    )

    ap.add_argument("--debug-fast-checks", action="store_true",
                    help="Run fast drift diagnostics (prints k-growth, k=1 mesh-mapped errors, Δ magnitude, mapping sanity).")
    ap.add_argument("--debug-fast-checks-n", type=int, default=5,
                    help="How many transitions to probe for the fast drift checks (default: 5).")


    args = ap.parse_args()
    if args.runtime_knn_k is not None and int(args.runtime_knn_k) <= 0:
        raise ValueError("--runtime-knn-k must be > 0.")
    if args.runtime_interp_chunk is not None and int(args.runtime_interp_chunk) <= 0:
        raise ValueError("--runtime-interp-chunk must be > 0.")
    if args.runtime_update_every_steps is not None and int(args.runtime_update_every_steps) <= 0:
        raise ValueError("--runtime-update-every-steps must be >= 1.")
    if int(args.viz_raster_k) <= 0:
        raise ValueError("--viz-raster-k/--raster-k must be > 0.")
    if int(args.viz_raster_chunk) <= 0:
        raise ValueError("--viz-raster-chunk/--raster-chunk must be > 0.")

    # ----- Load checkpoint -----
    log(f"[INFO] Loading checkpoint from {args.checkpoint}")
    with Timer("torch.load(checkpoint)"):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Base config
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        log(f"[INFO] Using config from {args.config}")
    else:
        cfg = ckpt.get("cfg", None)
        if cfg is None:
            raise RuntimeError("Checkpoint has no 'cfg' and no --config was provided.")
        log("[INFO] Using cfg from checkpoint.")

    # ---- reconcile adapter intent with checkpoint contents ----
    loss_cfg = cfg.get("loss", {}) or {}
    sd = ckpt["model"] if "model" in ckpt else ckpt

    has_adapter_weights = any(k.startswith("parc_adapter.") for k in sd.keys())
    cfg_wants_adapter = bool(loss_cfg.get("parc_use_adapter", False))


    if cfg_wants_adapter and (not has_adapter_weights):
        log("[WARN] cfg requests parc_use_adapter=True but checkpoint has no parc_adapter.* weights. "
            "Forcing parc_use_adapter=False to avoid runtime errors.")
        loss_cfg["parc_use_adapter"] = False
        cfg["loss"] = loss_cfg

    # Override pt_path if requested
    if args.pt_path is not None:
        cfg.setdefault("data", {})["pt_path"] = args.pt_path
    if "data" not in cfg or "pt_path" not in cfg["data"]:
        raise RuntimeError("cfg['data']['pt_path'] must be set (or use --pt-path).")

    runtime_cli_flags = []
    if args.runtime_mesh_cnn_ckpt is not None:
        runtime_cli_flags.append("--runtime-mesh-cnn-ckpt")
    if args.runtime_mesh_spec_path is not None:
        runtime_cli_flags.append("--runtime-mesh-spec-path")
    if args.runtime_idw_backend is not None:
        runtime_cli_flags.append("--runtime-idw-backend")
    if args.runtime_knn_k is not None:
        runtime_cli_flags.append("--runtime-knn-k")
    if args.runtime_interp_chunk is not None:
        runtime_cli_flags.append("--runtime-interp-chunk")
    if args.runtime_update_every_steps is not None:
        runtime_cli_flags.append("--runtime-update-every-steps")
    if args.runtime_multires_dir is not None:
        runtime_cli_flags.append("--runtime-multires-dir")
    if args.infer_only:
        runtime_cli_flags.append("--infer-only")
    if runtime_cli_flags:
        raise RuntimeError(
            "This cartesian rollout script does not support runtime-mesh overrides. "
            f"Unsupported flags: {', '.join(runtime_cli_flags)}"
        )

    # Model device
    device_str = args.device if (args.device is not None) else cfg.get("device", "cpu")
    device = torch.device(device_str)
    log(f"[INFO] Using model device: {device}")

    # Optional: simple CUDA verification print
    if device.type == "cuda":
        log(f"[CUDA] available={torch.cuda.is_available()} current_device={torch.cuda.current_device()} name={torch.cuda.get_device_name(torch.cuda.current_device())}")

    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    # Domain params
    H = int(cfg["data"].get("H", 64))
    W = int(cfg["data"].get("W", 64))
    xmin, xmax, ymin, ymax = _get_bbox(cfg)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H

    start_t = int(args.start_t)
    horizon = int(args.horizon)
    if horizon <= 0:
        raise ValueError("--horizon must be > 0")

    # Force stride=1 so window_idx == start_t is valid
    cfg.setdefault("data", {})["window_size"] = horizon + 1
    cfg["data"]["stride"] = 1
    log(f"[INFO] Using window_size={cfg['data']['window_size']} (horizon={horizon}), stride=1")

    # ----- Load series -----
    with Timer("Load time series"):
        data_list = _load_pt_series(cfg["data"]["pt_path"])

    # If snapshots carry explicit base resolution (uniform HDF5 path), prefer it.
    if len(data_list) > 0:
        h_file, w_file = _extract_hw_from_snapshot(data_list[0])
        if (h_file is not None) and (w_file is not None) and (h_file > 0) and (w_file > 0):
            if (h_file != H) or (w_file != W):
                log(
                    "[INFO] Overriding cfg data resolution from file metadata: "
                    f"H,W=({H},{W}) -> ({h_file},{w_file})"
                )
            H = int(h_file)
            W = int(w_file)
            cfg.setdefault("data", {})["H"] = H
            cfg["data"]["W"] = W
            dx = (xmax - xmin) / W
            dy = (ymax - ymin) / H
    T = len(data_list)
    log(f"[INFO] Series length T={T}")

    if start_t < 0 or start_t + horizon >= T:
        raise ValueError(
            f"Invalid (start_t={start_t}, horizon={horizon}) for series length T={T}. "
            f"Need start_t+horizon < T."
        )

    # dt info (if present in snapshots)
    dt_transitions, dt_ref = _compute_dt_transitions(data_list)

    runtime_mesh_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    runtime_mesh_enabled = _cfg_bool_strict(
        runtime_mesh_cfg.get("enabled", False), key="train.runtime_mesh.enabled"
    )
    if runtime_mesh_enabled:
        raise RuntimeError(
            "Cartesian rollout requires train.runtime_mesh.enabled=false. "
            "Runtime remeshing paths are disabled in this repo."
        )
    runtime_backend = str(runtime_mesh_cfg.get("policy_backend", "gradient")).strip().lower()
    if runtime_backend in ("cnn_policy",):
        runtime_backend = "cnn"

    # For rollout with CNN or gradient_fast runtime mesh, default to rebuild every step.
    # CLI can explicitly override update_every_steps for faster inference experiments.
    if runtime_mesh_enabled and runtime_backend in ("cnn", "gradient_fast"):
        rt_cfg_mut = cfg.setdefault("train", {}).setdefault("runtime_mesh", {})
        prev_update_every = int(rt_cfg_mut.get("update_every_steps", 1))
        prev_warm_start = bool(rt_cfg_mut.get("warm_start_from_precompute", True))
        if args.runtime_update_every_steps is None:
            rt_cfg_mut["update_every_steps"] = 1
            update_note = "forcing update_every_steps=1"
        else:
            rt_cfg_mut["update_every_steps"] = int(args.runtime_update_every_steps)
            update_note = (
                "keeping CLI override "
                f"update_every_steps={int(args.runtime_update_every_steps)}"
            )
        rt_cfg_mut["warm_start_from_precompute"] = False
        curr_update_every = int(rt_cfg_mut.get("update_every_steps", 1))
        if (prev_update_every != curr_update_every) or prev_warm_start:
            log(
                f"[RUNTIME-MESH] {runtime_backend} rollout override: {update_note} "
                "and warm_start_from_precompute=false."
            )
        runtime_mesh_cfg = rt_cfg_mut

    if runtime_mesh_enabled:
        runtime_idw_cfg = runtime_mesh_cfg.get("idw", {}) or {}
        eff_runtime_knn_k = int(cfg.get("train", {}).get("knn_k", cfg.get("loss", {}).get("interp_k", 8)))
        eff_runtime_chunk = int(cfg.get("speed", {}).get("interp_chunk", 8192))
        log(
            "[RUNTIME-MESH] remap controls: "
            f"idw_backend={str(runtime_idw_cfg.get('backend', 'exact')).lower()}, "
            f"knn_k={eff_runtime_knn_k}, interp_chunk={eff_runtime_chunk}, "
            f"update_every_steps={int(runtime_mesh_cfg.get('update_every_steps', 1))}"
        )

    if bool((cfg.get("train", {}) or {}).get("use_precompute", False)):
        log(
            "[WARN] train.use_precompute=true in config, but cartesian rollout enforces "
            "no-precompute collate (static/uniform path)."
        )
    runtime_mesh_policy = None

    # ----- Build dataset (CPU) -----
    with Timer("Build dataset"):
        full_ds = CellRefineWindowDataset(
            series=data_list,
            cfg=cfg,
            window_size=cfg["data"]["window_size"],
            stride=cfg["data"]["stride"],
            H=H, W=W,
            device="cpu",
        )
    log(f"[INFO] Dataset windows: {len(full_ds)}")

    # Evaluate only the ONE window we want
    window_idx = start_t
    if window_idx >= len(full_ds):
        raise ValueError(f"start_t={start_t} maps to window_idx={window_idx}, but len(dataset)={len(full_ds)}")
    test_ds = Subset(full_ds, [window_idx])

    # ----- Collate (uniform/static, no AMR precompute) -----
    if (args.precomp_path is not None) or bool(args.recompute_precomp) or (args.save_precomp is not None):
        log(
            "[INFO] Ignoring --precomp-path/--recompute-precomp/--save-precomp for "
            "cartesian static rollout (no precompute required)."
        )
    log("[PRECOMP] disabled for cartesian rollout: using CollateWithUniformStaticNoPrecomp.")
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

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        sampler=SequentialSampler(test_ds),
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate,
    )

    # ----- Build model & load weights -----
    with Timer("Build model"):
        model = build_model_from_cfg(cfg, device)

    # ---- attach ParcFeatureAdapter when physics-input channels are enabled ----
    loss_cfg = cfg.get("loss", {}) or {}
    parc_use = _physics_inputs_active_from_loss_cfg(loss_cfg)
    use_adapter = bool(loss_cfg.get("parc_use_adapter", False))

    if parc_use and use_adapter:
        Fdim = int(cfg.get("features", {}).get("num_features", 4))
        la = len(dec.parc_select_feature_indices_adv(cfg, Fdim))
        ld = len(dec.parc_select_feature_indices_diff(cfg, Fdim))

        model.parc_adapter = ParcFeatureAdapter(
            la, ld,
            use_norm=bool(loss_cfg.get("parc_feat_norm", True)),
            clip_pre=float(loss_cfg.get("parc_feat_clip_pre", 10.0)),
            clip_post=float(loss_cfg.get("parc_feat_clip_post", 10.0)),
            momentum=float(loss_cfg.get("parc_feat_norm_momentum", 0.02)),
            per_channel_gates=bool(loss_cfg.get("parc_gate_per_channel", True)),
            gate_init=float(loss_cfg.get("parc_gate_init", -5.0)),
        ).to(device)
    else:
        model.parc_adapter = None

    sd = ckpt["model"] if "model" in ckpt else ckpt
    has_adapter_weights = any(k.startswith("parc_adapter.") for k in sd.keys())
    has_adapter_module  = getattr(model, "parc_adapter", None) is not None

    if has_adapter_weights and has_adapter_module:
        # load everything (adapter + model)
        missing, unexpected = model.load_state_dict(sd, strict=False)
    else:
        # strip adapter keys if either side doesn't have them
        sd_no_adapter = {k: v for k, v in sd.items() if not k.startswith("parc_adapter.")}
        missing, unexpected = model.load_state_dict(sd_no_adapter, strict=False)

    # sanity: allow only adapter-related mismatches
    missing_non_adapter = [k for k in missing if not k.startswith("parc_adapter.")]
    unexpected_non_adapter = [k for k in unexpected if not k.startswith("parc_adapter.")]
    if missing_non_adapter or unexpected_non_adapter:
        raise RuntimeError(
            "State-dict mismatch beyond parc_adapter.*\n"
            f"Missing: {missing_non_adapter}\n"
            f"Unexpected: {unexpected_non_adapter}"
        )

    if missing or unexpected:
        log(f"[INFO] load_state_dict strict=False (missing={len(missing)}, unexpected={len(unexpected)})")


    model.to(device)
    model.eval()

    # Output dir
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        save_dir = cfg.get("train", {}).get("save_dir", ".")
        out_dir = os.path.join(save_dir, f"rollout_t{start_t}_H{horizon}")
    os.makedirs(out_dir, exist_ok=True)

    budget_csv_path = os.path.join(out_dir, "rollout_budgets.csv")

    # Norm stats: prefer checkpoint payload, then fall back to config payload.
    norm_stats = ckpt.get("norm_stats", None) if isinstance(ckpt, dict) else None
    if (not norm_stats) or (norm_stats.get("mu") is None):
        cfg_norm = (cfg.get("features", {}) or {}).get("norm_stats", None)
        if cfg_norm and (cfg_norm.get("mu") is not None):
            norm_stats = cfg_norm
            log("[INFO] Using normalization stats from config.")

    if norm_stats is not None and norm_stats.get("mu") is not None:
        mu = torch.tensor(norm_stats["mu"], dtype=torch.float32, device=device)
        sigma = torch.tensor(norm_stats["sigma"], dtype=torch.float32, device=device)
        log("[INFO] Normalization is enabled for rollout.")
    else:
        mu = sigma = None
        log("[WARN] No normalization stats found in checkpoint/config; proceeding without normalization.")

    degree_diag_requested = (args.degree_diag_steps is not None) and (str(args.degree_diag_steps).strip() != "")

    # ----- Evaluate only this rollout window -----
    log(f"[INFO] Running multi-step evaluation for ONE window at start_t={start_t} (horizon={horizon})...")
    if bool(args.infer_only):
        log("[INFO] rollout_viz infer-only enabled.")
    with Timer("evaluate_one_epoch_multi_step"):
        test_loss, test_stats = evaluate_one_epoch_multi_step(
            model,
            test_loader,
            cfg,
            device,
            H=H, W=W,
            dx=dx, dy=dy,
            mu=mu,
            sigma=sigma,
            collect_examples=True,
            collect_example_edges=bool(degree_diag_requested),
            budget_csv_path=budget_csv_path, 
            write_budgets=True,
            runtime_mesh_policy=runtime_mesh_policy,
            infer_only=bool(args.infer_only),
        )
    log(f"[TEST] loss={test_loss:.4e}")

    _maybe_empty_device_cache(device)

    examples = test_stats.get("examples", [])
    if not examples:
        raise RuntimeError("evaluate_one_epoch_multi_step returned no examples.")

    # Ensure exactly 'horizon' steps, and stamp consistent 't'
    examples = examples[:horizon]
    for i, ex in enumerate(examples):
        ex["t"] = start_t + i
        if "bbox" not in ex:
            xmin, xmax, ymin, ymax = _get_bbox(cfg)
            ex["bbox"] = (float(xmin), float(xmax), float(ymin), float(ymax))

    feat_names = cfg.get("features", {}).get("names", None)
    degree_diag_steps = _parse_degree_diag_steps(args.degree_diag_steps, len(examples))

    if args.debug_fast_checks:
        bbox = examples[0]["bbox"] if len(examples) else tuple(_get_bbox(cfg))

        run_fast_drift_checks(
            test_stats,
            examples,
            feat_names=feat_names,
            bbox=bbox,
            knn_k=int(args.viz_raster_k),
            chunk=int(args.viz_raster_chunk),
            n_probe=int(args.debug_fast_checks_n),
        )

    # ----- Make GIFs -----
    if bool(args.truth_mode):
        if bool(args.include_deltas):
            raise ValueError("--truth-mode currently supports 2-panel output only. Disable --include-deltas.")
        log("[INFO] Making rollout GIFs (truth mode; physical-space rendering)...")
        with Timer("make_rollout_gifs_truth"):
            make_rollout_gifs_truth(
                examples=examples,
                cfg=cfg,
                raw_series=data_list,
                out_dir=out_dir,
                feature_names=feat_names,
                unify_clims=bool(args.unify_clims),
                fps=int(args.fps),
                dpi=int(args.dpi),
                progress_every=int(args.progress_every),
                two_panel_fig_w=float(args.two_panel_fig_w),
                two_panel_fig_h=float(args.two_panel_fig_h),
                truth_map_mode=str(args.truth_map_mode),
                truth_map_affine_tol=float(args.truth_map_affine_tol),
                truth_idw_backend=str(args.truth_idw_backend),
                truth_idw_k=int(args.truth_idw_k),
                truth_idw_chunk=int(args.truth_idw_chunk),
                truth_gt_ref_ratio=str(args.truth_gt_ref_ratio),
                truth_gt_render_level=str(args.truth_gt_render_level),
                truth_zoom_features=str(args.truth_zoom_features),
                truth_zoom_w_frac=float(args.truth_zoom_w_frac),
                truth_zoom_h_frac=float(args.truth_zoom_h_frac),
                truth_gt_comp_view=bool(args.truth_gt_comp_view),
            )
    else:
        log("[INFO] Making rollout GIFs (raster) with fixed clims..." if args.unify_clims else "[INFO] Making rollout GIFs (raster)...")
        with Timer("make_rollout_gifs_raster"):
            make_rollout_gifs_raster(
                examples=examples,
                cfg=cfg,
                out_dir=out_dir,
                feature_names=feat_names,
                unify_clims=bool(args.unify_clims),
                fps=int(args.fps),
                dpi=int(args.dpi),
                raster_mode=str(args.raster_mode),
                raster_bins=int(args.raster_bins),
                raster_k=int(args.viz_raster_k),
                raster_chunk=int(args.viz_raster_chunk),
                raster_lmax=(int(args.raster_lmax) if args.raster_lmax is not None else None),
                delta_scale=str(args.delta_scale),
                progress_every=int(args.progress_every),
                include_deltas=bool(args.include_deltas),
                two_panel_fig_w=float(args.two_panel_fig_w),
                two_panel_fig_h=float(args.two_panel_fig_h),
            )

    log(f"[INFO] Done. GIFs are in: {out_dir}")
    
    # Plot metrics (RelL2w all-features + MAEw 2x2) when available.
    if not bool(test_stats.get("infer_only", False)):
        plot_rollout_metrics_maew_rell2w(
            test_stats,
            feature_names=feat_names,
            start_t=start_t,
            save_dir=out_dir,          # optional; remove if you don't want saving yet
            prefix=f"metrics_t{start_t}_H{horizon}",
            dpi=150,
            show=False,
        )
    else:
        log("[INFO] infer-only rollout: skipping metric plots that require GT-mapped targets.")

    if degree_diag_steps is not None:
        if len(degree_diag_steps) == 0:
            log("[WARN] --degree-diag-steps was provided but no valid step tokens were parsed; skipping degree diagnostics.")
        else:
            diag_out_dir = out_dir
            log(f"[INFO] Making node-degree error bar plots for rollout steps: {degree_diag_steps}")
            with Timer("make_degree_error_bar_plots"):
                outputs = make_degree_error_bar_plots(
                    examples=examples,
                    step_indices=degree_diag_steps,
                    out_dir=diag_out_dir,
                    knn_k=int(args.viz_raster_k),
                    chunk=int(args.viz_raster_chunk),
                    dpi=max(120, int(args.dpi)),
                    log_fn=log,
                )
            if outputs:
                log(f"[INFO] Degree diagnostics saved to: {diag_out_dir}")
            else:
                log("[WARN] Degree diagnostics requested, but no plots were produced.")
    

if __name__ == "__main__":
    main()
