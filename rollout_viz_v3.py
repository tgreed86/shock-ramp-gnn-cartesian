#!/usr/bin/env python
"""
rollout_viz.py (replacement)

Long-rollout visualization script for trained models (memory-safe + faster).

What it produces:
- One GIF per feature, named: rollout_<feature_name>.gif
- Each frame is a 2x3 panel:
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
  --interp-k 4            Smaller kNN for delta interpolation (speedup).
  --interp-chunk 4096     Chunk size for interpolation (often affects speed a lot).

Notes:
- Uses Agg backend to avoid GUI/memory issues on macOS.
- Assumes these repo imports exist:
    dataset.CellRefineWindowDataset
    pretrain.precompute_pred_mesh_and_interps_for_rollout, pretrain.CollateWithPrecompute
    plots.compute_plot_deltas, plots._recover_parent_mask, plots._unwrap_delta, plots._fidx_or_none, plots._draw_amr_cells
    train.build_model_from_cfg, train.evaluate_one_epoch_multi_step, train._get_bbox
"""

import os
import io
import gc
import json
import time
import argparse
import zipfile
import inspect
from typing import Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # critical on macOS for headless/stability
import matplotlib.pyplot as plt

import imageio.v2 as imageio
from torch.utils.data import DataLoader, SequentialSampler, Subset

# Project imports – must exist in your repo.
from dataset import CellRefineWindowDataset
from pretrain import precompute_pred_mesh_and_interps_for_rollout, CollateWithPrecompute
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
)
from utils_geom import build_idw_map, apply_idw_map
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

def _load_pt_series(pt_path: str):
    """Load raw list[Data] time series (.pt or .zip containing .pt)."""
    if pt_path.endswith(".zip"):
        with zipfile.ZipFile(pt_path, "r") as zf:
            member = next(m for m in zf.namelist() if m.endswith(".pt") or m.endswith(".pth"))
            with zf.open(member, "r") as fh:
                buf = io.BytesIO(fh.read())
        data_list = torch.load(buf, map_location="cpu")
    else:
        data_list = torch.load(pt_path, map_location="cpu")
    return data_list

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

def _build_collate(precomp, dt_transitions, dt_ref):
    """Construct CollateWithPrecompute, passing dt metadata when supported."""
    try:
        inspect.signature(CollateWithPrecompute)
    except Exception:
        pass

    for ctor in (
        lambda: CollateWithPrecompute(precomp, dt_transitions=dt_transitions, dt_ref=dt_ref),
        lambda: CollateWithPrecompute(precomp, dt_transitions=dt_transitions),
        lambda: CollateWithPrecompute(precomp),
    ):
        try:
            return ctor()
        except TypeError:
            continue
    raise RuntimeError("Could not construct CollateWithPrecompute with available arguments.")

def _maybe_empty_device_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


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
):
    """
    Raster/imshow rollout GIF writer using the 2x3 panel layout:

      Top row:    [GT(t), Pred(t+1), GT(t+1)]
      Bottom row: [GT(t+1)-GT(t), Pred(t+1)-GT(t+1), Pred(t+1)-GT(t)]

    Fixed-scale behavior (when unify_clims=True):
      - Top-row scale is fixed across all timesteps PER FEATURE, determined ONLY from GT(t+1) plots.
      - Delta-row scale is fixed across all timesteps PER FEATURE, determined ONLY from GT(t+1)-GT(t) plots.
      - All three delta panels share the same symmetric scale.

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
        AMR block replication onto a common fine grid sized (H*2^Lmax, W*2^Lmax).
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

        scale = 1 << int(Lmax)
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
            b = 1 << (Lmax - l)  # block size in fine pixels
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
        global_d_abs_f   = torch.zeros((F,), dtype=torch.float32)

        for step, ex in enumerate(examples):
            H = int(ex["H"]); W = int(ex["W"])
            bbox = tuple(ex["bbox"])

            step_lmax = _infer_lmax(ex)
            if raster_lmax is not None:
                step_lmax = min(int(step_lmax), int(raster_lmax))

            if not hasattr(make_rollout_gifs_raster, "_printed_keys"):
                make_rollout_gifs_raster._printed_keys = True
                print("[ROLLOUT-CHK] ex keys:", sorted(list(ex.keys())))

            # Pull GT(t) and GT(t+1) for TOP + GTΔ clims
            A_cent = ex["centers_t"]
            B_cent = ex["centers_tp1"]

            # Prefer level_t/level_tp1 if present; block needs them
            A_lev = ex.get("level_t", None)
            B_lev = ex.get("level_tp1", None)

            A_val = ex["gt_t"]
            B_val = ex["gt_tp1"]

            if raster_mode.lower() == "idw":
                A_img, HH, WW = _rasterize_idw(A_cent, A_val, raster_bins, raster_k, bbox, raster_chunk)
                B_img, _,  _  = _rasterize_idw(B_cent, B_val, raster_bins, raster_k, bbox, raster_chunk)
            else:
                if A_lev is None or B_lev is None:
                    raise KeyError(
                        "unify_clims=True with raster_mode='block' requires ex['level_t'] and ex['level_tp1'] "
                        "in every example."
                    )
                A_img, HH, WW = _rasterize_block_common(A_cent, A_lev, A_val, H, W, bbox, step_lmax)
                B_img, _,  _  = _rasterize_block_common(B_cent, B_lev, B_val, H, W, bbox, step_lmax)

            A_img = _fill_internal_nans_nearest(A_img, HH, WW)
            B_img = _fill_internal_nans_nearest(B_img, HH, WW)

            # TOP clims from GT(t+1) ONLY
            global_top_min_f = torch.minimum(global_top_min_f, _nanmin_per_feature(B_img))
            global_top_max_f = torch.maximum(global_top_max_f, _nanmax_per_feature(B_img))

            # DELTA clims from GTΔ ONLY
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

        global_d_abs_f = torch.where((~torch.isfinite(global_d_abs_f)) | (global_d_abs_f <= 0.0),
                                     torch.tensor(1e-12),
                                     global_d_abs_f)

        print("[CLIM] Done.")
        print(f"[CLIM] Example (first 3 feats):")
        for f in range(min(3, F)):
            print(f"       f={f}: top=[{float(global_top_min_f[f]):.6e}, {float(global_top_max_f[f]):.6e}] "
                  f"delta=±{float(global_d_abs_f[f]):.6e}")

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
            A_cent = ex["centers_t"]
            B_cent = ex["centers_tp1"]
            P_cent = ex["pred_centers"]

            A_lev = ex.get("level_t", ex.get("pred_levels", None))
            B_lev = ex.get("level_tp1", ex.get("pred_levels", None))
            P_lev = ex["pred_levels"]

            A_val = ex["gt_t"]
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
            
            # rasterize three “top-row” fields onto a common grid
            if raster_mode.lower() == "idw":
                A_img, HH, WW = _rasterize_idw(A_cent, A_val, raster_bins, raster_k, bbox, raster_chunk)
                B_img, _,  _  = _rasterize_idw(B_cent, B_val, raster_bins, raster_k, bbox, raster_chunk)
                P_img, _,  _  = _rasterize_idw(P_cent, P_val, raster_bins, raster_k, bbox, raster_chunk)
                mesh_label = f"(idw {raster_bins}×{raster_bins}, k={raster_k})"
            else:
                if A_lev is None or B_lev is None:
                    raise KeyError("Block raster requires level_t and level_tp1 (or compatible) in each example.")
                A_img, HH, WW = _rasterize_block_common(A_cent, A_lev, A_val, H, W, bbox, step_lmax)
                B_img, _,  _  = _rasterize_block_common(B_cent, B_lev, B_val, H, W, bbox, step_lmax)
                P_img, _,  _  = _rasterize_block_common(P_cent, P_lev, P_val, H, W, bbox, step_lmax)
                mesh_label = f"(block @ {HH}×{WW}, Lmax={step_lmax})"
            
            A_img = _fill_internal_nans_nearest(A_img, HH, WW)
            B_img = _fill_internal_nans_nearest(B_img, HH, WW)

            # Diagnostic: rasterized level map (what refinement is actually present)
            P_level_img, HH, WW = _rasterize_block_common(P_cent, P_lev, P_lev.to(torch.float32)[:, None], H, W, bbox, step_lmax)
            B_level_img, B_HH, B_WW = _rasterize_block_common(B_cent, B_lev, B_lev.to(torch.float32)[:, None], H, W, bbox, step_lmax)

            # deltas on the same raster grid
            #Dgt = (B_img - A_img)  # GT(t+1)-GT(t)
            #Dpg = (P_img - B_img)  # Pred(t+1)-GT(t+1)
            #Dpt = (P_img - A_img)  # Pred(t+1)-GT(t)

            # deltas on the same raster grid
            Dgt = (B_img - A_img)  # GT(t+1)-GT(t)

            if raster_mode.lower() == "block":
                # Map GT(t+1) features onto Pred(t+1) centers (IDW on centers)
                idx_tp1, w_tp1 = build_idw_map(
                    dst_xy=P_cent.to(dtype=torch.float32, device="cpu"),
                    src_xy=B_cent.to(dtype=torch.float32, device="cpu"),
                    k=raster_k,          # or choose a separate map_k if desired
                    chunk=raster_chunk,  # keep consistent with raster settings
                )
                gt_tp1_on_pred = apply_idw_map(
                    idx_tp1, w_tp1, B_val.to(dtype=torch.float32, device="cpu")
                )  # (N_pred, F)

                # Rasterize that mapped GT(t+1) using the *Pred mesh geometry*
                B_on_pred_img, _, _ = _rasterize_block_common(
                    centers=P_cent,
                    levels=P_lev,
                    values=gt_tp1_on_pred,
                    H=H, W=W, bbox=bbox, Lmax=step_lmax,
                )

                # Δ₂ on Pred mesh (rasterized)
                Dpg = (P_img - B_on_pred_img)
            else:
                # IDW raster already yields dense grids; the direct diff is fine
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

                # DELTA scale: determined by GTΔ ONLY (Dgt), shared by all 3 delta panels
                if unify_clims:
                    m = float(global_d_abs_f[f].item())
                else:
                    m = float(torch.nan_to_num(Dgt[:, f].abs(), nan=0.0).max().item())

                # If user insists on "each" and unify_clims is off, allow it; otherwise force "gt"
                if (not unify_clims) and (delta_scale.lower() == "each"):
                    m1 = float(torch.nan_to_num(Dgt[:, f].abs(), nan=0.0).max().item())
                    m2 = float(torch.nan_to_num(Dpg[:, f].abs(), nan=0.0).max().item())
                    m3 = float(torch.nan_to_num(Dpt[:, f].abs(), nan=0.0).max().item())
                    lims = [(-m1, +m1), (-m2, +m2), (-m3, +m3)]
                else:
                    lims = [(-m, +m), (-m, +m), (-m, +m)]

                fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=int(dpi), constrained_layout=True)
                fig.suptitle(f"t={t} — {feature_names[f]} {mesh_label}", fontsize=12)

                # Top row: GT(t), Pred(t+1), GT(t+1) — all share GT(t+1)-anchored scale
                im0 = _imshow_flat(axs[0, 0], A_img[:, f], HH, WW, "GT(t)",     vmin=tmin, vmax=tmax, cmap="viridis")
                im1 = _imshow_flat(axs[0, 1], P_img[:, f], HH, WW, "Pred(t+1)", vmin=tmin, vmax=tmax, cmap="viridis")
                #im1 = _imshow_flat(axs[0, 1], P_level_img[:, 0], HH, WW, "Pred mesh level", vmin=0, vmax=step_lmax, cmap="viridis")
                im2 = _imshow_flat(axs[0, 2], B_img[:, f], HH, WW, "GT(t+1)",   vmin=tmin, vmax=tmax, cmap="viridis")
                #im2 = _imshow_flat(axs[0, 2], B_level_img[:, 0], B_HH, B_WW, "GT mesh level", vmin=0, vmax=step_lmax, cmap="viridis")

                fig.colorbar(im0, ax=axs[0, 0], shrink=0.75)
                fig.colorbar(im1, ax=axs[0, 1], shrink=0.75)
                fig.colorbar(im2, ax=axs[0, 2], shrink=0.75)

                # Bottom row: deltas — all share GTΔ-anchored symmetric scale
                im3 = _imshow_flat(axs[1, 0], Dgt[:, f], HH, WW, "GT(t+1) − GT(t)",
                                   vmin=lims[0][0], vmax=lims[0][1], cmap="coolwarm")
                im4 = _imshow_flat(axs[1, 1], Dpg[:, f], HH, WW, "Pred(t+1) − GT(t+1)",
                                   vmin=lims[1][0], vmax=lims[1][1], cmap="coolwarm")
                im5 = _imshow_flat(axs[1, 2], Dpt[:, f], HH, WW, "Pred(t+1) − GT(t)",
                                   vmin=lims[2][0], vmax=lims[2][1], cmap="coolwarm")

                fig.colorbar(im3, ax=axs[1, 0], shrink=0.75)
                fig.colorbar(im4, ax=axs[1, 1], shrink=0.75)
                fig.colorbar(im5, ax=axs[1, 2], shrink=0.75)

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


# ----------------- Main rollout script ----------------- #

def main():
    import argparse
    import json
    import os
    import time
    import torch
    from torch.utils.data import DataLoader, Subset, SequentialSampler

    ap = argparse.ArgumentParser(description="Long-rollout visualization with GIFs (raster version + fixed clims).")

    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint .pt (e.g. best_model.pt or last_model.pt).")
    ap.add_argument("--config", type=str, default=None,
                    help="Optional JSON config; if omitted, uses cfg from checkpoint.")
    ap.add_argument("--pt-path", type=str, default=None,
                    help="Optional override for cfg['data']['pt_path'].")

    ap.add_argument("--horizon", type=int, default=20,
                    help="Number of predicted steps to visualize (rollout length).")
    ap.add_argument("--start-t", type=int, default=48,
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

    # DataLoader
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Number of workers for DataLoader.")

    # Precomp controls
    ap.add_argument("--precomp-path", type=str, default=None,
                    help="Optional path to load precomp (torch.load).")
    ap.add_argument("--save-precomp", type=str, default=None,
                    help="If set and precomp is computed, save it to this path (torch.save).")
    ap.add_argument("--recompute-precomp", action="store_true",
                    help="Force recompute of precomp instead of using checkpoint precomp.")
    ap.add_argument("--precompute-scope", type=str, default="all", choices=["window", "all"],
                    help="Compute precomp only for required timesteps ('window') or for all ('all').")
    ap.add_argument("--precompute-device", type=str, default="cpu",
                    help="Device to use for precompute (typically cpu).")

    # Raster controls
    ap.add_argument("--raster-mode", type=str, default="block", choices=["block", "idw"],
                    help="Rasterization mode for plots.")
    ap.add_argument("--raster-lmax", type=int, default=None,
                    help="Clamp raster Lmax (e.g. 3). If omitted, infer per step.")
    ap.add_argument("--delta-scale", type=str, default="gt", choices=["gt", "each"],
                    help="Delta scaling when unify_clims is OFF. When unify_clims is ON, we force 'gt' behavior.")

    ap.add_argument("--raster-bins", type=int, default=256, help="IDW grid bins (only for raster-mode=idw).")
    ap.add_argument("--raster-k", type=int, default=8, help="IDW kNN k (only for raster-mode=idw).")
    ap.add_argument("--raster-chunk", type=int, default=32768, help="IDW chunk (only for raster-mode=idw).")

    args = ap.parse_args()

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

    # Model device
    device_str = args.device if (args.device is not None) else cfg.get("device", "cpu")
    device = torch.device(device_str)
    log(f"[INFO] Using model device: {device}")

    # Optional: simple CUDA verification print
    if device.type == "cuda":
        log(f"[CUDA] available={torch.cuda.is_available()} current_device={torch.cuda.current_device()} name={torch.cuda.get_device_name(torch.cuda.current_device())}")

    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    # Check for adapter weights in checkpoint
    '''
    sd = ckpt["model"] if "model" in ckpt else ckpt
    has_adapter_weights = any(k.startswith("parc_adapter.") for k in sd.keys())
    # Build model with/without adapter accordingly
    cfg["loss"]["parc_use_adapter"] = bool(has_adapter_weights)
    print(f"[INFO] Checkpoint has_adapter_weights={has_adapter_weights}; setting cfg['loss']['parc_use_adapter']={cfg['loss']['parc_use_adapter']}")
    '''

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
    T = len(data_list)
    log(f"[INFO] Series length T={T}")

    if start_t < 0 or start_t + horizon >= T:
        raise ValueError(
            f"Invalid (start_t={start_t}, horizon={horizon}) for series length T={T}. "
            f"Need start_t+horizon < T."
        )

    # dt info (if present in snapshots)
    dt_transitions, dt_ref = _compute_dt_transitions(data_list)

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

    # ----- Precomp -----
    precomp = None
    print("args.precomp_path:", args.precomp_path)
    if args.precomp_path is not None:
        log(f"[INFO] Loading precomp from {args.precomp_path}")
        precomp = torch.load(args.precomp_path, map_location="cpu", weights_only=False)
    elif (not args.recompute_precomp) and ("precomp" in ckpt):
        log("[INFO] Using precomp from checkpoint.")
        precomp = ckpt["precomp"]
    print("precomp:", precomp)

    if precomp is None:
        steps = getattr(full_ds, "steps", None)
        if steps is None:
            raise RuntimeError("Dataset has no attribute 'steps'; cannot precompute precomp.")

        pre_device = torch.device(str(args.precompute_device))
        if args.precompute_scope == "window":
            lo = window_idx
            hi = window_idx + horizon + 1
            steps_to_precompute = steps[lo:hi]
            log(f"[INFO] Precomputing precomp for window steps [{lo}:{hi}] on {pre_device}...")
        else:
            steps_to_precompute = steps
            log(f"[INFO] Precomputing precomp for ALL steps ({len(steps_to_precompute)}) on {pre_device}...")

        with Timer("Precompute predicted meshes + maps"):
            precomp = precompute_pred_mesh_and_interps_for_rollout(
                steps=steps_to_precompute,
                cfg=cfg,
                H=H, W=W,
                dx=dx, dy=dy,
                device=pre_device,
                progress=True,
            )

        if args.save_precomp is not None:
            os.makedirs(os.path.dirname(args.save_precomp) or ".", exist_ok=True)
            torch.save(precomp, args.save_precomp)
            log(f"[INFO] Saved precomp to {args.save_precomp}")

    collate = _build_collate(precomp, dt_transitions, dt_ref)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        sampler=SequentialSampler(test_ds),
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate,
    )

    # ----- Build model & load weights -----
    '''
    with Timer("Build model"):
        model = build_model_from_cfg(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    '''
    with Timer("Build model"):
        model = build_model_from_cfg(cfg, device)

    # ---- attach ParcFeatureAdapter if PARC is enabled (mirrors training main) ----
    loss_cfg = cfg.get("loss", {}) or {}
    parc_use = bool(loss_cfg.get("parc", False) or loss_cfg.get("parc_inputs", False))
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

    '''
    # ----- Load weights (strip parc_adapter keys if present) -----
    sd = ckpt["model"] if "model" in ckpt else ckpt

    # Remove adapter keys unconditionally — safe for all checkpoints
    sd_no_adapter = {k: v for k, v in sd.items() if not k.startswith("parc_adapter.")}

    missing, unexpected = model.load_state_dict(sd_no_adapter, strict=False)

    # Sanity check: nothing except adapter should be missing/unexpected
    missing_non_adapter = [k for k in missing if not k.startswith("parc_adapter.")]
    unexpected_non_adapter = [k for k in unexpected if not k.startswith("parc_adapter.")]

    if missing_non_adapter or unexpected_non_adapter:
        raise RuntimeError(
            "State-dict mismatch beyond parc_adapter.*\n"
            f"Missing: {missing_non_adapter}\n"
            f"Unexpected: {unexpected_non_adapter}"
        )

    if missing or unexpected:
        print(
            f"[INFO] Loaded checkpoint ignoring adapter keys "
            f"(missing_adapter={len(missing)}, unexpected_adapter={len(unexpected)})"
        )
    '''
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

    # Norm stats
    norm_stats = ckpt.get("norm_stats", None)
    if norm_stats is not None and norm_stats.get("mu") is not None:
        mu = torch.tensor(norm_stats["mu"], dtype=torch.float32, device=device)
        sigma = torch.tensor(norm_stats["sigma"], dtype=torch.float32, device=device)
        log("[INFO] Using normalization stats from checkpoint.")
    else:
        mu = sigma = None
        log("[WARN] No normalization stats in checkpoint; proceeding without normalization.")

    # ----- Evaluate only this rollout window -----
    log(f"[INFO] Running multi-step evaluation for ONE window at start_t={start_t} (horizon={horizon})...")
    with Timer("evaluate_one_epoch_multi_step"):
        #test_loss, test_mae, test_stats = evaluate_one_epoch_multi_step(
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
            budget_csv_path=budget_csv_path, 
            write_budgets=True,
        )
    #log(f"[TEST] loss={test_loss:.4e}, MAE={test_mae:.4e}")
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

    # ----- Make GIFs (RASTER) -----
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
            raster_k=int(args.raster_k),
            raster_chunk=int(args.raster_chunk),
            raster_lmax=(int(args.raster_lmax) if args.raster_lmax is not None else None),
            delta_scale=str(args.delta_scale),
            progress_every=int(args.progress_every),
        )

    log(f"[INFO] Done. GIFs are in: {out_dir}")
    
    # Plot metrics (RelL2w all-features + MAEw 2x2)
    plot_rollout_metrics_maew_rell2w(
        test_stats,
        feature_names=feat_names,
        start_t=start_t,
        save_dir=out_dir,          # optional; remove if you don't want saving yet
        prefix=f"metrics_t{start_t}_H{horizon}",
        dpi=150,
        show=False,
    )
    

if __name__ == "__main__":
    main()

