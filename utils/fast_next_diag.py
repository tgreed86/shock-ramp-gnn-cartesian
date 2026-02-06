# utils/fast_next_diag.py
# Drop-in “fastest next diagnostic” for:
#   (1) scale bug (~dt or ~1/dt),
#   (2) channel permutation,
#   (3) normalization mismatch
#
# Usage: call maybe_run_fast_next_diag(...) once from inside your training loop
# right after you have pred_rate, gt_t_feats, gt_tp1_feats, and dt for a step.

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def _as_tensor(x, device=None, dtype=None) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def _flatten_NF(x: torch.Tensor) -> torch.Tensor:
    # Accept (..., F) and flatten to (M, F)
    if x.ndim < 2:
        raise ValueError(f"Expected tensor with at least 2 dims (..., F). Got shape={tuple(x.shape)}")
    F = x.shape[-1]
    return x.reshape(-1, F)


def _broadcast_dt(dt: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    dt can be scalar, (B,), (N,), (B,1), (N,1), (...,1), etc.
    Returns dt broadcastable to like.shape[:-1] (feature-last).
    """
    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(dt, device=like.device, dtype=like.dtype)
    else:
        dt = dt.to(device=like.device, dtype=like.dtype)

    # If scalar, make shape (1,)
    if dt.ndim == 0:
        dt = dt.reshape(1)

    # Try to reshape dt to match like without touching last dim (F)
    # We want dt shape (..., 1) broadcastable with like (..., F).
    # If dt already ends with 1, good; else append singleton.
    if dt.shape[-1] != 1:
        dt = dt.unsqueeze(-1)

    # Now ensure dt is broadcastable to like.shape
    # We only need broadcastable to like.shape[:-1] + (1,)
    target = like.shape[:-1] + (1,)
    try:
        dt = dt.expand(target)
    except Exception:
        # fallback: try flattening dt to scalar median
        dt_scalar = torch.median(dt.reshape(-1)).to(like.device, like.dtype)
        dt = dt_scalar.reshape(1, 1).expand(target)
    return dt


def _safe_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach()
    xf = x.reshape(-1)
    if xf.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    mean = xf.mean()
    std = xf.std(unbiased=False)
    return {
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "mean": float(mean.item()),
        "std": float(std.item()),
    }


def _mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean()


def _rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((a - b) ** 2).mean())


def _rel_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.linalg.norm(a - b)
    den = torch.linalg.norm(b).clamp_min(eps)
    return num / den


def _per_channel_metrics(pred: torch.Tensor, tgt: torch.Tensor) -> Dict[str, List[float]]:
    # pred,tgt: (M,F)
    M, F = pred.shape
    out = {"mae": [], "rmse": [], "rel_l2": []}
    for j in range(F):
        pj = pred[:, j]
        tj = tgt[:, j]
        out["mae"].append(float(_mae(pj, tj).item()))
        out["rmse"].append(float(_rmse(pj, tj).item()))
        out["rel_l2"].append(float(_rel_l2(pj, tj).item()))
    return out


def _corr_matrix(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    pred,tgt: (M,F). Returns (F,F) correlation between pred[:,i] and tgt[:,j].
    """
    pred = pred.detach()
    tgt = tgt.detach()
    Mp, F = pred.shape
    pred0 = pred - pred.mean(dim=0, keepdim=True)
    tgt0 = tgt - tgt.mean(dim=0, keepdim=True)
    pred_std = pred0.std(dim=0, unbiased=False).clamp_min(eps)
    tgt_std = tgt0.std(dim=0, unbiased=False).clamp_min(eps)
    predn = pred0 / pred_std
    tgtn = tgt0 / tgt_std
    # corr(i,j) = mean(predn[:,i]*tgtn[:,j])
    return (predn.T @ tgtn) / Mp


def _best_perm_by_cost(cost: torch.Tensor) -> Tuple[Tuple[int, ...], float]:
    """
    cost: (F,F), minimize sum_i cost[i, perm[i]]
    brute-force for F<=8 (your case F=4).
    """
    F = cost.shape[0]
    assert cost.shape[1] == F
    best_p = None
    best_val = float("inf")
    for p in itertools.permutations(range(F)):
        val = 0.0
        for i in range(F):
            val += float(cost[i, p[i]].item())
        if val < best_val:
            best_val = val
            best_p = p
    return best_p, best_val


def _apply_mu_sigma_norm(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # x: (...,F); mu/sigma: (F,) or (1,F) etc.
    mu = mu.to(device=x.device, dtype=x.dtype)
    sigma = sigma.to(device=x.device, dtype=x.dtype).clamp_min(1e-12)
    while mu.ndim < x.ndim:
        mu = mu.unsqueeze(0)
    while sigma.ndim < x.ndim:
        sigma = sigma.unsqueeze(0)
    return (x - mu) / sigma


def _invert_mu_sigma_norm(xn: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    mu = mu.to(device=xn.device, dtype=xn.dtype)
    sigma = sigma.to(device=xn.device, dtype=xn.dtype)
    while mu.ndim < xn.ndim:
        mu = mu.unsqueeze(0)
    while sigma.ndim < xn.ndim:
        sigma = sigma.unsqueeze(0)
    return xn * sigma + mu


def _try_scaler_forward(x: torch.Tensor, scaler: Any) -> Optional[torch.Tensor]:
    """
    Best-effort: support scaler.forward(x), scaler.transform(x), scaler(x)
    """
    if scaler is None:
        return None
    try:
        if hasattr(scaler, "forward"):
            y = scaler.forward(x)
            if isinstance(y, torch.Tensor):
                return y
        if hasattr(scaler, "transform"):
            y = scaler.transform(x)
            if isinstance(y, torch.Tensor):
                return y
        if callable(scaler):
            y = scaler(x)
            if isinstance(y, torch.Tensor):
                return y
    except Exception:
        return None
    return None


def _try_scaler_inverse(x: torch.Tensor, scaler: Any) -> Optional[torch.Tensor]:
    """
    Best-effort: support scaler.reverse(x), scaler.inverse_transform(x)
    """
    if scaler is None:
        return None
    try:
        if hasattr(scaler, "reverse"):
            y = scaler.reverse(x)
            if isinstance(y, torch.Tensor):
                return y
        if hasattr(scaler, "inverse_transform"):
            y = scaler.inverse_transform(x)
            if isinstance(y, torch.Tensor):
                return y
    except Exception:
        return None
    return None


@dataclass
class FastDiagResult:
    # High-signal summary
    dt_stats: Dict[str, float]
    scale_summary: Dict[str, Any]
    perm_summary: Dict[str, Any]
    norm_summary: Dict[str, Any]

    # More details for saving
    details: Dict[str, Any]


@torch.no_grad()
def run_fast_next_diagnostics(
    *,
    pred_rate: torch.Tensor,         # model output (rate or delta pretending to be rate)
    gt_t_feats: torch.Tensor,        # GT features at time t
    gt_tp1_feats: torch.Tensor,      # GT features at time t+1 (same nodes as gt_t_feats for this check)
    dt: torch.Tensor | float,        # dt for this step (scalar or broadcastable)
    mu: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    scaler: Any = None,              # optional (e.g., your d_scaler style)
    feature_names: Optional[List[str]] = None,
    tag: str = "",
    save_json_path: Optional[str] = None,
) -> FastDiagResult:
    """
    Assumes gt_t_feats and gt_tp1_feats correspond to the *same mesh / indexing* for a single-step target.
    (This is the fastest sanity check; if you are mapping meshes, pass the already-mapped tensors.)

    Returns a FastDiagResult and optionally writes a JSON blob.
    """

    device = pred_rate.device
    dtype = pred_rate.dtype

    # shape checks
    if gt_t_feats.shape != gt_tp1_feats.shape:
        raise ValueError(f"gt_t_feats shape {gt_t_feats.shape} != gt_tp1_feats shape {gt_tp1_feats.shape}")
    if pred_rate.shape[-1] != gt_t_feats.shape[-1]:
        raise ValueError(f"pred_rate F={pred_rate.shape[-1]} != gt_feats F={gt_t_feats.shape[-1]}")

    # dt broadcast
    dt_b = _broadcast_dt(_as_tensor(dt, device=device, dtype=dtype), like=gt_t_feats)

    # flatten to (M,F)
    pr = _flatten_NF(pred_rate)
    g0 = _flatten_NF(gt_t_feats)
    g1 = _flatten_NF(gt_tp1_feats)
    dtf = _flatten_NF(dt_b).mean(dim=1, keepdim=True)  # (M,1)

    F = pr.shape[1]
    if feature_names is None:
        feature_names = [f"ch{i}" for i in range(F)]

    # Basic targets
    gt_delta = g1 - g0
    gt_rate = gt_delta / dtf.clamp_min(1e-12)

    # --- 1) SCALE BUG CHECKS (dt vs 1/dt vs delta-vs-rate) ---
    # Compare pred_rate against:
    #   (A) gt_rate
    #   (B) gt_delta  (means you're outputting delta but labeling it rate)
    #   (C) gt_rate * dt  (means you're ~1/dt off)
    #   (D) gt_delta / dt (same as gt_rate; included for clarity)
    pred_as_delta = pr * dtf  # if model is correct rate, this should match gt_delta

    # Global errors
    scale_candidates = {
        "pred_rate vs gt_rate": (pr, gt_rate),
        "pred_rate vs gt_delta (dt missing)": (pr, gt_delta),
        "pred_rate vs gt_rate*dt (≈1/dt error)": (pr, gt_rate * dtf),
        "pred_delta(pred_rate*dt) vs gt_delta": (pred_as_delta, gt_delta),
        "pred_delta(pred_rate*dt) vs gt_rate (≈dt error)": (pred_as_delta, gt_rate),
    }

    scale_err = {}
    for k, (a, b) in scale_candidates.items():
        scale_err[k] = {
            "mae": float(_mae(a, b).item()),
            "rmse": float(_rmse(a, b).item()),
            "rel_l2": float(_rel_l2(a, b).item()),
        }

    # Per-channel best scalar alpha that maps pred_rate -> gt_rate (least squares)
    # alpha_j = (p·g)/(p·p)
    alphas = []
    for j in range(F):
        pj = pr[:, j]
        gj = gt_rate[:, j]
        denom = (pj * pj).mean().clamp_min(1e-12)
        alpha = (pj * gj).mean() / denom
        alphas.append(float(alpha.item()))

    # Median dt for quick heuristics
    dt_med = float(torch.median(dtf.reshape(-1)).item())
    # If alpha ~ dt or ~1/dt, it's a strong dt scaling clue
    def _near(x, y, tol=0.30):
        if not (math.isfinite(x) and math.isfinite(y)) or y == 0:
            return False
        return abs(x - y) / abs(y) < tol

    scale_flags = {
        "alpha≈dt? (suggest pred is ~delta, not rate)": [_near(a, dt_med) for a in alphas],
        "alpha≈1/dt? (suggest pred is ~rate/dt)": [_near(a, (1.0 / dt_med) if dt_med != 0 else float("inf")) for a in alphas],
    }

    scale_summary = {
        "dt_median": dt_med,
        "candidate_errors": scale_err,
        "per_channel_alpha(pred_rate→gt_rate)": dict(zip(feature_names, alphas)),
        "heuristic_flags": {k: dict(zip(feature_names, v)) for k, v in scale_flags.items()},
    }

    # --- 2) CHANNEL PERMUTATION CHECK ---
    # Cost matrix: MAE(pred_i, gt_j) normalized by gt_j std (so scales don’t dominate)
    gt_std = gt_rate.std(dim=0, unbiased=False).clamp_min(1e-12)  # (F,)
    cost = torch.zeros((F, F), device=device, dtype=dtype)
    corr = _corr_matrix(pr, gt_rate)  # (F,F)

    for i in range(F):
        for j in range(F):
            cost[i, j] = (pr[:, i] - gt_rate[:, j]).abs().mean() / gt_std[j]

    best_p, best_cost = _best_perm_by_cost(cost)
    # Identity cost
    id_cost = float(sum(float(cost[i, i].item()) for i in range(F)))

    # Apply best permutation to pred channels (perm maps pred_i -> gt_{perm[i]})
    # For easier interpretation, build mapping gt_j gets pred_{invperm[j]}
    invperm = [None] * F
    for i, j in enumerate(best_p):
        invperm[j] = i

    pr_perm = pr[:, invperm]  # now pr_perm[:,j] aligns with gt_rate[:,j] under best mapping

    perm_summary = {
        "identity_norm_mae_sum": id_cost,
        "best_perm_norm_mae_sum": float(best_cost),
        "improvement_factor": float(id_cost / best_cost) if best_cost > 0 else float("inf"),
        "best_mapping_gt_channel <- pred_channel": {
            feature_names[j]: feature_names[invperm[j]] for j in range(F)
        },
        "corr(pred_i, gt_j)": {
            feature_names[i]: {feature_names[j]: float(corr[i, j].item()) for j in range(F)}
            for i in range(F)
        },
        "post_perm_errors(pred_rate_perm vs gt_rate)": {
            "mae": float(_mae(pr_perm, gt_rate).item()),
            "rmse": float(_rmse(pr_perm, gt_rate).item()),
            "rel_l2": float(_rel_l2(pr_perm, gt_rate).item()),
            "per_channel": _per_channel_metrics(pr_perm, gt_rate),
        },
        "pre_perm_errors(pred_rate vs gt_rate)": {
            "mae": float(_mae(pr, gt_rate).item()),
            "rmse": float(_rmse(pr, gt_rate).item()),
            "rel_l2": float(_rel_l2(pr, gt_rate).item()),
            "per_channel": _per_channel_metrics(pr, gt_rate),
        },
    }

    # --- 3) NORMALIZATION MISMATCH CHECK ---
    # We test whether pred_rate matches:
    #   - physical gt_rate (already computed)
    #   - normalized gt_rate if mu/sigma or scaler exists
    norm_tests = {}
    norm_mode = []

    # mu/sigma route
    mu_t = _as_tensor(mu, device=device, dtype=dtype)
    sigma_t = _as_tensor(sigma, device=device, dtype=dtype)
    if mu_t is not None and sigma_t is not None:
        g0n = _apply_mu_sigma_norm(g0, mu_t, sigma_t)
        g1n = _apply_mu_sigma_norm(g1, mu_t, sigma_t)
        gt_delta_n = g1n - g0n
        gt_rate_n = gt_delta_n / dtf.clamp_min(1e-12)

        norm_tests["pred_rate vs gt_rate_norm(mu/sigma)"] = {
            "mae": float(_mae(pr, gt_rate_n).item()),
            "rmse": float(_rmse(pr, gt_rate_n).item()),
            "rel_l2": float(_rel_l2(pr, gt_rate_n).item()),
        }
        norm_tests["pred_delta(pred_rate*dt) vs gt_delta_norm(mu/sigma)"] = {
            "mae": float(_mae(pred_as_delta, gt_delta_n).item()),
            "rmse": float(_rmse(pred_as_delta, gt_delta_n).item()),
            "rel_l2": float(_rel_l2(pred_as_delta, gt_delta_n).item()),
        }
        norm_mode.append("mu/sigma")

    # scaler route (best-effort)
    g0s = _try_scaler_forward(g0, scaler)
    g1s = _try_scaler_forward(g1, scaler)
    if (g0s is not None) and (g1s is not None):
        gt_delta_s = g1s - g0s
        gt_rate_s = gt_delta_s / dtf.clamp_min(1e-12)
        norm_tests["pred_rate vs gt_rate_norm(scaler)"] = {
            "mae": float(_mae(pr, gt_rate_s).item()),
            "rmse": float(_rmse(pr, gt_rate_s).item()),
            "rel_l2": float(_rel_l2(pr, gt_rate_s).item()),
        }
        norm_tests["pred_delta(pred_rate*dt) vs gt_delta_norm(scaler)"] = {
            "mae": float(_mae(pred_as_delta, gt_delta_s).item()),
            "rmse": float(_rmse(pred_as_delta, gt_delta_s).item()),
            "rel_l2": float(_rel_l2(pred_as_delta, gt_delta_s).item()),
        }
        norm_mode.append("scaler")

    # Decide “most likely mismatch” by comparing errors
    phys_mae = scale_err["pred_rate vs gt_rate"]["mae"]
    best_norm_key = None
    best_norm_mae = float("inf")
    for k, v in norm_tests.items():
        if v["mae"] < best_norm_mae:
            best_norm_mae = v["mae"]
            best_norm_key = k

    norm_summary = {
        "available_norm_modes": norm_mode,
        "phys_space_pred_rate_vs_gt_rate": scale_err["pred_rate vs gt_rate"],
        "norm_space_tests": norm_tests,
        "best_norm_match_key": best_norm_key,
        "best_norm_match_mae": best_norm_mae if best_norm_key is not None else None,
        "phys_mae": phys_mae,
        "strong_signal_pred_is_in_normalized_space": (
            (best_norm_key is not None) and (best_norm_mae < 0.6 * phys_mae)
        ),
        "stats_pred_rate": _safe_stats(pr),
        "stats_gt_rate_phys": _safe_stats(gt_rate),
        "stats_gt_delta_phys": _safe_stats(gt_delta),
    }

    # dt stats
    dt_stats = _safe_stats(dtf)

    details = {
        "tag": tag,
        "feature_names": feature_names,
        "dt_stats": dt_stats,
        "scale_summary": scale_summary,
        "perm_summary": perm_summary,
        "norm_summary": norm_summary,
        # include the raw per-channel candidate errors (high value)
        "scale_candidate_errors_per_channel": {
            name: _per_channel_metrics(a, b) for name, (a, b) in scale_candidates.items()
        },
    }

    res = FastDiagResult(
        dt_stats=dt_stats,
        scale_summary=scale_summary,
        perm_summary=perm_summary,
        norm_summary=norm_summary,
        details=details,
    )

    # Print a compact, high-signal report
    print("\n" + "=" * 88)
    print(f"[FAST_DIAG]{' ' + tag if tag else ''}")
    print(f"  dt stats: min={dt_stats['min']:.6g}  med={scale_summary['dt_median']:.6g}  max={dt_stats['max']:.6g}")
    print("  SCALE CANDIDATES (lower is better):")
    for k, v in scale_err.items():
        print(f"    - {k:45s}  MAE={v['mae']:.6g}  RMSE={v['rmse']:.6g}  RelL2={v['rel_l2']:.6g}")
    print("  per-channel alpha mapping pred_rate→gt_rate (LS fit):")
    for fn in feature_names:
        a = scale_summary["per_channel_alpha(pred_rate→gt_rate)"][fn]
        print(f"    - {fn:12s}: alpha={a:.6g}")
    print("  CHANNEL PERMUTATION:")
    print(f"    identity_norm_mae_sum={perm_summary['identity_norm_mae_sum']:.6g}")
    print(f"    best_perm_norm_mae_sum={perm_summary['best_perm_norm_mae_sum']:.6g}")
    print(f"    improvement_factor={perm_summary['improvement_factor']:.6g}")
    print("    best mapping (gt <- pred):")
    for k, v in perm_summary["best_mapping_gt_channel <- pred_channel"].items():
        print(f"      {k:12s} <- {v}")
    if norm_mode:
        print("  NORMALIZATION:")
        print(f"    modes detected: {norm_mode}")
        print(f"    phys MAE (pred_rate vs gt_rate): {phys_mae:.6g}")
        if best_norm_key is not None:
            print(f"    best norm match: {best_norm_key}  MAE={best_norm_mae:.6g}")
            print(f"    strong signal pred in normalized space? {norm_summary['strong_signal_pred_is_in_normalized_space']}")
    else:
        print("  NORMALIZATION: no mu/sigma and no usable scaler detected in this call.")
    print("=" * 88 + "\n")

    if save_json_path is not None:
        # Ensure JSON-serializable
        with open(save_json_path, "w") as f:
            json.dump(details, f, indent=2)
        print(f"[FAST_DIAG] wrote JSON: {save_json_path}")

    return res


def maybe_run_fast_next_diag(
    *,
    cfg: Dict[str, Any],
    state: Dict[str, Any],
    pred_rate: torch.Tensor,
    gt_t_feats: torch.Tensor,
    gt_tp1_feats: torch.Tensor,
    dt: torch.Tensor | float,
    mu: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    scaler: Any = None,
    feature_names: Optional[List[str]] = None,
    tag: str = "",
) -> None:
    """
    Gate this with cfg["debug"]["fast_next_diag"] and run only once per process.

    In your train loop, create:
        diag_state = {}
    and call maybe_run_fast_next_diag(..., state=diag_state, ...)
    """
    dbg = (cfg.get("debug", {}) or {})
    if not bool(dbg.get("fast_next_diag", False)):
        return
    if state.get("ran", False):
        return

    out_path = dbg.get("fast_next_diag_json", None)  # optional path string
    run_fast_next_diagnostics(
        pred_rate=pred_rate,
        gt_t_feats=gt_t_feats,
        gt_tp1_feats=gt_tp1_feats,
        dt=dt,
        mu=mu,
        sigma=sigma,
        scaler=scaler,
        feature_names=feature_names,
        tag=tag,
        save_json_path=out_path,
    )
    state["ran"] = True
