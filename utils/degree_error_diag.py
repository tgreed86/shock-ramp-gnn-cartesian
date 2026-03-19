from __future__ import annotations

import csv
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from plots import compute_plot_deltas, _recover_parent_mask
from utils_geom import unique_undirected


@torch.no_grad()
def _node_degree_from_edge_index(pred_ei: torch.Tensor | None, num_nodes: int) -> torch.Tensor:
    if pred_ei is None or (not torch.is_tensor(pred_ei)) or pred_ei.numel() == 0:
        return torch.zeros((num_nodes,), dtype=torch.long)

    ei = pred_ei.detach().cpu().to(torch.long)
    ei = unique_undirected(ei, int(num_nodes)).to(torch.long).cpu()
    if ei.numel() == 0:
        return torch.zeros((num_nodes,), dtype=torch.long)

    deg = torch.bincount(torch.cat([ei[0], ei[1]], dim=0), minlength=int(num_nodes))
    return deg.to(torch.long)


@torch.no_grad()
def _mean_abs_error_on_pred_mesh(
    ex: dict,
    *,
    knn_k: int,
    chunk: int,
) -> torch.Tensor:
    pred = ex["pred_tp1"].detach().cpu().to(torch.float32)
    H, W = int(ex["H"]), int(ex["W"])
    mask_pred_parent = _recover_parent_mask(ex, H, W)

    deltas = compute_plot_deltas(
        gt_t_centers=ex["centers_t"].detach().cpu().to(torch.float32),
        gt_t_feats=ex["gt_t"].detach().cpu().to(torch.float32),
        gt_tp1_centers=ex["centers_tp1"].detach().cpu().to(torch.float32),
        gt_tp1_feats=ex["gt_tp1"].detach().cpu().to(torch.float32),
        pred_centers=ex["pred_centers"].detach().cpu().to(torch.float32),
        pred_levels=ex["pred_levels"].detach().cpu().to(torch.int64),
        pred_parents=ex["pred_parents"].detach().cpu().to(torch.int64),
        mask_pred=mask_pred_parent,
        pred_feats=pred,
        H=H,
        W=W,
        knn_k=int(knn_k),
        chunk=int(chunk),
    )
    gt_tp1_on_pred = deltas["gt_tp1_on_pred"].detach().cpu().to(torch.float32)
    if pred.shape != gt_tp1_on_pred.shape:
        raise RuntimeError(
            f"Pred/GT mapped shape mismatch: pred={tuple(pred.shape)} vs gt_on_pred={tuple(gt_tp1_on_pred.shape)}"
        )
    return (pred - gt_tp1_on_pred).abs().mean(dim=1)


def _aggregate_degree_error(
    degree: torch.Tensor,
    err_node: torch.Tensor,
) -> list[tuple[int, int, float]]:
    d = degree.detach().cpu().numpy().astype(np.int64, copy=False)
    e = err_node.detach().cpu().numpy().astype(np.float64, copy=False)
    uniq = np.unique(d)

    out: list[tuple[int, int, float]] = []
    for deg in uniq.tolist():
        m = (d == deg)
        n = int(np.sum(m))
        if n == 0:
            continue
        out.append((int(deg), n, float(np.mean(e[m]))))
    return out


def _write_degree_error_csv(
    csv_path: str,
    rows: Iterable[tuple[int, int, float]],
):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degree", "node_count", "mean_abs_error"])
        for degree, node_count, mean_abs_error in rows:
            writer.writerow([int(degree), int(node_count), float(mean_abs_error)])


def _plot_degree_error_bar(
    png_path: str,
    rows: list[tuple[int, int, float]],
    *,
    step_idx: int,
    t_abs: int,
    dpi: int,
):
    deg = [r[0] for r in rows]
    mean_err = [r[2] for r in rows]
    total_nodes = int(sum(r[1] for r in rows))

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=int(dpi))
    ax.bar(deg, mean_err, color="#2a9d8f", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Mean absolute error (avg over features)")
    ax.set_title(f"Prediction Error vs Node Degree | k={step_idx}, t={t_abs}, N={total_nodes}")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    if len(deg) <= 30:
        ax.set_xticks(deg)
    fig.tight_layout()
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_degree_count_hist(
    png_path: str,
    rows: list[tuple[int, int, float]],
    *,
    step_idx: int,
    t_abs: int,
    dpi: int,
):
    deg = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    total_nodes = int(sum(counts))

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=int(dpi))
    ax.bar(deg, counts, color="#264653", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Node count")
    ax.set_title(f"Node Degree Histogram | k={step_idx}, t={t_abs}, N={total_nodes}")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    if len(deg) <= 30:
        ax.set_xticks(deg)
    fig.tight_layout()
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def make_degree_error_bar_plots(
    *,
    examples: list[dict],
    step_indices: list[int],
    out_dir: str,
    knn_k: int = 8,
    chunk: int = 8192,
    dpi: int = 150,
    log_fn=print,
) -> list[tuple[str, str, str]]:
    os.makedirs(out_dir, exist_ok=True)

    outputs: list[tuple[str, str, str]] = []
    for step_idx in step_indices:
        if step_idx < 0 or step_idx >= len(examples):
            raise ValueError(f"degree diag step {step_idx} is out of range [0, {len(examples)-1}]")

        ex = examples[step_idx]
        if "pred_ei" not in ex:
            log_fn(f"[WARN] degree diag: step {step_idx} has no pred_ei; skipping.")
            continue

        t_abs = int(ex.get("t", step_idx))
        n_nodes = int(ex["pred_tp1"].shape[0])
        degree = _node_degree_from_edge_index(ex.get("pred_ei"), n_nodes)
        err_node = _mean_abs_error_on_pred_mesh(ex, knn_k=int(knn_k), chunk=int(chunk))
        rows = _aggregate_degree_error(degree, err_node)
        if not rows:
            log_fn(f"[WARN] degree diag: no rows for step {step_idx}; skipping.")
            continue

        base = f"degree_error_bar_k{step_idx:03d}_t{t_abs}"
        png_path = os.path.join(out_dir, f"{base}.png")
        hist_png_path = os.path.join(out_dir, f"degree_hist_k{step_idx:03d}_t{t_abs}.png")
        csv_path = os.path.join(out_dir, f"{base}.csv")

        _write_degree_error_csv(csv_path, rows)
        _plot_degree_error_bar(
            png_path,
            rows,
            step_idx=int(step_idx),
            t_abs=int(t_abs),
            dpi=int(dpi),
        )
        _plot_degree_count_hist(
            hist_png_path,
            rows,
            step_idx=int(step_idx),
            t_abs=int(t_abs),
            dpi=int(dpi),
        )
        outputs.append((png_path, hist_png_path, csv_path))
        log_fn(f"[DEGREE-DIAG] wrote: {png_path}")
        log_fn(f"[DEGREE-DIAG] wrote: {hist_png_path}")

    return outputs
