#!/usr/bin/env python3
"""
Plot rollout "budget" (global integral) diagnostics saved by evaluate_one_epoch_multi_step.

Expected CSV columns (minimum):
  t_abs, step_k, kind, mass, mom_x, mom_y, energy, area

Where:
  - kind is typically "gt" or "pred"
  - t_abs is the absolute time index in the dataset (if available)
  - step_k is the rollout step index (0..H-1)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_QUANTS = ("mass", "mom_x", "mom_y", "energy", "area")


def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] CSV missing required columns: {missing}\n"
                         f"        Found columns: {list(df.columns)}")


def _sort_kind(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer consistent ordering: gt then pred (if present)
    kind_order = {"gt": 0, "pred": 1}
    if "kind" in df.columns:
        df = df.copy()
        df["_kind_order"] = df["kind"].map(kind_order).fillna(99).astype(int)
        df = df.sort_values(["_kind_order", "t_abs", "step_k"]).drop(columns=["_kind_order"])
    return df


def plot_budget_scatter(
    df: pd.DataFrame,
    *,
    x_key: str,
    out_dir: str,
    prefix: str = "budget",
    quants: Sequence[str] = DEFAULT_QUANTS,
    title_suffix: str | None = None,
    show: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    _require_cols(df, ["kind", x_key])
    for q in quants:
        if q not in df.columns:
            # Skip quietly, so the script is robust to extra/changed fields
            continue

        fig = plt.figure()
        ax = plt.gca()

        # Plot GT and Pred separately (scatter)
        for kind in sorted(df["kind"].unique()):
            sub = df[df["kind"] == kind]
            ax.scatter(sub[x_key].to_numpy(), sub[q].to_numpy(), label=str(kind))

        ax.set_xlabel(x_key)
        ax.set_ylabel(q)
        ttl = f"{q} vs {x_key}"
        if title_suffix:
            ttl += f" ({title_suffix})"
        ax.set_title(ttl)
        ax.legend()

        out_png = os.path.join(out_dir, f"{prefix}_{q}_vs_{x_key}.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot rollout budget CSV as scatter plots.")
    ap.add_argument("--csv", type=str, required=True, help="Path to rollout_budgets.csv")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Directory to save PNGs. Default: <csv_dir>/budget_plots")
    ap.add_argument("--prefix", type=str, default="budget", help="Filename prefix for outputs")
    ap.add_argument("--x", type=str, default="t_abs", choices=["t_abs", "step_k"],
                    help="X-axis column to use")
    ap.add_argument("--quants", type=str, default=",".join(DEFAULT_QUANTS),
                    help=f"Comma-separated list of columns to plot (default: {DEFAULT_QUANTS})")
    ap.add_argument("--show", action="store_true", help="Display interactive windows (optional)")
    args = ap.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise SystemExit(f"[ERROR] CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Basic expectations
    _require_cols(df, ["t_abs", "step_k", "kind"])
    df = _sort_kind(df)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "budget_plots")

    quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    # Helpful title context: infer rollout length
    title_suffix = None
    try:
        H = int(df["step_k"].max()) + 1
        t0 = int(df["t_abs"].min())
        t1 = int(df["t_abs"].max())
        title_suffix = f"H={H}, t={t0}..{t1}"
    except Exception:
        pass

    plot_budget_scatter(
        df,
        x_key=args.x,
        out_dir=out_dir,
        prefix=args.prefix,
        quants=quants,
        title_suffix=title_suffix,
        show=bool(args.show),
    )

    print(f"[OK] Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
