# interpolation.py
# Back-compat wrappers around utils_geom interpolation helpers.

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from utils_geom import (
    _targeted_map_to_pred as _targeted_map_to_pred_impl,
    apply_idw_map as _apply_idw_map_impl,
    build_idw_map as _build_idw_map_impl,
)


@torch.no_grad()
def kNN_indices(
    dst_xy: torch.Tensor,
    src_xy: torch.Tensor,
    k: int = 8,
    chunk: int = 8192,
) -> torch.Tensor:
    """
    Return k-nearest source indices for each destination point.
    """
    n_dst = int(dst_xy.shape[0])
    all_idx = []
    for start in range(0, n_dst, chunk):
        end = min(n_dst, start + chunk)
        d = torch.cdist(dst_xy[start:end], src_xy, p=2)
        tk = torch.topk(d, k=min(k, d.shape[1]), dim=1, largest=False)
        all_idx.append(tk.indices)
    return torch.cat(all_idx, dim=0)


@torch.no_grad()
def build_idw_map(
    dst_xy: torch.Tensor,
    src_xy: torch.Tensor,
    k: int = 8,
    chunk: int = 8192,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Back-compat wrapper for utils_geom.build_idw_map.
    """
    del eps  # utils_geom handles epsilon internally
    return _build_idw_map_impl(dst_xy, src_xy, k=k, chunk=chunk)


@torch.no_grad()
def apply_idw_map(idx: torch.Tensor, w: torch.Tensor, src_feat: torch.Tensor) -> torch.Tensor:
    """
    Back-compat wrapper for utils_geom.apply_idw_map.
    """
    return _apply_idw_map_impl(idx, w, src_feat)


@torch.no_grad()
def map_points_idw(
    dst_xy: torch.Tensor,
    src_xy: torch.Tensor,
    src_feat: torch.Tensor,
    k: int = 8,
    chunk: int = 8192,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convenience wrapper: build + apply IDW in one call.
    """
    idx, w = build_idw_map(dst_xy, src_xy, k=k, chunk=chunk, eps=eps)
    return apply_idw_map(idx, w, src_feat)


@torch.no_grad()
def targeted_map_to_pred(
    pred_centers: torch.Tensor,
    pred_levels: torch.Tensor,
    pred_parents: torch.Tensor,
    mask_pred: Optional[torch.Tensor],
    src_centers: torch.Tensor,
    src_feats: torch.Tensor,
    *,
    mask_src_parent: Optional[torch.Tensor] = None,
    src_parent_feats: Optional[torch.Tensor] = None,
    knn_k: int = 8,
    chunk: int = 8192,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Back-compat wrapper for utils_geom._targeted_map_to_pred.
    Falls back to pure IDW if mask_pred is not provided.
    """
    del eps  # utils_geom handles epsilon internally

    if mask_pred is None:
        out = map_points_idw(pred_centers, src_centers, src_feats, k=knn_k, chunk=chunk)
        return out, {"copied_same_coarse": 0, "idw_points": int(pred_centers.size(0))}

    if mask_pred.ndim == 2:
        H, W = int(mask_pred.shape[0]), int(mask_pred.shape[1])
    else:
        H, W = 1, int(mask_pred.numel())

    return _targeted_map_to_pred_impl(
        pred_centers=pred_centers,
        pred_levels=pred_levels,
        pred_parents=pred_parents,
        mask_pred=mask_pred,
        src_centers=src_centers,
        src_feats=src_feats,
        H=H,
        W=W,
        mask_src_parent=mask_src_parent,
        src_parent_feats=src_parent_feats,
        knn_k=knn_k,
        chunk=chunk,
    )
