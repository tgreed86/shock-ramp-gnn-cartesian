# interpolation.py
# Robust nearest-neighbor and IDW utilities for AMR point sets.

from __future__ import annotations
import torch
from typing import Tuple, Optional, Dict

@torch.no_grad()
def kNN_indices(
    dst_xy: torch.Tensor,    # (N,2)
    src_xy: torch.Tensor,    # (M,2)
    k: int = 8,
    chunk: int = 8192,
) -> torch.Tensor:
    """
    Return indices of the K NEAREST neighbors in src for each dst point.
    Chunked to bound memory. Shape: (N, k), dtype long.
    """
    N = int(dst_xy.shape[0])
    all_idx = []
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        # topk on *negative distance* yields smallest distances
        d = torch.cdist(dst_xy[s:e], src_xy)  # (b,M)
        _, idx = torch.topk(-d, k=min(k, d.shape[1]), dim=1)
        all_idx.append(idx)
    return torch.cat(all_idx, dim=0)


@torch.no_grad()
def build_idw_map(
    dst_xy: torch.Tensor,    # (N,2)
    src_xy: torch.Tensor,    # (M,2)
    k: int = 8,
    chunk: int = 8192,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Correct IDW map:
      1) pick K nearest neighbors,
      2) compute weights from their distances only,
      3) normalize weights so each row sums to 1,
      4) handle exact matches as one-hot.

    Returns:
        idx : (N, k) long  indices into src_xy
        w   : (N, k) float weights, row-normalized
    """
    N = int(dst_xy.shape[0])
    all_idx, all_w = [], []

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        d_full = torch.cdist(dst_xy[s:e], src_xy)                     # (b,M)

        # 1) KNN
        d_neg, idx = torch.topk(-d_full, k=min(k, d_full.shape[1]), dim=1)  # (b,k)
        d = -d_neg                                                           # (b,k), smallest distances

        # 2) exact-match handling: if any distance == 0, make that neighbor one-hot
        zero_mask = (d <= eps)
        if zero_mask.any():
            # set all weights to 0, then place 1 at the exact-match neighbor (first one)
            w = torch.zeros_like(d)
            first = torch.argmax(zero_mask.to(torch.int64), dim=1)  # (b,)
            w.scatter_(1, first.view(-1,1), 1.0)
        else:
            # 3) standard IDW on the K neighbors only
            w = 1.0 / (d + eps)
            w = w / (w.sum(dim=1, keepdim=True) + eps)

        all_idx.append(idx)
        all_w.append(w)

    return torch.cat(all_idx, dim=0), torch.cat(all_w, dim=0)


@torch.no_grad()
def apply_idw_map(idx: torch.Tensor, w: torch.Tensor, src_feat: torch.Tensor) -> torch.Tensor:
    """
    src_feat: (M, F) → returns (N, F) via the mapping (idx, w).
    Assumes w rows sum to 1 (or are one-hot in exact-match cases).
    """
    gathered = src_feat[idx]              # (N, k, F)
    return (gathered * w.unsqueeze(-1)).sum(dim=1)


@torch.no_grad()
def map_points_idw(
    dst_xy: torch.Tensor,   # (N,2)
    src_xy: torch.Tensor,   # (M,2)
    src_feat: torch.Tensor, # (M,F)
    k: int = 8,
    chunk: int = 8192,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convenience wrapper: build + apply the IDW map in one call.
    """
    idx, w = build_idw_map(dst_xy, src_xy, k=k, chunk=chunk, eps=eps)
    return apply_idw_map(idx, w, src_feat)


@torch.no_grad()
def targeted_map_to_pred(
    # destination (predicted t+1 mesh)
    pred_centers: torch.Tensor,        # [N_pred, 2]
    pred_levels : torch.Tensor,        # [N_pred]
    pred_parents: torch.Tensor,        # [N_pred] parent index in [0..H*W-1]
    mask_pred   : Optional[torch.Tensor],  # [H,W] or [H*W] bool, optional

    # source (t or GT(t+1))
    src_centers : torch.Tensor,        # [N_src, 2]
    src_feats   : torch.Tensor,        # [N_src, F]
    *,
    # optional: parent-level "unchanged coarse" shortcut
    mask_src_parent: Optional[torch.Tensor] = None,  # [H,W] or [H*W] bool; True = refined
    src_parent_feats: Optional[torch.Tensor] = None, # [H*W, F] (coarse aggregate)
    knn_k: int = 8,
    chunk: int = 8192,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Hybrid mapping:
      - COPY parent-aggregated values for coarse cells where both src/pred are unrefined
      - IDW for everything else (using the corrected IDW implementation above)
    """
    device = pred_centers.device
    N_pred, F = pred_centers.size(0), src_feats.size(1)
    out = pred_centers.new_zeros((N_pred, F))
    stats = {"copied_same_coarse": 0, "idw_points": 0}

    pred_is_coarse = (pred_levels == 0)
    parent_ix = pred_parents.long()

    # 1) Coarse identity copies (optional)
    if mask_src_parent is not None and src_parent_feats is not None and mask_pred is not None:
        m_src = mask_src_parent.view(-1).bool()      # True = refined
        m_dst = mask_pred.view(-1).bool()
        same_coarse = pred_is_coarse & (~m_src[parent_ix]) & (~m_dst[parent_ix])
        if same_coarse.any():
            ii = same_coarse.nonzero(as_tuple=True)[0]
            out[ii] = src_parent_feats[parent_ix[ii]]
            stats["copied_same_coarse"] = int(ii.numel())
        need_idw_mask = ~same_coarse
    else:
        need_idw_mask = torch.ones(N_pred, dtype=torch.bool, device=device)

    # 2) Correct IDW for the remainder
    if need_idw_mask.any():
        q = need_idw_mask.nonzero(as_tuple=True)[0]
        vals = map_points_idw(pred_centers[q], src_centers, src_feats, k=knn_k, chunk=chunk, eps=eps)
        out[q] = vals
        stats["idw_points"] = int(q.numel())

    return out, stats
