# dataset.py
# -----------------------------------------------------------------------------
# AMR Temporal Dataset (multi-level, arbitrary depth)
#
# This file replaces the prior 2-level-specific dataset with a level-agnostic
# version that supports any L_max >= 1. It is config-driven and robust to
# minor variations in the serialized .pt structure produced by your AMR
# preprocessor. It yields (t, t+1) pairs with per-level masks and parent maps.
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import math
import os
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils_geom import unique_undirected


# ------------------------------- Utilities ---------------------------------- #

def _torch_device(device: Union[str, torch.device] = "cpu") -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _load_pt_or_zip(path: str) -> Any:
    """
    Load a torch object from:
      - a .pt / .pth file directly, or
      - a .zip that contains exactly one .pt / .pth file (or the first one).
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"PT path not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".pth"):
        return torch.load(path, map_location="cpu")

    if ext == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            # Pick the first .pt/.pth in the archive if multiple exist.
            pt_names = [n for n in zf.namelist() if n.lower().endswith((".pt", ".pth"))]
            if not pt_names:
                raise RuntimeError(f"No .pt/.pth file found inside zip: {path}")
            with zf.open(pt_names[0], "r") as f:
                buf = f.read()
            return torch.load(io.BytesIO(buf), map_location="cpu")

    raise RuntimeError(f"Unsupported file extension: {ext} (expected .pt/.pth/.zip)")


def _as_int_tuple(x: Any, name: str) -> Tuple[int, int]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    raise ValueError(f"Expected {name} to be a 2-tuple, got {type(x)}: {x}")


def _maybe_from(cfg: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """
    Safe retrieval from nested dict with several candidate paths.
    The first existing path wins; otherwise returns default.
    Example keys: ["data.HW", "dataset.HW"]
    """
    for dotted in keys:
        node: Any = cfg
        ok = True
        for k in dotted.split("."):
            if not isinstance(node, dict) or k not in node:
                ok = False
                break
            node = node[k]
        if ok:
            return node
    return default


def _infer_HW_from_level0_count(n0: int, prefer_square: bool = True) -> Tuple[int, int]:
    """
    Infer (H, W) from the number of level-0 cells. If prefer_square=True, try a
    perfect square; otherwise factor into near-square dimensions.
    """
    if n0 <= 0:
        raise ValueError("Cannot infer H/W: count of level-0 cells is zero.")

    if prefer_square:
        root = int(round(math.sqrt(n0)))
        if root * root == n0:
            return root, root

    # Fallback: factor into near-square (greedy search)
    best = (1, n0)
    best_ratio = float("inf")
    for h in range(1, int(math.sqrt(n0)) + 1):
        if n0 % h == 0:
            w = n0 // h
            ratio = abs(w - h)
            if ratio < best_ratio:
                best_ratio = ratio
                best = (h, w)
    return best


def _linear_id(i: torch.Tensor, j: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert (i, j) index on an HxW raster into a flat linear index (row-major).
    i: [N], j: [N], 0 <= i < H, 0 <= j < W
    Returns: [N] long tensor
    """
    return (i.long() * W) + j.long()


def _ensure_long(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.long else x.long()


def _ensure_bool(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.bool else x.to(torch.bool)


# ----------------------------- Data Adapters -------------------------------- #

@dataclass
class Snapshot:
    """
    A normalized view of one AMR time step.
    Required fields (tensors on CPU):
      - features: [N, F]
      - level:    [N] (0..L)
      - ij:       [N, 2] integer grid coordinates on each level's raster
    Optional (if available in the .pt):
      - xy / pos: [N, 2] (continuous coordinates)
      - H, W:     ints (coarse grid shape)
      - dx, dy:   floats (coarse spacings)
      - meta:     dict of arbitrary extras
    """
    features: torch.Tensor
    level: torch.Tensor
    ij: torch.Tensor
    xy: Optional[torch.Tensor] = None
    H: Optional[int] = None
    W: Optional[int] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


def _normalize_snapshot(obj: Any) -> Snapshot:
    """
    Accept common formats produced by AMR preprocessors and normalize.
    Supports:
      - dict-like snapshots (existing behavior)
      - torch_geometric.data.Data snapshots (new)
    """
    # Already normalized?
    if isinstance(obj, Snapshot):
        return obj

    # --- NEW: torch_geometric.data.Data support ---
    # We detect by module name instead of importing torch_geometric unconditionally.
    mod = type(obj).__module__
    if mod.startswith("torch_geometric"):
        data = obj  # PyG Data
        # features
        feats = getattr(data, "x", None)
        if feats is None:
            raise KeyError("PyG Data is missing 'x' tensor for features.")
        feats = torch.as_tensor(feats)

        # level vector
        level = getattr(data, "level", None)
        if level is None:
            level = getattr(data, "levels", None)
        if level is None:
            level = getattr(data, "l", None)
        if level is None:
            raise KeyError("PyG Data is missing 'level' tensor (e.g., 'level', 'levels', or 'l').")
        level = _ensure_long(torch.as_tensor(level).view(-1))

        # ij indices (N,2)
        ij = getattr(data, "ij", None)
        if ij is None:
            # some pipelines store i/j separately
            i = getattr(data, "i", None)
            j = getattr(data, "j", None)
            if i is not None and j is not None:
                ij = torch.stack((torch.as_tensor(i).view(-1),
                                  torch.as_tensor(j).view(-1)), dim=1)
            else:
                # or 'indices'
                ij = getattr(data, "indices", None)
        if ij is None:
            raise KeyError("PyG Data is missing 'ij' (Nx2), or ('i','j'), or 'indices'.")

        ij = torch.as_tensor(ij)
        if ij.ndim != 2 or ij.size(-1) != 2:
            raise ValueError(f"'ij' must be [N,2], got {ij.shape}")

        # Optional xy / pos
        xy = getattr(data, "xy", None)
        if xy is None:
            xy = getattr(data, "pos", None)
        if xy is not None:
            xy = torch.as_tensor(xy)

        # Optional H/W/dx/dy attributes (handle both attr and dict-like)
        def _getattr_or_item(d, k, default=None):
            return getattr(d, k, d.__dict__.get(k, default))

        H = _getattr_or_item(data, "H", _getattr_or_item(data, "coarse_H", None))
        W = _getattr_or_item(data, "W", _getattr_or_item(data, "coarse_W", None))
        dx = _getattr_or_item(data, "dx", None)
        dy = _getattr_or_item(data, "dy", None)

        # Meta: stash all extra fields except the standard ones
        standard = {"x", "features", "level", "levels", "l", "ij", "indices",
                    "i", "j", "xy", "pos", "H", "W", "coarse_H", "coarse_W", "dx", "dy"}
        meta = {}
        for k, v in data.__dict__.items():
            if k.startswith("_"):  # skip PyG internal
                continue
            if k not in standard:
                meta[k] = v

        return Snapshot(
            features=feats,
            level=level,
            ij=ij,
            xy=torch.as_tensor(xy) if xy is not None else None,
            H=int(H) if H is not None else None,
            W=int(W) if W is not None else None,
            dx=float(dx) if dx is not None else None,
            dy=float(dy) if dy is not None else None,
            meta=meta if meta else None,
        )

    # --- Original dict-like path (kept as-is, just below) ---
    if not isinstance(obj, dict):
        raise TypeError(f"Snapshot must be a dict-like object, got {type(obj)}")

    feats = obj.get("features", obj.get("x"))
    level = obj.get("level", obj.get("levels"))
    ij = obj.get("ij", obj.get("indices"))

    if feats is None or level is None or ij is None:
        raise KeyError(
            "Snapshot is missing required keys; need 'features' (or 'x'), "
            "'level' (or 'levels'), and 'ij' (or 'indices'). Found keys: "
            f"{list(obj.keys())}"
        )

    feats = torch.as_tensor(feats)
    level = _ensure_long(torch.as_tensor(level).view(-1))
    ij = torch.as_tensor(ij)
    if ij.ndim != 2 or ij.size(-1) != 2:
        raise ValueError(f"'ij' must be [N,2], got {ij.shape}")

    xy = obj.get("xy", obj.get("pos", obj.get("centers")))
    if xy is not None:
        xy = torch.as_tensor(xy)

    H = obj.get("H", obj.get("coarse_H"))
    W = obj.get("W", obj.get("coarse_W"))
    dx = obj.get("dx")
    dy = obj.get("dy")

    meta = {k: v for k, v in obj.items()
            if k not in ("features", "x", "level", "levels", "ij", "indices", "xy",
                         "pos", "centers", "H", "W", "coarse_H", "coarse_W", "dx", "dy")}

    return Snapshot(
        features=feats,
        level=level,
        ij=ij,
        xy=xy if xy is not None else None,
        H=int(H) if H is not None else None,
        W=int(W) if W is not None else None,
        dx=float(dx) if dx is not None else None,
        dy=float(dy) if dy is not None else None,
        meta=meta if meta else None,
    )


def _extract_sequence(obj: Any) -> List[Snapshot]:
    """
    The .pt may store:
      - a list[dict or PyG Data] of snapshots, or
      - {'snapshots': [...]}, etc., or
      - a single PyG Data (wrap as a one-element list).
    """
    # Single PyG Data
    if type(obj).__module__.startswith("torch_geometric"):
        return [_normalize_snapshot(obj)]

    if isinstance(obj, list):
        return [_normalize_snapshot(s) for s in obj]

    if isinstance(obj, dict):
        for k in ("snapshots", "steps", "time_steps", "sequence", "data_list"):
            if k in obj and isinstance(obj[k], list):
                return [_normalize_snapshot(s) for s in obj[k]]
        # Looks like a single snapshot dict
        if {"features", "x", "level", "levels", "ij", "indices"}.intersection(obj.keys()):
            return [_normalize_snapshot(obj)]

    raise ValueError(
        "Could not find a list of snapshots inside the loaded .pt. "
        "Expected a list, a dict with 'snapshots'/similar, or a single PyG Data."
    )

def _ensure_hw_from_cfg_or_data(ds0: Data, cfg: Dict[str, Any], H: int | None, W: int | None) -> Tuple[int,int]:
    if H is not None and W is not None:
        return int(H), int(W)
    # prefer attributes saved in PT
    if hasattr(ds0, "H") and hasattr(ds0, "W"):
        return int(ds0.H), int(ds0.W)
    # final fallback to cfg
    dom = cfg.get("domain", {})
    Hc = int(dom.get("H", 64))
    Wc = int(dom.get("W", 64))
    return Hc, Wc

def _select_feature_columns(x: Tensor, cfg: Dict[str, Any]) -> Tensor:
    use_cols = list(cfg.get("features", {}).get("use_columns", [0, 1, 3]))  # default: density, x-mom, energy
    use_cols = [int(c) for c in use_cols]
    if x.size(1) >= max(use_cols) + 1:
        return x[:, use_cols].to(torch.float32)
    return x.to(torch.float32)

@torch.no_grad()
def level_ij_to_centers_bbox(levels: torch.Tensor,
                             ij: torch.Tensor,
                             H: int, W: int,
                             bbox: tuple[float, float, float, float]
                             ) -> torch.Tensor:
    """
    Convert per-cell (level, i, j) to (x,y) centers in bbox units.
    i,j are integer cell indices at that level's resolution (H*2^l, W*2^l).

    Returns (N,2) tensor of [x,y] in [xmin,xmax]×[ymin,ymax].
    """
    xmin, xmax, ymin, ymax = bbox
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    levels = levels.view(-1).long()
    i = ij[:, 0].to(torch.float32)
    j = ij[:, 1].to(torch.float32)
    scale = torch.pow(torch.tensor(2.0, device=levels.device), levels.to(torch.float32))  # 2^l

    # Per-level cell size
    dx = dx0 / scale
    dy = dy0 / scale

    x = xmin + (j + 0.5) * dx
    y = ymin + (i + 0.5) * dy
    return torch.stack([x, y], dim=1)

@torch.no_grad()
def _parents_from_level_ij(levels: torch.Tensor,
                           ij: torch.Tensor,
                           H: int, W: int) -> torch.Tensor:
    """
    Map each level-l cell with indices (i,j) at its own resolution (H*2^l, W*2^l)
    to its coarse parent index in a flattened (H*W) coarse grid.

    Works for arbitrary l >= 0. Assumes ij are integer cell indices at the cell's level.
    """
    if levels.dim() != 1:
        levels = levels.view(-1)
    i = ij[:, 0].long()
    j = ij[:, 1].long()

    # scale = 2^level, elementwise
    scale = torch.pow(torch.tensor(2, device=levels.device, dtype=torch.long), levels)
    # integer floor division to get coarse parent row/col
    row = torch.div(i, scale, rounding_mode='floor').clamp_(0, H - 1)
    col = torch.div(j, scale, rounding_mode='floor').clamp_(0, W - 1)

    parents = row * W + col
    return parents.long()


@torch.no_grad()
def _coarse_mask_from_level_parents(levels: torch.Tensor,
                                    parents: torch.Tensor,
                                    H: int, W: int) -> torch.Tensor:
    """
    Produce a coarse-grid (H*W,) boolean mask where True means the coarse cell
    has at least one child at level >= 1. Works for L>2 as well.
    """
    levels = levels.view(-1).long()
    parents = parents.view(-1).long()
    mask = torch.zeros(H * W, dtype=torch.bool, device=levels.device)
    if parents.numel() == 0:
        return mask
    refined = levels >= 1
    if refined.any():
        mask.index_fill_(0, parents[refined], True)
    return mask  # caller may .view(H, W)

# ----------------------------- Level Helpers -------------------------------- #

def _masks_by_level_from_ij(
    level: torch.Tensor,
    ij: torch.Tensor,
    H: int,
    W: int,
    L_max: int,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Build boolean per-level masks on canonical rasters:
      - Level l mask lives on shape (H*2**l, W*2**l).
    Only cells that are present in the snapshot at that level are True.

    Returns: dict {l: mask_l bool[H*2**l, W*2**l]} for l in 0..L_max,
             but note level==0 will be a full HxW True mask by definition.
    """
    masks: Dict[int, torch.Tensor] = {}
    level = _ensure_long(level)
    ij = torch.as_tensor(ij).to(torch.long)

    for l in range(0, L_max + 1):
        Hl = H * (2 ** l)
        Wl = W * (2 ** l)
        mask = torch.zeros((Hl * Wl,), dtype=torch.bool, device=device)

        sel = (level == l).nonzero(as_tuple=False).view(-1)
        if sel.numel() > 0:
            ij_l = ij.index_select(0, sel)
            i, j = ij_l[:, 0], ij_l[:, 1]
            lin = _linear_id(i, j, Hl, Wl)
            mask[lin] = True

        masks[l] = mask.view(Hl, Wl)

    return masks


def _safe_edge_index(d: Data) -> Tensor:
    if hasattr(d, "edge_index") and d.edge_index.numel() > 0:
        return unique_undirected(d.edge_index.clone(), d.pos.size(0))
    # fallback: empty graph
    return torch.empty(2, 0, dtype=torch.long)


def _parents_root_and_immediate(
    level: torch.Tensor,
    ij: torch.Tensor,
    H: int,
    W: int,
    L_max: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute two parent maps for each cell:
      - parent_root:   index (on level-0 raster flattened) of the *coarse* parent
      - parent_immed:  index (on level-(l-1) raster flattened) of the immediate parent
    For level==0, parent_immed == parent_root.

    Returns:
      parent_root:  [N] long (0..H*W-1)
      parent_immed: [N] long (for l>0, inside (H*2**(l-1))*(W*2**(l-1)))
    """
    level = _ensure_long(level).to(device)
    ij = torch.as_tensor(ij, device=device).to(torch.long)
    N = ij.size(0)

    parent_root = torch.empty((N,), dtype=torch.long, device=device)
    parent_immed = torch.empty((N,), dtype=torch.long, device=device)

    for l in range(0, L_max + 1):
        sel = (level == l).nonzero(as_tuple=False).view(-1)
        if sel.numel() == 0:
            continue

        i = ij[sel, 0]
        j = ij[sel, 1]

        # Root (coarse-level) parent: divide coordinates by 2**l
        if l == 0:
            lin_root = _linear_id(i, j, H, W)
        else:
            i0 = i // (2 ** l)
            j0 = j // (2 ** l)
            lin_root = _linear_id(i0, j0, H, W)

        parent_root[sel] = lin_root

        # Immediate parent: if l == 0, itself on coarse; else divide by 2 once
        if l == 0:
            lin_immed = _linear_id(i, j, H, W)
        else:
            Hi = H * (2 ** (l - 1))
            Wi = W * (2 ** (l - 1))
            i1 = i // 2
            j1 = j // 2
            lin_immed = _linear_id(i1, j1, Hi, Wi)

        parent_immed[sel] = lin_immed

    return parent_root, parent_immed


# ------------------------------ Main Dataset -------------------------------- #

class AMRTemporalDataset(Dataset):
    """
    AMR sequence dataset yielding (t, t+1) pairs with multi-level structure.

    Usage:
      ds = AMRTemporalDataset(
              pt_path=cfg["data"]["pt_path"],
              cfg=cfg,
              device="cpu" or "cuda",
           )
      sample = ds[idx]  # dict with keys shown below

    Returned dict (tensors on the chosen device):
      - x_t:           [N_t, F]       features at time t (subset by feature_idx if set)
      - level_t:       [N_t]          0..L levels for nodes at t
      - ij_t:          [N_t, 2]       integer indices on their level rasters
      - parent_root_t: [N_t]          linear parent on level-0 raster
      - parent_im_t:   [N_t]          linear parent on level-(l-1) raster
      - masks_t:       dict{l: bool[H*2**l, W*2**l]} (0..L_max)
      - x_tp1, level_tp1, ij_tp1, parent_root_tp1, parent_im_tp1, masks_tp1: same at t+1
      - H, W:          ints (coarse shape)
      - dx, dy:        floats (coarse spacings if available, else 1.0)
      - L_max:         int (max level encountered or from cfg)
      - meta_t, meta_tp1: optional per-snapshot metadata dicts

    Config keys recognized (all optional):
      - data.pt_path                : string (.pt/.pth or .zip)
      - data.L_max                  : int (if absent -> infer)
      - data.H, data.W              : ints (if absent -> infer from level==0 count)
      - features.use_columns        : list[int] (feature subset)
      - data.feature_idx            : list[int] (fallback path)
    """

    def __init__(
        self,
        pt_path: str,
        cfg: Optional[Dict[str, Any]] = None,
        device: Union[str, torch.device] = "cpu",
        H: Optional[int] = None,
        W: Optional[int] = None,
        L_max: Optional[int] = None,
        feature_idx: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.device = _torch_device(device)
        self.cfg = cfg or {}

        # Allow args to override config; otherwise read from config.
        if pt_path is None:
            pt_path = _maybe_from(self.cfg, ["data.pt_path", "dataset.pt_path"])
            if pt_path is None:
                raise ValueError("pt_path must be provided (arg or cfg['data']['pt_path']).")
        self.pt_path = os.path.expanduser(pt_path)

        self.seq_raw = _extract_sequence(_load_pt_or_zip(self.pt_path))
        if len(self.seq_raw) < 2:
            raise ValueError(f"Need at least 2 snapshots for (t, t+1) pairs; found {len(self.seq_raw)}.")

        # Normalize and collect basic info
        self.seq: List[Snapshot] = [_normalize_snapshot(s) for s in self.seq_raw]

        # Feature subset selection
        if feature_idx is None:
            feature_idx = _maybe_from(self.cfg, ["features.use_columns", "data.feature_idx"], default=None)
        self.feature_idx = list(feature_idx) if feature_idx is not None else None

        # Coarse shape (H, W)
        if H is None or W is None:
            H_cfg = _maybe_from(self.cfg, ["data.H"])
            W_cfg = _maybe_from(self.cfg, ["data.W"])
            if H_cfg is not None and W_cfg is not None:
                H, W = int(H_cfg), int(W_cfg)

        if H is None or W is None:
            # Try to grab from the first snapshot if present
            H0 = self.seq[0].H
            W0 = self.seq[0].W
            if H0 is not None and W0 is not None:
                H, W = int(H0), int(W0)
            else:
                # Infer from the count of level==0 cells
                lvl0 = (self.seq[0].level == 0).sum().item()
                H, W = _infer_HW_from_level0_count(lvl0, prefer_square=True)

        self.H, self.W = int(H), int(W)

        # dx, dy (coarse spacings)
        dx = self.seq[0].dx if self.seq[0].dx is not None else 1.0
        dy = self.seq[0].dy if self.seq[0].dy is not None else 1.0
        self.dx, self.dy = float(dx), float(dy)

        # L_max (max desired level). If not provided, infer from data.
        if L_max is None:
            L_cfg = _maybe_from(self.cfg, ["data.L_max"])
            L_max = int(L_cfg) if L_cfg is not None else None

        if L_max is None:
            L_max = 0
            for s in self.seq:
                L_max = max(L_max, int(torch.as_tensor(s.level).max().item()))
        self.L_max = int(L_max)

        # Pre-validate snapshots: shapes and key presence
        for idx, s in enumerate(self.seq):
            N = s.features.shape[0]
            if s.level.numel() != N or s.ij.shape[0] != N:
                raise ValueError(
                    f"Snapshot {idx}: inconsistent lengths: "
                    f"features={s.features.shape}, level={s.level.shape}, ij={s.ij.shape}"
                )

        # Indices yielded: 0..len-2 produce (t, t+1)
        self._num_pairs = len(self.seq) - 1

    # ------------------------------- Dunder ---------------------------------- #

    def __len__(self) -> int:
        return self._num_pairs

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Yield a dict with t and t+1 snapshot tensors placed on self.device and
        sliced to the chosen feature columns (if any).
        """
        if idx < 0 or idx >= self._num_pairs:
            raise IndexError(f"Index {idx} out of range [0, {self._num_pairs}).")

        s_t = self.seq[idx]
        s_tp1 = self.seq[idx + 1]

        # Move & slice features
        x_t = torch.as_tensor(s_t.features, device=self.device, dtype=torch.float32)
        x_tp1 = torch.as_tensor(s_tp1.features, device=self.device, dtype=torch.float32)

        if self.feature_idx is not None:
            x_t = x_t[:, self.feature_idx]
            x_tp1 = x_tp1[:, self.feature_idx]

        level_t = _ensure_long(torch.as_tensor(s_t.level, device=self.device))
        ij_t = _ensure_long(torch.as_tensor(s_t.ij, device=self.device))
        level_tp1 = _ensure_long(torch.as_tensor(s_tp1.level, device=self.device))
        ij_tp1 = _ensure_long(torch.as_tensor(s_tp1.ij, device=self.device))

        # Per-level masks
        masks_t = _masks_by_level_from_ij(level_t, ij_t, self.H, self.W, self.L_max, self.device)
        masks_tp1 = _masks_by_level_from_ij(level_tp1, ij_tp1, self.H, self.W, self.L_max, self.device)

        # Parent mappings (root & immediate)
        parent_root_t, parent_im_t = _parents_root_and_immediate(
            level_t, ij_t, self.H, self.W, self.L_max, self.device
        )
        parent_root_tp1, parent_im_tp1 = _parents_root_and_immediate(
            level_tp1, ij_tp1, self.H, self.W, self.L_max, self.device
        )

        # Optional XY
        xy_t = None if s_t.xy is None else torch.as_tensor(s_t.xy, device=self.device, dtype=torch.float32)
        xy_tp1 = None if s_tp1.xy is None else torch.as_tensor(s_tp1.xy, device=self.device, dtype=torch.float32)

        sample: Dict[str, Any] = {
            # time t
            "x_t": x_t,
            "level_t": level_t,
            "ij_t": ij_t,
            "xy_t": xy_t,
            "parent_root_t": parent_root_t,
            "parent_im_t": parent_im_t,
            "masks_t": masks_t,  # {l: bool[H*2**l, W*2**l]}

            # time t+1
            "x_tp1": x_tp1,
            "level_tp1": level_tp1,
            "ij_tp1": ij_tp1,
            "xy_tp1": xy_tp1,
            "parent_root_tp1": parent_root_tp1,
            "parent_im_tp1": parent_im_tp1,
            "masks_tp1": masks_tp1,

            # shared
            "H": self.H,
            "W": self.W,
            "dx": torch.tensor(self.dx, device=self.device, dtype=torch.float32),
            "dy": torch.tensor(self.dy, device=self.device, dtype=torch.float32),
            "L_max": torch.tensor(self.L_max, device=self.device, dtype=torch.long),

            # raw meta (kept on CPU to avoid blowing up GPU memory)
            "meta_t": s_t.meta,
            "meta_tp1": s_tp1.meta,
        }

        # --- enforce tensor types / defaults and drop None ---

        # (A) If you really need edge indices, ensure empty but valid tensors.
        # If your model does NOT use edges, you can comment this whole (A) section out.
        if "edge_index_t" not in sample or sample["edge_index_t"] is None:
            sample["edge_index_t"] = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        else:
            ei = sample["edge_index_t"]
            ei = torch.as_tensor(ei, device=self.device, dtype=torch.long)
            if ei.ndim == 1:
                ei = ei.view(2, -1)
            sample["edge_index_t"] = ei

        if "edge_index_tp1" not in sample or sample["edge_index_tp1"] is None:
            sample["edge_index_tp1"] = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        else:
            ei = sample["edge_index_tp1"]
            ei = torch.as_tensor(ei, device=self.device, dtype=torch.long)
            if ei.ndim == 1:
                ei = ei.view(2, -1)
            sample["edge_index_tp1"] = ei

        # (B) Ensure xy_* (cell centers) exist: synthesize from (level_*, ij_*) if missing.
        if sample.get("xy_t", None) is None:
            level = sample["level_t"].to(torch.long).view(-1)
            ij    = sample["ij_t"].to(torch.long)
            i = ij[:, 0].to(torch.float32)
            j = ij[:, 1].to(torch.float32)
            Wl = (self.W * (2.0 ** level.to(torch.float32)))
            Hl = (self.H * (2.0 ** level.to(torch.float32)))
            x = (i + 0.5) / Wl
            y = (j + 0.5) / Hl
            sample["xy_t"] = torch.stack([x, y], dim=1).to(self.device)

        if sample.get("xy_tp1", None) is None:
            level = sample["level_tp1"].to(torch.long).view(-1)
            ij    = sample["ij_tp1"].to(torch.long)
            i = ij[:, 0].to(torch.float32)
            j = ij[:, 1].to(torch.float32)
            Wl = (self.W * (2.0 ** level.to(torch.float32)))
            Hl = (self.H * (2.0 ** level.to(torch.float32)))
            x = (i + 0.5) / Wl
            y = (j + 0.5) / Hl
            sample["xy_tp1"] = torch.stack([x, y], dim=1).to(self.device)

        # (C) Normalize dtypes for the time-stamped fields your model actually uses.
        to_float = ["x_t", "x_tp1"]
        to_long  = ["level_t", "ij_t", "level_tp1", "ij_tp1"]

        for k in to_float:
            v = sample.get(k, None)
            if v is not None:
                sample[k] = torch.as_tensor(v, device=self.device, dtype=torch.float32)

        for k in to_long:
            v = sample.get(k, None)
            if v is not None:
                sample[k] = torch.as_tensor(v, device=self.device, dtype=torch.long)

        # Scalars already tensors; ensure device/dtype
        sample["dx"]    = torch.as_tensor(self.dx, device=self.device, dtype=torch.float32)
        sample["dy"]    = torch.as_tensor(self.dy, device=self.device, dtype=torch.float32)
        sample["L_max"] = torch.as_tensor(self.L_max, device=self.device, dtype=torch.long)

        # (D) Remove non-tensor metadata so default_collate won't try to stack it.
        # If you need meta, fetch it from the base dataset via indices instead of batching it.
        sample.pop("meta_t",   None)
        sample.pop("meta_tp1", None)

        # (E) Final cleanup: drop any remaining None entries
        sample = {k: v for k, v in sample.items() if v is not None}

        return sample


    # ----------------------------- Convenience ------------------------------- #

    @property
    def coarse_shape(self) -> Tuple[int, int]:
        return self.H, self.W

    @property
    def spacing(self) -> Tuple[float, float]:
        return self.dx, self.dy

    def describe(self) -> str:
        return (
            f"AMRTemporalDataset(num_pairs={len(self)}, HxW={self.H}x{self.W}, "
            f"L_max={self.L_max}, features={self.feature_idx if self.feature_idx is not None else 'all'})"
        )
    

class CellRefineTemporalDataset(Dataset):
    """
    Yields per-sample dictionaries for consecutive timesteps (t -> t+1),
    with features and geometry defined at **cell centers**.

    Required Data fields in each element of data_list:
      - pos: (N,2) center coordinates in [xmin, xmax]×[ymin, ymax]
      - x:   (N,F) per-center features
      - level: (N,) 0 for coarse, 1 for fine
      - ij:  (N,2) integer indices; for L1 these are on a 2x finer lattice
      - edge_index: (2,E) center-graph edges (optional; empty ok)
      - (optional) H, W attributes; else take from cfg.domain.{H,W}
    """
    def __init__(self, data_list: List[Data], cfg: Dict[str, Any],
                 H: int | None = None, W: int | None = None, device: str = "cpu"):
        super().__init__()
        assert isinstance(data_list, list) and len(data_list) >= 2, "Need a list of ≥2 time steps"
        self.ds   = data_list
        self.cfg  = cfg
        self.dev  = device  # kept for parity; tensors are left on CPU

        # domain / grid setup
        self.H, self.W = _ensure_hw_from_cfg_or_data(data_list[0], cfg, H, W)

        # Prebuild a lightweight cache of (t, t+1) pairs
        self.cache: List[Dict[str, Tensor]] = []
        for t in range(len(self.ds) - 1):
            dt  = self.ds[t]
            dt1 = self.ds[t + 1]

            # --- centers and features (select requested columns) ---
            centers_t      = dt.pos[:, :2].contiguous()
            centers_tp1    = dt1.pos[:, :2].contiguous()
            center_feat_t  = _select_feature_columns(dt.x,   cfg)
            center_feat_tp1= _select_feature_columns(dt1.x,  cfg)

            # --- levels, ij, parents, masks (on coarse grid) ---
            level_t  = getattr(dt,  "level",  torch.zeros(centers_t.size(0), dtype=torch.long))
            level_tp1= getattr(dt1, "level",  torch.zeros(centers_tp1.size(0), dtype=torch.long))
            ij_t     = getattr(dt,  "ij",     None)
            ij_tp1   = getattr(dt1, "ij",     None)
            if ij_t is None or ij_tp1 is None:
                raise RuntimeError("Expected 'ij' indices per center in amr_cells.pt for robust parent/mask logic.")

            dyn_parents = _parents_from_level_ij(level_t, ij_t, self.H, self.W)      # (N_t,)
            mask_t      = _coarse_mask_from_level_parents(level_t,  dyn_parents, self.H, self.W)  # (H*W,)
            parents_tp1 = _parents_from_level_ij(level_tp1, ij_tp1, self.H, self.W)  # only for mask at t+1
            mask_tp1    = _coarse_mask_from_level_parents(level_tp1, parents_tp1, self.H, self.W)

            #print("level_t unique:", torch.unique(level_t))

            # --- edges (center graph) ---
            ei_t   = _safe_edge_index(dt)
            ei_tp1 = _safe_edge_index(dt1)

            # after you compute ei_t / ei_tp1 in CellRefineTemporalDataset.__init__
            if ei_t.numel() == 0:
                print("[WARN] ei_t is empty for t sample")
            if ei_tp1.numel() == 0:
                print("[WARN] ei_tp1 is empty for t+1 sample")
            if ei_t.dtype != torch.long: ei_t = ei_t.long()
            if ei_tp1.dtype != torch.long: ei_tp1 = ei_tp1.long()

            # --- pack sample dict ---
            sample: Dict[str, Tensor] = {
                # time index
                "t": torch.tensor([t], dtype=torch.long),

                # center-based geometry + features at t and t+1
                "centers_t":       centers_t,          # (N_t, 2)
                "center_feat_t":   center_feat_t,      # (N_t, F)
                "level_t":         level_t.to(torch.long),
                "ij_t":            ij_t.to(torch.long),

                "centers_tp1":     centers_tp1,        # (N_tp1, 2)
                "center_feat_tp1": center_feat_tp1,    # (N_tp1, F)
                "level_tp1":       level_tp1.to(torch.long),
                "ij_tp1":          ij_tp1.to(torch.long),

                # edges of the dynamic cell graphs (t and t+1)
                "ei_t":            ei_t,               # (2, E_t)
                "ei_tp1":          ei_tp1,             # (2, E_tp1)

                # parent mapping (t) and coarse masks (t, t+1)
                "dyn_parents":     dyn_parents,        # (N_t,)
                "mask_t":          mask_t,             # (H*W,) bool
                "mask_tp1":        mask_tp1,           # (H*W,) bool
            }

            # --- compatibility aliases (keep older code/plots working without changes) ---
            sample["pos_t"]        = sample["centers_t"]
            sample["pos_tp1"]      = sample["centers_tp1"]
            sample["dyn_feat_t"]   = sample["center_feat_t"]
            sample["dyn_feat_tp1"] = sample["center_feat_tp1"]

            self.cache.append(sample)

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # Return CPU tensors; training loop moves them to device.
        return self.cache[idx]


def preprocess_timesteps_once(
    raw_series: List[Data] | str,
    cfg: Dict[str, Any],
    out_path: str,
    H: int | None = None,
    W: int | None = None,
) -> str:
    """
    Compute per-step fields ONCE (feature slice, parents, coarse mask, edge types)
    and save a compact processed series.

    Saves:
      {"H": H, "W": W, "timesteps": List[Data]}
    where each Data has: pos, x (selected columns), level, ij, edge_index, parents, mask, H, W
    """
    if isinstance(raw_series, str):
        ds_in: List[Data] = torch.load(raw_series, map_location="cpu")
    else:
        ds_in = raw_series
    assert isinstance(ds_in, list) and len(ds_in) >= 2, "Need ≥2 timesteps"

    H, W = _ensure_hw_from_cfg_or_data(ds_in[0], cfg, H, W)
    processed: List[Data] = []

    for k, dt in enumerate(ds_in):
        centers = dt.pos[:, :2].contiguous()
        x_sel   = _select_feature_columns(dt.x, cfg)

        level = getattr(dt, "level", torch.zeros(centers.size(0), dtype=torch.long))
        ij    = getattr(dt, "ij",    None)
        if ij is None:
            raise RuntimeError("Expected 'ij' per center for robust parent/mask logic.")

        parents = _parents_from_level_ij(level, ij, H, W)
        mask    = _coarse_mask_from_level_parents(level, parents, H, W)
        ei      = _safe_edge_index(dt)
        if ei.dtype != torch.long: ei = ei.long()
        if ei.numel() == 0:
            print(f"[WARN] empty edge_index at t={k}")

        d = Data(
            pos=centers,
            x=x_sel,
            level=level.long(),
            ij=ij.long(),
            edge_index=ei,
        )
        d.parents = parents
        d.mask    = mask
        d.H = H
        d.W = W
        processed.append(d)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"H": H, "W": W, "timesteps": processed}, out_path)
    return out_path


class CellRefineWindowDataset(Dataset):
    """
    Yields contiguous windows of K timesteps (default K=2). Each timestep is
    preprocessed exactly once by preprocess_timesteps_once() or on-the-fly here.

    If K==2, it also adds pair-compatible keys so your current training code
    can consume it unchanged.

    __getitem__(i) → a dict:
      Core (always):
        - "t0": int start index
        - "t_indices": LongTensor[K]
        - "H","W"
        - "{centers,feat,level,ij,ei,parents,mask}_list": lists of length K

      Compatibility extras (only if K==2):
        - centers_t, center_feat_t, level_t, ij_t, ei_t, dyn_parents, mask_t
        - centers_tp1, center_feat_tp1, level_tp1, ij_tp1, ei_tp1, mask_tp1
        - pos_t, pos_tp1, dyn_feat_t, dyn_feat_tp1
        - t (tensor[[t0]])
    """
    def __init__(
        self,
        series: str | List[Data],
        cfg: Dict[str, Any],
        window_size: int = 2,
        stride: int = 1,
        H: int | None = None,
        W: int | None = None,
        device: str = "cpu",
        *,
        is_processed_file: bool | None = None,
    ):
        super().__init__()
        assert window_size >= 2, "window_size must be ≥2"
        assert stride >= 1, "stride must be ≥1"

        self.cfg = cfg
        self.dev = device
        self.K   = int(window_size)
        self.S   = int(stride)
        self.H   = int(H)
        self.W   = int(W)

        # Load processed payload or raw list
        if isinstance(series, str):
            payload = torch.load(series, map_location="cpu")
            if isinstance(payload, dict) and "timesteps" in payload:
                self.steps: List[Data] = payload["timesteps"]
                self.H = int(payload.get("H", 0)) or H
                self.W = int(payload.get("W", 0)) or W
                print("H, W:", self.H, self.W)
                is_proc = True
            else:
                self.steps = payload
                is_proc = False
        else:
            self.steps = series
            is_proc = False if is_processed_file is None else not is_processed_file

        self.H, self.W = _ensure_hw_from_cfg_or_data(self.steps[0], cfg, self.H, self.W)

        # inside CellRefineWindowDataset.__init__ ...
        # after we've set: self.steps = series_or_loaded_list; self.H, self.W established

        if not is_proc:
            proc = []
            for k, dt in enumerate(self.steps):
                # allow either PyG Data-like or dict-like inputs
                def _get(attr, default=None):
                    if isinstance(dt, dict):
                        return dt.get(attr, default)
                    # PyG-like object with attributes
                    return getattr(dt, attr, default)

                centers = _get("pos")
                x_full  = _get("x")
                level   = _get("level")
                ij      = _get("ij")
                ei      = _get("edge_index")

                if centers is None or x_full is None:
                    raise RuntimeError(f"t={k}: expected 'pos' and 'x'")

                centers = centers[:, :2].contiguous()
                x_sel   = _select_feature_columns(x_full, self.cfg)  # your existing helper

                if level is None:
                    level = torch.zeros(centers.size(0), dtype=torch.long)
                else:
                    level = level.long()

                if ij is None:
                    raise RuntimeError("expected 'ij' indices per center for robust parent/mask logic.")
                ij = ij.long()

                parents = _parents_from_level_ij(level, ij, self.H, self.W)     # existing helper
                mask    = _coarse_mask_from_level_parents(level, parents, self.H, self.W)

                if ei is None or (torch.is_tensor(ei) and ei.numel() == 0):
                    # your existing edge builder / sanitizer
                    ei = _safe_edge_index({"pos": centers, "level": level, "ij": ij})
                if ei.dtype != torch.long:
                    ei = ei.long()

                step = {
                    "pos": centers,           # (N,2)
                    "x": x_sel,               # (N,F_sel)
                    "level": level,           # (N,)
                    "ij": ij,                 # (N,2)
                    "edge_index": ei,         # (2,E)
                    "parents": parents,       # (N,)
                    "mask": mask,             # (H*W,)
                    "H": self.H, "W": self.W,
                }
                proc.append(step)

            self.steps = proc

        self.N = len(self.steps)
        assert self.N >= self.K, f"Need at least {self.K} timesteps, found {self.N}"

        # number of windows with chosen stride
        self.n_windows = 1 + (self.N - self.K) // self.S

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, i: int):
        t0 = i * self.S
        t1 = t0 + self.K
        ws = self.steps[t0:t1]

        # lists for the window
        centers_list  = [d["pos"]        for d in ws]
        feat_list     = [d["x"]          for d in ws]
        level_list    = [d["level"]      for d in ws]
        ij_list       = [d["ij"]         for d in ws]
        ei_list       = [d["edge_index"] for d in ws]
        parents_list  = [d["parents"]    for d in ws]
        mask_list     = [d["mask"]       for d in ws]

        out = {
            "t0": t0,
            "t_indices": torch.arange(t0, t1, dtype=torch.long),
            "H": self.H, "W": self.W,
            "centers_list": centers_list,
            "feat_list":    feat_list,
            "level_list":   level_list,
            "ij_list":      ij_list,
            "ei_list":      ei_list,
            "parents_list": parents_list,
            "mask_list":    mask_list,
        }

        # Back-compat pair keys for K==2 (so existing code keeps working)
        if self.K == 2:
            a, b = ws[0], ws[1]
            out.update({
                "t": torch.tensor([t0], dtype=torch.long),

                "centers_t":       a["pos"],
                "center_feat_t":   a["x"],
                "level_t":         a["level"],
                "ij_t":            a["ij"],
                "ei_t":            a["edge_index"],
                "dyn_parents":     a["parents"],
                "mask_t":          a["mask"],

                "centers_tp1":     b["pos"],
                "center_feat_tp1": b["x"],
                "level_tp1":       b["level"],
                "ij_tp1":          b["ij"],
                "ei_tp1":          b["edge_index"],
                "mask_tp1":        b["mask"],

                "pos_t":           a["pos"],
                "pos_tp1":         b["pos"],
                "dyn_feat_t":      a["x"],
                "dyn_feat_tp1":    b["x"],
            })
            
        return out



