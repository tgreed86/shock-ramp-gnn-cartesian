# ------------------------------
# PRECOMPUTE: mesh + GT→pred maps
# ------------------------------
# ---- rollout_precompute.py ----
from __future__ import annotations
from typing import Dict, List, Any
import os, json, hashlib, time, torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.collections import LineCollection
from utils.extract_dmr_boundaries import build_amr_mesh_for_wedge

from amr_policy import (
    predict_masks_hierarchical_from_gt_gradients, 
    _parents_from_level_ij,
)
from utils_geom import (
    dynamic_cells_from_parent_masks,
    _targeted_map_to_pred,
    parents_from_pos, build_idw_map,
    infer_refine_ratio_from_level_ij_pos,
)
from utils.precomp_h5 import precomp_h5_is_usable

_WEDGE_CLIP_LOOKUP_CACHE: Dict[str, Dict[str, Any]] = {}

def _cfg_hash_for_cache(cfg):
    # Small hash so different policy/interp settings produce separate caches
    key = {
        "policy": cfg.get("policy", {}),
        "interp": {
            "method": cfg.get("loss", {}).get("interp_method", "idw"),
            "k":      int(cfg.get("loss", {}).get("interp_k", 8)),
            "delta":  float(cfg.get("loss", {}).get("huber_delta", 0.05)),
        },
    }
    s = json.dumps(key, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _abs_or_none(path_like: Any) -> str | None:
    if path_like is None:
        return None
    s = str(path_like).strip()
    if s == "":
        return None
    return os.path.abspath(os.path.expanduser(s))


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (bytes, np.bytes_)):
        return obj.decode("utf-8", errors="replace")
    return obj


def _drop_comment_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ks = str(k)
            if ks.startswith("_comment"):
                continue
            out[ks] = _drop_comment_keys(v)
        return out
    if isinstance(obj, list):
        return [_drop_comment_keys(v) for v in obj]
    return obj


def _precomp_repro_json_path(h5_path: str) -> str:
    root, _ = os.path.splitext(os.path.abspath(os.path.expanduser(str(h5_path))))
    return f"{root}.repro.json"


def _write_precomp_repro_json(h5_path: str, payload: Dict[str, Any]) -> str:
    out_path = _precomp_repro_json_path(h5_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, sort_keys=True, indent=2)
        f.write("\n")
    os.replace(tmp_path, out_path)
    print(f"[PRECOMP] Saved reproduction config JSON: {out_path}")
    return out_path


def _build_precomp_repro_payload(
    *,
    mode: str,
    cache_path: str,
    cfg: dict,
    H: int,
    W: int,
    dx: float,
    dy: float,
    T: int,
    gt_refine_ratio: int,
    refine_ratio: int,
    cfg_sha1: str,
    geometry_mode: str | None = None,
    mesh_spec_path: str | None = None,
    mesh_spec_sha1: str | None = None,
    uniform_signature_sha1: str | None = None,
) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    mesh_cfg = cfg.get("mesh", {}) or {}
    pol_cfg_raw = cfg.get("policy", {}) or {}
    edges_cfg_raw = cfg.get("edges", {}) or {}
    idw_cfg_raw = cfg.get("idw", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    speed_cfg = cfg.get("speed", {}) or {}

    pol_cfg = _drop_comment_keys(pol_cfg_raw)
    edges_cfg = _drop_comment_keys(edges_cfg_raw)

    bbox = data_cfg.get("bbox", [0.0, 1.0, 0.0, 1.0])
    bbox = [float(v) for v in bbox]

    controls: Dict[str, Any] = {
        "refine_ratio": int(refine_ratio),
        "policy": pol_cfg,
        "edges": edges_cfg,
        "loss": {"interp_k": int(loss_cfg.get("interp_k", 8))},
        "speed": {
            "interp_chunk": int(speed_cfg.get("interp_chunk", 8192)),
            "idw_on_cpu": speed_cfg.get("idw_on_cpu", None),
        },
    }
    if mode == "uniform":
        controls["idw"] = {
            "backend": idw_cfg_raw.get("backend", "exact"),
            "allow_fallback_to_exact": idw_cfg_raw.get("allow_fallback_to_exact", True),
            "faiss_nlist": idw_cfg_raw.get("faiss_nlist", 256),
            "faiss_nprobe": idw_cfg_raw.get("faiss_nprobe", 16),
            "faiss_cache": idw_cfg_raw.get("faiss_cache", True),
            "faiss_cache_max_entries": idw_cfg_raw.get("faiss_cache_max_entries", 4),
            "k": idw_cfg_raw.get("k", 8),
            "chunk": idw_cfg_raw.get("chunk", speed_cfg.get("interp_chunk", 8192)),
        }
    if geometry_mode is not None:
        controls["geometry_mode"] = str(geometry_mode)

    checks: Dict[str, Any] = {"cfg_sha1": str(cfg_sha1)}
    if mesh_spec_sha1 is not None:
        checks["mesh_spec_sha1"] = str(mesh_spec_sha1)
    if uniform_signature_sha1 is not None:
        checks["uniform_signature_sha1"] = str(uniform_signature_sha1)

    return {
        "schema": "physics-aware-precomp-repro-v1",
        "mode": str(mode),
        "output_h5_path": os.path.abspath(os.path.expanduser(str(cache_path))),
        "input": {
            "pt_path": _abs_or_none(data_cfg.get("pt_path", None)),
            "starting_mesh_path": _abs_or_none(mesh_spec_path if mesh_spec_path is not None else mesh_cfg.get("starting_mesh_path", None)),
            "H": int(H),
            "W": int(W),
            "dx": float(dx),
            "dy": float(dy),
            "bbox": bbox,
            "T": int(T),
            "gt_refine_ratio": int(gt_refine_ratio),
            "source_t_start": train_cfg.get("precompute_t_start", None),
            "source_t_end": train_cfg.get("precompute_t_end", None),
        },
        "controls": controls,
        "checks": checks,
    }


def _get_refine_ratio(cfg: dict) -> int:
    pol = cfg.get("policy", {}) or {}
    mesh = cfg.get("mesh", {}) or {}
    rr = pol.get("refine_ratio", mesh.get("refine_ratio", 2))
    rr = int(rr)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {rr}")
    return rr


def _step_like_get(step: Any, key: str, default: Any = None) -> Any:
    if isinstance(step, dict):
        return step.get(key, default)
    return getattr(step, key, default)


def _infer_gt_refine_ratio_from_steps(
    steps: List[Any],
    cfg: dict,
    H: int,
    W: int,
    *,
    fallback_ratio: int,
    log_prefix: str = "[PRECOMP][GT-RR]",
) -> int:
    data_cfg = cfg.get("data", {}) or {}
    bbox_raw = data_cfg.get("bbox", (0.0, float(W), 0.0, float(H)))
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        bbox = tuple(float(v) for v in bbox_raw)
    else:
        bbox = (0.0, float(W), 0.0, float(H))

    cand_raw = data_cfg.get("gt_refine_ratio_candidates", (2, 4, 8))
    if isinstance(cand_raw, (list, tuple)):
        candidates = tuple(int(v) for v in cand_raw if int(v) >= 2) or (2, 4, 8)
    else:
        candidates = (2, 4, 8)

    for idx, step in enumerate(steps):
        level = _step_like_get(step, "level", None)
        ij = _step_like_get(step, "ij", None)
        pos = _step_like_get(step, "pos", None)
        if pos is None:
            pos = _step_like_get(step, "xy", None)
        if level is None or ij is None or pos is None:
            continue

        rr, info = infer_refine_ratio_from_level_ij_pos(
            levels=torch.as_tensor(level),
            ij=torch.as_tensor(ij),
            pos_xy=torch.as_tensor(pos),
            H=int(H),
            W=int(W),
            bbox=bbox,
            candidate_ratios=tuple(candidates),
            fallback_ratio=int(fallback_ratio),
        )
        print(
            f"{log_prefix} inferred refine_ratio={int(rr)} "
            f"(sample={idx} rmse={float(info.get('rmse', float('nan'))):.3e} "
            f"orient={info.get('orientation', 'row_col')} ok={bool(info.get('ok', False))})"
        )
        return int(rr)

    rr = max(2, int(fallback_ratio))
    print(f"{log_prefix} fallback refine_ratio={rr} (missing level/ij/pos)")
    return rr

def plot_precomputed_mesh_from_h5(
    h5_path: str,
    *,
    t: int = 1,
    out_path: str | None = None,
    show: bool = False,
    max_cells: int = 250_000,
    linewidth: float = 0.05,
    title: str | None = None,
    wedge_path=None,          # optional: overlay wedge polygon if it has .vertices
):
    """
    Plot the *predicted/training* mesh stored in the precompute HDF5.

    - Reads group t{t:05d} (default t=1, which is the first 'predicted step' mesh).
    - Uses pred_centers + pred_levels and meta dx/dy/bbox to draw rectangles.
    - For large meshes, decimates to at most `max_cells` cells.

    Parameters
    ----------
    h5_path : str
        Path to precomp_rollout.h5
    t : int
        Which group to plot (1..T-1). If missing, falls back to the first available group.
    out_path : str | None
        If provided, saves a PNG/PDF/etc. via matplotlib.
    show : bool
        If True, calls plt.show().
    max_cells : int
        Max number of cells to draw (decimates uniformly by index if exceeded).
    linewidth : float
        Line width for cell edges.
    title : str | None
        Plot title override.
    wedge_path : optional
        If provided and has attribute `.vertices`, overlays polygon boundary.
    """

    with h5py.File(h5_path, "r") as f:
        if "meta" not in f:
            raise RuntimeError(f"H5 file missing 'meta' group: {h5_path}")

        meta = f["meta"].attrs
        dx = float(meta["dx"])
        dy = float(meta["dy"])
        rr = int(meta.get("refine_ratio", 2))
        bbox = np.asarray(meta["bbox"], dtype=np.float64).reshape(-1)
        if bbox.size != 4:
            raise RuntimeError(f"Expected bbox with 4 entries, got shape {bbox.shape}")
        xmin, xmax, ymin, ymax = map(float, bbox)

        gname = f"t{int(t):05d}"
        if gname not in f:
            # fall back to first available txxxxx group
            keys = sorted([k for k in f.keys() if k.startswith("t")])
            if not keys:
                raise RuntimeError("No txxxxx groups found in H5; nothing to plot.")
            gname = keys[0]

        g = f[gname]
        centers = np.asarray(g["pred_centers"][...], dtype=np.float32)  # [N,2]
        levels = np.asarray(g["pred_levels"][...], dtype=np.int32)      # [N]

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise RuntimeError(f"pred_centers has unexpected shape: {centers.shape}")
    if levels.ndim != 1 or levels.shape[0] != centers.shape[0]:
        raise RuntimeError(f"pred_levels shape mismatch: levels={levels.shape}, centers={centers.shape}")

    N = centers.shape[0]
    if N == 0:
        raise RuntimeError("Mesh has zero cells; cannot plot.")

    # Optional decimation (important for large refined meshes)
    if max_cells is not None and N > int(max_cells):
        idx = np.linspace(0, N - 1, int(max_cells), dtype=np.int64)
        centers = centers[idx]
        levels = levels[idx]
        N = centers.shape[0]

    # Cell half-widths/half-heights by refinement level:
    # level L => cell size = (dx / rr^L, dy / rr^L)
    scale = np.power(float(rr), levels.astype(np.float32))
    hx = (dx / scale) * 0.5
    hy = (dy / scale) * 0.5

    x = centers[:, 0]
    y = centers[:, 1]
    x0 = x - hx
    x1 = x + hx
    y0 = y - hy
    y1 = y + hy

    # Build 4 line segments per cell (bottom, top, left, right)
    segs = np.empty((N * 4, 2, 2), dtype=np.float32)

    # bottom: (x0,y0) -> (x1,y0)
    segs[0::4, 0, 0] = x0
    segs[0::4, 0, 1] = y0
    segs[0::4, 1, 0] = x1
    segs[0::4, 1, 1] = y0

    # top: (x0,y1) -> (x1,y1)
    segs[1::4, 0, 0] = x0
    segs[1::4, 0, 1] = y1
    segs[1::4, 1, 0] = x1
    segs[1::4, 1, 1] = y1

    # left: (x0,y0) -> (x0,y1)
    segs[2::4, 0, 0] = x0
    segs[2::4, 0, 1] = y0
    segs[2::4, 1, 0] = x0
    segs[2::4, 1, 1] = y1

    # right: (x1,y0) -> (x1,y1)
    segs[3::4, 0, 0] = x1
    segs[3::4, 0, 1] = y0
    segs[3::4, 1, 0] = x1
    segs[3::4, 1, 1] = y1

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)

    lc = LineCollection(segs, linewidths=float(linewidth))
    ax.add_collection(lc)

    # Optional wedge overlay if wedge_path is a matplotlib.path.Path-like object
    if wedge_path is not None and hasattr(wedge_path, "vertices"):
        v = np.asarray(wedge_path.vertices, dtype=np.float32)
        if v.ndim == 2 and v.shape[1] == 2 and v.shape[0] >= 2:
            ax.plot(v[:, 0], v[:, 1], linewidth=2.0)

    pad_x = 0.02 * (xmax - xmin)
    pad_y = 0.02 * (ymax - ymin)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (physical)")
    ax.set_ylabel("y (physical)")

    if title is None:
        title = f"Precomputed training mesh from {os.path.basename(h5_path)} @ {gname} (N={N})"
    ax.set_title(title)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)

def plot_precomputed_mesh_with_edges_from_h5(
    h5_path: str,
    *,
    t: int = 1,
    out_path: str | None = None,
    show: bool = False,
    # zoom control:
    zoom_bbox: tuple[float, float, float, float] | None = None,  # (xmin,xmax,ymin,ymax) in physical coords
    zoom_cells: float = 20.0,   # size of auto-zoom window in units of coarse-cell widths/heights
    # drawing control:
    max_cells: int = 200_000,   # cap ONLY after cropping
    max_edges: int = 400_000,   # cap ONLY after cropping
    cell_linewidth: float = 0.25,
    edge_linewidth: float = 0.40,
    node_size: float = 3.0,
    title: str | None = None,
    wedge_path=None,            # optional: overlay wedge polygon if it has .vertices
):
    """
    Zoomed mesh plot that overlays the *saved* graph edges and node centers.

    - Reads group t{t:05d}.
    - Uses pred_centers + pred_levels + meta dx/dy/bbox to draw cell rectangles.
    - Uses *saved* pred_ei (edge_index) to draw graph edges (no adjacency recomputation).
    - Auto-zooms to a coarse–fine interface (an edge connecting nodes of different levels),
      unless zoom_bbox is explicitly provided.
    """

    def _fix_ei_shape(ei_np: np.ndarray) -> np.ndarray:
        # expected (2,E); accept (E,2)
        ei_np = np.asarray(ei_np)
        if ei_np.ndim != 2:
            raise RuntimeError(f"pred_ei must be 2D, got shape {ei_np.shape}")
        if ei_np.shape[0] == 2:
            return ei_np
        if ei_np.shape[1] == 2:
            return ei_np.T
        raise RuntimeError(f"pred_ei must be (2,E) or (E,2), got shape {ei_np.shape}")

    def _auto_zoom_bbox_from_level_interface(
        centers: np.ndarray,
        levels: np.ndarray,
        ei: np.ndarray,
        dx0: float,
        dy0: float,
        bbox_full: tuple[float, float, float, float],
        zoom_cells: float,
    ):
        """
        Find one edge (u,v) in saved ei where levels differ, then build a zoom bbox around it.
        """
        xmin_full, xmax_full, ymin_full, ymax_full = bbox_full
        if ei.size == 0:
            return None, None

        u = ei[0].astype(np.int64, copy=False)
        v = ei[1].astype(np.int64, copy=False)

        # Keep only valid indices
        N = int(centers.shape[0])
        valid = (u >= 0) & (u < N) & (v >= 0) & (v < N) & (u != v)
        if not valid.any():
            return None, None
        u = u[valid]; v = v[valid]

        # Find an interface edge: different refinement levels
        diff = (levels[u] != levels[v])
        if not diff.any():
            return None, None

        # pick the first such edge
        u0 = int(u[diff][0]); v0 = int(v[diff][0])

        # center the view on the midpoint of the two node centers
        c_mid = 0.5 * (centers[u0] + centers[v0])

        # scale the window using the *coarser* of the two levels
        Lc = int(min(levels[u0], levels[v0]))
        w = dx0 / (float(rr) ** Lc)
        h = dy0 / (float(rr) ** Lc)

        half_w = 0.5 * float(zoom_cells) * float(w)
        half_h = 0.5 * float(zoom_cells) * float(h)

        xmin = float(c_mid[0] - half_w)
        xmax = float(c_mid[0] + half_w)
        ymin = float(c_mid[1] - half_h)
        ymax = float(c_mid[1] + half_h)

        # clamp to full bbox
        xmin = max(xmin, xmin_full); xmax = min(xmax, xmax_full)
        ymin = max(ymin, ymin_full); ymax = min(ymax, ymax_full)

        return (xmin, xmax, ymin, ymax), (u0, v0)

    # ------------------- read H5 -------------------
    with h5py.File(h5_path, "r") as f:
        if "meta" not in f:
            raise RuntimeError(f"H5 file missing 'meta' group: {h5_path}")

        meta = f["meta"].attrs
        dx0 = float(meta["dx"])
        dy0 = float(meta["dy"])
        rr = int(meta.get("refine_ratio", 2))
        bbox = np.asarray(meta["bbox"], dtype=np.float64).reshape(-1)
        if bbox.size != 4:
            raise RuntimeError(f"Expected bbox with 4 entries, got shape {bbox.shape}")
        xmin_full, xmax_full, ymin_full, ymax_full = map(float, bbox)
        bbox_full = (xmin_full, xmax_full, ymin_full, ymax_full)

        gname = f"t{int(t):05d}"
        if gname not in f:
            keys = sorted([k for k in f.keys() if k.startswith("t")])
            if not keys:
                raise RuntimeError("No txxxxx groups found in H5; nothing to plot.")
            gname = keys[0]

        g = f[gname]
        centers = np.asarray(g["pred_centers"][...], dtype=np.float32)   # (N,2)
        levels  = np.asarray(g["pred_levels"][...], dtype=np.int32)      # (N,)
        if "pred_ei" not in g:
            raise RuntimeError(f"Group {gname} missing pred_ei; cannot plot saved edges.")
        ei = _fix_ei_shape(np.asarray(g["pred_ei"][...], dtype=np.int64))

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise RuntimeError(f"pred_centers has unexpected shape: {centers.shape}")
    if levels.ndim != 1 or levels.shape[0] != centers.shape[0]:
        raise RuntimeError(f"pred_levels shape mismatch: levels={levels.shape}, centers={centers.shape}")

    N = int(centers.shape[0])
    if N == 0:
        raise RuntimeError("Mesh has zero cells; cannot plot.")

    # ------------------- choose zoom bbox -------------------
    anchor = None
    if zoom_bbox is None:
        zoom_bbox, anchor = _auto_zoom_bbox_from_level_interface(
            centers=centers,
            levels=levels,
            ei=ei,
            dx0=dx0,
            dy0=dy0,
            bbox_full=bbox_full,
            zoom_cells=zoom_cells,
        )
        if zoom_bbox is None:
            # fallback: just zoom to the center of the domain
            cx = 0.5 * (xmin_full + xmax_full)
            cy = 0.5 * (ymin_full + ymax_full)
            # window ~ zoom_cells * level-0 cell sizes
            half_w = 0.5 * float(zoom_cells) * dx0
            half_h = 0.5 * float(zoom_cells) * dy0
            zoom_bbox = (
                max(xmin_full, cx - half_w),
                min(xmax_full, cx + half_w),
                max(ymin_full, cy - half_h),
                min(ymax_full, cy + half_h),
            )

    zx0, zx1, zy0, zy1 = map(float, zoom_bbox)

    # ------------------- crop nodes to zoom region -------------------
    x = centers[:, 0]
    y = centers[:, 1]
    keep = (x >= zx0) & (x <= zx1) & (y >= zy0) & (y <= zy1)
    keep_idx = np.nonzero(keep)[0].astype(np.int64)

    if keep_idx.size == 0:
        raise RuntimeError(f"No cells found inside zoom_bbox={zoom_bbox}")

    keep_set = np.zeros((N,), dtype=np.bool_)
    keep_set[keep_idx] = True

    # cap cells (after crop), preserving anchor and its 1-hop neighborhood if available
    if keep_idx.size > int(max_cells):
        mandatory = set()
        if anchor is not None:
            u0, v0 = anchor
            if keep_set[u0]:
                mandatory.add(u0)
            if keep_set[v0]:
                mandatory.add(v0)

            # add 1-hop neighbors via SAVED edges
            u = ei[0].astype(np.int64, copy=False)
            v = ei[1].astype(np.int64, copy=False)
            for a in (u0, v0):
                m = (u == a) | (v == a)
                nbr = np.unique(np.concatenate([u[m], v[m]], axis=0))
                for n in nbr.tolist():
                    if 0 <= n < N and keep_set[n]:
                        mandatory.add(int(n))

        mandatory = np.array(sorted(mandatory), dtype=np.int64)
        remaining = np.setdiff1d(keep_idx, mandatory, assume_unique=False)

        # uniform downsample remaining
        need = int(max_cells) - int(mandatory.size)
        if need <= 0:
            keep_idx = mandatory
        else:
            sel = np.linspace(0, max(0, remaining.size - 1), need, dtype=np.int64)
            keep_idx = np.concatenate([mandatory, remaining[sel]], axis=0)

        keep_set[:] = False
        keep_set[keep_idx] = True

    # ------------------- build cell-wall segments for kept nodes -------------------
    lev_k = levels[keep_idx].astype(np.int32, copy=False)
    c_k = centers[keep_idx]

    scale = np.power(float(rr), lev_k.astype(np.float32))
    hx = (dx0 / scale) * 0.5
    hy = (dy0 / scale) * 0.5

    x0 = c_k[:, 0] - hx
    x1 = c_k[:, 0] + hx
    y0 = c_k[:, 1] - hy
    y1 = c_k[:, 1] + hy

    Nc = int(c_k.shape[0])
    segs = np.empty((Nc * 4, 2, 2), dtype=np.float32)

    segs[0::4, 0, 0] = x0; segs[0::4, 0, 1] = y0
    segs[0::4, 1, 0] = x1; segs[0::4, 1, 1] = y0

    segs[1::4, 0, 0] = x0; segs[1::4, 0, 1] = y1
    segs[1::4, 1, 0] = x1; segs[1::4, 1, 1] = y1

    segs[2::4, 0, 0] = x0; segs[2::4, 0, 1] = y0
    segs[2::4, 1, 0] = x0; segs[2::4, 1, 1] = y1

    segs[3::4, 0, 0] = x1; segs[3::4, 0, 1] = y0
    segs[3::4, 1, 0] = x1; segs[3::4, 1, 1] = y1

    # ------------------- filter SAVED edges to kept nodes -------------------
    u = ei[0].astype(np.int64, copy=False)
    v = ei[1].astype(np.int64, copy=False)
    valid = (u >= 0) & (u < N) & (v >= 0) & (v < N) & (u != v)
    u = u[valid]; v = v[valid]

    in_view = keep_set[u] & keep_set[v]
    u = u[in_view]; v = v[in_view]

    # dedup as undirected for plotting clarity
    if u.size > 0:
        uu = np.minimum(u, v)
        vv = np.maximum(u, v)
        keys = uu.astype(np.int64) * np.int64(N) + vv.astype(np.int64)
        uniq = np.unique(keys)
        uu = (uniq // np.int64(N)).astype(np.int64)
        vv = (uniq %  np.int64(N)).astype(np.int64)
        u, v = uu, vv

    # cap edges after filtering
    if u.size > int(max_edges):
        sel = np.linspace(0, u.size - 1, int(max_edges), dtype=np.int64)
        u = u[sel]; v = v[sel]

    edge_segs = None
    if u.size > 0:
        edge_segs = np.empty((u.size, 2, 2), dtype=np.float32)
        edge_segs[:, 0, :] = centers[u]
        edge_segs[:, 1, :] = centers[v]

    # ------------------- plot -------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=180)

    # cell walls (solid)
    lc_cells = LineCollection(segs, linewidths=float(cell_linewidth))
    ax.add_collection(lc_cells)

    # graph edges (dashed + distinct)
    if edge_segs is not None and edge_segs.shape[0] > 0:
        lc_edges = LineCollection(edge_segs, linewidths=float(edge_linewidth))
        lc_edges.set_linestyle((0, (3.0, 3.0)))  # dashed
        lc_edges.set_alpha(0.9)
        lc_edges.set_color("tab:red")
        ax.add_collection(lc_edges)

    # node centers
    ax.scatter(c_k[:, 0], c_k[:, 1], s=float(node_size), alpha=0.85)

    # optional wedge overlay
    if wedge_path is not None and hasattr(wedge_path, "vertices"):
        vtx = np.asarray(wedge_path.vertices, dtype=np.float32)
        if vtx.ndim == 2 and vtx.shape[1] == 2 and vtx.shape[0] >= 2:
            ax.plot(vtx[:, 0], vtx[:, 1], linewidth=2.0)

    # view
    pad_x = 0.02 * (zx1 - zx0)
    pad_y = 0.02 * (zy1 - zy0)
    ax.set_xlim(zx0 - pad_x, zx1 + pad_x)
    ax.set_ylim(zy0 - pad_y, zy1 + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (physical)")
    ax.set_ylabel("y (physical)")

    if title is None:
        title = (
            f"Zoomed mesh + saved edges/nodes: {os.path.basename(h5_path)} @ {gname} "
            f"(cells_in_view={Nc}, edges_in_view={0 if edge_segs is None else edge_segs.shape[0]})"
        )
    ax.set_title(title)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _normalize_builder_output_to_mesh(
    out,
    *,
    cfg: dict,
    H: int,
    W: int,
    device: torch.device,
    refine_ratio: int = 2,
):
    """
    Normalize various possible builder outputs into:
      centers (N,2) float32,
      levels  (N,)  int64,
      parents (N,)  int64 (coarse parent flat index in [0, H*W)),
      edge_index (2,E) int64,
      mask_parent (H,W) bool

    Supports:
      - dict with centers/levels/(parents)/(edge_index)
      - tuple/list of length 4 or 5
      - list-of-cells (len >> 5), where each cell is dict/tuple-like
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    bbox = tuple(cfg["data"]["bbox"])
    xmin, xmax, ymin, ymax = bbox

    def _as_tensor(x, dtype=None):
        if torch.is_tensor(x):
            t = x
        else:
            t = torch.as_tensor(x)
        if dtype is not None:
            t = t.to(dtype)
        return t

    # -------------------------
    # Case A: dict-style output
    # -------------------------
    if isinstance(out, dict):
        # common key aliases
        def pick(*names):
            for n in names:
                if n in out and out[n] is not None:
                    return out[n]
            return None

        centers = pick("centers", "pred_centers", "pos", "xy", "cell_centers")
        levels  = pick("levels", "pred_levels", "level", "cell_levels")
        parents = pick("parents", "pred_parents", "parent", "parent_flat", "dyn_parents")
        ei      = pick("edge_index", "pred_ei", "ei")

        if centers is None or levels is None:
            raise RuntimeError(f"Builder dict missing centers/levels. Keys={list(out.keys())}")

        centers = _as_tensor(centers, torch.float32).to(device)
        levels  = _as_tensor(levels).to(torch.int64).to(device)

        if parents is None:
            parents = parents_from_pos(centers, H, W, xmin, xmax, ymin, ymax)
        else:
            parents = _as_tensor(parents).to(torch.int64).to(device)

        # if parents are (N,2) ij on coarse grid, flatten
        if parents.ndim == 2 and parents.size(-1) == 2:
            parents = parents[:, 1] * W + parents[:, 0]
        parents = parents.view(-1)

        if ei is not None:
            ei = _as_tensor(ei).to(torch.int64).to(device)
        else:
            ei = build_amr_local_knn_edges(
                centers, parents, H, W,
                k_local=int(cfg.get("edges", {}).get("k_local", 4)),
                max_local=int(cfg.get("edges", {}).get("max_local", 2048)),
            ).to(device)

        mask_parent = _mask_from_parents(parents, H, W).to(device)
        return centers, levels, parents, ei, mask_parent

    # -----------------------------------
    # Case B: short tuple/list (4 or 5)
    # -----------------------------------
    if isinstance(out, (tuple, list)) and len(out) in (4, 5):
        centers, levels, parents, ei = out[:4]
        mask_parent = out[4] if len(out) == 5 else None

        centers = _as_tensor(centers, torch.float32).to(device)
        levels  = _as_tensor(levels).to(torch.int64).to(device)
        parents = _as_tensor(parents).to(torch.int64).to(device)

        if parents.ndim == 2 and parents.size(-1) == 2:
            parents = parents[:, 1] * W + parents[:, 0]
        parents = parents.view(-1)

        if ei is not None:
            ei = _as_tensor(ei).to(torch.int64).to(device)
        else:
            ei = build_amr_local_knn_edges(
                centers, parents, H, W,
                k_local=int(cfg.get("edges", {}).get("k_local", 4)),
                max_local=int(cfg.get("edges", {}).get("max_local", 2048)),
            ).to(device)
        if mask_parent is None:
            mask_parent = _mask_from_parents(parents, H, W).to(device)
        else:
            mask_parent = _as_tensor(mask_parent).to(torch.bool).to(device).view(H, W)

        return centers, levels, parents, ei, mask_parent

    # Case: builder returned a list of per-cell dicts (common for wedge mesh builders)
    if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], dict):
        first = out[0]
        keys = list(first.keys())

        # Required minimal keys from your builder: level, i, j
        if not (("level" in first) and ("i" in first) and ("j" in first)):
            raise RuntimeError(
                f"Cell dict missing required (level,i,j). First cell keys={keys}"
            )

        # Extract arrays
        levels = torch.as_tensor([c["level"] for c in out], dtype=torch.int64, device=device).view(-1)
        ii     = torch.as_tensor([c["i"]     for c in out], dtype=torch.int64, device=device).view(-1)
        jj     = torch.as_tensor([c["j"]     for c in out], dtype=torch.int64, device=device).view(-1)

        centers = _centers_from_level_ij(
            levels=levels,
            row=jj,
            col=ii,
            H=H,
            W=W,
            bbox=bbox,
            refine_ratio=rr,
        ).to(dtype=torch.float32)

        # Parents: coarse parent index on level-0 grid.
        # For a level-L cell, its level-0 parent is (i // rr^L, j // rr^L).
        scale_i = torch.pow(torch.tensor(rr, device=device, dtype=torch.int64), levels)
        i0 = torch.div(ii, scale_i, rounding_mode="floor")
        j0 = torch.div(jj, scale_i, rounding_mode="floor")
        parents = (j0 * int(W) + i0).to(torch.int64).view(-1)

        # Coarse mask
        mask_parent = _mask_from_parents(parents, H, W).to(device=device, dtype=torch.bool)

        # If no explicit edge list is provided by the builder, rebuild local edges.
        edge_index = build_amr_local_knn_edges(
            centers,
            parents,
            H,
            W,
            k_local=int(cfg.get("edges", {}).get("k_local", 4)),
            max_local=int(cfg.get("edges", {}).get("max_local", 2048)),
        ).to(device=device, dtype=torch.int64)

        return centers, levels, parents, edge_index, mask_parent

    raise RuntimeError(f"Unsupported builder return type: {type(out)}")

def amr_cell_wh_area_from_levels(
    levels: torch.Tensor,
    *,
    dx0: float,
    dy0: float,
    refine_ratio: int = 2,
):
    """
    levels: (N,) int64
    Returns:
      w: (N,) float32
      h: (N,) float32
      area: (N,) float32
      hx: (N,) float32
      hy: (N,) float32
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    lv = levels.to(dtype=torch.float32)
    scale = torch.pow(torch.tensor(float(rr), device=levels.device), lv)
    w = (float(dx0) / scale).to(torch.float32)
    h = (float(dy0) / scale).to(torch.float32)
    area = (w * h).to(torch.float32)
    hx = (0.5 * w).to(torch.float32)
    hy = (0.5 * h).to(torch.float32)
    return w, h, area, hx, hy


def dec_edge_attr_for_dyadic_quads(
    centers: torch.Tensor,     # (N,2) float32, physical
    levels: torch.Tensor,      # (N,) int64
    edge_index: torch.Tensor,  # (2,E) int64, directed edges (both directions allowed)
    *,
    dx0: float,
    dy0: float,
    refine_ratio: int = 2,
):
    """
    Returns edge_attr: (E,5) float32 with columns:
      [nx, ny, face_len, dual_len, tau]
    """
    device = centers.device
    ei = edge_index.to(device=device, dtype=torch.int64)
    u = ei[0]
    v = ei[1]

    w, h, _area, hx, hy = amr_cell_wh_area_from_levels(
        levels, dx0=dx0, dy0=dy0, refine_ratio=refine_ratio
    )

    du = centers[v] - centers[u]  # (E,2)
    dx = du[:, 0]
    dy = du[:, 1]

    # Expected normal separations (orthogonal dyadic quads)
    exp_dx = hx[u] + hx[v]
    exp_dy = hy[u] + hy[v]

    # Decide whether this edge corresponds to a vertical face (left/right) or horizontal face (up/down).
    # Use "which component matches the expected separation better" rather than abs(dx)>abs(dy),
    # because coarse-fine neighbors can have tangential offsets.
    err_x = (dx.abs() - exp_dx).abs()
    err_y = (dy.abs() - exp_dy).abs()
    is_lr = err_x <= err_y  # True => left/right neighbor across vertical face

    # Unit normal from u->v
    nx = torch.zeros_like(dx, dtype=torch.float32)
    ny = torch.zeros_like(dy, dtype=torch.float32)

    # Dual length (center-to-center distance along normal)
    dual_len = torch.empty_like(dx, dtype=torch.float32)

    # Shared face length
    face_len = torch.empty_like(dx, dtype=torch.float32)

    # Left/right edges (vertical face): normal is ±x, face length is overlap in y => min(h_u,h_v)
    nx[is_lr] = torch.sign(dx[is_lr]).to(torch.float32)
    dual_len[is_lr] = exp_dx[is_lr].to(torch.float32)
    face_len[is_lr] = torch.minimum(h[u[is_lr]], h[v[is_lr]]).to(torch.float32)

    # Up/down edges (horizontal face): normal is ±y, face length is overlap in x => min(w_u,w_v)
    inv = ~is_lr
    ny[inv] = torch.sign(dy[inv]).to(torch.float32)
    dual_len[inv] = exp_dy[inv].to(torch.float32)
    face_len[inv] = torch.minimum(w[u[inv]], w[v[inv]]).to(torch.float32)

    # Hodge-star ratio / diffusion weight
    eps = torch.tensor(1e-12, device=device, dtype=torch.float32)
    tau = face_len / torch.maximum(dual_len, eps)

    edge_attr = torch.stack([nx, ny, face_len, dual_len, tau], dim=1).to(torch.float32)
    return edge_attr

def _centers_from_level_ij(
    level_1d: torch.Tensor,  # (N,) int64
    i_1d: torch.Tensor,      # (N,) int64
    j_1d: torch.Tensor,      # (N,) int64
    *,
    H: int,
    W: int,
    bbox: tuple[float, float, float, float],
    device: torch.device,
    refine_ratio: int = 2,
) -> torch.Tensor:
    """
    Compute physical cell centers from (level, i, j) where i,j are indices
    on the level-L grid (0 <= i < W*refine_ratio^L, 0 <= j < H*refine_ratio^L).

    Assumes bbox spans the full level-0 domain discretized into HxW cells.
    """
    xmin, xmax, ymin, ymax = map(float, bbox)
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    # cell size at level L: dx0/rr^L, dy0/rr^L
    lv = level_1d.to(device=device, dtype=torch.float32)
    scale = torch.pow(torch.tensor(float(rr), device=device), lv)

    dxL = dx0 / scale
    dyL = dy0 / scale

    # center = lower_left + (i+0.5)*dxL, (j+0.5)*dyL
    x = torch.tensor(xmin, device=device) + (i_1d.to(device=device, dtype=torch.float32) + 0.5) * dxL
    y = torch.tensor(ymin, device=device) + (j_1d.to(device=device, dtype=torch.float32) + 0.5) * dyL
    return torch.stack([x, y], dim=1)  # (N,2)

@torch.no_grad()
def debug_plot_policy_gradients(
    grad_debug: dict,
    cfg: dict,
    t: int | None = None,
    use_pooled_up: bool = False,
    out_dir: str = "./debug_grad_plots",
    prefix: str = "grads",
):
    """
    Plot gradient magnitude maps used by the AMR policy.

    grad_debug should be the dict that predict_masks_hierarchical_from_gt_gradients
    filled when called with debug_out=...:
        grad_debug["G"]         : raw per-level magnitudes (dict[L] -> (h_L, w_L))
        grad_debug["pooled_up"] : pooled-up maps used for parent thresholds

    If use_pooled_up=False, plots G[L]; if True, plots pooled_up[L].

    Produces one PNG with subplots for all levels.
    """

    os.makedirs(out_dir, exist_ok=True)

    maps = grad_debug["pooled_up"] if use_pooled_up else grad_debug["G"]
    if not maps:
        print("[DEBUG GRADS] No gradient maps in grad_debug; nothing to plot.")
        return

    xmin, xmax, ymin, ymax = cfg["data"]["bbox"]

    # Convert to numpy, collect vmin/vmax over finite entries
    maps_np = {}
    vmin = float("inf")
    vmax = float("-inf")

    for L, m in maps.items():
        if torch.is_tensor(m):
            m_np = m.detach().cpu().numpy()
        else:
            m_np = np.asarray(m)
        maps_np[L] = m_np
        finite = np.isfinite(m_np)
        if finite.any():
            vmin = min(vmin, float(m_np[finite].min()))
            vmax = max(vmax, float(m_np[finite].max()))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0

    # non-negative mags: clamp min at 0 for color scale
    vmin = max(0.0, vmin)

    levels = sorted(maps_np.keys())
    nL = len(levels)

    fig, axes = plt.subplots(
        1, nL, figsize=(4 * nL, 4), squeeze=False,
        sharex=True, sharey=True,
    )

    for idx, L in enumerate(levels):
        ax = axes[0, idx]
        m_np = maps_np[L]

        im = ax.imshow(
            m_np,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_title(f"L{L} {'(pooled)' if use_pooled_up else ''}")
        ax.set_xlabel("x")
        if idx == 0:
            ax.set_ylabel("y")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if t is None:
        t_tag = "tXXXX"
    else:
        t_tag = f"t{int(t):04d}"

    fname = f"{prefix}_{t_tag}_{'pooled' if use_pooled_up else 'raw'}.png"
    out_path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[DEBUG GRADS] saved {out_path}")

def _cell_corners_from_centers_levels(
    centers: torch.Tensor,  # (N,2)
    levels: torch.Tensor,   # (N,)
    H: int, W: int,
    bbox: tuple[float, float, float, float],
    refine_ratio: int = 2,
) -> torch.Tensor:
    """
    Return corners for each AMR cell (axis-aligned) as (N,4,2) in physical coords.
    Corner order: (-x,-y), (-x,+y), (+x,-y), (+x,+y) relative to center.
    """
    xmin, xmax, ymin, ymax = bbox
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    lv = levels.to(torch.float32)
    # cell width at level l is dx0 / rr^l, so half-width is dx0 / (2*rr^l)
    halfx = (dx0 * 0.5) * torch.pow(torch.tensor(float(rr), device=centers.device), -lv)
    halfy = (dy0 * 0.5) * torch.pow(torch.tensor(float(rr), device=centers.device), -lv)

    # (N,1) for broadcasting
    halfx = halfx.view(-1, 1)
    halfy = halfy.view(-1, 1)

    # 4 corners
    sx = torch.tensor([-1.0, -1.0,  1.0,  1.0], device=centers.device).view(1, 4)
    sy = torch.tensor([-1.0,  1.0, -1.0,  1.0], device=centers.device).view(1, 4)

    corners = torch.empty((centers.shape[0], 4, 2), device=centers.device, dtype=centers.dtype)
    corners[..., 0] = centers[:, 0:1] + sx * halfx
    corners[..., 1] = centers[:, 1:2] + sy * halfy
    return corners

def _build_starting_mesh_from_spec(
    mesh_spec_path_or_spec,
    cfg: Dict[str, Any],
    H: int,
    W: int,
    dx: float,
    dy: float,
    device: torch.device,
):
    """
    Build an explicit AMR wedge mesh (centers/levels/parents/edge_index) from the wedge spec.

    Accepts either:
      - mesh_spec_path_or_spec: str path to .pt spec, OR a dict already loaded

    Returns:
      centers: (N,2) float32
      levels : (N,)  int64
      parents_flat: (N,) int64  parent indices on coarse HxW grid in [0, H*W)
      edge_index: (2,E) int64
      mask_parent: (H,W) bool
    """
    import inspect

    # --- load spec + keep path if available ---
    mesh_spec_path = mesh_spec_path_or_spec if isinstance(mesh_spec_path_or_spec, str) else None
    spec = _load_mesh_spec(mesh_spec_path) if mesh_spec_path is not None else mesh_spec_path_or_spec
    if not isinstance(spec, dict):
        raise RuntimeError(f"Expected dict mesh spec, got {type(spec)}")

    bbox = tuple(map(float, cfg["data"]["bbox"]))
    n0_x = int(spec.get("n0_x", spec.get("n0", W)))
    n0_y = int(spec.get("n0_y", spec.get("n0", H)))
    max_level = int(spec.get("max_level", cfg.get("policy", {}).get("max_level", 3)))

    # --- call build_amr_mesh_for_wedge with signature-agnostic kwargs ---
    """
    fn = build_amr_mesh_for_wedge
    params = inspect.signature(fn).parameters
    kwargs = {}

    # pass spec or path depending on what the function accepts
    if "spec" in params:
        kwargs["spec"] = spec
    elif "mesh_spec" in params:
        kwargs["mesh_spec"] = spec
    elif "mesh_spec_path" in params or "spec_path" in params or "path" in params:
        if mesh_spec_path is None:
            raise RuntimeError(
                "build_amr_mesh_for_wedge appears to require a file path, "
                "but _build_starting_mesh_from_spec was given a dict."
            )
        if "mesh_spec_path" in params:
            kwargs["mesh_spec_path"] = mesh_spec_path
        elif "spec_path" in params:
            kwargs["spec_path"] = mesh_spec_path
        else:
            kwargs["path"] = mesh_spec_path
    else:
        # last resort: try first positional arg as spec
        # (we’ll do this by calling with **kwargs empty below)
        pass

    # common optional args
    if "bbox" in params:
        kwargs["bbox"] = bbox
    if "H" in params:
        kwargs["H"] = int(H)
    if "W" in params:
        kwargs["W"] = int(W)
    has_rect_params = ("n0_x" in params) or ("n0_y" in params)
    if "n0_x" in params:
        kwargs["n0_x"] = int(n0_x)
    if "n0_y" in params:
        kwargs["n0_y"] = int(n0_y)
    if "n0" in params and (not has_rect_params):
        if int(n0_x) != int(n0_y):
            raise RuntimeError(
                "Mesh spec is rectangular (n0_x != n0_y), but build_amr_mesh_for_wedge "
                "does not expose n0_x/n0_y parameters."
            )
        kwargs["n0"] = int(n0_x)
    if "max_level" in params:
        kwargs["max_level"] = max_level
    if "device" in params:
        kwargs["device"] = device

    try:
        out = fn(**kwargs) if kwargs else fn(spec)
    except TypeError:
        # fallback: try passing path if we have one
        if mesh_spec_path is not None:
            out = fn(mesh_spec_path)
        else:
            raise
    """
        # --- call build_amr_mesh_for_wedge with signature-agnostic kwargs ---
    fn = build_amr_mesh_for_wedge
    params = inspect.signature(fn).parameters
    kwargs = {}

    # This builder wants a dict called "boundaries"
    if "boundaries" in params:
        kwargs["boundaries"] = spec
    elif "spec" in params:
        kwargs["spec"] = spec
    elif "mesh_spec" in params:
        kwargs["mesh_spec"] = spec
    else:
        # If there is no obvious dict parameter, try passing spec positionally
        # (but DO NOT pass the path, because we already know this builder indexes like a dict)
        #out = fn(spec)
        # then skip to normalize output
        # (return handling continues below)
        # NOTE: put `out = fn(spec)` in a small else block in your file
        # For clarity, we’ll fall through by setting `kwargs` empty and calling below.
        kwargs = {}
    
    # common optional args
    if "bbox" in params:
        kwargs["bbox"] = bbox
    if "H" in params:
        kwargs["H"] = int(H)
    if "W" in params:
        kwargs["W"] = int(W)
    has_rect_params = ("n0_x" in params) or ("n0_y" in params)
    if "n0_x" in params:
        kwargs["n0_x"] = int(n0_x)
    if "n0_y" in params:
        kwargs["n0_y"] = int(n0_y)
    if "n0" in params and (not has_rect_params):
        if int(n0_x) != int(n0_y):
            raise RuntimeError(
                "Mesh spec is rectangular (n0_x != n0_y), but build_amr_mesh_for_wedge "
                "does not expose n0_x/n0_y parameters."
            )
        kwargs["n0"] = int(n0_x)
    if "max_level" in params:
        kwargs["max_level"] = max_level
    if "device" in params:
        kwargs["device"] = device

    # Call it. IMPORTANT: do not fall back to fn(mesh_spec_path).
    out = fn(**kwargs) if kwargs else fn(spec)
    # --- normalize any builder output into the canonical mesh tensors ---
    centers, levels, parents, edge_index, mask_parent = _normalize_builder_output_to_mesh(
        out,
        cfg=cfg,
        H=H,
        W=W,
        device=torch.device(device) if not isinstance(device, torch.device) else device,
        refine_ratio=_get_refine_ratio(cfg),
    )

    # Optional: enforce expected shapes early
    if centers.ndim != 2 or centers.size(1) != 2:
        raise RuntimeError(f"normalized centers must be (N,2), got {tuple(centers.shape)}")
    if levels.ndim != 1 or levels.size(0) != centers.size(0):
        raise RuntimeError(f"normalized levels must be (N,), got {tuple(levels.shape)} for N={centers.size(0)}")
    if parents.ndim != 1 or parents.size(0) != centers.size(0):
        raise RuntimeError(f"normalized parents must be (N,), got {tuple(parents.shape)} for N={centers.size(0)}")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise RuntimeError(f"normalized edge_index must be (2,E), got {tuple(edge_index.shape)}")
    if mask_parent.ndim != 2 or tuple(mask_parent.shape) != (H, W):
        raise RuntimeError(f"normalized mask_parent must be (H,W)={(H,W)}, got {tuple(mask_parent.shape)}")

    return centers, levels, parents, edge_index, mask_parent


def _path_contains_points_torch(wedge_path, pts: torch.Tensor, radius: float = 0.0) -> torch.Tensor:
    """
    pts: (N,2) torch tensor on any device.
    Returns: (N,) bool torch tensor on same device.
    """
    pts_np = pts.detach().cpu().numpy()
    inside_np = wedge_path.contains_points(pts_np, radius=radius)
    return torch.from_numpy(inside_np).to(device=pts.device)


def _wedge_lookup_cache_key(
    *,
    wedge_path,
    H: int,
    W: int,
    Lmax: int,
    refine_ratio: int,
    bbox: tuple[float, float, float, float],
    radius: float,
) -> str:
    verts = np.asarray(wedge_path.vertices, dtype=np.float64)
    vhash = hashlib.sha1(verts.tobytes()).hexdigest()[:20]
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    return (
        f"v={vhash}|H={int(H)}|W={int(W)}|Lmax={int(Lmax)}|rr={int(refine_ratio)}|"
        f"bbox={xmin:.16g},{xmax:.16g},{ymin:.16g},{ymax:.16g}|r={float(radius):.16g}"
    )


def _build_wedge_clip_level_lookup(
    *,
    wedge_path,
    H: int,
    W: int,
    Lmax: int,
    refine_ratio: int,
    bbox: tuple[float, float, float, float],
    radius: float,
) -> Dict[str, Any]:
    """
    Precompute per-level cell classification masks:
      - full[L][j,i]: all four corners inside wedge
      - intersect[L][j,i]: center-inside OR any-corner-inside
    """
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    xspan = float(xmax - xmin)
    yspan = float(ymax - ymin)
    rr = int(refine_ratio)

    full_cpu: Dict[int, torch.Tensor] = {}
    intersect_cpu: Dict[int, torch.Tensor] = {}

    for l in range(0, int(Lmax) + 1):
        HH = int(H) * (rr ** int(l))
        WW = int(W) * (rr ** int(l))
        dxL = xspan / float(WW)
        dyL = yspan / float(HH)

        # Cell-vertex inclusion: evaluate once per vertex grid, then derive per-cell corner logic.
        x_edges = xmin + np.arange(WW + 1, dtype=np.float64) * dxL
        y_edges = ymin + np.arange(HH + 1, dtype=np.float64) * dyL
        xv, yv = np.meshgrid(x_edges, y_edges, indexing="xy")
        v_pts = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)
        v_in = wedge_path.contains_points(v_pts, radius=float(radius)).reshape(HH + 1, WW + 1)

        c00 = v_in[:-1, :-1]
        c10 = v_in[1:, :-1]
        c01 = v_in[:-1, 1:]
        c11 = v_in[1:, 1:]
        full = c00 & c10 & c01 & c11
        any_corner = c00 | c10 | c01 | c11

        # Cell-center inclusion.
        x_ctr = xmin + (np.arange(WW, dtype=np.float64) + 0.5) * dxL
        y_ctr = ymin + (np.arange(HH, dtype=np.float64) + 0.5) * dyL
        xc, yc = np.meshgrid(x_ctr, y_ctr, indexing="xy")
        c_pts = np.stack([xc.reshape(-1), yc.reshape(-1)], axis=1)
        center_in = wedge_path.contains_points(c_pts, radius=float(radius)).reshape(HH, WW)

        intersect = center_in | any_corner
        full_cpu[int(l)] = torch.from_numpy(full.astype(np.bool_))
        intersect_cpu[int(l)] = torch.from_numpy(intersect.astype(np.bool_))

    return {
        "full_cpu": full_cpu,
        "intersect_cpu": intersect_cpu,
        "device_cache": {},  # keyed by device string
    }


def _get_wedge_clip_level_lookup(
    *,
    wedge_path,
    H: int,
    W: int,
    Lmax: int,
    refine_ratio: int,
    bbox: tuple[float, float, float, float],
    radius: float,
) -> Dict[str, Any]:
    key = _wedge_lookup_cache_key(
        wedge_path=wedge_path,
        H=H,
        W=W,
        Lmax=Lmax,
        refine_ratio=refine_ratio,
        bbox=bbox,
        radius=radius,
    )
    entry = _WEDGE_CLIP_LOOKUP_CACHE.get(key, None)
    if entry is None:
        entry = _build_wedge_clip_level_lookup(
            wedge_path=wedge_path,
            H=H,
            W=W,
            Lmax=Lmax,
            refine_ratio=refine_ratio,
            bbox=bbox,
            radius=radius,
        )
        _WEDGE_CLIP_LOOKUP_CACHE[key] = entry
        print(
            f"[WEDGE-LOOKUP] built cache for H={int(H)} W={int(W)} Lmax={int(Lmax)} rr={int(refine_ratio)}",
            flush=True,
        )
    return entry


def _lookup_level_masks_on_device(
    lookup: Dict[str, Any],
    device: torch.device,
) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    dev = torch.device(device)
    if dev.type == "cpu":
        return lookup["full_cpu"], lookup["intersect_cpu"]

    dkey = f"{dev.type}:{dev.index if dev.index is not None else -1}"
    dcache = lookup.get("device_cache", {})
    got = dcache.get(dkey, None)
    if got is None:
        got = {
            "full": {L: m.to(device=dev, dtype=torch.bool) for L, m in lookup["full_cpu"].items()},
            "intersect": {L: m.to(device=dev, dtype=torch.bool) for L, m in lookup["intersect_cpu"].items()},
        }
        dcache[dkey] = got
        lookup["device_cache"] = dcache
    return got["full"], got["intersect"]


def _cell_level_ij_from_centers(
    *,
    centers: torch.Tensor,
    levels: torch.Tensor,
    H: int,
    W: int,
    bbox: tuple[float, float, float, float],
    refine_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    col_f = ((centers[:, 0].to(torch.float32) - xmin) * ww_l / xs) - 0.5
    row_f = ((centers[:, 1].to(torch.float32) - ymin) * hh_l / ys) - 0.5

    col = torch.round(col_f).to(torch.long)
    row = torch.round(row_f).to(torch.long)
    ww_i = ww_l.to(torch.long)
    hh_i = hh_l.to(torch.long)
    col = torch.minimum(torch.maximum(col, torch.zeros_like(col)), ww_i - 1)
    row = torch.minimum(torch.maximum(row, torch.zeros_like(row)), hh_i - 1)
    return row, col


def _centers_from_level_ij(
    *,
    levels: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    H: int,
    W: int,
    bbox: tuple[float, float, float, float],
    refine_ratio: int,
) -> torch.Tensor:
    if levels.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=levels.device)

    dev = levels.device
    lv = levels.view(-1).long()
    max_l = int(torch.clamp(lv.max(), min=0).item())
    rr = int(refine_ratio)
    rr_pows = torch.tensor([rr ** i for i in range(max_l + 1)], device=dev, dtype=torch.float32)
    ww_l = (float(W) * rr_pows).index_select(0, lv.clamp(0, max_l))
    hh_l = (float(H) * rr_pows).index_select(0, lv.clamp(0, max_l))

    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    xs = float(xmax - xmin)
    ys = float(ymax - ymin)
    x = xmin + (col.to(torch.float32) + 0.5) * (xs / ww_l)
    y = ymin + (row.to(torch.float32) + 0.5) * (ys / hh_l)
    return torch.stack([x, y], dim=1).to(dtype=torch.float32)


def _lookup_mask_values_by_level(
    mask_by_level: Dict[int, torch.Tensor],
    *,
    levels: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros(levels.shape[0], dtype=torch.bool, device=levels.device)
    if levels.numel() == 0:
        return out
    for L in torch.unique(levels).tolist():
        Lint = int(L)
        sel = (levels == Lint)
        M = mask_by_level[Lint]
        out[sel] = M[row[sel], col[sel]]
    return out


def _refine_cells_one_level_ij(
    *,
    row: torch.Tensor,
    col: torch.Tensor,
    levels: torch.Tensor,
    parents: torch.Tensor,
    refine_mask: torch.Tensor,
    refine_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rr = int(refine_ratio)
    keep_mask = ~refine_mask

    row_keep = row[keep_mask]
    col_keep = col[keep_mask]
    lvl_keep = levels[keep_mask]
    par_keep = parents[keep_mask]

    row_ref = row[refine_mask]
    col_ref = col[refine_mask]
    lvl_ref = levels[refine_mask]
    par_ref = parents[refine_mask]
    if row_ref.numel() == 0:
        return row, col, levels, parents

    offs = torch.arange(rr, device=row.device, dtype=torch.long)
    off_r, off_c = torch.meshgrid(offs, offs, indexing="ij")
    child_row = row_ref.view(-1, 1, 1) * rr + off_r.view(1, rr, rr)
    child_col = col_ref.view(-1, 1, 1) * rr + off_c.view(1, rr, rr)

    child_count = rr * rr
    row_child = child_row.reshape(-1)
    col_child = child_col.reshape(-1)
    lvl_child = (lvl_ref + 1).repeat_interleave(child_count)
    par_child = par_ref.repeat_interleave(child_count)

    new_row = torch.cat([row_keep, row_child], dim=0)
    new_col = torch.cat([col_keep, col_child], dim=0)
    new_lvl = torch.cat([lvl_keep, lvl_child], dim=0)
    new_par = torch.cat([par_keep, par_child], dim=0)
    return new_row, new_col, new_lvl, new_par

def _load_mesh_spec(mesh_spec_path: str) -> dict:
    spec = torch.load(mesh_spec_path, map_location="cpu")
    if not isinstance(spec, dict):
        raise RuntimeError(f"Expected dict from mesh spec, got {type(spec)}")
    return spec

def _refine_cells_one_level(
    centers: torch.Tensor,     # (N,2)
    levels: torch.Tensor,      # (N,)
    parents: torch.Tensor,     # (N,)
    refine_mask: torch.Tensor, # (N,) bool
    H: int, W: int,
    bbox: tuple[float, float, float, float],
    refine_ratio: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replace each selected cell by its refine_ratio^2 children (level+1).
    Non-selected cells are kept.
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    xmin, xmax, ymin, ymax = bbox
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    keep_mask = ~refine_mask
    c_keep = centers[keep_mask]
    l_keep = levels[keep_mask]
    p_keep = parents[keep_mask]

    c_ref = centers[refine_mask]
    l_ref = levels[refine_mask]
    p_ref = parents[refine_mask]

    if c_ref.numel() == 0:
        return centers, levels, parents

    # Parent cell size at level l is dx0/rr^l, dy0/rr^l.
    lv = l_ref.to(torch.float32)
    parent_w = float(dx0) * torch.pow(torch.tensor(float(rr), device=centers.device), -lv)
    parent_h = float(dy0) * torch.pow(torch.tensor(float(rr), device=centers.device), -lv)

    # Child-center offsets in parent-cell units.
    frac = ((torch.arange(rr, device=centers.device, dtype=centers.dtype) + 0.5) / float(rr)) - 0.5
    gx, gy = torch.meshgrid(frac, frac, indexing="xy")

    child = torch.empty((c_ref.shape[0], rr, rr, 2), device=centers.device, dtype=centers.dtype)
    child[..., 0] = c_ref[:, 0].view(-1, 1, 1) + parent_w.view(-1, 1, 1) * gx.view(1, rr, rr)
    child[..., 1] = c_ref[:, 1].view(-1, 1, 1) + parent_h.view(-1, 1, 1) * gy.view(1, rr, rr)
    child_centers = child.reshape(-1, 2)

    child_count = rr * rr
    child_levels = (l_ref + 1).repeat_interleave(child_count)
    child_parents = p_ref.repeat_interleave(child_count)

    new_centers = torch.cat([c_keep, child_centers], dim=0)
    new_levels  = torch.cat([l_keep, child_levels], dim=0)
    new_parents = torch.cat([p_keep, child_parents], dim=0)
    return new_centers, new_levels, new_parents


def _mask_from_parents(parents_flat: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    parents_flat: (H*W,) or (N_dynamic,) parent indices into H*W in [0, H*W), or -1 for inactive.
    Returns boolean (H, W) mask where True = active parent.
    """
    pf = parents_flat.view(-1)
    if pf.numel() == H * W:
        # already parent-grid length; treat >=0 as active
        m = (pf >= 0)
        return m.view(H, W)
    # dynamic-length: scatter to parent-grid
    m = torch.zeros(H * W, dtype=torch.bool, device=pf.device)
    valid = (pf >= 0)
    if valid.any():
        m[pf[valid].long()] = True
    return m.view(H, W)

@torch.no_grad()
def cell_area_from_levels(
    levels: torch.Tensor,
    H: int,
    W: int,
    bbox: tuple[float, float, float, float],
    refine_ratio: int = 2,
) -> torch.Tensor:
    """
    Compute area per cell for axis-aligned AMR quads:
      area(L) = (dx0 * dy0) / (refine_ratio^2)^L
    where dx0, dy0 are level-0 cell sizes from bbox and (H,W).
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    xmin, xmax, ymin, ymax = map(float, bbox)
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)
    levels_f = levels.to(torch.float32)
    area = (dx0 * dy0) * torch.pow(torch.tensor(float(rr * rr), device=levels.device), -levels_f)
    return area.to(torch.float32)


@torch.no_grad()
def build_amr_face_adjacency_edges(
    centers: torch.Tensor,      # (N,2) physical cell centers
    levels: torch.Tensor,       # (N,) int
    H: int,
    W: int,
    bbox: tuple[float, float, float, float],
    *,
    max_L_guard: int = 6,
    return_edge_attr: bool = True,
    refine_ratio: int = 2,
):
    """
    Face-adjacency edge builder for axis-aligned AMR quads.

    Uses a max-grid occupancy rasterization to detect which *original* cells share a face,
    including coarse–fine partial overlaps. Nodes remain original cells (true centers).

    Returns:
      edge_index: (2,E) int64, directed, includes both directions, deduplicated, no self-loops.

    If return_edge_attr=True, also returns:
      edge_attr: (E,5) float32 columns:
        [nx, ny, face_len, center_dist, w_diff]
      where:
        (nx,ny)     = unit outward normal for the directed edge src->dst (axis-aligned),
        face_len    = shared face length between src and dst (exact on max-grid),
        center_dist = ||c_dst - c_src||,
        w_diff      = face_len / center_dist   (canonical two-point diffusion weight).
    """
    device = centers.device
    N = int(centers.shape[0])
    if N == 0:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
        if return_edge_attr:
            ea = torch.empty((0, 5), dtype=torch.float32, device=device)
            return ei, ea
        return ei

    xmin, xmax, ymin, ymax = map(float, bbox)
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    levels_i64 = levels.to(torch.int64)
    Lmax = int(levels_i64.max().item()) if levels_i64.numel() else 0
    if Lmax > int(max_L_guard):
        raise RuntimeError(
            f"Lmax={Lmax} exceeds guard={max_L_guard}. "
            "Raster-based face adjacency will be too large; "
            "raise guard or use a hierarchical neighbor finder."
        )

    # Level-0 cell sizes
    dx0 = (xmax - xmin) / float(W)
    dy0 = (ymax - ymin) / float(H)

    # Max-grid resolution
    Hmax = int(H) * (rr ** Lmax)
    Wmax = int(W) * (rr ** Lmax)
    dx_max = dx0 / (float(rr) ** Lmax)
    dy_max = dy0 / (float(rr) ** Lmax)

    # occupancy grid storing ORIGINAL cell id
    occ = np.full((Hmax, Wmax), fill_value=-1, dtype=np.int32)

    c = centers.detach().cpu().to(torch.float32).numpy()
    l = levels_i64.detach().cpu().numpy()

    # Rasterize: stamp each AMR cell id into the max-grid region it occupies
    for cid in range(N):
        L = int(l[cid])
        scale = rr ** (Lmax - L)  # block size in max-grid pixels along each axis

        dxL = dx0 / (float(rr) ** L)
        dyL = dy0 / (float(rr) ** L)

        # integer index in the level-L grid
        ixL = int(np.floor((c[cid, 0] - xmin) / dxL))
        iyL = int(np.floor((c[cid, 1] - ymin) / dyL))

        # clamp for numerical edge cases at domain boundary
        ixL = max(0, min(ixL, (W * (rr ** L)) - 1))
        iyL = max(0, min(iyL, (H * (rr ** L)) - 1))

        ix0 = ixL * scale
        iy0 = iyL * scale

        occ[iy0:iy0 + scale, ix0:ix0 + scale] = cid

    # Helper: aggregate boundary pixel transitions into (src,dst,face_len)
    def _agg_directed(src_ids: np.ndarray, dst_ids: np.ndarray, seg_len: float):
        # Count how many max-grid segments each directed pair owns
        keys = src_ids.astype(np.int64) * N + dst_ids.astype(np.int64)
        uniq, counts = np.unique(keys, return_counts=True)
        src_u = (uniq // N).astype(np.int64)
        dst_u = (uniq %  N).astype(np.int64)
        face_len = counts.astype(np.float32) * float(seg_len)
        return src_u, dst_u, face_len

    # Horizontal scan: left pixel vs right pixel -> vertical face segments (length dy_max)
    left = occ[:, :-1]
    right = occ[:, 1:]
    m = (left != right) & (left >= 0) & (right >= 0)
    src_lr = left[m]
    dst_lr = right[m]
    # directed pairs: left->right and right->left
    s1, d1, fl1 = _agg_directed(src_lr, dst_lr, dy_max)
    s2, d2, fl2 = _agg_directed(dst_lr, src_lr, dy_max)

    # Vertical scan: bottom pixel vs top pixel -> horizontal face segments (length dx_max)
    bot = occ[:-1, :]
    top = occ[1:, :]
    m = (bot != top) & (bot >= 0) & (top >= 0)
    src_bt = bot[m]
    dst_bt = top[m]
    # directed pairs: bottom->top and top->bottom
    s3, d3, fl3 = _agg_directed(src_bt, dst_bt, dx_max)
    s4, d4, fl4 = _agg_directed(dst_bt, src_bt, dx_max)

    src_all = np.concatenate([s1, s2, s3, s4], axis=0)
    dst_all = np.concatenate([d1, d2, d3, d4], axis=0)
    fl_all  = np.concatenate([fl1, fl2, fl3, fl4], axis=0)

    # Remove self-loops (should not happen, but safe)
    keep = (src_all != dst_all)
    src_all = src_all[keep]
    dst_all = dst_all[keep]
    fl_all  = fl_all[keep]

    if src_all.size == 0:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
        if return_edge_attr:
            ea = torch.empty((0, 5), dtype=torch.float32, device=device)
            return ei, ea
        return ei

    # Final dedup: sum face lengths if duplicates exist
    keys = src_all.astype(np.int64) * N + dst_all.astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    if uniq.size != keys.size:
        fl_sum = np.zeros((uniq.size,), dtype=np.float32)
        np.add.at(fl_sum, inv, fl_all)
        src_all = (uniq // N).astype(np.int64)
        dst_all = (uniq %  N).astype(np.int64)
        fl_all  = fl_sum

    edge_index = torch.as_tensor(
        np.stack([src_all, dst_all], axis=0),
        dtype=torch.int64,
        device=device,
    )

    if not return_edge_attr:
        return edge_index

    # Edge geometry for operator-style message passing
    c_t = centers.to(torch.float32)
    src_t = edge_index[0]
    dst_t = edge_index[1]
    delta = c_t[dst_t] - c_t[src_t]
    dist = torch.linalg.norm(delta, dim=1).clamp_min(1e-12)

    # For axis-aligned dyadic quads, neighbor centers differ predominantly in x or y.
    # Use the dominant component to set the outward normal for src->dst.
    use_x = (delta[:, 0].abs() >= delta[:, 1].abs())
    nx = torch.zeros((edge_index.shape[1],), dtype=torch.float32, device=device)
    ny = torch.zeros((edge_index.shape[1],), dtype=torch.float32, device=device)
    nx[use_x] = torch.sign(delta[use_x, 0])
    ny[~use_x] = torch.sign(delta[~use_x, 1])

    face_len = torch.as_tensor(fl_all, dtype=torch.float32, device=device)
    w_diff = face_len / dist

    edge_attr = torch.stack([nx, ny, face_len, dist, w_diff], dim=1)  # (E,5)
    return edge_index, edge_attr

@torch.no_grad()
def build_amr_local_knn_edges(
    centers: torch.Tensor,          # (N,2)
    parents_flat: torch.Tensor,     # (N,) parent idx in [0, H*W) (>=0 valid)
    H: int, W: int,
    k_local: int = 4,
    max_local: int = 2048,
):
    """
    Build edges by restricting KNN to each coarse parent and its 4-neighborhood.
    Returns undirected, self-loop free, deduplicated (2,E) int64 edge_index.
    MPS-safe (no unique(dim=...)).
    """
    out_device = centers.device
    N = int(centers.size(0))
    if N == 0:
        return torch.empty(2, 0, dtype=torch.long, device=out_device)

    # On accelerator backends (cuda/mps), this routine is Python-loop heavy and
    # uses many small cdist/topk blocks; CPU execution is often faster and safer.
    use_cpu = (out_device.type != "cpu")
    work_device = torch.device("cpu") if use_cpu else out_device
    centers_w = centers.to(device=work_device, dtype=torch.float32)
    pf = parents_flat.to(device=work_device, dtype=torch.long).view(-1)

    valid_mask = (pf >= 0) & (pf < (H * W))
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    if valid_idx.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=out_device)

    pf_valid = pf.index_select(0, valid_idx)
    order = torch.argsort(pf_valid)
    pf_sorted = pf_valid.index_select(0, order)
    idx_sorted = valid_idx.index_select(0, order)

    counts = torch.bincount(pf_sorted, minlength=H * W)
    offsets = torch.zeros(H * W + 1, dtype=torch.long, device=work_device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    active_parents = torch.nonzero(counts > 0, as_tuple=False).view(-1)

    def _slice_for_parent(p: int) -> torch.Tensor:
        s = int(offsets[p].item())
        e = int(offsets[p + 1].item())
        if e <= s:
            return idx_sorted.new_empty((0,), dtype=torch.long)
        return idx_sorted[s:e]

    def _neighbor_parents(p: int) -> list[int]:
        r, c = divmod(p, W)
        out = [p]
        if c > 0:
            out.append(p - 1)
        if c + 1 < W:
            out.append(p + 1)
        if r > 0:
            out.append(p - W)
        if r + 1 < H:
            out.append(p + W)
        return out

    rows_parts: list[torch.Tensor] = []
    cols_parts: list[torch.Tensor] = []
    k_take = int(max(1, k_local + 1))

    for p in active_parents.tolist():
        src_idx = _slice_for_parent(int(p))
        if src_idx.numel() == 0:
            continue

        cand_parts = []
        for q in _neighbor_parents(int(p)):
            part = _slice_for_parent(int(q))
            if part.numel() > 0:
                cand_parts.append(part)
        if len(cand_parts) == 0:
            continue
        cand_idx = torch.cat(cand_parts, dim=0)
        if cand_idx.numel() <= 1:
            continue
        if int(cand_idx.numel()) > int(max_local):
            cand_idx = cand_idx[: int(max_local)]

        S = centers_w.index_select(0, src_idx)   # (S,2)
        C = centers_w.index_select(0, cand_idx)  # (M,2)
        D = torch.cdist(S, C)                    # (S,M)
        k_eff = min(k_take, int(C.size(0)))
        nbr_pos = torch.topk(D, k_eff, largest=False).indices  # (S,k_eff)

        nbr_idx = cand_idx.index_select(0, nbr_pos.reshape(-1)).view(src_idx.size(0), k_eff)
        src_rep = src_idx.view(-1, 1).expand(-1, k_eff)
        keep = (nbr_idx != src_rep)
        if keep.any():
            rows_parts.append(src_rep[keep])
            cols_parts.append(nbr_idx[keep])

    if len(rows_parts) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=out_device)

    u0 = torch.cat(rows_parts, dim=0).to(torch.long)
    v0 = torch.cat(cols_parts, dim=0).to(torch.long)

    # make undirected
    u = torch.cat([u0, v0], dim=0)
    v = torch.cat([v0, u0], dim=0)

    # drop self loops
    keep = (u != v)
    if not keep.any():
        return torch.empty(2, 0, dtype=torch.long, device=out_device)
    u = u[keep]
    v = v[keep]

    # dedup directed edges via 1-D keys (CPU unique is robust/safe)
    keys = (u.to(torch.int64) * int(N) + v.to(torch.int64)).to("cpu")
    keys = torch.unique(keys)
    u = (keys // int(N)).to(out_device, torch.long)
    v = (keys % int(N)).to(out_device, torch.long)
    return torch.stack([u, v], dim=0)


@torch.no_grad()
def _build_pred_mesh_from_gt_gradients(
    ex: dict,
    cfg: dict,
    H: int,
    W: int,
    dx: float,
    dy: float,
    device,
):
    """
    Predict the mesh at t+1 using the gradient-based AMR policy.

    IMPORTANT:
      - This function is *rectangular-domain only*: it knows nothing about the
        wedge geometry.  Wedge clipping is handled upstream in
        precompute_pred_mesh_and_interps_for_rollout via _clip_pred_mesh_to_wedge.
      - It returns:
          pred_centers : (N,2)  physical centers
          pred_levels  : (N,)   refinement level of each leaf
          parent_flat  : (N,)   coarse parent index in [0, H*W)
          pred_ei      : (2,E)  edge_index (AMR-local KNN)
          mask_pred_parent : (H,W) bool, mask of which coarse parents are occupied
    """
    dev = torch.device(device)
    rr = _get_refine_ratio(cfg)
    gt_rr = int(ex.get("gt_refine_ratio", rr)) if isinstance(ex, dict) else int(rr)
    if gt_rr < 2:
        gt_rr = int(rr)

    # -------------------------------
    # 1. GT(t) centers and features
    # -------------------------------
    centers_t = ex["centers_t"]
    if not torch.is_tensor(centers_t):
        centers_t = torch.as_tensor(centers_t)
    centers_t = centers_t.to(dev)

    feat_t = ex.get("center_feat_t", ex.get("dyn_feat_t"))
    if feat_t is None:
        raise KeyError("Expected 'center_feat_t' or 'dyn_feat_t' in dataset sample.")
    if not torch.is_tensor(feat_t):
        feat_t = torch.as_tensor(feat_t)
    feat_t = feat_t.to(dev).to(torch.float32)

    # -------------------------------
    # 2. Parents on coarse (H,W) grid
    # -------------------------------
    if "dyn_parents" in ex and ex["dyn_parents"] is not None:
        parents_t = ex["dyn_parents"].to(dev)
    elif "level_t" in ex and ex["level_t"] is not None and \
       "ij_t" in ex and ex["ij_t"] is not None:
        parents_t = _parents_from_level_ij(
            ex["level_t"].to(dev),
            ex["ij_t"].to(dev),
            H,
            W,
            refine_ratio=gt_rr,
        )
    else:
        xmin, xmax, ymin, ymax = cfg["data"]["bbox"]
        parents_t = parents_from_pos(
            centers_t,
            H,
            W,
            xmin,
            xmax,
            ymin,
            ymax,
        ).to(dev)

    # Coarse mask at t
    if "mask_t" in ex and ex["mask_t"] is not None:
        mask_t_parent = ex["mask_t"].to(dev).view(H, W)
    else:
        mask_t_parent = _mask_from_parents(parents_t, H, W)

    # -------------------------------
    # 3. Build batch for policy call
    # -------------------------------
    batch_like = {
        "centers_t":     centers_t,
        "center_feat_t": feat_t,
        "dyn_feat_t":    feat_t,
        "dyn_parents":   parents_t,
        "mask_t":        mask_t_parent.view(-1),
    }
    if "level_t" in ex:
        batch_like["level_t"] = ex["level_t"]
    if "ij_t" in ex:
        batch_like["ij_t"] = ex["ij_t"]
    if "ei_t" in ex and ex["ei_t"] is not None:
        batch_like["ei_t"] = ex["ei_t"]

    # -------------------------------
    # 4. Gradient-based masks
    # -------------------------------
    dbg_cfg = cfg.get("debug", {}) or {}
    # Keep precompute fast by default: only enable per-step gradient capture/plots
    # when explicitly requested.
    want_grad_plots = bool(dbg_cfg.get("plot_gradients", False)) and bool(
        dbg_cfg.get("plot_gradients_during_precompute", False)
    )
    grad_debug = {} if want_grad_plots else None

    masks_pred_by_level = predict_masks_hierarchical_from_gt_gradients(
        batch_like,
        cfg,
        H,
        W,
        dx,
        dy,
        device=device,
        debug_out=grad_debug,
    )

    if want_grad_plots and isinstance(grad_debug, dict):
        t_idx = ex.get("t", None)
        if torch.is_tensor(t_idx):
            t_idx = int(t_idx.item())
        debug_plot_policy_gradients(
            grad_debug,
            cfg=cfg,
            t=t_idx,
            use_pooled_up=False,   
            out_dir="./debug_grad_plots",
            prefix="policy_grads",
        )
    # -------------------------------
    # 5. Masks -> dynamic leaf mesh
    # -------------------------------
    xmin, xmax, ymin, ymax = cfg["data"]["bbox"]
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

    # Flatten parents to [N] indices on level-0 grid
    if pred_parents.ndim == 2 and pred_parents.size(1) == 2:
        # pred_parents is (i,j) at level-0 resolution
        parent_flat = pred_parents[:, 1].long() * W + pred_parents[:, 0].long()
    else:
        parent_flat = pred_parents.view(-1).long()

    # -------------------------------
    # 6. Ensure edges exist
    # -------------------------------
    edge_method = str(cfg.get("edges", {}).get("method", "knn")).lower()
    #print(f"[precompute] building pred_ei using method='{edge_method}'")
    if ("face" in edge_method):
        pred_ei, pred_ea = build_amr_face_adjacency_edges(
            pred_centers,
            pred_levels,
            H,
            W,
            bbox=tuple(cfg["data"]["bbox"]),
            return_edge_attr=True,  # keep identical downstream contract for now
            refine_ratio=rr,
        )
        pred_ei = pred_ei.to(dev, dtype=torch.long)
        pred_ea = pred_ea.to(dev, dtype=torch.float32)      
    else:
        pred_ei = build_amr_local_knn_edges(
            pred_centers,
            parent_flat,
            H,
            W,
            k_local=int(cfg.get("edges", {}).get("k_local", 4)),
            max_local=int(cfg.get("edges", {}).get("max_local", 2048)),
        )
        pred_ei = pred_ei.to(dev).long()

    if pred_ei.ndim != 2 or pred_ei.size(0) != 2 or pred_ei.size(1) == 0:
        sid = ex.get("t", None)
        sid_int = int(sid.item()) if torch.is_tensor(sid) else int(sid) if sid is not None else -1
        raise RuntimeError(f"[precompute] empty pred_ei for sample t={sid_int}")

    # -------------------------------
    # 7. Parent-grid occupancy mask
    # -------------------------------
    mask_pred_parent = _mask_from_parents(parent_flat, H, W)

    return pred_centers, pred_levels, parent_flat, pred_ei, mask_pred_parent


@torch.no_grad()
def _map_gt_on_pred_mesh_once(
    *,
    src_centers, src_feats,            # centers/features on GT mesh
    mask_src_parent, parents_src,      # (H,W) mask of active parents & parent indices for src
    pred_centers, pred_levels, pred_parents, mask_pred_parent,
    H, W, device,
    knn_k: int = 8,
    knn_chunk: int = 8192,
    knn_backend: str = "exact",
    knn_backend_kwargs: dict | None = None,
):

    src_feats = src_feats.to(device)
    feat_on_pred, _stat = _targeted_map_to_pred(
        pred_centers=pred_centers,
        pred_levels=pred_levels,
        pred_parents=pred_parents,
        mask_pred=mask_pred_parent,
        src_centers=src_centers,
        src_feats=src_feats,
        H=H, W=W,
        #mask_src_parent=mask_src_parent,
        mask_src_parent=None,
        #src_parent_feats=Xc,
        src_parent_feats=None,
        knn_k=int(knn_k),
        chunk=int(knn_chunk),
        knn_backend=str(knn_backend),
        knn_backend_kwargs=dict(knn_backend_kwargs or {}),
    )
    return feat_on_pred


def make_collate_with_precompute(pre_dir: str):
    def _collate(samples):
        ex = samples[0]  # bs=1
        sid = int(ex["t"].item()) if torch.is_tensor(ex["t"]) else int(ex["t"])
        p = os.path.join(pre_dir, f"{sid:07d}.pt")
        if os.path.exists(p):
            blob = torch.load(p, map_location="cpu", weights_only=False)
            # ---- normalize types right here ----
            if "pred_ei" in blob:
                blob["pred_ei"] = blob["pred_ei"].to(torch.long).contiguous()
            if "mask_pred_parent" in blob and blob["mask_pred_parent"].dtype != torch.bool:
                blob["mask_pred_parent"] = (blob["mask_pred_parent"] > 0)
            ex.update(blob)
        return ex
    return _collate


@torch.no_grad()
def _load_wedge_path_from_spec(mesh_spec_path: str, cfg: Dict[str, Any]):
    """
    Load the wedge polygon from wedge_mesh_spec.pt and return a
    matplotlib.path.Path object, rescaled into the same coordinate
    system as cfg["data"]["bbox"].

    We infer the native wedge extents directly from outer_polygon_ccw,
    rather than relying on spec["left"/"right"/"bottom"/"top"].
    """
    import numpy as np

    if mesh_spec_path is None:
        raise ValueError("mesh_spec_path is None in _load_wedge_path_from_spec")

    # Load the saved spec
    spec = torch.load(mesh_spec_path, map_location="cpu", weights_only=False)
    if "outer_polygon_ccw" not in spec:
        raise KeyError(
            f"Expected key 'outer_polygon_ccw' in mesh spec {mesh_spec_path}, "
            f"found keys={list(spec.keys())}"
        )

    outer = spec["outer_polygon_ccw"]
    if isinstance(outer, torch.Tensor):
        outer = outer.detach().cpu().numpy()
    else:
        outer = np.asarray(outer, dtype=np.float64)

    if outer.ndim != 2 or outer.shape[1] != 2:
        raise ValueError(
            f"'outer_polygon_ccw' must be (M,2), got shape {outer.shape}"
        )

    # Native wedge extents inferred from the polygon itself
    left   = float(outer[:, 0].min())
    right  = float(outer[:, 0].max())
    bottom = float(outer[:, 1].min())
    top    = float(outer[:, 1].max())

    # Target extents from the training config
    xmin, xmax, ymin, ymax = cfg["data"]["bbox"]

    dx_src = right  - left
    dy_src = top    - bottom
    if dx_src == 0.0 or dy_src == 0.0:
        raise ValueError(
            f"Degenerate wedge extents inferred from outer_polygon_ccw: "
            f"left={left}, right={right}, bottom={bottom}, top={top}"
        )

    dx_tgt = xmax - xmin
    dy_tgt = ymax - ymin

    # Affine map: [left,right] x [bottom,top] -> [xmin,xmax] x [ymin,ymax]
    outer_scaled = outer.copy()
    outer_scaled[:, 0] = xmin + (outer[:, 0] - left)   / dx_src * dx_tgt
    outer_scaled[:, 1] = ymin + (outer[:, 1] - bottom) / dy_src * dy_tgt

    return Path(outer_scaled)


@torch.no_grad()
def _clip_pred_mesh_to_wedge(
    pred_centers: torch.Tensor,     # (N,2), physical coordinates
    pred_levels: torch.Tensor,      # (N,)
    pred_parents: torch.Tensor,     # (N,) parent indices in [0, H*W)
    pred_ei: torch.Tensor,          # (2,E) edge_index
    H: int,
    W: int,
    wedge_path,                     # matplotlib.path.Path or None
    cfg: Dict[str, Any],
    device: torch.device | str,
    timing_out: Dict[str, float] | None = None,
):
    """
    Restrict a predicted mesh to the wedge domain, with optional boundary refinement.

    New behavior (when wedge_path is not None):
      - Keep cells that intersect the wedge (center-in OR any-corner-in).
      - Refine boundary-intersecting cells until they are fully inside (all corners in),
        up to policy.max_level.
      - Drop any remaining cells that are not fully inside at max_level.
      - Rebuild edges and parent mask.
    """
    dev = torch.device(device)

    timing = {
        "lookup_classify_s": 0.0,
        "lookup_refine_s": 0.0,
        "edge_build_s": 0.0,
        "legacy_geom_s": 0.0,
    }

    # Normalize parents to flat long
    parent_flat = pred_parents.view(-1).long().to(pred_centers.device)

    if wedge_path is None:
        if not torch.is_tensor(pred_ei):
            pred_ei = torch.as_tensor(pred_ei, dtype=torch.long, device=dev)
        else:
            pred_ei = pred_ei.to(dev, dtype=torch.long)
        mask_pred_parent = _mask_from_parents(parent_flat, H, W)
        if isinstance(timing_out, dict):
            timing_out.clear()
            timing_out.update(timing)
        return pred_centers, pred_levels, parent_flat, pred_ei, mask_pred_parent

    pol = cfg.get("policy", {})
    Lmax = int(pol.get("max_level", 3))
    rr = _get_refine_ratio(cfg)
    bbox = tuple(cfg["data"]["bbox"])

    # Small positive radius expands the polygon slightly so boundary points count "inside"
    # (important to avoid over-pruning/refining due to numerical edge cases).
    xspan = float(bbox[1] - bbox[0])
    yspan = float(bbox[3] - bbox[2])
    radius = float(pol.get("wedge_clip_radius", 1e-9 * max(xspan, yspan)))

    rt_cfg = cfg.get("train", {}).get("runtime_mesh", {}) or {}
    use_lookup = bool(rt_cfg.get("wedge_clip_use_lookup", True))
    did_fast_clip = False

    if use_lookup:
        t_cls0 = time.perf_counter()
        lookup = _get_wedge_clip_level_lookup(
            wedge_path=wedge_path,
            H=H,
            W=W,
            Lmax=Lmax,
            refine_ratio=rr,
            bbox=bbox,
            radius=radius,
        )
        full_by_level, intersect_by_level = _lookup_level_masks_on_device(lookup, pred_centers.device)

        row, col = _cell_level_ij_from_centers(
            centers=pred_centers,
            levels=pred_levels,
            H=H,
            W=W,
            bbox=bbox,
            refine_ratio=rr,
        )
        lvl = pred_levels.view(-1).long()

        # ---------- Step 1 (lookup): keep cells that intersect wedge ----------
        keep0 = _lookup_mask_values_by_level(
            intersect_by_level, levels=lvl, row=row, col=col
        )
        if not keep0.any():
            raise RuntimeError(
                "Wedge clipping removed all predicted cells. "
                "Check wedge_mesh_spec.pt vs cfg['data']['bbox']."
            )

        row = row[keep0]
        col = col[keep0]
        lvl = lvl[keep0]
        parent_flat = parent_flat[keep0]
        timing["lookup_classify_s"] += float(time.perf_counter() - t_cls0)

        # ---------- Step 2 (lookup): refine boundary-crossing cells ----------
        t_ref0 = time.perf_counter()
        for _ in range(Lmax + 1):
            fully_inside = _lookup_mask_values_by_level(
                full_by_level, levels=lvl, row=row, col=col
            )
            crossing = ~fully_inside
            refine_mask = crossing & (lvl < Lmax)
            if not refine_mask.any():
                break

            row, col, lvl, parent_flat = _refine_cells_one_level_ij(
                row=row,
                col=col,
                levels=lvl,
                parents=parent_flat,
                refine_mask=refine_mask,
                refine_ratio=rr,
            )
        timing["lookup_refine_s"] += float(time.perf_counter() - t_ref0)

        # ---------- Step 3 (lookup): final strict keep ----------
        t_cls1 = time.perf_counter()
        fully_inside = _lookup_mask_values_by_level(
            full_by_level, levels=lvl, row=row, col=col
        )
        row = row[fully_inside]
        col = col[fully_inside]
        lvl = lvl[fully_inside]
        parent_flat = parent_flat[fully_inside]

        if row.numel() == 0:
            raise RuntimeError(
                "After boundary refinement, no fully-inside cells remained. "
                "This usually means Lmax is too low for the wedge geometry or the wedge/bbox mismatch."
            )

        pred_levels = lvl
        pred_centers = _centers_from_level_ij(
            levels=pred_levels,
            row=row,
            col=col,
            H=H,
            W=W,
            bbox=bbox,
            refine_ratio=rr,
        ).to(pred_centers.device, dtype=pred_centers.dtype)
        did_fast_clip = True
        timing["lookup_classify_s"] += float(time.perf_counter() - t_cls1)

    if not did_fast_clip:
        t_legacy0 = time.perf_counter()
        # ---------- Step 1: keep cells that intersect the wedge ----------
        center_inside = _path_contains_points_torch(wedge_path, pred_centers, radius=radius)

        corners = _cell_corners_from_centers_levels(
            pred_centers, pred_levels, H, W, bbox, refine_ratio=rr
        )  # (N,4,2)
        corner_inside = _path_contains_points_torch(wedge_path, corners.reshape(-1, 2), radius=radius).view(-1, 4)
        any_corner_inside = corner_inside.any(dim=1)

        keep0 = center_inside | any_corner_inside
        if not keep0.any():
            raise RuntimeError(
                "Wedge clipping removed all predicted cells. "
                "Check wedge_mesh_spec.pt vs cfg['data']['bbox']."
            )

        pred_centers = pred_centers[keep0]
        pred_levels  = pred_levels[keep0]
        parent_flat  = parent_flat[keep0]

        # ---------- Step 2: boundary refinement until cells are fully inside ----------
        # A cell is "fully inside" if all 4 corners are inside.
        # Refine any cell that is not fully inside, up to Lmax.
        for _ in range(Lmax + 1):
            corners = _cell_corners_from_centers_levels(
                pred_centers, pred_levels, H, W, bbox, refine_ratio=rr
            )
            corner_inside = _path_contains_points_torch(wedge_path, corners.reshape(-1, 2), radius=radius).view(-1, 4)
            fully_inside = corner_inside.all(dim=1)

            # Cells that still cross boundary
            crossing = ~fully_inside

            # Refine those that can still be refined
            refine_mask = crossing & (pred_levels < Lmax)

            if not refine_mask.any():
                break

            pred_centers, pred_levels, parent_flat = _refine_cells_one_level(
                pred_centers, pred_levels, parent_flat,
                refine_mask=refine_mask,
                H=H, W=W, bbox=bbox,
                refine_ratio=rr,
            )

        # ---------- Step 3: final strict keep (no cells outside wedge) ----------
        corners = _cell_corners_from_centers_levels(
            pred_centers, pred_levels, H, W, bbox, refine_ratio=rr
        )
        corner_inside = _path_contains_points_torch(wedge_path, corners.reshape(-1, 2), radius=radius).view(-1, 4)
        fully_inside = corner_inside.all(dim=1)

        pred_centers = pred_centers[fully_inside]
        pred_levels  = pred_levels[fully_inside]
        parent_flat  = parent_flat[fully_inside]

        if pred_centers.numel() == 0:
            raise RuntimeError(
                "After boundary refinement, no fully-inside cells remained. "
                "This usually means Lmax is too low for the wedge geometry or the wedge/bbox mismatch."
            )
        timing["legacy_geom_s"] += float(time.perf_counter() - t_legacy0)

    # ---------- Step 4: rebuild edges + mask ----------
    t_edge0 = time.perf_counter()
    k_local   = int(cfg.get("edges", {}).get("k_local", 4))
    max_local = int(cfg.get("edges", {}).get("max_local", 2048))

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
        )
        pred_ei = pred_ei.to(dev, dtype=torch.long)
    else:
        pred_ei = build_amr_local_knn_edges(
            pred_centers,
            parent_flat,
            H,
            W,
            k_local=k_local,
            max_local=max_local,
        ).to(dev, dtype=torch.long)
    if pred_ei.numel() == 0 or int(pred_ei.shape[1]) == 0:
        raise RuntimeError("Empty edge_index after wedge clipping/refinement.")
    timing["edge_build_s"] += float(time.perf_counter() - t_edge0)

    mask_pred_parent = _mask_from_parents(parent_flat, H, W)
    if isinstance(timing_out, dict):
        timing_out.clear()
        timing_out.update(timing)
    return pred_centers, pred_levels, parent_flat, pred_ei, mask_pred_parent


def debug_print_precomp_h5_meta(path: str):

    print(f"[DEBUG H5] opening: {path}")
    with h5py.File(path, "r") as f:
        if "meta" not in f:
            print("[DEBUG H5] missing group: /meta")
            print("[DEBUG H5] top-level keys:", list(f.keys()))
            return

        meta = f["meta"]
        print("[DEBUG H5] meta attrs:")
        for k, v in meta.attrs.items():
            # make bytes readable
            if isinstance(v, (bytes, np.bytes_)):
                try:
                    v = v.decode("utf-8")
                except Exception:
                    pass
            print(f"  - {k}: {v}")

        print("[DEBUG H5] top-level keys:", list(f.keys())[:20], "..." if len(f.keys()) > 20 else "")
        # If you store timesteps under a group, print that too
        if "steps" in f:
            print("[DEBUG H5] /steps keys count:", len(f["steps"].keys()))
            # print a few keys
            ks = sorted(list(f["steps"].keys()))
            print("[DEBUG H5] /steps first/last:", ks[:3], ks[-3:])

@torch.no_grad()
def precompute_pred_mesh_and_interps_for_rollout(
    steps,
    cfg,
    H: int,
    W: int,
    dx: float,
    dy: float,
    device: str = "cpu",
    progress: bool = True,
    cache_path: str | None = None,
    force_recompute: bool = False,
):
    """
    Streaming precompute to ONE HDF5 file (no in-memory accumulation).

    Geometry modes (cfg['mesh']['geometry_mode']):
      1) "gt_gradients" (default): per-step predicted mesh from GT feature gradients (existing behavior)
      2) "starting_refine": static mesh from starting_mesh_path, deterministically refined to a target level

    In "starting_refine" mode:
      - The SAME pred mesh is used for every dst=t+1 group.
      - pred2pred maps are written as identity maps (since pred(t) == pred(t+1) geometry).
    """
    import numpy as np

    try:
        import h5py
    except ImportError as e:
        raise ImportError("This streaming mode requires h5py. Install with: pip install h5py") from e

    import torch

    # ----------------- cfg helpers -----------------
    def _cfg_sha1(_cfg: dict) -> str:
        s = json.dumps(_cfg, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(s).hexdigest()
    
    dbg = cfg.get("debug", {})
    refine_ratio = _get_refine_ratio(cfg)

    def _h5_is_usable(path: str, cfg: dict, expected_T: int) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with h5py.File(path, "r") as f:
                if "meta" not in f:
                    return False
                T_in = int(f["meta"].attrs.get("T", -1))
                if T_in != int(expected_T):
                    print("[H5 CHECK] reject: missing groups or wrong T")
                    return False
                for t in range(1, expected_T):
                    if f"t{t:05d}" not in f:
                        print("[H5 CHECK] reject: missing groups")
                        return False
                return True
        except Exception:
            print("[H5 CHECK] reject: unreadable")
            return False

    def _ensure_group(f: "h5py.File", t: int):
        gname = f"t{int(t):05d}"
        if gname in f:
            return f[gname]
        return f.create_group(gname)

    def _write_ds(g, name: str, arr: np.ndarray, *, compress=True):
        if name in g:
            del g[name]
        kwargs = {}
        if compress:
            kwargs = dict(compression="gzip", compression_opts=4, shuffle=True, chunks=True)
        g.create_dataset(name, data=arr, **kwargs)

    def _to_np(x: torch.Tensor, dtype=None) -> np.ndarray:
        a = x.detach().cpu().numpy()
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        return a

    def _read_centers(f: "h5py.File", t: int) -> torch.Tensor:
        g = f[f"t{int(t):05d}"]
        return torch.from_numpy(g["pred_centers"][...]).to(device=torch.device("cpu"), dtype=torch.float32)

    # ----------------- load starting mesh + deterministic refinement -----------------
    def _load_starting_mesh_from_spec(mesh_spec_path: str):
        spec = torch.load(mesh_spec_path, map_location="cpu")
        if not isinstance(spec, dict):
            raise RuntimeError(f"Expected dict in mesh spec, got: {type(spec)}")

        def _pick(*names):
            for n in names:
                if n in spec:
                    return spec[n]
            return None

        centers = _pick("pred_centers", "centers", "pos", "centers_u", "centers_selected")
        levels  = _pick("pred_levels", "levels", "level", "levels_selected")
        parents = _pick("pred_parents", "parents", "parent", "dyn_parents")
        ei      = _pick("pred_ei", "edge_index", "ei")

        if centers is None or levels is None:
            raise RuntimeError(
                f"Mesh spec missing centers/levels. Keys found: {sorted(list(spec.keys()))}"
            )

        centers = centers if torch.is_tensor(centers) else torch.as_tensor(centers)
        levels  = levels  if torch.is_tensor(levels)  else torch.as_tensor(levels)

        centers = centers.to(dtype=torch.float32)
        if levels.dtype not in (torch.int16, torch.int32, torch.int64):
            levels = levels.to(torch.int64)

        # Parents are optional in the spec; if absent, compute coarse parents from pos.
        if parents is None:
            xmin, xmax, ymin, ymax = cfg["data"]["bbox"]
            parents = parents_from_pos(centers, H, W, xmin, xmax, ymin, ymax)
        else:
            parents = parents if torch.is_tensor(parents) else torch.as_tensor(parents)
            if parents.dtype not in (torch.int16, torch.int32, torch.int64):
                parents = parents.to(torch.int64)

        if ei is not None:
            ei = ei if torch.is_tensor(ei) else torch.as_tensor(ei)
            if ei.dtype not in (torch.int32, torch.int64):
                ei = ei.to(torch.int64)
        return centers, levels, parents, ei

    def _refine_cells_one_level(centers: torch.Tensor, levels: torch.Tensor, parents: torch.Tensor):
        """
        Refine every cell in (centers, levels, parents) by one level
        (split into refine_ratio^2 children).
        Returns children arrays.
        """
        xmin, xmax, ymin, ymax = map(float, cfg["data"]["bbox"])

        L = levels.to(torch.int64)
        rr = int(refine_ratio)
        rr_t = torch.tensor(float(rr), device=L.device)

        # cell size at current level: dx_L = dx / rr^L, dy_L = dy / rr^L
        dx_L = (torch.tensor(float(dx)) / torch.pow(rr_t, L.to(torch.float32))).to(torch.float32)
        dy_L = (torch.tensor(float(dy)) / torch.pow(rr_t, L.to(torch.float32))).to(torch.float32)

        c = centers.to(torch.float32)
        x0 = c[:, 0]
        y0 = c[:, 1]

        frac = ((torch.arange(rr, dtype=torch.float32, device=c.device) + 0.5) / float(rr)) - 0.5
        gx, gy = torch.meshgrid(frac, frac, indexing="xy")

        child = torch.empty((c.shape[0], rr, rr, 2), dtype=torch.float32, device=c.device)
        child[..., 0] = x0.view(-1, 1, 1) + dx_L.view(-1, 1, 1) * gx.view(1, rr, rr)
        child[..., 1] = y0.view(-1, 1, 1) + dy_L.view(-1, 1, 1) * gy.view(1, rr, rr)

        child_centers = child.reshape(-1, 2)
        child_count = rr * rr
        child_levels = (L + 1).repeat_interleave(child_count)
        child_parents = parents.repeat_interleave(child_count)

        return child_centers, child_levels, child_parents

    def _refine_mesh_to_target(centers, levels, parents, target_level: int, policy: str):
        """
        Deterministic refinement:
          - policy="min_level": refine every cell with level < target until all >= target
          - policy="level0_only": refine only cells that started at level==0 up to target
        """
        target_level = int(target_level)
        if target_level <= 0:
            return centers, levels, parents

        centers = centers.clone()
        levels  = levels.clone().to(torch.int64)
        parents = parents.clone().to(torch.int64)

        if policy not in ("min_level", "level0_only"):
            raise ValueError(f"Unknown starting_refine_policy='{policy}'")

        if policy == "level0_only":
            base_is_level0 = (levels == 0)

        # iterative refine
        while True:
            if policy == "min_level":
                need = (levels < target_level)
            else:
                need = base_is_level0 & (levels < target_level)

            if not bool(need.any().item()):
                break

            keep = ~need
            c_keep = centers[keep]
            l_keep = levels[keep]
            p_keep = parents[keep]

            c_need = centers[need]
            l_need = levels[need]
            p_need = parents[need]

            c_child, l_child, p_child = _refine_cells_one_level(c_need, l_need, p_need)

            centers = torch.cat([c_keep, c_child], dim=0)
            levels  = torch.cat([l_keep, l_child], dim=0)
            parents = torch.cat([p_keep, p_child], dim=0)

            if policy == "level0_only":
                # base_is_level0 must track new children of originally level0 cells
                base_keep = base_is_level0[keep]
                base_child = torch.ones((c_child.shape[0],), dtype=torch.bool, device=base_keep.device)
                base_is_level0 = torch.cat([base_keep, base_child], dim=0)

        return centers, levels, parents

    def _build_edge_index_from_maxgrid_partition(centers: torch.Tensor, levels: torch.Tensor):
        """
        Build face-adjacency edges by rasterizing the AMR partition onto the max-level grid.

        Works for axis-aligned AMR meshes with integer refine_ratio.
        """
        xmin, xmax, ymin, ymax = map(float, cfg["data"]["bbox"])
        rr = int(refine_ratio)

        levels_i64 = levels.to(torch.int64)
        Lmax = int(levels_i64.max().item()) if levels_i64.numel() > 0 else 0
        if Lmax > 6:
            raise RuntimeError(f"Lmax={Lmax} is too large for max-grid raster adjacency (guard).")

        Hmax = int(H) * (rr ** Lmax)
        Wmax = int(W) * (rr ** Lmax)

        # occupancy grid of cell IDs at max resolution
        occ = np.full((Hmax, Wmax), fill_value=-1, dtype=np.int32)

        c = centers.detach().cpu().to(torch.float32).numpy()
        l = levels_i64.detach().cpu().numpy()

        for cid in range(c.shape[0]):
            L = int(l[cid])
            scale = rr ** (Lmax - L)

            dxL = float(dx) / (float(rr) ** L)
            dyL = float(dy) / (float(rr) ** L)

            # integer indices at level L grid
            ixL = int(np.floor((c[cid, 0] - xmin) / dxL))
            iyL = int(np.floor((c[cid, 1] - ymin) / dyL))

            ixL = max(0, min(ixL, (W * (rr ** L)) - 1))
            iyL = max(0, min(iyL, (H * (rr ** L)) - 1))

            ix0 = ixL * scale
            iy0 = iyL * scale

            # fill block at max level
            occ[iy0:iy0 + scale, ix0:ix0 + scale] = cid

        # collect adjacency by scanning max-grid neighbors
        a = occ[:, :-1]
        b = occ[:,  1:]
        m = (a != b) & (a >= 0) & (b >= 0)
        pairs_h = np.stack([a[m], b[m]], axis=1) if np.any(m) else np.zeros((0, 2), dtype=np.int32)

        a = occ[:-1, :]
        b = occ[ 1:, :]
        m = (a != b) & (a >= 0) & (b >= 0)
        pairs_v = np.stack([a[m], b[m]], axis=1) if np.any(m) else np.zeros((0, 2), dtype=np.int32)

        pairs = np.concatenate([pairs_h, pairs_v], axis=0)
        if pairs.shape[0] == 0:
            # no edges
            return torch.empty((2, 0), dtype=torch.int64)

        u = np.minimum(pairs[:, 0], pairs[:, 1])
        v = np.maximum(pairs[:, 0], pairs[:, 1])
        uv = np.stack([u, v], axis=1)

        # unique undirected edges
        uv_unique = np.unique(uv, axis=0)

        # symmetrize
        src = uv_unique[:, 0]
        dst = uv_unique[:, 1]
        e0 = np.stack([src, dst], axis=0)
        e1 = np.stack([dst, src], axis=0)
        e = np.concatenate([e0, e1], axis=1)

        return torch.from_numpy(e).to(torch.int64)

    def _identity_pred2pred_map(N_dst: int, k: int, device: torch.device):
        """
        Identity IDW map: output at each dst node copies the same-index src node.
        Shapes:
          idx: [N_dst, k] int32, w: [N_dst, k] float16
        """
        k = int(k)
        if k <= 0:
            raise ValueError("interp_k must be >= 1 for pred2pred maps.")

        idx0 = torch.arange(N_dst, device=device, dtype=torch.int32).view(-1, 1)  # [N,1]
        idx  = idx0.repeat(1, k)  # [N,k]
        w = torch.full((N_dst, k), fill_value=(1.0 / float(k)), device=device, dtype=torch.float16)
        return idx, w

    # ----------------- basic checks -----------------
    dev = torch.device(device)
    T = len(steps)
    if T < 2:
        raise RuntimeError("Need at least 2 timesteps for rollout precompute.")
    gt_refine_ratio = _infer_gt_refine_ratio_from_steps(
        steps,
        cfg,
        H,
        W,
        fallback_ratio=refine_ratio,
        log_prefix="[PRECOMP][GT-RR]",
    )

    # ----------------- determine HDF5 cache path -----------------
    if cache_path is None:
        cache_dir = cfg.get("train", {}).get("cache_dir", cfg.get("train", {}).get("save_dir", "."))
        cache_path = os.path.join(cache_dir, "precomp_rollout.h5")

    if not str(cache_path).endswith(".h5"):
        raise ValueError(f"Streaming mode writes a single HDF5 file; cache_path must end with .h5, got: {cache_path}")

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    # ----------------- optional reuse -----------------
    #if (not force_recompute) and _h5_is_usable(cache_path, cfg, expected_T=T):
    #if (not force_recompute) and precomp_h5_is_usable(cache_path, cfg, expected_steps=T, H=H, W=W,
    #                                              require_dec=True, require_pred2pred=True, verbose=True):
    loss_cfg = cfg.get("loss", {}) or {}
    want_mls = (str(loss_cfg.get("physics_backend", "")).lower() == "mls")

    if (not force_recompute) and precomp_h5_is_usable(
            cache_path, cfg,
            expected_steps=T, H=H, W=W,
            require_dec=True,
            require_pred2pred=True,
            require_mls=want_mls,     # <--- ADD
            verbose=True
        ):
        print(f"[PRECOMP] Using existing H5 cache: {cache_path}")

        if cfg.get("debug", {}).get("print_dec_checks", False):
            with h5py.File(cache_path, "r") as f:
                # pick one timestep group
                gname = next(k for k in f.keys() if k.startswith("t"))
                g = f[gname]
                print("Group:", gname)
                print("Datasets:", list(g.keys()))

                # common expected names
                #for k in ["pred_edge_attr", "pred_ea", "pred_edge_attr_layout", "pred_cell_area", "pred_cell_wh"]:
                for k in [
                    "pred_edge_attr", "pred_ea", "pred_edge_attr_layout", "pred_cell_area", "pred_cell_wh",
                    "mls_grad_M_inv", "mls_grad_dX", "mls_lap_w", "mls_edge_index",  # <--- ADD
                ]:
                    if k in g:
                        print(k, "shape=", g[k].shape, "dtype=", g[k].dtype)
                    else:
                        print(k, "MISSING")

        return {
            "type": "h5",
            "path": cache_path,
            "T": int(T),
            "H": int(H),
            "W": int(W),
            "dx": float(dx),
            "dy": float(dy),
            "bbox": cfg["data"]["bbox"],
        }

    print("[PRECOMP] No valid cache found; computing precomputed meshes + interps...")

    # ----------------- load wedge polygon -----------------
    mesh_cfg = cfg.get("mesh", {})
    mesh_spec_path = mesh_cfg.get("starting_mesh_path", None)

    wedge_path = None
    if mesh_spec_path is not None:
        wedge_path = _load_wedge_path_from_spec(mesh_spec_path, cfg=cfg)
        print(f"[GEOM] Wedge path loaded from mesh spec (rescaled): {mesh_spec_path}")
    else:
        print("[GEOM] No starting_mesh_path provided; wedge clipping disabled.")

    # ----------------- choose IDW mapping device (GT→pred) -----------------
    speed = cfg.get("speed", {})
    idw_on_cpu = bool(speed.get("idw_on_cpu", dev.type == "mps"))
    map_dev = torch.device("cpu") if idw_on_cpu else dev

    # ----------------- geometry mode -----------------
    mesh_policy = cfg.get("policy", {})
    geometry_mode = str(mesh_policy.get("geometry_mode", "gt_gradients")).lower()
    refine_to_level = int(mesh_policy.get("starting_refine_to_level", 0))
    refine_policy = str(mesh_policy.get("starting_refine_policy", "min_level")).lower()
    hysteresis_prev_source = str(mesh_policy.get("hysteresis_prev_source", "gt")).strip().lower()

    # In starting_refine mode, build ONE static pred mesh now
    static_pred = None
    if geometry_mode in ("starting_refine", "starting", "static_refine", "refine_test"):
        if mesh_spec_path is None:
            raise RuntimeError("starting_refine mode requires cfg['mesh']['starting_mesh_path'].")

        base_c, base_l, base_p, base_ei, base_mask_parent = _build_starting_mesh_from_spec(
            mesh_spec_path, cfg, H, W, dx, dy, device=dev
        )
        #base_c, base_l, base_p, base_ei = _load_starting_mesh_from_spec(mesh_spec_path)

        pred_c, pred_l, pred_p = _refine_mesh_to_target(base_c, base_l, base_p, refine_to_level, refine_policy)

        # build edges (reuse if no refinement happened AND spec had edges)
        if (refine_to_level <= 0) and (base_ei is not None):
            pred_e = base_ei.to(torch.int64)
        else:
            pred_e = _build_edge_index_from_maxgrid_partition(pred_c, pred_l)

        # mask on coarse parents (domain mask)
        mask_p = _mask_from_parents(pred_p.to(dev), H, W).to(torch.bool)

        # optional clip to wedge (keeps behavior aligned with your pipeline)
        if wedge_path is not None:
            pred_c, pred_l, pred_p, pred_e, mask_p = _clip_pred_mesh_to_wedge(
                pred_c.to(dev),
                pred_l.to(dev),
                pred_p.to(dev),
                pred_e.to(dev),
                H, W,
                wedge_path=wedge_path,
                cfg=cfg,
                device=dev,
            )
            # move back to CPU tensors for reuse
            pred_c = pred_c.detach().cpu()
            pred_l = pred_l.detach().cpu()
            pred_p = pred_p.detach().cpu()
            pred_e = pred_e.detach().cpu()
            mask_p = mask_p.detach().cpu()

        static_pred = (pred_c, pred_l, pred_p, pred_e, mask_p)

        # Build device-ready versions ONCE (avoid per-timestep .to(...) overhead)
        pred_c_cpu, pred_l_cpu, pred_p_cpu, pred_e_cpu, mask_p_cpu = static_pred

        print(f"[PRECOMP][STATIC] Using starting mesh refined to min_level={refine_to_level} (policy={refine_policy})")
        print(f"[PRECOMP][STATIC] N={int(pred_c.shape[0])}, E={int(pred_e.shape[1])}")

    elif geometry_mode in ("gt_gradients", "gradients", "gradient_policy"):
        static_pred = None
        # existing per-step behavior will run in loop
    else:
        raise ValueError(f"Unknown cfg['policy']['geometry_mode'] = '{geometry_mode}'")

    use_predicted_prev_for_hysteresis = (
        static_pred is None
        and geometry_mode in ("gt_gradients", "gradients", "gradient_policy")
        and hysteresis_prev_source in ("predicted", "pred", "previous_pred", "previous_predicted")
    )
    if use_predicted_prev_for_hysteresis:
        print(
            "[PRECOMP] Hysteresis previous-state source: predicted mesh (temporal smoothing enabled).",
            flush=True,
        )

    # ----------------- open HDF5 (overwrite) -----------------
    f = h5py.File(cache_path, "w")
    try:
        meta = f.create_group("meta")
        meta.attrs["H"] = int(H)
        meta.attrs["W"] = int(W)
        meta.attrs["dx"] = float(dx)
        meta.attrs["dy"] = float(dy)
        meta.attrs["bbox"] = np.asarray(cfg["data"]["bbox"], dtype=np.float64)
        meta.attrs["refine_ratio"] = int(refine_ratio)
        meta.attrs["T"] = int(T)
        meta.attrs["cfg_sha1"] = _cfg_sha1(cfg)
        meta.attrs["geometry_mode"] = str(geometry_mode)
        meta.attrs["starting_refine_to_level"] = int(refine_to_level)
        meta.attrs["starting_refine_policy"] = str(refine_policy)
        t_start = cfg.get("train", {}).get("precompute_t_start", None)
        t_end = cfg.get("train", {}).get("precompute_t_end", None)
        if t_start is not None:
            meta.attrs["source_t_start"] = int(t_start)
        if t_end is not None:
            meta.attrs["source_t_end"] = int(t_end)
        meta.create_dataset("cfg_json", data=np.bytes_(json.dumps(cfg, sort_keys=True, default=str)))

        # iterator
        it = range(T - 1)
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(it, desc="[rollout precompute] mesh @ t+1 and GT→pred mappings")
            except Exception:
                pass

        # ---------- FIRST PASS: meshes + GT→pred(t+1) ----------
        prev_pred_level_hyst = None
        prev_pred_ij_hyst = None
        prev_pred_parents_hyst = None
        prev_pred_mask_parent_hyst = None
        for t in it:
            dst = t + 1

            # pull GT(t)
            s = steps[t]
            centers_t = s["pos"]
            centers_t = centers_t if torch.is_tensor(centers_t) else torch.as_tensor(centers_t)
            feat_t = s["x"]
            feat_t = feat_t if torch.is_tensor(feat_t) else torch.as_tensor(feat_t)
            level_t = s.get("level")
            ij_t = s.get("ij")
            ei_t = s.get("edge_index")

            centers_t = centers_t.to(dev)
            feat_t = feat_t.to(dev, dtype=torch.float32)

            if level_t is not None and ij_t is not None:
                parents_t = _parents_from_level_ij(
                    level_t.to(dev), ij_t.to(dev), H, W, refine_ratio=gt_refine_ratio
                )
            elif "parents" in s:
                parents_t = s["parents"].to(dev)
            else:
                xmin, xmax, ymin, ymax = cfg["data"]["bbox"]
                parents_t = parents_from_pos(centers_t, H, W, xmin, xmax, ymin, ymax).to(dev)

            if "mask" in s:
                mask_t_parent = s["mask"].to(dev).view(H, W)
            else:
                mask_t_parent = _mask_from_parents(parents_t, H, W)

            level_t_gt = level_t.to(dev) if level_t is not None else None
            ij_t_gt = ij_t.to(dev) if ij_t is not None else None
            ei_t_gt = ei_t.to(dev) if (ei_t is not None and torch.is_tensor(ei_t)) else None

            # pull GT(t+1)
            s_next = steps[dst]
            centers_tp1 = s_next["pos"]
            centers_tp1 = centers_tp1 if torch.is_tensor(centers_tp1) else torch.as_tensor(centers_tp1)
            feat_tp1 = s_next["x"]
            feat_tp1 = feat_tp1 if torch.is_tensor(feat_tp1) else torch.as_tensor(feat_tp1)
            level_tp1 = s_next.get("level")
            ij_tp1 = s_next.get("ij")

            centers_tp1 = centers_tp1.to(dev)
            feat_tp1 = feat_tp1.to(dev, dtype=torch.float32)

            if level_tp1 is not None and ij_tp1 is not None:
                parents_tp1 = _parents_from_level_ij(
                    level_tp1.to(dev), ij_tp1.to(dev), H, W, refine_ratio=gt_refine_ratio
                )
            elif "parents" in s_next:
                parents_tp1 = s_next["parents"].to(dev)
            else:
                xmin, xmax, ymin, ymax = cfg["data"]["bbox"]
                parents_tp1 = parents_from_pos(centers_tp1, H, W, xmin, xmax, ymin, ymax).to(dev)

            if "mask" in s_next:
                mask_tp1_parent = s_next["mask"].to(dev).view(H, W)
            else:
                mask_tp1_parent = _mask_from_parents(parents_tp1, H, W)

            # -------- choose pred mesh for this dst --------
            if static_pred is not None:
                pred_c, pred_l, pred_p, pred_e, mask_p = static_pred
                # use CPU copies; map to map_dev for IDW
                pred_c_dev = pred_c.to(map_dev)
                pred_l_dev = pred_l.to(map_dev)
                pred_p_dev = pred_p.to(map_dev)
                pred_e_dev = pred_e.to(dev)  # edges are used downstream on model device; store as int32 in H5
                mask_p_dev = mask_p.to(map_dev).view(-1).to(torch.bool)
            else:
                # ----- EXISTING per-step predicted mesh from gradients (your current pipeline) -----
                ex_compat = {
                    "centers_t": centers_t,
                    "center_feat_t": feat_t,
                    "level_t": level_t_gt,
                    "ij_t": ij_t_gt,
                    "gt_refine_ratio": int(gt_refine_ratio),
                    "dyn_parents": parents_t,
                    "mask_t": mask_t_parent.view(-1),
                    "ei_t": ei_t_gt,
                    "t": s.get("t", torch.tensor(t)),
                }
                if use_predicted_prev_for_hysteresis and (prev_pred_level_hyst is not None) and (prev_pred_ij_hyst is not None):
                    # Keep gradients tied to GT(t), but source hysteresis from
                    # previous predicted mesh occupancy.
                    ex_compat["prev_level_t"] = prev_pred_level_hyst
                    ex_compat["prev_ij_t"] = prev_pred_ij_hyst
                    ex_compat["prev_dyn_parents"] = prev_pred_parents_hyst
                    ex_compat["prev_mask_t"] = prev_pred_mask_parent_hyst.view(-1)

                pred_c_rect, pred_l_rect, pred_p_rect, pred_e_rect, _mask_p_rect = _build_pred_mesh_from_gt_gradients(
                    ex_compat, cfg, H, W, dx, dy, dev
                )

                # Save current predicted rectangular mesh as hysteresis state for
                # the next timestep when temporal stabilization is enabled.
                if use_predicted_prev_for_hysteresis:
                    row_prev, col_prev = _cell_level_ij_from_centers(
                        centers=pred_c_rect.to(dev),
                        levels=pred_l_rect.to(dev),
                        H=H,
                        W=W,
                        bbox=tuple(cfg["data"]["bbox"]),
                        refine_ratio=int(refine_ratio),
                    )
                    prev_pred_level_hyst = pred_l_rect.to(dev, dtype=torch.long).view(-1).detach()
                    prev_pred_ij_hyst = torch.stack([row_prev, col_prev], dim=1).to(dev, dtype=torch.long).detach()
                    prev_pred_parents_hyst = _parents_from_level_ij(
                        prev_pred_level_hyst,
                        prev_pred_ij_hyst,
                        H,
                        W,
                        refine_ratio=int(refine_ratio),
                    ).detach()
                    prev_pred_mask_parent_hyst = _mask_from_parents(prev_pred_parents_hyst, H, W).view(H, W).detach()

                pred_c_dev, pred_l_dev, pred_p_dev, pred_e_dev, mask_p_dev = _clip_pred_mesh_to_wedge(
                    pred_c_rect,
                    pred_l_rect,
                    pred_p_rect,
                    pred_e_rect,
                    H,
                    W,
                    wedge_path=wedge_path,
                    cfg=cfg,
                    device=dev,
                )

                pred_c_dev = pred_c_dev.to(map_dev)
                pred_l_dev = pred_l_dev.to(map_dev)
                pred_p_dev = pred_p_dev.to(map_dev)
                mask_p_dev = mask_p_dev.to(map_dev).view(-1).to(torch.bool)

            # -------- GT(t) -> pred(dst) --------
            ft_on_pred = _map_gt_on_pred_mesh_once(
                src_centers=centers_t.to(map_dev),
                src_feats=feat_t.to(map_dev),
                mask_src_parent=mask_t_parent.to(map_dev),
                parents_src=parents_t.to(map_dev),
                pred_centers=pred_c_dev,
                pred_levels=pred_l_dev,
                pred_parents=pred_p_dev,
                mask_pred_parent=mask_p_dev.view(H, W),
                H=H,
                W=W,
                device=map_dev,
            )

            # -------- GT(t+1) -> pred(dst) --------
            f1_on_pred = _map_gt_on_pred_mesh_once(
                src_centers=centers_tp1.to(map_dev),
                src_feats=feat_tp1.to(map_dev),
                mask_src_parent=mask_tp1_parent.to(map_dev),
                parents_src=parents_tp1.to(map_dev),
                pred_centers=pred_c_dev,
                pred_levels=pred_l_dev,
                pred_parents=pred_p_dev,
                mask_pred_parent=mask_p_dev.view(H, W),
                H=H,
                W=W,
                device=map_dev,
            )

            # write dst group
            g = _ensure_group(f, dst)

            _write_ds(g, "pred_centers", _to_np(pred_c_dev.to("cpu"), dtype=np.float32))
            _write_ds(g, "pred_levels",  _to_np(pred_l_dev.to("cpu").to(torch.int16), dtype=np.int16))
            _write_ds(g, "pred_parents", _to_np(pred_p_dev.to("cpu").to(torch.int32), dtype=np.int32))
            _write_ds(g, "pred_ei",      _to_np(pred_e_dev.to("cpu").to(torch.int32), dtype=np.int32))

            mask_flat_u8 = mask_p_dev.to("cpu").view(-1).to(torch.uint8)
            _write_ds(g, "mask_pred_parent_flat_u8", _to_np(mask_flat_u8, dtype=np.uint8))

            _write_ds(g, "feat_t_on_pred",   _to_np(ft_on_pred.to("cpu"), dtype=np.float32))
            _write_ds(g, "feat_tp1_on_pred", _to_np(f1_on_pred.to("cpu"), dtype=np.float32))

            # --- Discrete Exterior Calculus related stuff ---
            # Work on CPU for consistency and to avoid device-mismatch headaches
            c_cpu  = pred_c_dev.to("cpu", dtype=torch.float32)
            l_cpu  = pred_l_dev.to("cpu", dtype=torch.int64)
            ei_cpu = pred_e_dev.to("cpu", dtype=torch.int64)

            # dx,dy in your meta are level-0 cell sizes (you already store them in /meta attrs)
            w_cpu, h_cpu, area_cpu, *_ = amr_cell_wh_area_from_levels(
                l_cpu, dx0=float(dx), dy0=float(dy), refine_ratio=refine_ratio
            )
            cell_wh = torch.stack([w_cpu, h_cpu], dim=1)  # (N,2)

            edge_attr = dec_edge_attr_for_dyadic_quads(
                c_cpu, l_cpu, ei_cpu,
                dx0=float(dx), dy0=float(dy),
                refine_ratio=refine_ratio,
            )

            _write_ds(g, "pred_cell_area", _to_np(area_cpu, dtype=np.float32))
            _write_ds(g, "pred_cell_wh",   _to_np(cell_wh,  dtype=np.float32))
            _write_ds(g, "pred_edge_attr", _to_np(edge_attr, dtype=np.float32))

            g.attrs["pred_edge_attr_layout"] = np.bytes_("nx,ny,face_len,dual_len,tau")
            g.attrs["N_pred"] = int(pred_c_dev.shape[0])
            g.attrs["E"] = int(pred_e_dev.shape[1]) if (pred_e_dev is not None and pred_e_dev.numel() > 0) else 0

            f.flush()

            # cleanup
            del ft_on_pred, f1_on_pred
            del centers_t, feat_t, parents_t, mask_t_parent
            del centers_tp1, feat_tp1, parents_tp1, mask_tp1_parent

            if dev.type == "mps":
                torch.mps.empty_cache()

        # ---------- SECOND PASS: pred->pred IDW maps ----------
        knn_k = int(cfg["loss"].get("interp_k", 8))
        chunk = int(cfg.get("speed", {}).get("interp_chunk", 8192))
        total_maps = max(0, int(T - 2))
        print(f"[PRECOMP] SECOND PASS: building pred->pred maps ({total_maps} transitions)...")

        def _second_pass_iter():
            it2 = range(1, T - 1)
            used_tqdm = False
            if progress:
                try:
                    from tqdm import tqdm
                    it2 = tqdm(it2, desc="[rollout precompute] pred->pred maps")
                    used_tqdm = True
                except Exception:
                    pass
            return it2, used_tqdm

        if static_pred is not None:
            # geometry is constant => identity maps for every t
            # store maps in groups t=1..T-2 (mapping pred(t)->pred(t+1))
            # use N from any group (t=00001 exists)
            g1 = f.get("t00001", None)
            if g1 is None:
                raise RuntimeError("Missing t00001 group after first pass.")
            N_dst = int(g1.attrs.get("N_pred", g1["pred_centers"].shape[0]))

            idx_map, w_map = _identity_pred2pred_map(N_dst=N_dst, k=knn_k, device=torch.device("cpu"))

            it2, used_tqdm = _second_pass_iter()
            for i, t in enumerate(it2, start=1):
                g = _ensure_group(f, t)
                _write_ds(g, "pred2pred_idx_to_next", _to_np(idx_map.to(torch.int32), dtype=np.int32))
                _write_ds(g, "pred2pred_w_to_next",   _to_np(w_map.to(torch.float16), dtype=np.float16))
                f.flush()
                if progress and (not used_tqdm) and total_maps > 0:
                    stride = max(1, total_maps // 20)
                    if (i % stride == 0) or (i == total_maps):
                        print(f"[rollout precompute] pred->pred maps {i}/{total_maps}")

            del idx_map, w_map

        else:
            # existing behavior: build pred(t) -> pred(t+1) maps from centers
            knn_device = torch.device("cpu")

            it2, used_tqdm = _second_pass_iter()
            for i, t in enumerate(it2, start=1):
                if f"t{t:05d}" not in f or f"t{t+1:05d}" not in f:
                    continue

                src_c = _read_centers(f, t).to(knn_device)
                dst_c = _read_centers(f, t + 1).to(knn_device)

                idx_map, w_map = build_idw_map(
                    dst_c,
                    src_c,
                    k=knn_k,
                    chunk=chunk,
                )

                g = _ensure_group(f, t)
                _write_ds(g, "pred2pred_idx_to_next", _to_np(idx_map.to(torch.int32), dtype=np.int32))
                _write_ds(g, "pred2pred_w_to_next",   _to_np(w_map.to(torch.float16), dtype=np.float16))
                f.flush()
                if progress and (not used_tqdm) and total_maps > 0:
                    stride = max(1, total_maps // 20)
                    if (i % stride == 0) or (i == total_maps):
                        print(f"[rollout precompute] pred->pred maps {i}/{total_maps}")

                del src_c, dst_c, idx_map, w_map

        f.flush()

    finally:
        f.close()

    print(f"[PRECOMP] Saved H5 precompute to {cache_path}")
    _write_precomp_repro_json(
        cache_path,
        _build_precomp_repro_payload(
            mode="predicted",
            cache_path=cache_path,
            cfg=cfg,
            H=H,
            W=W,
            dx=dx,
            dy=dy,
            T=T,
            gt_refine_ratio=gt_refine_ratio,
            refine_ratio=refine_ratio,
            cfg_sha1=_cfg_sha1(cfg),
            geometry_mode=geometry_mode,
            mesh_spec_path=mesh_spec_path,
        ),
    )

    # plot the first predicted-step mesh (t00001). For static refinement tests, all groups are the same anyway.
    out_png = os.path.join(os.path.dirname(cache_path) or ".", "precomp_mesh_t00001.png")
    plot_precomputed_mesh_from_h5(
        cache_path,
        t=1,
        out_path=out_png,
        show=bool(dbg.get("show_precomp_mesh", False)),
        max_cells=int(dbg.get("plot_precomp_max_cells", 250_000)),
        linewidth=float(dbg.get("plot_precomp_linewidth", 0.15)),
        wedge_path=wedge_path,  # if in-scope in your function; otherwise drop this arg
    )
    print(f"[PRECOMP] Saved mesh plot: {out_png}")

    out_png_zoom = os.path.join(os.path.dirname(cache_path) or ".", "precomp_mesh_edges_zoom_t00001.png")
    plot_precomputed_mesh_with_edges_from_h5(
        cache_path,
        t=1,
        out_path=out_png_zoom,
        show=bool(dbg.get("show_precomp_mesh", False)),
        zoom_bbox=None,          # None => auto-zoom to a coarse–fine interface using SAVED pred_ei
        zoom_cells=20.0,         # adjust window size as needed
        max_cells=200_000,
        max_edges=400_000,
        cell_linewidth=0.20,
        edge_linewidth=0.45,
        node_size=3.0,
        wedge_path=wedge_path,
    )
    print(f"[PRECOMP] Saved zoomed mesh+edges plot: {out_png_zoom}")

    return {
        "type": "h5",
        "path": cache_path,
        "T": int(T),
        "H": int(H),
        "W": int(W),
        "dx": float(dx),
        "dy": float(dy),
        "bbox": cfg["data"]["bbox"],
    }


@torch.no_grad()
def precompute_uniform_mesh_in_memory(
    steps: List[Dict[str, Any]],
    cfg: dict,
    H: int,
    W: int,
    dx: float,
    dy: float,
    *,
    device: str | torch.device = "cpu",
    progress: bool = True,
    cache_path: str | None = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Build one static (uniform-mode) predicted mesh and reuse it for all timesteps.

    If cache_path is provided, this streams precompute to one H5 file and returns
    a lazy H5 handle descriptor: {"type":"h5", "path":...}. If cache_path is None,
    it returns the in-memory dict-of-lists expected by CollateWithPrecompute.
    """
    t_uniform_total_0 = time.perf_counter()
    timing_uniform = {
        "static_mesh_build_s": 0.0,
        "static_h5_write_s": 0.0,
        "gt_map_compute_s": 0.0,
        "gt_map_write_s": 0.0,
    }

    dev = torch.device(device)
    rr = _get_refine_ratio(cfg)
    T = len(steps)
    if T < 2:
        raise RuntimeError("Need at least 2 timesteps for in-memory uniform precompute.")
    gt_refine_ratio = _infer_gt_refine_ratio_from_steps(
        steps,
        cfg,
        H,
        W,
        fallback_ratio=rr,
        log_prefix="[PRECOMP][UNIFORM][GT-RR]",
    )

    mesh_cfg = cfg.get("mesh", {}) or {}
    mesh_spec_raw = mesh_cfg.get("starting_mesh_path", None)
    if not mesh_spec_raw:
        raise RuntimeError("Uniform mesh mode requires cfg['mesh']['starting_mesh_path'].")
    mesh_spec_path = os.path.abspath(os.path.expanduser(str(mesh_spec_raw)))
    if not os.path.exists(mesh_spec_path):
        raise FileNotFoundError(f"Uniform mesh mode expected mesh spec at: {mesh_spec_path}")

    pol = cfg.get("policy", {}) or {}
    refine_to_level = int(pol.get("starting_refine_to_level", 0))
    refine_policy = str(pol.get("starting_refine_policy", "min_level")).strip().lower()
    if refine_policy not in ("min_level", "level0_only"):
        raise ValueError(f"Unknown starting_refine_policy='{refine_policy}'")
    interp_k_cfg = int(cfg.get("loss", {}).get("interp_k", 8))

    if cache_path is not None:
        if not str(cache_path).endswith(".h5"):
            raise ValueError(f"Uniform mesh precompute cache_path must end with .h5, got: {cache_path}")
        cache_path = os.path.abspath(os.path.expanduser(str(cache_path)))
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    def _sha1_file(path: str, chunk_bytes: int = 1 << 20) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk_bytes)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    mesh_spec_sha1 = _sha1_file(mesh_spec_path)

    def _cfg_sha1(_cfg: dict) -> str:
        s = json.dumps(_cfg, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    def _to_np(x: torch.Tensor, dtype=None) -> np.ndarray:
        a = x.detach().cpu().numpy()
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        return a

    def _write_ds(g, name: str, arr: np.ndarray, *, compress=True):
        if name in g:
            del g[name]
        kwargs = {}
        if compress:
            kwargs = dict(compression="gzip", compression_opts=4, shuffle=True, chunks=True)
        g.create_dataset(name, data=arr, **kwargs)

    def _uniform_h5_signature() -> str:
        payload = {
            "mesh_mode": "uniform",
            "mesh_spec_sha1": mesh_spec_sha1,
            "starting_refine_to_level": int(refine_to_level),
            "starting_refine_policy": str(refine_policy),
            "refine_ratio": int(rr),
            "H": int(H),
            "W": int(W),
            "T": int(T),
            "dx": float(dx),
            "dy": float(dy),
            "bbox": [float(v) for v in cfg["data"]["bbox"]],
            "edges_method": str(cfg.get("edges", {}).get("method", "face")).lower(),
            "interp_k": int(interp_k_cfg),
        }
        s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    def _uniform_h5_usable(path: str) -> bool:
        ok_base = precomp_h5_is_usable(
            path,
            cfg,
            expected_steps=T,
            H=H,
            W=W,
            require_dec=True,
            require_pred2pred=True,
            require_mls=False,
            verbose=True,
        )
        if not ok_base:
            return False
        try:
            sig = _uniform_h5_signature()
            with h5py.File(path, "r") as f:
                if "meta" not in f:
                    print("[PRECOMP][UNIFORM] reject cache: missing /meta")
                    return False
                meta = f["meta"]
                mode = meta.attrs.get("mesh_mode", "")
                if isinstance(mode, (bytes, np.bytes_)):
                    mode = mode.decode("utf-8")
                mode = str(mode)
                if mode != "uniform":
                    print(f"[PRECOMP][UNIFORM] reject cache: mesh_mode='{mode}' (expected 'uniform')")
                    return False

                sig_stored = meta.attrs.get("uniform_signature_sha1", "")
                if isinstance(sig_stored, (bytes, np.bytes_)):
                    sig_stored = sig_stored.decode("utf-8")
                sig_stored = str(sig_stored)
                if sig_stored != sig:
                    # Backward-compatible relaxed acceptance for older uniform caches:
                    # accept when core mesh construction controls match, even if old
                    # signature included path/mtime that differs across jobs.
                    lvl_ok = int(meta.attrs.get("starting_refine_to_level", -999)) == int(refine_to_level)
                    pol_stored = meta.attrs.get("starting_refine_policy", "")
                    if isinstance(pol_stored, (bytes, np.bytes_)):
                        pol_stored = pol_stored.decode("utf-8")
                    pol_ok = str(pol_stored) == str(refine_policy)
                    rr_ok = int(meta.attrs.get("refine_ratio", -1)) == int(rr)

                    interp_stored = meta.attrs.get("interp_k", None)
                    interp_ok = True
                    if interp_stored is not None:
                        interp_ok = int(interp_stored) == int(interp_k_cfg)

                    mesh_sha_stored = meta.attrs.get("mesh_spec_sha1", "")
                    if isinstance(mesh_sha_stored, (bytes, np.bytes_)):
                        mesh_sha_stored = mesh_sha_stored.decode("utf-8")
                    mesh_sha_stored = str(mesh_sha_stored)
                    mesh_sha_ok = (mesh_sha_stored == "") or (mesh_sha_stored == mesh_spec_sha1)

                    if lvl_ok and pol_ok and rr_ok and interp_ok and mesh_sha_ok:
                        print(
                            "[PRECOMP][UNIFORM] cache signature mismatch but accepted via relaxed match "
                            "(legacy or path/mtime-only difference)."
                        )
                        return True

                    print(
                        "[PRECOMP][UNIFORM] reject cache: signature mismatch "
                        f"(stored={sig_stored}, current={sig})"
                    )
                    return False
            return True
        except Exception as e:
            print(f"[PRECOMP][UNIFORM] reject cache: unreadable meta/signature ({e})")
            return False

    if cache_path is not None and (not force_recompute) and _uniform_h5_usable(cache_path):
        print(f"[PRECOMP][UNIFORM] Using existing H5 cache: {cache_path}")
        return {
            "type": "h5",
            "path": cache_path,
            "T": int(T),
            "H": int(H),
            "W": int(W),
            "dx": float(dx),
            "dy": float(dy),
            "bbox": cfg["data"]["bbox"],
        }

    def _refine_cells_one_level(
        centers: torch.Tensor,
        levels: torch.Tensor,
        parents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lv = levels.to(torch.int64)
        rr_t = torch.tensor(float(rr), device=lv.device)
        dx_L = (torch.tensor(float(dx), device=lv.device) / torch.pow(rr_t, lv.to(torch.float32))).to(torch.float32)
        dy_L = (torch.tensor(float(dy), device=lv.device) / torch.pow(rr_t, lv.to(torch.float32))).to(torch.float32)

        c = centers.to(torch.float32)
        x0 = c[:, 0]
        y0 = c[:, 1]

        frac = ((torch.arange(rr, dtype=torch.float32, device=c.device) + 0.5) / float(rr)) - 0.5
        gx, gy = torch.meshgrid(frac, frac, indexing="xy")
        child = torch.empty((c.shape[0], rr, rr, 2), dtype=torch.float32, device=c.device)
        child[..., 0] = x0.view(-1, 1, 1) + dx_L.view(-1, 1, 1) * gx.view(1, rr, rr)
        child[..., 1] = y0.view(-1, 1, 1) + dy_L.view(-1, 1, 1) * gy.view(1, rr, rr)

        child_centers = child.reshape(-1, 2)
        child_count = rr * rr
        child_levels = (lv + 1).repeat_interleave(child_count)
        child_parents = parents.repeat_interleave(child_count)
        return child_centers, child_levels, child_parents

    def _refine_mesh_to_target(
        centers: torch.Tensor,
        levels: torch.Tensor,
        parents: torch.Tensor,
        target_level: int,
        policy_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_level = int(target_level)
        if target_level <= 0:
            return centers, levels.to(torch.int64), parents.to(torch.int64)

        c = centers.clone().to(torch.float32)
        l = levels.clone().to(torch.int64)
        p = parents.clone().to(torch.int64)

        if policy_name == "level0_only":
            base_is_level0 = (l == 0)

        while True:
            if policy_name == "min_level":
                need = (l < target_level)
            else:
                need = base_is_level0 & (l < target_level)
            if not bool(need.any().item()):
                break

            keep = ~need
            c_keep, l_keep, p_keep = c[keep], l[keep], p[keep]
            c_need, l_need, p_need = c[need], l[need], p[need]
            c_child, l_child, p_child = _refine_cells_one_level(c_need, l_need, p_need)

            c = torch.cat([c_keep, c_child], dim=0)
            l = torch.cat([l_keep, l_child], dim=0)
            p = torch.cat([p_keep, p_child], dim=0)

            if policy_name == "level0_only":
                base_keep = base_is_level0[keep]
                base_child = torch.ones((c_child.shape[0],), dtype=torch.bool, device=base_keep.device)
                base_is_level0 = torch.cat([base_keep, base_child], dim=0)

        return c, l, p

    t_static_mesh_0 = time.perf_counter()

    # Build base wedge mesh once.
    base_c, base_l, base_p, base_ei, _base_mask_parent = _build_starting_mesh_from_spec(
        mesh_spec_path,
        cfg,
        H,
        W,
        dx,
        dy,
        device=dev,
    )
    pred_c, pred_l, pred_p = _refine_mesh_to_target(
        base_c,
        base_l,
        base_p,
        refine_to_level,
        refine_policy,
    )

    # Build/reuse edges once.
    if (refine_to_level <= 0) and (base_ei is not None):
        pred_e = base_ei.to(torch.int64)
    else:
        edge_method = str(cfg.get("edges", {}).get("method", "face")).lower()
        if "face" in edge_method:
            pred_e = build_amr_face_adjacency_edges(
                pred_c,
                pred_l,
                H,
                W,
                bbox=tuple(cfg["data"]["bbox"]),
                return_edge_attr=False,
                refine_ratio=rr,
            ).to(torch.int64)
        else:
            pred_e = build_amr_local_knn_edges(
                pred_c,
                pred_p,
                H,
                W,
                k_local=int(cfg.get("edges", {}).get("k_local", 4)),
                max_local=int(cfg.get("edges", {}).get("max_local", 2048)),
            ).to(torch.int64)

    wedge_path = _load_wedge_path_from_spec(mesh_spec_path, cfg=cfg)
    pred_c, pred_l, pred_p, pred_e, mask_p = _clip_pred_mesh_to_wedge(
        pred_c.to(dev),
        pred_l.to(dev),
        pred_p.to(dev),
        pred_e.to(dev),
        H,
        W,
        wedge_path=wedge_path,
        cfg=cfg,
        device=dev,
    )

    # Static geometry on CPU for storage/reuse.
    pred_c_cpu = pred_c.detach().cpu().to(torch.float32)
    pred_l_cpu = pred_l.detach().cpu().to(torch.int64)
    pred_p_cpu = pred_p.detach().cpu().to(torch.int64)
    pred_e_cpu = pred_e.detach().cpu().to(torch.int64)
    mask_p_cpu = mask_p.detach().cpu().view(-1).to(torch.bool)

    if pred_c_cpu.numel() == 0 or pred_e_cpu.numel() == 0:
        raise RuntimeError("Uniform in-memory precompute produced empty mesh geometry.")

    # Static DEC geometry once.
    w_cpu, h_cpu, area_cpu, *_ = amr_cell_wh_area_from_levels(
        pred_l_cpu,
        dx0=float(dx),
        dy0=float(dy),
        refine_ratio=rr,
    )
    cell_wh_cpu = torch.stack([w_cpu, h_cpu], dim=1).to(torch.float32)
    edge_attr_cpu = dec_edge_attr_for_dyadic_quads(
        pred_c_cpu,
        pred_l_cpu,
        pred_e_cpu,
        dx0=float(dx),
        dy0=float(dy),
        refine_ratio=rr,
    ).to(torch.float32)
    timing_uniform["static_mesh_build_s"] += float(time.perf_counter() - t_static_mesh_0)

    # Identity pred->pred map once (mesh is constant across all t).
    k_interp = int(cfg.get("loss", {}).get("interp_k", 8))
    if k_interp < 1:
        raise ValueError("loss.interp_k must be >=1 for uniform in-memory precompute.")
    N = int(pred_c_cpu.shape[0])
    idx_id = torch.arange(N, dtype=torch.int64).view(-1, 1).repeat(1, k_interp)
    w_id = torch.full((N, k_interp), fill_value=(1.0 / float(k_interp)), dtype=torch.float32)

    # Mapping device for GT->pred interpolation.
    speed = cfg.get("speed", {}) or {}
    idw_on_cpu = bool(speed.get("idw_on_cpu", dev.type == "mps"))
    map_dev = torch.device("cpu") if idw_on_cpu else dev
    pred_c_map = pred_c_cpu.to(map_dev)
    pred_l_map = pred_l_cpu.to(map_dev)
    pred_p_map = pred_p_cpu.to(map_dev)
    mask_p_map_2d = mask_p_cpu.view(H, W).to(map_dev)

    # Reuse existing top-level cfg["idw"] settings for GT->pred mapping.
    idw_cfg = cfg.get("idw", {}) or {}
    if not isinstance(idw_cfg, dict):
        raise ValueError("cfg['idw'] must be an object when provided.")

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
    map_knn_backend = backend_aliases.get(raw_backend, raw_backend)
    if map_knn_backend not in ("exact", "faiss_flat", "faiss_ivf"):
        raise ValueError("cfg['idw']['backend'] must be one of: exact, faiss_flat, faiss_ivf.")

    map_knn_backend_kwargs: Dict[str, Any] = {}
    if map_knn_backend in ("faiss_flat", "faiss_ivf"):
        map_knn_backend_kwargs["faiss_nlist"] = max(1, int(idw_cfg.get("faiss_nlist", 256)))
        map_knn_backend_kwargs["faiss_nprobe"] = max(1, int(idw_cfg.get("faiss_nprobe", 16)))
        map_knn_backend_kwargs["faiss_cache"] = bool(idw_cfg.get("faiss_cache", True))
        map_knn_backend_kwargs["faiss_cache_max_entries"] = max(1, int(idw_cfg.get("faiss_cache_max_entries", 4)))

        allow_fallback = bool(idw_cfg.get("allow_fallback_to_exact", True))
        try:
            import faiss  # type: ignore  # noqa: F401
        except Exception:
            if allow_fallback:
                print(
                    "[PRECOMP][UNIFORM] idw.backend requested FAISS but import failed; "
                    "falling back to exact.",
                    flush=True,
                )
                map_knn_backend = "exact"
                map_knn_backend_kwargs = {}
            else:
                raise RuntimeError(
                    "cfg['idw']['backend'] requests FAISS, but faiss import failed "
                    "and allow_fallback_to_exact=false."
                )

    map_knn_k = max(1, int(idw_cfg.get("k", 8)))
    map_knn_chunk = max(1, int(idw_cfg.get("chunk", speed.get("interp_chunk", 8192))))

    precomp: Dict[str, Any] | None = None
    h5_file = None
    if cache_path is None:
        # Build precomp dict in-memory (list indexed by absolute timestep).
        precomp = {
            "pred_centers": [None] * T,
            "pred_levels": [None] * T,
            "pred_parents": [None] * T,
            "pred_ei": [None] * T,
            "mask_pred": [None] * T,
            "feat_t_on_pred": [None] * T,
            "feat_tp1_on_pred": [None] * T,
            "pred2pred_idx": [None] * T,
            "pred2pred_w": [None] * T,
            "pred_edge_attr": [None] * T,
            "pred_cell_wh": [None] * T,
            "pred_cell_area": [None] * T,
            "pred_edge_attr_layout": "nx,ny,face_len,dual_len,tau",
        }

        # Geometry is constant for all usable destination timesteps (1..T-1).
        for t in range(1, T):
            precomp["pred_centers"][t] = pred_c_cpu
            precomp["pred_levels"][t] = pred_l_cpu
            precomp["pred_parents"][t] = pred_p_cpu
            precomp["pred_ei"][t] = pred_e_cpu
            precomp["mask_pred"][t] = mask_p_cpu
            precomp["pred2pred_idx"][t] = idx_id
            precomp["pred2pred_w"][t] = w_id
            precomp["pred_edge_attr"][t] = edge_attr_cpu
            precomp["pred_cell_wh"][t] = cell_wh_cpu
            precomp["pred_cell_area"][t] = area_cpu.to(torch.float32)
    else:
        print(f"[PRECOMP][UNIFORM] Writing H5 cache: {cache_path}")
        t_static_h5_0 = time.perf_counter()
        h5_file = h5py.File(cache_path, "w")
        meta = h5_file.create_group("meta")
        meta.attrs["H"] = int(H)
        meta.attrs["W"] = int(W)
        meta.attrs["dx"] = float(dx)
        meta.attrs["dy"] = float(dy)
        meta.attrs["bbox"] = np.asarray(cfg["data"]["bbox"], dtype=np.float64)
        meta.attrs["refine_ratio"] = int(rr)
        meta.attrs["T"] = int(T)
        meta.attrs["cfg_sha1"] = _cfg_sha1(cfg)
        meta.attrs["mesh_mode"] = np.bytes_("uniform")
        meta.attrs["uniform_signature_sha1"] = np.bytes_(_uniform_h5_signature())
        meta.attrs["starting_refine_to_level"] = int(refine_to_level)
        meta.attrs["starting_refine_policy"] = np.bytes_(str(refine_policy))
        meta.attrs["starting_mesh_path"] = np.bytes_(mesh_spec_path)
        meta.attrs["mesh_spec_sha1"] = np.bytes_(mesh_spec_sha1)
        t_start = cfg.get("train", {}).get("precompute_t_start", None)
        t_end = cfg.get("train", {}).get("precompute_t_end", None)
        if t_start is not None:
            meta.attrs["source_t_start"] = int(t_start)
        if t_end is not None:
            meta.attrs["source_t_end"] = int(t_end)
        meta.attrs["interp_k"] = int(interp_k_cfg)
        meta.attrs["pred_edge_attr_layout"] = np.bytes_("nx,ny,face_len,dual_len,tau")
        meta.attrs["uniform_static_geometry"] = np.uint8(1)
        meta.attrs["uniform_static_pred2pred"] = np.uint8(1)
        meta.create_dataset("cfg_json", data=np.bytes_(json.dumps(cfg, sort_keys=True, default=str)))

        idx_id_h5 = idx_id.to(torch.int32)
        w_id_h5 = w_id.to(torch.float16)
        mask_flat_u8 = mask_p_cpu.to(torch.uint8).view(-1)
        area_f32 = area_cpu.to(torch.float32)

        # Store static geometry only once.
        g_static = h5_file.create_group("static")
        _write_ds(g_static, "pred_centers", _to_np(pred_c_cpu, dtype=np.float32))
        _write_ds(g_static, "pred_levels", _to_np(pred_l_cpu.to(torch.int16), dtype=np.int16))
        _write_ds(g_static, "pred_parents", _to_np(pred_p_cpu.to(torch.int32), dtype=np.int32))
        _write_ds(g_static, "pred_ei", _to_np(pred_e_cpu.to(torch.int32), dtype=np.int32))
        _write_ds(g_static, "mask_pred_parent_flat_u8", _to_np(mask_flat_u8, dtype=np.uint8))
        _write_ds(g_static, "pred_edge_attr", _to_np(edge_attr_cpu, dtype=np.float32))
        _write_ds(g_static, "pred_cell_wh", _to_np(cell_wh_cpu, dtype=np.float32))
        _write_ds(g_static, "pred_cell_area", _to_np(area_f32, dtype=np.float32))
        _write_ds(g_static, "pred2pred_idx_to_next", _to_np(idx_id_h5, dtype=np.int32))
        _write_ds(g_static, "pred2pred_w_to_next", _to_np(w_id_h5, dtype=np.float16))
        g_static.attrs["pred_edge_attr_layout"] = np.bytes_("nx,ny,face_len,dual_len,tau")
        g_static.attrs["N_pred"] = int(pred_c_cpu.shape[0])
        g_static.attrs["E"] = int(pred_e_cpu.shape[1]) if pred_e_cpu.numel() > 0 else 0

        # Keep timestep groups for per-step mapped features.
        for t in range(1, T):
            g = h5_file.create_group(f"t{int(t):05d}")
            g.attrs["N_pred"] = int(pred_c_cpu.shape[0])
            g.attrs["E"] = int(pred_e_cpu.shape[1]) if pred_e_cpu.numel() > 0 else 0
        h5_file.flush()
        timing_uniform["static_h5_write_s"] += float(time.perf_counter() - t_static_h5_0)

    # Map each absolute GT timestep once, then reuse adjacent maps for (t -> t+1) pairs.
    def _map_gt_abs_t(abs_t: int) -> torch.Tensor:
        s = steps[int(abs_t)]
        centers_t = (s["pos"] if torch.is_tensor(s["pos"]) else torch.as_tensor(s["pos"])).to(map_dev, dtype=torch.float32)
        feat_t = (s["x"] if torch.is_tensor(s["x"]) else torch.as_tensor(s["x"])).to(map_dev, dtype=torch.float32)
        level_t = s.get("level", None)
        ij_t = s.get("ij", None)
        if level_t is not None and ij_t is not None:
            parents_t = _parents_from_level_ij(
                level_t.to(map_dev),
                ij_t.to(map_dev),
                H,
                W,
                refine_ratio=gt_refine_ratio,
            )
        elif "parents" in s:
            parents_t = (s["parents"] if torch.is_tensor(s["parents"]) else torch.as_tensor(s["parents"])).to(
                map_dev,
                dtype=torch.long,
            )
        else:
            xmin, xmax, ymin, ymax = cfg["data"]["bbox"]
            parents_t = parents_from_pos(centers_t, H, W, xmin, xmax, ymin, ymax).to(map_dev, dtype=torch.long)
        if "mask" in s:
            mask_t_parent = (s["mask"] if torch.is_tensor(s["mask"]) else torch.as_tensor(s["mask"])).to(map_dev).view(H, W).to(torch.bool)
        else:
            mask_t_parent = _mask_from_parents(parents_t, H, W).to(map_dev).to(torch.bool)

        t_map0 = time.perf_counter()
        mapped = _map_gt_on_pred_mesh_once(
            src_centers=centers_t,
            src_feats=feat_t,
            mask_src_parent=mask_t_parent,
            parents_src=parents_t,
            pred_centers=pred_c_map,
            pred_levels=pred_l_map,
            pred_parents=pred_p_map,
            mask_pred_parent=mask_p_map_2d,
            H=H,
            W=W,
            device=map_dev,
            knn_k=map_knn_k,
            knn_chunk=map_knn_chunk,
            knn_backend=map_knn_backend,
            knn_backend_kwargs=map_knn_backend_kwargs,
        )
        timing_uniform["gt_map_compute_s"] += float(time.perf_counter() - t_map0)
        return mapped

    print(
        "[PRECOMP][UNIFORM] GT mapping settings:",
        f"idw.backend={map_knn_backend}",
        f"k={int(map_knn_k)}",
        f"chunk={int(map_knn_chunk)}",
        flush=True,
    )

    it = range(1, T)
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc="[uniform precompute] GT->static-mesh mappings")
        except Exception:
            pass

    mapped_prev = _map_gt_abs_t(0)
    for dst in it:
        mapped_cur = _map_gt_abs_t(dst)
        if precomp is not None:
            precomp["feat_t_on_pred"][dst] = mapped_prev.detach().cpu().to(torch.float32)
            precomp["feat_tp1_on_pred"][dst] = mapped_cur.detach().cpu().to(torch.float32)
        else:
            t_write0 = time.perf_counter()
            g = h5_file[f"t{int(dst):05d}"]
            _write_ds(g, "feat_t_on_pred", _to_np(mapped_prev.to("cpu"), dtype=np.float32))
            _write_ds(g, "feat_tp1_on_pred", _to_np(mapped_cur.to("cpu"), dtype=np.float32))
            timing_uniform["gt_map_write_s"] += float(time.perf_counter() - t_write0)
            if (dst % 8) == 0:
                h5_file.flush()
        mapped_prev = mapped_cur

    lv_u, lv_c = torch.unique(pred_l_cpu, return_counts=True)
    lv_summary = ", ".join([f"L{int(u)}={int(c)}" for u, c in zip(lv_u.tolist(), lv_c.tolist())])
    print(
        "[PRECOMP][UNIFORM] built static mesh:",
        f"cells={int(pred_c_cpu.shape[0])}",
        f"edges={int(pred_e_cpu.shape[1])}",
        f"levels=({lv_summary})",
        f"starting_refine_to_level={refine_to_level}",
        f"starting_refine_policy={refine_policy}",
        flush=True,
    )

    total_uniform_s = float(time.perf_counter() - t_uniform_total_0)
    tracked_uniform_s = (
        float(timing_uniform["static_mesh_build_s"])
        + float(timing_uniform["static_h5_write_s"])
        + float(timing_uniform["gt_map_compute_s"])
        + float(timing_uniform["gt_map_write_s"])
    )
    other_uniform_s = max(0.0, total_uniform_s - tracked_uniform_s)

    def _pct(v: float) -> float:
        return (100.0 * float(v) / total_uniform_s) if total_uniform_s > 0.0 else 0.0

    print(
        "[PRECOMP][UNIFORM][TIMING]",
        f"total={total_uniform_s:.3f}s",
        f"static_mesh_build={timing_uniform['static_mesh_build_s']:.3f}s ({_pct(timing_uniform['static_mesh_build_s']):.1f}%)",
        f"static_h5_write={timing_uniform['static_h5_write_s']:.3f}s ({_pct(timing_uniform['static_h5_write_s']):.1f}%)",
        f"gt_map_compute={timing_uniform['gt_map_compute_s']:.3f}s ({_pct(timing_uniform['gt_map_compute_s']):.1f}%)",
        f"gt_map_write={timing_uniform['gt_map_write_s']:.3f}s ({_pct(timing_uniform['gt_map_write_s']):.1f}%)",
        f"other={other_uniform_s:.3f}s ({_pct(other_uniform_s):.1f}%)",
        flush=True,
    )

    if h5_file is not None:
        try:
            h5_file.flush()
        finally:
            h5_file.close()
        print(f"[PRECOMP][UNIFORM] Saved H5 precompute to {cache_path}")
        _write_precomp_repro_json(
            cache_path,
            _build_precomp_repro_payload(
                mode="uniform",
                cache_path=cache_path,
                cfg=cfg,
                H=H,
                W=W,
                dx=dx,
                dy=dy,
                T=T,
                gt_refine_ratio=gt_refine_ratio,
                refine_ratio=rr,
                cfg_sha1=_cfg_sha1(cfg),
                geometry_mode="starting_refine",
                mesh_spec_path=mesh_spec_path,
                mesh_spec_sha1=mesh_spec_sha1,
                uniform_signature_sha1=_uniform_h5_signature(),
            ),
        )
        return {
            "type": "h5",
            "path": cache_path,
            "T": int(T),
            "H": int(H),
            "W": int(W),
            "dx": float(dx),
            "dy": float(dy),
            "bbox": cfg["data"]["bbox"],
        }
    return precomp


class CollateWithPrecompute:
    #def __init__(self, precomp: Dict[str, List], *, dt_transitions: torch.Tensor, dt_ref: torch.Tensor | float | None = None):
    #    self.precomp = precomp
    #    self.dt_transitions = dt_transitions  # length T-1, CPU tensor is fine
    #    self.dt_ref = dt_ref

    def __init__(self, precomp, *, dt_transitions, dt_ref=None):
        self.precomp = precomp
        if torch.is_tensor(dt_transitions):
            self.dt_transitions = dt_transitions.detach().cpu()
        else:
            self.dt_transitions = torch.as_tensor(dt_transitions, dtype=torch.float32, device="cpu")
        self.dt_ref = dt_ref

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # identity (batch_size=1) + enrich with precomputed lists for the window
        #ex = batch[0]
        ex0 = batch[0]
        ex = dict(ex0)
        idxs = ex["t_indices"].tolist()  # absolute time indices in the window
        K = len(idxs)

        # dt per transition in this window: dt_list[j] corresponds to idxs[j] -> idxs[j+1]
        # dt_transitions[t] is dt from t -> t+1
        dt_list = []
        for j in range(K - 1):
            t_src = idxs[j]
            #dt_list.append(self.dt_transitions[t_src])  # scalar tensor on CPU
            #dt_list.append(float(self.dt_transitions[t_src].item()))
            v = self.dt_transitions[t_src]
            dt_list.append(float(v.item()) if torch.is_tensor(v) else float(v))


        ex["dt_list"] = dt_list
        if self.dt_ref is not None:
            ex["dt_ref"] = float(self.dt_ref) if not torch.is_tensor(self.dt_ref) else float(self.dt_ref.item())

        pc    = self.precomp["pred_centers"]
        pl    = self.precomp["pred_levels"]
        pp    = self.precomp["pred_parents"]
        pei   = self.precomp["pred_ei"]
        pea = self.precomp.get("pred_edge_attr", None)
        mp    = self.precomp["mask_pred"]
        ft    = self.precomp["feat_t_on_pred"]
        ftp1  = self.precomp["feat_tp1_on_pred"]

        # ----- geometry + GT→pred lists (length = K) -----
        ex["pred_centers_list"]    = [pc[i]   for i in idxs]
        ex["pred_levels_list"]     = [pl[i]   for i in idxs]
        ex["pred_parents_list"]    = [pp[i]   for i in idxs]
        ex["pred_ei_list"]         = [pei[i]  for i in idxs]
        if pea is not None:
            ex["pred_edge_attr_list"] = [pea[i] for i in idxs]

        ex["mask_pred_list"]       = [mp[i]   for i in idxs]

        ex["feat_t_on_pred_list"]   = [ft[i]   for i in idxs]
        ex["feat_tp1_on_pred_list"] = [ftp1[i] for i in idxs]

        if getattr(self, "_printed_dec_once", False) is False:
            if "pred_edge_attr_list" in ex:
                pea0 = ex["pred_edge_attr_list"][0]
                pei0 = ex["pred_ei_list"][0]
                print("[DEC-CHK] Collate attached pred_edge_attr_list.")
                if torch.is_tensor(pei0):
                    print(f"[DEC-CHK] batch pred_ei_list[0] shape={tuple(pei0.shape)} dtype={pei0.dtype} dev={pei0.device}")
                if torch.is_tensor(pea0):
                    print(f"[DEC-CHK] batch pred_edge_attr_list[0] shape={tuple(pea0.shape)} dtype={pea0.dtype} dev={pea0.device}")
                    if pea0.ndim == 2 and pea0.size(1) >= 5:
                        # quick sanity stats on tau column
                        tau = pea0[:, 4]
                        print(f"[DEC-CHK] tau stats: min={tau.min().item():.3e} max={tau.max().item():.3e} mean={tau.mean().item():.3e}")
            else:
                print("[DEC-CHK] Collate did NOT attach pred_edge_attr_list (key missing).")
            self._printed_dec_once = True

        # ----- optional pred→pred IDW maps (length = K-1) -----
        pred2pred_idx = self.precomp.get("pred2pred_idx", None)
        pred2pred_w   = self.precomp.get("pred2pred_w",   None)

        if pred2pred_idx is not None and pred2pred_w is not None:
            pred2pred_idx_list = []
            pred2pred_w_list   = []

            # For training step k = 1..K-1 (our loop is k=1..K-2),
            # we want a map from mesh at idxs[k] -> mesh at idxs[k+1].
            # Training then uses pred2pred_idx_list[k-1] at step k.
            for k in range(1, K):
                t_src = idxs[k]          # "source" time for this map
                pred2pred_idx_list.append(pred2pred_idx[t_src])
                pred2pred_w_list.append(pred2pred_w[t_src])

            ex["pred2pred_idx_list"] = pred2pred_idx_list
            ex["pred2pred_w_list"]   = pred2pred_w_list

            # Debug (optional)
            # print("K =", K)
            # for j, (idx_j, w_j) in enumerate(zip(pred2pred_idx_list, pred2pred_w_list)):
            #     if idx_j is None or w_j is None:
            #         print(f"  step {j}: idx=None or w=None")
            #     else:
#                 print(f"  step {j}: idx shape={idx_j.shape}, w shape={w_j.shape}")
        return ex


class CollateWithDtOnly:
    """
    Lightweight collate for runtime mesh mode.

    Keeps dataset window tensors untouched and only appends:
      - dt_list: list[float], length K-1
      - dt_ref : optional float scalar
    """

    def __init__(self, *, dt_transitions, dt_ref=None):
        if torch.is_tensor(dt_transitions):
            self.dt_transitions = dt_transitions.detach().cpu()
        else:
            self.dt_transitions = torch.as_tensor(dt_transitions, dtype=torch.float32, device="cpu")
        self.dt_ref = dt_ref

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ex0 = batch[0]
        ex = dict(ex0)

        idxs = ex["t_indices"].tolist()  # absolute time indices in this window
        K = len(idxs)

        dt_list = []
        for j in range(K - 1):
            t_src = idxs[j]
            v = self.dt_transitions[t_src]
            dt_list.append(float(v.item()) if torch.is_tensor(v) else float(v))

        ex["dt_list"] = dt_list
        if self.dt_ref is not None:
            ex["dt_ref"] = float(self.dt_ref) if not torch.is_tensor(self.dt_ref) else float(self.dt_ref.item())
        return ex
