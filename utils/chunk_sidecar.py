from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError("chunk sidecar builder requires h5py (pip install h5py).") from e


@dataclass(frozen=True)
class ChunkBuilderConfig:
    precomp_path: str
    sidecar_path: str
    overwrite: bool
    progress: bool
    num_chunks: int
    halo_hops: int
    partition_mode: str
    compute_activity: bool
    activity_feature_idx: Optional[List[int]]
    activity_threshold: float
    max_timesteps: Optional[int]


def _to_abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def derive_sidecar_path(precomp_path: str, configured_path: str | None) -> str:
    if configured_path:
        return _to_abs_path(configured_path)
    root, ext = os.path.splitext(_to_abs_path(precomp_path))
    if ext.lower() != ".h5":
        return root + ".chunks.h5"
    return root + "_chunks.h5"


def _resolve_cfg(cfg: Dict[str, Any]) -> ChunkBuilderConfig:
    train_cfg = cfg.get("train", {}) or {}
    chunk_cfg = cfg.get("chunk", {}) or {}
    builder_cfg = chunk_cfg.get("builder", {}) or {}

    precomp_raw = str(train_cfg.get("precomp_cache_path", "")).strip()
    if not precomp_raw:
        raise ValueError("Missing train.precomp_cache_path in config.")
    precomp_path = _to_abs_path(precomp_raw)
    if not os.path.exists(precomp_path):
        raise FileNotFoundError(f"Precompute H5 not found: {precomp_path}")

    num_chunks = int(chunk_cfg.get("num_chunks", 16))
    if num_chunks < 1:
        raise ValueError(f"chunk.num_chunks must be >= 1, got {num_chunks}")

    halo_hops = int(chunk_cfg.get("halo_hops", 2))
    if halo_hops < 0:
        raise ValueError(f"chunk.halo_hops must be >= 0, got {halo_hops}")

    partition_mode = str(chunk_cfg.get("partition_mode", "parent_grid")).strip().lower()
    if partition_mode != "parent_grid":
        raise ValueError(
            f"Unsupported chunk.partition_mode='{partition_mode}'. "
            "Only 'parent_grid' is implemented in this standalone builder."
        )

    configured_sidecar = builder_cfg.get("sidecar_path", "")
    sidecar_path = derive_sidecar_path(precomp_path, configured_sidecar)
    overwrite = bool(builder_cfg.get("overwrite", False))
    progress = bool(builder_cfg.get("progress", True))
    compute_activity = bool(builder_cfg.get("compute_activity", True))
    activity_threshold = float(chunk_cfg.get("activity_threshold", 0.0))

    raw_feat_idx = builder_cfg.get("activity_feature_idx", None)
    activity_feature_idx: Optional[List[int]] = None
    if isinstance(raw_feat_idx, (list, tuple)) and len(raw_feat_idx) > 0:
        activity_feature_idx = sorted({int(v) for v in raw_feat_idx if int(v) >= 0})
    elif isinstance(raw_feat_idx, str) and raw_feat_idx.strip():
        activity_feature_idx = sorted(
            {int(tok.strip()) for tok in raw_feat_idx.split(",") if tok.strip()}
        )

    max_timesteps = builder_cfg.get("max_timesteps", None)
    if max_timesteps is not None:
        max_timesteps = int(max_timesteps)
        if max_timesteps <= 0:
            max_timesteps = None

    return ChunkBuilderConfig(
        precomp_path=precomp_path,
        sidecar_path=sidecar_path,
        overwrite=overwrite,
        progress=progress,
        num_chunks=num_chunks,
        halo_hops=halo_hops,
        partition_mode=partition_mode,
        compute_activity=compute_activity,
        activity_feature_idx=activity_feature_idx,
        activity_threshold=activity_threshold,
        max_timesteps=max_timesteps,
    )


def _write_ds(g: "h5py.Group", name: str, arr: np.ndarray) -> None:
    if name in g:
        del g[name]
    g.create_dataset(
        name,
        data=arr,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=True,
    )


def _factor_grid(num_chunks: int, H: int, W: int) -> Tuple[int, int]:
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


def _tile_bounds(num_chunks: int, H: int, W: int) -> List[Tuple[int, int, int, int, int]]:
    rows, cols = _factor_grid(num_chunks, H, W)
    y_edges = np.linspace(0, H, rows + 1, dtype=np.int64)
    x_edges = np.linspace(0, W, cols + 1, dtype=np.int64)

    out: List[Tuple[int, int, int, int, int]] = []
    cid = 0
    for ry in range(rows):
        y0, y1 = int(y_edges[ry]), int(y_edges[ry + 1])
        for cx in range(cols):
            x0, x1 = int(x_edges[cx]), int(x_edges[cx + 1])
            out.append((cid, y0, y1, x0, x1))
            cid += 1
    return out


def _core_mask_from_parents(parents: np.ndarray, W: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    row = parents // int(W)
    col = parents % int(W)
    return (row >= y0) & (row < y1) & (col >= x0) & (col < x1)


def _expand_halo(core_mask: np.ndarray, edge_index: np.ndarray, hops: int) -> np.ndarray:
    if hops <= 0 or core_mask.size == 0:
        return core_mask.copy()
    if edge_index.size == 0:
        return core_mask.copy()

    full_mask = core_mask.copy()
    frontier = core_mask.copy()

    src = edge_index[0]
    dst = edge_index[1]

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


def _build_local_edge_index(
    edge_index: np.ndarray,
    full_mask: np.ndarray,
    full_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if edge_index.size == 0 or full_idx.size == 0:
        return np.zeros((2, 0), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    src = edge_index[0]
    dst = edge_index[1]
    keep = full_mask[src] & full_mask[dst]
    if not np.any(keep):
        return np.zeros((2, 0), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    lut = np.full((full_mask.size,), fill_value=-1, dtype=np.int32)
    lut[full_idx] = np.arange(full_idx.size, dtype=np.int32)

    src_l = lut[src[keep]]
    dst_l = lut[dst[keep]]
    edge_local = np.stack([src_l, dst_l], axis=0).astype(np.int32, copy=False)
    edge_ids = np.nonzero(keep)[0].astype(np.int32, copy=False)
    return edge_local, edge_ids


def _compute_activity_score(
    feat_t: Optional[np.ndarray],
    feat_tp1: Optional[np.ndarray],
    area: Optional[np.ndarray],
    core_idx: np.ndarray,
    feature_idx: Optional[List[int]],
) -> float:
    if core_idx.size == 0:
        return float("nan")
    if feat_t is None or feat_tp1 is None:
        return float("nan")
    if feat_t.shape != feat_tp1.shape:
        return float("nan")

    delta = np.abs(feat_tp1.astype(np.float64, copy=False) - feat_t.astype(np.float64, copy=False))
    if delta.ndim != 2 or delta.shape[0] == 0:
        return float("nan")

    if feature_idx:
        valid_idx = [i for i in feature_idx if 0 <= int(i) < int(delta.shape[1])]
        if len(valid_idx) == 0:
            return float("nan")
        delta = delta[:, valid_idx]

    per_node = np.mean(delta, axis=1)
    core_vals = per_node[core_idx]

    if area is None or area.shape[0] != per_node.shape[0]:
        return float(np.mean(core_vals))

    w = area.astype(np.float64, copy=False)[core_idx]
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return float(np.mean(core_vals))
    return float(np.sum(core_vals * w) / w_sum)


def _sha1_file(path: str, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_optional(g: "h5py.Group", name: str) -> Optional[np.ndarray]:
    if name not in g:
        return None
    return g[name][...]


def _read_group_or_static(g_t: "h5py.Group", g_static: Optional["h5py.Group"], name: str) -> Optional[np.ndarray]:
    arr = _read_optional(g_t, name)
    if arr is not None:
        return arr
    if g_static is not None:
        return _read_optional(g_static, name)
    return None


def _collect_timestep_groups(f: "h5py.File", T_meta: Optional[int]) -> List[int]:
    ts: List[int] = []
    for key in f.keys():
        if not (len(key) == 6 and key.startswith("t")):
            continue
        idx_str = key[1:]
        if not idx_str.isdigit():
            continue
        t = int(idx_str)
        if t <= 0:
            continue
        if T_meta is not None and t >= int(T_meta):
            continue
        ts.append(t)
    ts = sorted(set(ts))
    return ts


def build_chunk_sidecar(cfg: Dict[str, Any]) -> Dict[str, Any]:
    c = _resolve_cfg(cfg)
    os.makedirs(os.path.dirname(c.sidecar_path) or ".", exist_ok=True)

    if os.path.exists(c.sidecar_path) and (not c.overwrite):
        raise FileExistsError(
            f"Sidecar exists and chunk.builder.overwrite=false: {c.sidecar_path}"
        )

    with h5py.File(c.precomp_path, "r") as fin:
        if "meta" not in fin:
            raise RuntimeError(f"Precompute file missing /meta: {c.precomp_path}")

        meta_in = fin["meta"]
        H = int(meta_in.attrs.get("H", -1))
        W = int(meta_in.attrs.get("W", -1))
        T_meta_raw = meta_in.attrs.get("T", None)
        T_meta = int(T_meta_raw) if T_meta_raw is not None else None

        if H <= 0 or W <= 0:
            raise RuntimeError(
                f"Invalid H/W in source precomp meta: H={H}, W={W}, path={c.precomp_path}"
            )

        max_chunks = int(H * W)
        if c.num_chunks > max_chunks:
            raise ValueError(
                f"chunk.num_chunks={c.num_chunks} exceeds H*W={max_chunks} parent cells."
            )

        tile_spec = _tile_bounds(c.num_chunks, H, W)
        chunk_bounds = np.asarray([[y0, y1, x0, x1] for (_cid, y0, y1, x0, x1) in tile_spec], dtype=np.int32)

        timesteps = _collect_timestep_groups(fin, T_meta)
        if c.max_timesteps is not None:
            timesteps = timesteps[: int(c.max_timesteps)]
        if len(timesteps) == 0:
            raise RuntimeError("No timestep groups found in source precomp (expected t00001+).")

        g_static = fin.get("static", None)

        with h5py.File(c.sidecar_path, "w") as fout:
            meta_out = fout.create_group("meta")
            meta_out.attrs["source_precomp_path"] = np.bytes_(c.precomp_path)
            meta_out.attrs["source_precomp_sha1"] = np.bytes_(_sha1_file(c.precomp_path))
            meta_out.attrs["generated_utc"] = np.bytes_(_dt.datetime.utcnow().isoformat(timespec="seconds"))
            meta_out.attrs["H"] = int(H)
            meta_out.attrs["W"] = int(W)
            meta_out.attrs["n_timesteps"] = int(len(timesteps))
            meta_out.attrs["n_chunks"] = int(c.num_chunks)
            meta_out.attrs["halo_hops"] = int(c.halo_hops)
            meta_out.attrs["partition_mode"] = np.bytes_(str(c.partition_mode))
            meta_out.attrs["compute_activity"] = np.uint8(1 if c.compute_activity else 0)
            meta_out.attrs["activity_threshold"] = float(c.activity_threshold)
            meta_out.attrs["activity_feature_idx"] = np.bytes_(
                ",".join(str(v) for v in (c.activity_feature_idx or []))
            )

            _write_ds(meta_out, "timesteps", np.asarray(timesteps, dtype=np.int32))
            _write_ds(meta_out, "chunk_bounds_yx", chunk_bounds)
            meta_out.create_dataset(
                "builder_cfg_json",
                data=np.bytes_(
                    json.dumps(
                        {
                            "precomp_path": c.precomp_path,
                            "sidecar_path": c.sidecar_path,
                            "overwrite": c.overwrite,
                            "progress": c.progress,
                            "num_chunks": c.num_chunks,
                            "halo_hops": c.halo_hops,
                            "partition_mode": c.partition_mode,
                            "compute_activity": c.compute_activity,
                            "activity_feature_idx": c.activity_feature_idx,
                            "activity_threshold": c.activity_threshold,
                            "max_timesteps": c.max_timesteps,
                        },
                        sort_keys=True,
                    )
                ),
            )

            chunks_root = fout.create_group("chunks")
            n_total_chunks = int(len(timesteps) * c.num_chunks)
            write_count = 0

            for it, t in enumerate(timesteps, start=1):
                g_in = fin[f"t{int(t):05d}"]

                parents = _read_group_or_static(g_in, g_static, "pred_parents")
                ei = _read_group_or_static(g_in, g_static, "pred_ei")
                if parents is None or ei is None:
                    if c.progress:
                        print(f"[CHUNK-SIDECAR][WARN] skip t={t}: missing pred_parents/pred_ei")
                    continue

                parents = parents.astype(np.int64, copy=False).reshape(-1)
                ei = ei.astype(np.int64, copy=False)
                if ei.ndim != 2:
                    raise RuntimeError(f"t={t}: pred_ei must be 2D, got shape {ei.shape}")
                if ei.shape[0] != 2 and ei.shape[1] == 2:
                    ei = ei.T
                if ei.shape[0] != 2:
                    raise RuntimeError(f"t={t}: pred_ei must be shape (2,E), got {ei.shape}")

                N = int(parents.shape[0])
                if ei.size > 0:
                    if int(ei.min()) < 0 or int(ei.max()) >= N:
                        raise RuntimeError(
                            f"t={t}: pred_ei out of bounds for N={N} "
                            f"(min={int(ei.min())}, max={int(ei.max())})."
                        )

                feat_t = _read_optional(g_in, "feat_t_on_pred") if c.compute_activity else None
                feat_tp1 = _read_optional(g_in, "feat_tp1_on_pred") if c.compute_activity else None
                area = _read_group_or_static(g_in, g_static, "pred_cell_area") if c.compute_activity else None
                if area is not None:
                    area = area.astype(np.float64, copy=False).reshape(-1)

                g_out_t = chunks_root.create_group(f"t{int(t):05d}")
                g_out_t.attrs["N_pred"] = int(N)
                g_out_t.attrs["E_pred"] = int(ei.shape[1]) if ei.size > 0 else 0

                for cid, y0, y1, x0, x1 in tile_spec:
                    core_mask = _core_mask_from_parents(parents, W=W, y0=y0, y1=y1, x0=x0, x1=x1)
                    full_mask = _expand_halo(core_mask, edge_index=ei, hops=c.halo_hops)
                    halo_mask = full_mask & (~core_mask)

                    core_idx = np.nonzero(core_mask)[0].astype(np.int32, copy=False)
                    halo_idx = np.nonzero(halo_mask)[0].astype(np.int32, copy=False)
                    full_idx = np.concatenate([core_idx, halo_idx], axis=0).astype(np.int32, copy=False)

                    edge_local, edge_ids_local = _build_local_edge_index(
                        edge_index=ei,
                        full_mask=full_mask,
                        full_idx=full_idx,
                    )
                    core_mask_local = np.zeros((full_idx.size,), dtype=np.uint8)
                    core_mask_local[: core_idx.size] = 1

                    activity = _compute_activity_score(
                        feat_t=feat_t,
                        feat_tp1=feat_tp1,
                        area=area,
                        core_idx=core_idx.astype(np.int64, copy=False),
                        feature_idx=c.activity_feature_idx,
                    )
                    is_active = bool(np.isfinite(activity) and (activity >= c.activity_threshold))

                    g_chunk = g_out_t.create_group(f"chunk_{int(cid):04d}")
                    _write_ds(g_chunk, "core_idx", core_idx)
                    _write_ds(g_chunk, "halo_idx", halo_idx)
                    _write_ds(g_chunk, "full_idx", full_idx)
                    _write_ds(g_chunk, "core_mask_local_u8", core_mask_local)
                    _write_ds(g_chunk, "edge_index_local", edge_local.astype(np.int32, copy=False))
                    _write_ds(g_chunk, "edge_ids_local", edge_ids_local.astype(np.int32, copy=False))

                    g_chunk.attrs["n_core"] = int(core_idx.size)
                    g_chunk.attrs["n_halo"] = int(halo_idx.size)
                    g_chunk.attrs["n_full"] = int(full_idx.size)
                    g_chunk.attrs["e_local"] = int(edge_local.shape[1])
                    g_chunk.attrs["activity_score"] = float(activity)
                    g_chunk.attrs["is_active"] = np.uint8(1 if is_active else 0)
                    g_chunk.attrs["tile_y0"] = int(y0)
                    g_chunk.attrs["tile_y1"] = int(y1)
                    g_chunk.attrs["tile_x0"] = int(x0)
                    g_chunk.attrs["tile_x1"] = int(x1)

                    write_count += 1

                if c.progress:
                    if it == 1 or it == len(timesteps) or (it % max(1, len(timesteps) // 20) == 0):
                        print(
                            f"[CHUNK-SIDECAR] timesteps {it}/{len(timesteps)} "
                            f"(chunks written {write_count}/{n_total_chunks})"
                        )

            fout.flush()

    return {
        "sidecar_path": c.sidecar_path,
        "source_precomp_path": c.precomp_path,
        "num_chunks": int(c.num_chunks),
        "halo_hops": int(c.halo_hops),
        "timesteps": int(len(timesteps)),
    }


def build_chunk_sidecar_from_config_path(config_path: str) -> Dict[str, Any]:
    config_path = _to_abs_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return build_chunk_sidecar(cfg)


class ChunkSidecarH5:
    """
    Lightweight reader for chunk sidecar metadata.

    Expected structure:
      /chunks/t00001/chunk_0000/{core_idx,halo_idx,full_idx,core_mask_local_u8,edge_index_local,edge_ids_local?}
    """

    def __init__(self, sidecar_path: str):
        self.path = _to_abs_path(sidecar_path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Chunk sidecar H5 not found: {self.path}")
        self._f: Optional[h5py.File] = None
        self._n_chunks: Optional[int] = None
        self._timesteps: Optional[List[int]] = None

    def _open(self) -> None:
        if self._f is None:
            self._f = h5py.File(self.path, "r")

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _chunks_group(self) -> h5py.Group:
        self._open()
        if self._f is None or "chunks" not in self._f:
            raise RuntimeError(f"Invalid chunk sidecar (missing /chunks): {self.path}")
        return self._f["chunks"]

    @property
    def n_chunks(self) -> int:
        if self._n_chunks is not None:
            return int(self._n_chunks)
        self._open()
        if self._f is None or "meta" not in self._f:
            return 0
        meta = self._f["meta"]
        self._n_chunks = int(meta.attrs.get("n_chunks", 0))
        return int(self._n_chunks)

    @property
    def timesteps(self) -> List[int]:
        if self._timesteps is not None:
            return list(self._timesteps)
        self._open()
        if self._f is None or "meta" not in self._f:
            self._timesteps = []
            return []
        meta = self._f["meta"]
        if "timesteps" in meta:
            arr = np.asarray(meta["timesteps"][...], dtype=np.int64).reshape(-1)
            self._timesteps = [int(v) for v in arr.tolist()]
            return list(self._timesteps)
        out: List[int] = []
        chunks = self._chunks_group()
        for key in chunks.keys():
            if len(key) == 6 and key.startswith("t") and key[1:].isdigit():
                out.append(int(key[1:]))
        self._timesteps = sorted(set(out))
        return list(self._timesteps)

    def has_timestep(self, t_abs: int) -> bool:
        key = f"t{int(t_abs):05d}"
        chunks = self._chunks_group()
        return key in chunks

    def get_timestep_chunks(self, t_abs: int) -> Optional[List[Dict[str, Any]]]:
        key = f"t{int(t_abs):05d}"
        chunks = self._chunks_group()
        if key not in chunks:
            return None
        g_t = chunks[key]

        out: List[Dict[str, Any]] = []
        # deterministic order by chunk id
        names = sorted(
            [k for k in g_t.keys() if str(k).startswith("chunk_")],
            key=lambda s: int(str(s).split("_")[-1]),
        )
        for name in names:
            g = g_t[name]
            core_idx = np.asarray(g["core_idx"][...], dtype=np.int64).reshape(-1)
            halo_idx = np.asarray(g["halo_idx"][...], dtype=np.int64).reshape(-1)
            full_idx = np.asarray(g["full_idx"][...], dtype=np.int64).reshape(-1)
            core_mask_local = np.asarray(g["core_mask_local_u8"][...], dtype=np.uint8).reshape(-1)

            ei = np.asarray(g["edge_index_local"][...], dtype=np.int64)
            if ei.ndim == 2 and ei.shape[0] != 2 and ei.shape[1] == 2:
                ei = ei.T
            edge_ids_local = None
            if "edge_ids_local" in g:
                edge_ids_local = np.asarray(g["edge_ids_local"][...], dtype=np.int64).reshape(-1)

            out.append(
                {
                    "chunk_name": str(name),
                    "core_idx": core_idx,
                    "halo_idx": halo_idx,
                    "full_idx": full_idx,
                    "core_mask_local_u8": core_mask_local,
                    "edge_index_local": ei,
                    "edge_ids_local": edge_ids_local,
                    "activity_score": float(g.attrs.get("activity_score", float("nan"))),
                    "is_active": bool(int(g.attrs.get("is_active", 0))),
                }
            )
        return out
