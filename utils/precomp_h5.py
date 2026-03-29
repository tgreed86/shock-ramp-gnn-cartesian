import os, json, hashlib
import numpy as np
import torch
from typing import Any, Dict, Optional

try:
    import h5py
except ImportError as e:
    raise ImportError("Option B.2 requires h5py (pip install h5py).") from e


def _normalize_device(dev: Any) -> torch.device:
    """
    Make serialized devices portable across machines.

    If an object was saved with device='cuda:0' on an HPC node, but we are loading
    on a machine where CUDA is not available (e.g., macOS), fall back to CPU.
    """
    d = torch.device(dev) if not isinstance(dev, torch.device) else dev

    if d.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    if d.type == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_ok:
            return torch.device("cpu")

    return d

def _cfg_sha1(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


class PrecompH5Writer:
    """
    One-file (HDF5) streaming writer: each timestep is a group t00000/, t00001/, ...
    """
    def __init__(self, path: str, cfg: dict, H: int, W: int, bbox, *, overwrite: bool = True):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mode = "w" if overwrite else "a"
        self.f = h5py.File(path, mode)

        # Write/overwrite meta
        if "meta" in self.f:
            del self.f["meta"]
        meta = self.f.create_group("meta")

        meta.attrs["H"] = int(H)
        meta.attrs["W"] = int(W)
        meta.attrs["bbox"] = np.asarray(bbox, dtype=np.float64)
        meta.attrs["cfg_sha1"] = _cfg_sha1(cfg)

        cfg_json = json.dumps(cfg, sort_keys=True, default=str)
        #meta.create_dataset("cfg_json", data=np.string_(cfg_json))
        meta.create_dataset("cfg_json", data=np.bytes_(json.dumps(cfg, sort_keys=True, default=str)))


        self.path = path

    def write_step(
        self,
        t: int,
        *,
        pred_centers: torch.Tensor,      # (N,2) float
        pred_levels: torch.Tensor,       # (N,)  int
        pred_parents: torch.Tensor,      # (N,)  int
        pred_ei: torch.Tensor,           # (2,E) int
        mask_pred_parent: torch.Tensor,  # (H,W) bool OR (H*W) bool
        feat_t_on_pred: torch.Tensor | None = None,     # (N,F) optional
        feat_tp1_on_pred: torch.Tensor | None = None,   # (N,F) optional
        stats: dict | None = None,
    ):
        gname = f"t{int(t):05d}"
        if gname in self.f:
            del self.f[gname]
        g = self.f.create_group(gname)

        # --- tensors -> numpy (CPU) ---
        pc = pred_centers.detach().cpu().to(torch.float32).numpy()
        pl = pred_levels.detach().cpu().to(torch.int16).numpy()
        pp = pred_parents.detach().cpu().to(torch.int32).numpy()

        ei = pred_ei.detach().cpu()
        # store edges as int32 to save space
        if ei.numel() == 0:
            ei_np = np.zeros((2, 0), dtype=np.int32)
        else:
            ei_np = ei.to(torch.int32).numpy()

        mp = mask_pred_parent.detach().cpu()
        mp = mp.view(-1) if mp.dim() == 2 else mp
        mp_np = mp.to(torch.uint8).numpy()  # 0/1

        # --- datasets (chunked + compressed) ---
        g.create_dataset("pred_centers", data=pc, compression="gzip", compression_opts=4, shuffle=True, chunks=True)
        g.create_dataset("pred_levels",  data=pl, compression="gzip", compression_opts=4, shuffle=True, chunks=True)
        g.create_dataset("pred_parents", data=pp, compression="gzip", compression_opts=4, shuffle=True, chunks=True)
        g.create_dataset("pred_ei",      data=ei_np, compression="gzip", compression_opts=4, shuffle=True, chunks=True)
        g.create_dataset("mask_pred_parent_flat_u8", data=mp_np, compression="gzip", compression_opts=4, shuffle=True, chunks=True)

        if feat_t_on_pred is not None:
            ft = feat_t_on_pred.detach().cpu().to(torch.float32).numpy()
            g.create_dataset("feat_t_on_pred", data=ft, compression="gzip", compression_opts=4, shuffle=True, chunks=True)

        if feat_tp1_on_pred is not None:
            f1 = feat_tp1_on_pred.detach().cpu().to(torch.float32).numpy()
            g.create_dataset("feat_tp1_on_pred", data=f1, compression="gzip", compression_opts=4, shuffle=True, chunks=True)

        if stats is not None:
            g.attrs["stats_json"] = json.dumps(stats, sort_keys=True, default=str)

        self.f.flush()

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()


class LazyPrecompH5(dict):
    """
    Dict-like drop-in replacement for the old dict-of-lists precomp.

    Examples:
      precomp["pred_centers"][t]      -> Tensor [N,2] or None
      precomp["pred_ei"][t]           -> Tensor [2,E] or None
      precomp["pred_edge_attr"][t]    -> Tensor [E?,C] or None   (we'll fix alignment later)
      precomp["pred_cell_wh"][t]      -> Tensor [N,2] or None
      precomp["pred_cell_area"][t]    -> Tensor [N]   or None
      precomp["pred2pred_idx"][t]     -> Tensor [Ndst,k] or None
      precomp["pred2pred_w"][t]       -> Tensor [Ndst,k] or None

    Non-indexed metadata:
      precomp["pred_edge_attr_layout"] -> str or None
    """
    def __init__(self, path: str, T: int, H: int, W: int, device: str | torch.device = "cpu"):
        super().__init__()
        if h5py is None:
            raise ImportError("LazyPrecompH5 requires h5py (pip install h5py).")

        self.path = str(path)
        self.T = int(T)
        self.H = int(H)
        self.W = int(W)
        #self.device = torch.device(device)
        self.device = _normalize_device(device)

        self._f = None  # lazy-open per process
        self._meta_cache = {}  # cache scalar attrs/values

        # expose the same keys as your old dict-of-lists PLUS DEC keys
        seq_keys = [
            "pred_centers", "pred_levels", "pred_parents", "pred_ei", "mask_pred",
            "feat_t_on_pred", "feat_tp1_on_pred",
            "pred2pred_idx", "pred2pred_w",
            # DEC additions
            "pred_edge_attr", "pred_cell_wh", "pred_cell_area",
        ]
        for k in seq_keys:
            super().__setitem__(k, _LazyPrecompSeq(self, k))

        # metadata / scalar keys (not time-indexed)
        super().__setitem__("pred_edge_attr_layout", _LazyPrecompScalar(self, "pred_edge_attr_layout"))

    def _open(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_f"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._f = None
        # Important: if this was pickled with device='cuda:0', fix it on machines without CUDA.
        self.device = _normalize_device(getattr(self, "device", "cpu"))

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None

    def _grp_name(self, t: int) -> str:
        return f"t{int(t):05d}"

    def _has_group(self, t: int) -> bool:
        self._open()
        return self._grp_name(t) in self._f

    def _read_np(self, t: int, name: str) -> Optional[np.ndarray]:
        self._open()
        gname = self._grp_name(t)
        if gname not in self._f:
            return None
        g = self._f[gname]
        if name not in g:
            return None
        return g[name][...]

    def _read_np_static(self, name: str) -> Optional[np.ndarray]:
        self._open()
        g = self._f.get("static", None)
        if g is None or name not in g:
            return None
        return g[name][...]

    def _to_torch(self, arr: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        dev = _normalize_device(device)
        return torch.from_numpy(arr).to(device=dev, dtype=dtype)

    # -------- scalar metadata --------
    def _get_scalar(self, key: str):
        # cache after first read
        if key in self._meta_cache:
            return self._meta_cache[key]

        self._open()

        # Prefer group attribute (t00001) if present; fallback to meta attrs
        val = None
        try:
            if "t00001" in self._f and key in self._f["t00001"].attrs:
                v = self._f["t00001"].attrs[key]
                val = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
            elif "meta" in self._f and key in self._f["meta"].attrs:
                v = self._f["meta"].attrs[key]
                val = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
        except Exception:
            val = None

        self._meta_cache[key] = val
        return val

    # -------- time-indexed tensors --------
    def _get(self, key: str, t: int):
        if t is None:
            return None
        t = int(t)
        if t < 0 or t >= self.T:
            return None

        # pred meshes / mapped features exist for t=1..T-1
        pred_series = (
            "pred_centers", "pred_levels", "pred_parents", "pred_ei", "mask_pred",
            "feat_t_on_pred", "feat_tp1_on_pred",
            # DEC series live in same groups
            "pred_edge_attr", "pred_cell_wh", "pred_cell_area",
        )
        if key in pred_series:
            if t == 0:
                return None
            if not self._has_group(t):
                return None

            def _read_group_or_static(ds_name: str) -> Optional[np.ndarray]:
                arr = self._read_np(t, ds_name)
                if arr is None:
                    arr = self._read_np_static(ds_name)
                return arr

            if key == "pred_centers":
                arr = _read_group_or_static("pred_centers")
                return None if arr is None else self._to_torch(arr, dtype=torch.float32, device=self.device)

            if key == "pred_levels":
                arr = _read_group_or_static("pred_levels")
                return None if arr is None else self._to_torch(arr, dtype=torch.long, device=self.device)

            if key == "pred_parents":
                arr = _read_group_or_static("pred_parents")
                return None if arr is None else self._to_torch(arr, dtype=torch.long, device=self.device)

            if key == "pred_ei":
                arr = _read_group_or_static("pred_ei")
                return None if arr is None else self._to_torch(arr, dtype=torch.long, device=self.device)

            if key == "mask_pred":
                arr = _read_group_or_static("mask_pred_parent_flat_u8")
                if arr is None:
                    return None
                m = torch.from_numpy(arr.astype(np.uint8, copy=False)).to(self.device)
                return m.view(self.H, self.W).to(torch.bool)

            if key == "feat_t_on_pred":
                arr = self._read_np(t, "feat_t_on_pred")
                return None if arr is None else self._to_torch(arr, dtype=torch.float32, device=self.device)

            if key == "feat_tp1_on_pred":
                arr = self._read_np(t, "feat_tp1_on_pred")
                return None if arr is None else self._to_torch(arr, dtype=torch.float32, device=self.device)

            # --- DEC additions ---
            if key == "pred_edge_attr":
                arr = _read_group_or_static("pred_edge_attr")
                return None if arr is None else self._to_torch(arr, dtype=torch.float32, device=self.device)

            if key == "pred_cell_wh":
                arr = _read_group_or_static("pred_cell_wh")
                return None if arr is None else self._to_torch(arr, dtype=torch.float32, device=self.device)

            if key == "pred_cell_area":
                arr = _read_group_or_static("pred_cell_area")
                return None if arr is None else self._to_torch(arr, dtype=torch.float32, device=self.device)

        # pred2pred maps exist for t=1..T-2 (stored in group t, mapping to t+1)
        if key in ("pred2pred_idx", "pred2pred_w"):
            if t < 1 or t >= self.T - 1:
                return None
            if not self._has_group(t):
                return None

            if key == "pred2pred_idx":
                arr = self._read_np(t, "pred2pred_idx_to_next")
                if arr is None:
                    arr = self._read_np_static("pred2pred_idx_to_next")
                return None if arr is None else self._to_torch(arr, dtype=torch.long, device=self.device)

            if key == "pred2pred_w":
                arr = self._read_np(t, "pred2pred_w_to_next")
                if arr is None:
                    arr = self._read_np_static("pred2pred_w_to_next")
                if arr is None:
                    return None
                return self._to_torch(arr.astype(np.float32, copy=False), dtype=torch.float32, device=self.device)

        return None
'''

def _normalize_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


class LazyPrecompH5(dict):
    """
    H5-backed, dict-like precomp with time-indexable sequences.

    Timestep groups are expected at /t00001..../t{T-1:05d}
    MLS datasets are expected under /tXXXXX/mls/*
    """
    def __init__(self, path: str, T: int, H: int, W: int, device: str | torch.device = "cpu"):
        super().__init__()
        if h5py is None:
            raise ImportError("LazyPrecompH5 requires h5py (pip install h5py).")

        self.path = str(path)
        self.T = int(T)
        self.H = int(H)
        self.W = int(W)
        self.device = _normalize_device(device)

        self._f = None
        self._meta_cache = {}

        # time-indexed keys
        seq_keys = [
            # base geometry + mappings
            "pred_centers", "pred_levels", "pred_parents", "pred_ei", "mask_pred",
            "feat_t_on_pred", "feat_tp1_on_pred",
            "pred2pred_idx", "pred2pred_w",
            # DEC
            "pred_edge_attr", "pred_cell_wh", "pred_cell_area",
            # MLS (mapped to subgroup datasets)
            "mls_grad_M_inv", "mls_grad_dX", "mls_lap_w",
            "mls_edge_index",              # convenience: edges used by MLS path
            "mls_grad_node_damp", "mls_lap_node_damp",
        ]
        for k in seq_keys:
            super().__setitem__(k, _LazyPrecompSeq(self, k))

        # non-indexed metadata keys (optional)
        super().__setitem__("pred_edge_attr_layout", _LazyPrecompScalar(self, "pred_edge_attr_layout"))

    def _ensure_open(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")

    def _group(self, t: int):
        """
        Returns group /t{t:05d} if exists, else None.
        Note: your writer starts at t00001. So t==0 returns None.
        """
        if t is None:
            return None
        t = int(t)
        if t <= 0:
            return None
        self._ensure_open()
        gname = f"t{t:05d}"
        if gname not in self._f:
            return None
        return self._f[gname]

    def _to_tensor(self, arr, *, dtype=None, device=None):
        t = torch.from_numpy(arr)
        if dtype is not None:
            t = t.to(dtype=dtype)
        if device is None:
            device = self.device
        return t.to(device=device)

    def _get(self, key: str, t: int):
        g = self._group(t)
        if g is None:
            return None

        # ---------- root datasets ----------
        if key == "pred_centers":
            if "pred_centers" not in g: return None
            return self._to_tensor(g["pred_centers"][...].astype(np.float32, copy=False), dtype=torch.float32)

        if key == "pred_levels":
            if "pred_levels" not in g: return None
            # store as int64 for downstream ops
            return self._to_tensor(g["pred_levels"][...].astype(np.int64, copy=False), dtype=torch.int64)

        if key == "pred_parents":
            if "pred_parents" not in g: return None
            return self._to_tensor(g["pred_parents"][...].astype(np.int64, copy=False), dtype=torch.int64)

        if key == "pred_ei":
            if "pred_ei" not in g: return None
            return self._to_tensor(g["pred_ei"][...].astype(np.int64, copy=False), dtype=torch.int64)

        if key == "mask_pred":
            # your writer uses mask_pred_parent_flat_u8
            if "mask_pred_parent_flat_u8" not in g: return None
            m = g["mask_pred_parent_flat_u8"][...].astype(np.uint8, copy=False)
            m = (m != 0)
            return self._to_tensor(m.astype(np.bool_, copy=False), dtype=torch.bool)

        if key == "feat_t_on_pred":
            if "feat_t_on_pred" not in g: return None
            return self._to_tensor(g["feat_t_on_pred"][...].astype(np.float32, copy=False), dtype=torch.float32)

        if key == "feat_tp1_on_pred":
            if "feat_tp1_on_pred" not in g: return None
            return self._to_tensor(g["feat_tp1_on_pred"][...].astype(np.float32, copy=False), dtype=torch.float32)

        if key == "pred2pred_idx":
            if "pred2pred_idx_to_next" not in g: return None
            return self._to_tensor(g["pred2pred_idx_to_next"][...].astype(np.int64, copy=False), dtype=torch.int64)

        if key == "pred2pred_w":
            if "pred2pred_w_to_next" not in g: return None
            # stored float16; keep float16 unless you prefer float32
            return self._to_tensor(g["pred2pred_w_to_next"][...].astype(np.float16, copy=False), dtype=torch.float16)

        # ---------- DEC ----------
        if key == "pred_edge_attr":
            if "pred_edge_attr" not in g: return None
            return self._to_tensor(g["pred_edge_attr"][...].astype(np.float32, copy=False), dtype=torch.float32)

        if key == "pred_cell_wh":
            if "pred_cell_wh" not in g: return None
            return self._to_tensor(g["pred_cell_wh"][...].astype(np.float32, copy=False), dtype=torch.float32)

        if key == "pred_cell_area":
            if "pred_cell_area" not in g: return None
            return self._to_tensor(g["pred_cell_area"][...].astype(np.float32, copy=False), dtype=torch.float32)

        # ---------- MLS subgroup ----------
        mg = g.get("mls", None)

        if key == "mls_grad_M_inv":
            if mg is None or "grad_M_inv" not in mg: return None
            return self._to_tensor(mg["grad_M_inv"][...].astype(np.float32, copy=False), dtype=torch.float32)

        if key == "mls_grad_dX":
            if mg is None or "grad_dX" not in mg: return None
            return self._to_tensor(mg["grad_dX"][...].astype(np.float32, copy=False), dtype=torch.float32)

        if key == "mls_lap_w":
            # your writer calls this "lap_weights"
            if mg is None or "lap_weights" not in mg: return None
            # could be float16 or float32 in file; upcast to float32 for math
            w = mg["lap_weights"][...]
            if w.dtype == np.float16:
                w = w.astype(np.float32, copy=False)
            else:
                w = w.astype(np.float32, copy=False)
            return self._to_tensor(w, dtype=torch.float32)

        if key == "mls_grad_node_damp":
            if mg is None or "grad_node_damp" not in mg: return None
            return self._to_tensor(mg["grad_node_damp"][...].astype(np.float32, copy=False), dtype=torch.float32)

        if key == "mls_lap_node_damp":
            if mg is None or "lap_node_damp" not in mg: return None
            return self._to_tensor(mg["lap_node_damp"][...].astype(np.float32, copy=False), dtype=torch.float32)

        if key == "mls_edge_index":
            # Prefer grad_ei_used (2-hop augmented), else lap_ei_base, else pred_ei
            if mg is not None:
                if "grad_ei_used" in mg:
                    return self._to_tensor(mg["grad_ei_used"][...].astype(np.int64, copy=False), dtype=torch.int64)
                if "lap_ei_base" in mg:
                    return self._to_tensor(mg["lap_ei_base"][...].astype(np.int64, copy=False), dtype=torch.int64)
            # fallback to face-adj pred_ei (still valid)
            if "pred_ei" in g:
                return self._to_tensor(g["pred_ei"][...].astype(np.int64, copy=False), dtype=torch.int64)
            return None

        return None

    def _get_scalar(self, key: str):
        self._ensure_open()
        meta = self._f.get("meta", None)
        if meta is None:
            return None
        if key in self._meta_cache:
            return self._meta_cache[key]

        # attr example for pred_edge_attr_layout stored as bytes
        v = meta.attrs.get(key, None)
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("utf-8")
        self._meta_cache[key] = v
        return v

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None

    def __del__(self):
        self.close()
'''

class _LazyPrecompSeq:
    """Sequence-like wrapper: seq[t] calls LazyPrecompH5._get(key,t)."""
    def __init__(self, owner: LazyPrecompH5, key: str):
        self.owner = owner
        self.key = key

    def __len__(self):
        return self.owner.T

    def __getitem__(self, t: int):
        return self.owner._get(self.key, t)


class _LazyPrecompScalar:
    """Scalar-like wrapper: precomp['pred_edge_attr_layout'] returns str/None."""
    def __init__(self, owner: LazyPrecompH5, key: str):
        self.owner = owner
        self.key = key

    def get(self):
        return self.owner._get_scalar(self.key)

    def __repr__(self):
        return repr(self.get())

    def __str__(self):
        v = self.get()
        return "" if v is None else str(v)

    # allow direct access pattern: precomp["pred_edge_attr_layout"]
    def __call__(self):
        return self.get()


def precomp_h5_is_usable(
    path: str,
    cfg: dict,
    expected_steps: int,
    *,
    H: int,
    W: int,
    require_dec: bool = False,
    require_pred2pred: bool = False,
    require_mls: bool = False,
    verbose: bool = False,
) -> bool:
    import os, h5py, numpy as np, json, hashlib

    def _v(msg):
        if verbose:
            print("[PRECOMP CHECK]", msg)

    if not os.path.exists(path):
        _v("reject: file does not exist")
        return False
    
    # MLS schema options (adjust if your writer used different names)
    MLS_REQUIRED_SCHEMAS = [
        # Schema A: explicit per-node/per-edge arrays
        ("mls_grad_M_inv", "mls_grad_dX", "mls_lap_w"),
        # Schema B: gradient only (if you sometimes precompute only grad pieces)
        # Uncomment if you want this to be accepted when require_mls=True.
        # ("mls_grad_M_inv", "mls_grad_dX"),
    ]

    try:
        with h5py.File(path, "r") as f:
            if "meta" not in f:
                _v("reject: missing /meta group")
                return False

            meta = f["meta"]
            static_g = f.get("static", None)

            # ---- T check
            stored_T = meta.attrs.get("T", None)
            if stored_T is None:
                _v("reject: meta.attrs['T'] missing")
                return False
            stored_T = int(stored_T)
            if stored_T != int(expected_steps):
                _v(f"reject: T mismatch stored_T={stored_T} expected_steps={expected_steps}")
                return False

            # ---- H/W check (matches what your writer stores)
            if int(meta.attrs.get("H", -1)) != int(H) or int(meta.attrs.get("W", -1)) != int(W):
                _v(f"reject: H/W mismatch stored=({meta.attrs.get('H')},{meta.attrs.get('W')}) expected=({H},{W})")
                return False

            # ---- IMPORTANT CHANGE: do NOT reject on cfg_sha1 mismatch
            stored_sha = meta.attrs.get("cfg_sha1", None)
            if isinstance(stored_sha, (bytes, np.bytes_)):
                stored_sha = stored_sha.decode("utf-8")
            if stored_sha is None:
                _v("warn: meta.attrs['cfg_sha1'] missing (continuing)")
            else:
                cfg_json = json.dumps(cfg, sort_keys=True, default=str)
                cur_sha = hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()
                if stored_sha != cur_sha:
                    _v(f"warn: cfg_sha1 mismatch stored={stored_sha} current={cur_sha} (continuing)")

            # ---- IMPORTANT CHANGE: validate streaming layout (t00001..t{T-1}), NOT '/steps'
            for t in range(1, stored_T):
                gname = f"t{t:05d}"
                if gname not in f:
                    _v(f"reject: missing group {gname}")
                    return False

                g = f[gname]

                def _has_ds(ds_name: str) -> bool:
                    if ds_name in g:
                        return True
                    return (static_g is not None) and (ds_name in static_g)

                # minimum datasets written in FIRST PASS
                required_geom = [
                    "pred_centers", "pred_levels", "pred_parents", "pred_ei",
                    "mask_pred_parent_flat_u8",
                ]
                required_feats = [
                    "feat_t_on_pred", "feat_tp1_on_pred",
                ]
                for ds in required_geom:
                    if not _has_ds(ds):
                        _v(f"reject: group {gname} missing dataset '{ds}'")
                        return False
                for ds in required_feats:
                    if ds not in g:
                        _v(f"reject: group {gname} missing dataset '{ds}'")
                        return False

                if require_dec:
                    for ds in ("pred_edge_attr", "pred_cell_wh", "pred_cell_area"):
                        if not _has_ds(ds):
                            _v(f"reject: group {gname} missing DEC dataset '{ds}'")
                            return False
                '''
                if require_mls:
                    # accept if ANY one of the schemas is satisfied
                    ok = False
                    for schema in MLS_REQUIRED_SCHEMAS:
                        if all((name in g) for name in schema):
                            ok = True
                            break
                    if not ok:
                        # helpful message: show which MLS keys are present
                        present = [k for k in g.keys() if str(k).startswith("mls_")]
                        _v(
                            f"reject: group {gname} missing required MLS datasets. "
                            f"Present mls_* keys: {present}"
                        )
                        return False
                '''
                '''
                if require_mls:
                    if "mls" not in g:
                        _v(f"reject: group {gname} missing 'mls' subgroup")
                        return False

                    mg = g["mls"]

                    # What your writer actually produces:
                    #   grad: grad_M_inv, grad_dX (and grad_ei_used optional)
                    #   lap : lap_weights (lap_ei_base optional but usually present)
                    need = ["grad_M_inv", "grad_dX", "lap_weights"]

                    missing = [k for k in need if k not in mg]
                    if missing:
                        present = list(mg.keys())
                        _v(
                            f"reject: group {gname} missing MLS datasets in /mls: {missing}. "
                            f"Present /mls keys: {present}"
                        )
                        return False
                '''

            if require_pred2pred:
                for t in range(1, stored_T - 1):
                    gname = f"t{t:05d}"
                    g = f[gname]
                    has_idx = ("pred2pred_idx_to_next" in g) or ((static_g is not None) and ("pred2pred_idx_to_next" in static_g))
                    has_w = ("pred2pred_w_to_next" in g) or ((static_g is not None) and ("pred2pred_w_to_next" in static_g))
                    if (not has_idx) or (not has_w):
                        _v(f"reject: missing pred2pred maps in {gname}")
                        return False

    except Exception as e:
        _v(f"reject: exception while reading H5: {e}")
        return False

    _v("accept: cache is usable")
    return True


def read_precomp_step_h5(path: str, t: int, *, device: torch.device, H: int, W: int):
    """
    Minimal reader for one timestep. Returns tensors on `device`.
    """
    gname = f"t{int(t):05d}"
    with h5py.File(path, "r") as f:
        g = f[gname]
        gs = f.get("static", None)

        def _read_ds(name: str):
            if name in g:
                return g[name][...]
            if gs is not None and name in gs:
                return gs[name][...]
            raise KeyError(f"Dataset '{name}' not found in '{gname}' or '/static'.")

        pred_centers = torch.from_numpy(_read_ds("pred_centers")).to(device)
        pred_levels  = torch.from_numpy(_read_ds("pred_levels")).to(device=device, dtype=torch.long)
        pred_parents = torch.from_numpy(_read_ds("pred_parents")).to(device=device, dtype=torch.long)
        pred_ei      = torch.from_numpy(_read_ds("pred_ei")).to(device=device, dtype=torch.long)
        mp_u8        = _read_ds("mask_pred_parent_flat_u8")
        mask_pred_parent = torch.from_numpy(mp_u8.astype(np.uint8)).to(device=device).view(H, W).bool()

        feat_t_on_pred = torch.from_numpy(g["feat_t_on_pred"][...]).to(device) if "feat_t_on_pred" in g else None
        feat_tp1_on_pred = torch.from_numpy(g["feat_tp1_on_pred"][...]).to(device) if "feat_tp1_on_pred" in g else None

    return pred_centers, pred_levels, pred_parents, pred_ei, mask_pred_parent, feat_t_on_pred, feat_tp1_on_pred
