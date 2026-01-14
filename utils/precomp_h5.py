import os, json, hashlib
import numpy as np
import torch
from typing import Any, Dict, Optional

try:
    import h5py
except ImportError as e:
    raise ImportError("Option B.2 requires h5py (pip install h5py).") from e


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


class _LazyPrecompSeq:
    """List-like object: seq[t] -> tensor (or None), loaded from HDF5 on demand."""
    def __init__(self, owner: "LazyPrecompH5", key: str):
        self._o = owner
        self._k = key

    def __len__(self) -> int:
        return self._o.T

    def __getitem__(self, t: int):
        return self._o._get(self._k, t)


class LazyPrecompH5(dict):
    """
    Dict-like drop-in replacement for the old in-memory precomp:
      precomp["pred_centers"][t] -> torch.Tensor or None
      precomp["pred2pred_idx"][t] -> torch.Tensor or None
    """
    def __init__(self, path: str, T: int, H: int, W: int, device: str | torch.device = "cpu"):
        super().__init__()
        if h5py is None:
            raise ImportError("LazyPrecompH5 requires h5py (pip install h5py).")

        self.path = str(path)
        self.T = int(T)
        self.H = int(H)
        self.W = int(W)
        self.device = torch.device(device)

        self._f = None  # lazy-open per process

        # expose the same keys as your old dict-of-lists
        for k in [
            "pred_centers", "pred_levels", "pred_parents", "pred_ei", "mask_pred",
            "feat_t_on_pred", "feat_tp1_on_pred",
            "pred2pred_idx", "pred2pred_w",
        ]:
            super().__setitem__(k, _LazyPrecompSeq(self, k))

    def _open(self):
        if self._f is None:
            # SWMR not required here; we read after writing finished.
            self._f = h5py.File(self.path, "r")

    def __getstate__(self):
        # make picklable for DataLoader workers: drop file handle
        d = dict(self.__dict__)
        d["_f"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._f = None

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

    def _to_torch(self, arr: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # torch.from_numpy shares memory; arr here is a new numpy array from h5py anyway
        return torch.from_numpy(arr).to(device=device, dtype=dtype)

    def _get(self, key: str, t: int):
        # preserve your original convention: index 0 is None for most series
        if t is None:
            return None
        t = int(t)
        if t < 0 or t >= self.T:
            return None

        # pred meshes / mapped features exist for t=1..T-1
        if key in ("pred_centers", "pred_levels", "pred_parents", "pred_ei", "mask_pred",
                   "feat_t_on_pred", "feat_tp1_on_pred"):
            if t == 0:
                return None
            if not self._has_group(t):
                return None

            if key == "pred_centers":
                arr = self._read_np(t, "pred_centers")
                return None if arr is None else self._to_torch(arr, dtype=torch.float32, device=self.device)

            if key == "pred_levels":
                arr = self._read_np(t, "pred_levels")
                return None if arr is None else self._to_torch(arr, dtype=torch.long, device=self.device)

            if key == "pred_parents":
                arr = self._read_np(t, "pred_parents")
                return None if arr is None else self._to_torch(arr, dtype=torch.long, device=self.device)

            if key == "pred_ei":
                arr = self._read_np(t, "pred_ei")
                return None if arr is None else self._to_torch(arr, dtype=torch.long, device=self.device)

            if key == "mask_pred":
                arr = self._read_np(t, "mask_pred_parent_flat_u8")
                if arr is None:
                    return None
                # stored as flat uint8 (H*W)
                m = torch.from_numpy(arr.astype(np.uint8, copy=False)).to(self.device)
                return m.view(self.H, self.W).to(torch.bool)

            if key == "feat_t_on_pred":
                arr = self._read_np(t, "feat_t_on_pred")
                return None if arr is None else self._to_torch(arr, dtype=torch.float32, device=self.device)

            if key == "feat_tp1_on_pred":
                arr = self._read_np(t, "feat_tp1_on_pred")
                return None if arr is None else self._to_torch(arr, dtype=torch.float32, device=self.device)

        # pred2pred maps exist for t=1..T-2 (stored in group t, mapping to t+1)
        if key in ("pred2pred_idx", "pred2pred_w"):
            if t < 1 or t >= self.T - 1:
                return None
            if not self._has_group(t):
                return None

            if key == "pred2pred_idx":
                arr = self._read_np(t, "pred2pred_idx_to_next")
                return None if arr is None else self._to_torch(arr, dtype=torch.long, device=self.device)

            if key == "pred2pred_w":
                arr = self._read_np(t, "pred2pred_w_to_next")
                if arr is None:
                    return None
                # stored float16; upcast to float32 for stable math
                return self._to_torch(arr.astype(np.float32, copy=False), dtype=torch.float32, device=self.device)

        return None


def precomp_h5_is_usable(path: str, cfg: dict, expected_steps: int, verbose: bool = False) -> bool:
    import os, h5py, numpy as np, json, hashlib

    def _v(msg):
        if verbose:
            print("[PRECOMP CHECK]", msg)

    if not os.path.exists(path):
        _v("reject: file does not exist")
        return False

    try:
        with h5py.File(path, "r") as f:
            if "meta" not in f:
                _v("reject: missing /meta group")
                return False

            meta = f["meta"]

            # ---- T check
            stored_T = meta.attrs.get("T", None)
            if stored_T is None:
                _v("reject: meta.attrs['T'] missing")
                return False
            stored_T = int(stored_T)
            if stored_T != int(expected_steps):
                _v(f"reject: T mismatch stored_T={stored_T} expected_steps={expected_steps}")
                return False

            # ---- cfg hash check (if you do one)
            stored_sha = meta.attrs.get("cfg_sha1", None)
            if stored_sha is None:
                _v("reject: meta.attrs['cfg_sha1'] missing")
                return False
            if isinstance(stored_sha, (bytes, np.bytes_)):
                stored_sha = stored_sha.decode("utf-8")

            # compute current sha exactly the same way you did when writing
            cfg_json = json.dumps(cfg, sort_keys=True, default=str)
            cur_sha = hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()

            if stored_sha != cur_sha:
                _v(f"reject: cfg_sha1 mismatch stored={stored_sha} current={cur_sha}")
                return False

            # ---- required groups check (example)
            # If your data is under f["steps"], ensure it exists and has enough keys
            if "steps" in f:
                n = len(f["steps"].keys())
                if n < (stored_T - 1):
                    _v(f"reject: /steps incomplete n={n} expected>={stored_T-1}")
                    return False
            else:
                _v("reject: missing /steps group (checker expects it)")
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
        pred_centers = torch.from_numpy(g["pred_centers"][...]).to(device)
        pred_levels  = torch.from_numpy(g["pred_levels"][...]).to(device=device, dtype=torch.long)
        pred_parents = torch.from_numpy(g["pred_parents"][...]).to(device=device, dtype=torch.long)
        pred_ei      = torch.from_numpy(g["pred_ei"][...]).to(device=device, dtype=torch.long)
        mp_u8        = g["mask_pred_parent_flat_u8"][...]
        mask_pred_parent = torch.from_numpy(mp_u8.astype(np.uint8)).to(device=device).view(H, W).bool()

        feat_t_on_pred = torch.from_numpy(g["feat_t_on_pred"][...]).to(device) if "feat_t_on_pred" in g else None
        feat_tp1_on_pred = torch.from_numpy(g["feat_tp1_on_pred"][...]).to(device) if "feat_tp1_on_pred" in g else None

    return pred_centers, pred_levels, pred_parents, pred_ei, mask_pred_parent, feat_t_on_pred, feat_tp1_on_pred
