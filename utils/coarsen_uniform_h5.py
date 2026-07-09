#!/usr/bin/env python3
"""
One-time coarsening utility for uniform-grid HDF5 training data.

Expected input schema:
  - snapshots: [T, H, W, C]
  - time:      [T]
  - xy:        [H, W, 2]
  - channel_names: [C]
Optional pass-through:
  - channel_units: [C]
  - ramp-angle metadata is preserved/inferred into output attrs

The script performs:
  1) symmetric center-crop to dimensions divisible by target size
  2) block-mean downsampling to the target (H_out, W_out)

Example:
  python utils/coarsen_uniform_h5.py \
    --in_h5 ./cache/DMR_7_0_45_.h5 \
    --out_h5 ./cache/DMR_7_0_45__250x500.h5 \
    --target_h 250 \
    --target_w 500
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Tuple

import h5py
import numpy as np

_RES_SUFFIX_RE = re.compile(r"_(\d+)x(\d+)$")


def _to_scalar_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (bytes, np.bytes_)):
        try:
            v = v.decode("utf-8")
        except Exception:
            return None
    arr = np.asarray(v)
    if arr.size != 1:
        return None
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return None


def _parse_compact_decimal_token(tok: str | None) -> float | None:
    if tok is None:
        return None
    s = str(tok).strip()
    if s == "":
        return None
    if ("p" in s.lower()) and ("." not in s):
        if s[0] in "+-":
            s = s[0] + s[1:].replace("p", ".").replace("P", ".")
        else:
            s = s.replace("p", ".").replace("P", ".")
    try:
        return float(s)
    except Exception:
        return None


def _extract_angle_from_path(path_like: str | Path | None) -> float | None:
    if not path_like:
        return None
    s = os.path.basename(str(path_like))

    patterns = (
        r"(?:^|[_\-])ramp[-_]?angle[-_]?(-?\d+(?:\.\d+)?)",
        r"(?:^|[_\-])angle[-_]?(-?\d+(?:\.\d+)?)",
    )
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m is not None:
            val = _parse_compact_decimal_token(m.group(1))
            if val is not None:
                return float(val)

    # Fallback for shock-ramp DMR files named like:
    #   DMR_8_0_60__125x250.h5  -> pressure/token 8_0, ramp angle 60 deg
    #   DMR_5_5_70__125x250.h5  -> pressure/token 5_5, ramp angle 70 deg
    m = re.search(
        r"^DMR[_\-]"
        r"(-?\d+(?:[pP]\d+|\.\d+)?)"
        r"[_\-]"
        r"(-?\d+(?:[pP]\d+|\.\d+)?)"
        r"[_\-]"
        r"(-?\d+(?:[pP]\d+|\.\d+)?)"
        r"(?=(?:[_\-]{1,2}|\.|$))",
        s,
        flags=re.IGNORECASE,
    )
    if m is not None:
        angle = _parse_compact_decimal_token(m.group(3))
        if angle is not None:
            return float(angle)
    return None


def _resolve_ramp_angle_deg(fin: h5py.File, in_h5: str) -> tuple[float | None, str | None]:
    degree_keys = (
        "ramp_angle_deg",
        "ramp_angle_degrees",
        "angle_deg",
        "angle_degrees",
        "theta_deg",
        "theta_degrees",
        "ramp_angle",
        "angle",
        "theta",
    )
    radian_keys = ("ramp_angle_rad", "angle_rad", "theta_rad", "ramp_angle_radians")

    for key in degree_keys:
        if key in fin.attrs:
            val = _to_scalar_float(fin.attrs[key])
            if val is not None:
                return float(val), f"input_attr:{key}"

    for key in radian_keys:
        if key in fin.attrs:
            val = _to_scalar_float(fin.attrs[key])
            if val is not None:
                return float(np.rad2deg(val)), f"input_attr:{key}"

    angle = _extract_angle_from_path(in_h5)
    if angle is not None:
        return float(angle), "filename"

    return None, None


def _is_h5_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in (".h5", ".hdf5")


def _with_resolution_suffix(path: Path, target_h: int, target_w: int) -> Path:
    """
    Return sibling path whose basename ends with _{target_h}x{target_w}<ext>.
    If basename already ends with _<int>x<int>, replace that suffix.
    """
    stem = path.stem
    m = _RES_SUFFIX_RE.search(stem)
    if m is not None:
        stem = stem[: m.start()]
    return path.with_name(f"{stem}_{int(target_h)}x{int(target_w)}{path.suffix}")


def _crop_plan(in_h: int, in_w: int, out_h: int, out_w: int) -> Tuple[int, int, int, int, int, int]:
    """
    Return (r0, r1, c0, c1, fh, fw) where:
      - [r0:r1, c0:c1] is a symmetric crop
      - crop_h = out_h * fh
      - crop_w = out_w * fw
    """
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"target_h/target_w must be > 0, got ({out_h}, {out_w})")
    if in_h < out_h or in_w < out_w:
        raise ValueError(
            f"Target resolution ({out_h}, {out_w}) is larger than input ({in_h}, {in_w})."
        )

    fh = in_h // out_h
    fw = in_w // out_w
    if fh < 1 or fw < 1:
        raise ValueError(
            f"Could not derive valid integer block factors for input ({in_h}, {in_w}) "
            f"and target ({out_h}, {out_w})."
        )

    crop_h = out_h * fh
    crop_w = out_w * fw
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(
            f"Computed non-positive crop size ({crop_h}, {crop_w}) from factors ({fh}, {fw})."
        )

    dh = in_h - crop_h
    dw = in_w - crop_w
    r0 = dh // 2
    c0 = dw // 2
    r1 = r0 + crop_h
    c1 = c0 + crop_w
    return r0, r1, c0, c1, fh, fw


def _block_mean_2d(arr_hwf: np.ndarray, out_h: int, out_w: int, fh: int, fw: int) -> np.ndarray:
    """
    Block-mean downsample for arrays shaped [H, W, F] -> [out_h, out_w, F].
    """
    if arr_hwf.ndim != 3:
        raise ValueError(f"Expected rank-3 array [H,W,F], got shape={arr_hwf.shape}")
    h, w, f = arr_hwf.shape
    if h != out_h * fh or w != out_w * fw:
        raise ValueError(
            f"Input shape {(h, w)} not compatible with target {(out_h, out_w)} "
            f"and factors {(fh, fw)}."
        )
    v = arr_hwf.reshape(out_h, fh, out_w, fw, f)
    return v.mean(axis=(1, 3), dtype=np.float64).astype(np.float32, copy=False)


def coarsen_uniform_h5(
    in_h5: str,
    out_h5: str,
    target_h: int,
    target_w: int,
    *,
    overwrite: bool = False,
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    in_h5 = os.path.abspath(os.path.expanduser(str(in_h5)))
    out_h5 = os.path.abspath(os.path.expanduser(str(out_h5)))

    if not os.path.exists(in_h5):
        raise FileNotFoundError(f"Input HDF5 not found: {in_h5}")
    if os.path.exists(out_h5) and not overwrite:
        raise FileExistsError(
            f"Output exists: {out_h5}. Use --overwrite to replace it."
        )
    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)

    with h5py.File(in_h5, "r") as fin:
        required = ("snapshots", "time", "xy", "channel_names")
        missing = [k for k in required if k not in fin]
        if missing:
            raise RuntimeError(f"Input missing required datasets: {missing}")

        d_snap = fin["snapshots"]
        d_time = fin["time"]
        d_xy = fin["xy"]
        d_names = fin["channel_names"]
        d_units = fin["channel_units"] if "channel_units" in fin else None

        if d_snap.ndim != 4:
            raise RuntimeError(f"snapshots must be [T,H,W,C], got shape={d_snap.shape}")
        if d_xy.ndim != 3 or int(d_xy.shape[-1]) < 2:
            raise RuntimeError(f"xy must be [H,W,2], got shape={d_xy.shape}")

        t_count, in_h, in_w, n_chan = [int(v) for v in d_snap.shape]
        if int(d_xy.shape[0]) != in_h or int(d_xy.shape[1]) != in_w:
            raise RuntimeError(
                f"xy shape {tuple(d_xy.shape)} incompatible with snapshots H/W ({in_h}, {in_w})"
            )
        if int(d_time.shape[0]) != t_count:
            raise RuntimeError(
                f"time length {int(d_time.shape[0])} does not match snapshots T {t_count}"
            )
        if int(d_names.shape[0]) != n_chan:
            raise RuntimeError(
                f"channel_names length {int(d_names.shape[0])} does not match C {n_chan}"
            )

        r0, r1, c0, c1, fh, fw = _crop_plan(in_h, in_w, int(target_h), int(target_w))
        crop_h = r1 - r0
        crop_w = c1 - c0
        ramp_angle_deg, ramp_angle_source = _resolve_ramp_angle_deg(fin, in_h5)

        print(
            "[COARSEN] input:",
            f"T={t_count}, H={in_h}, W={in_w}, C={n_chan}",
            flush=True,
        )
        print(
            "[COARSEN] crop:",
            f"rows [{r0}:{r1}] (size={crop_h}), cols [{c0}:{c1}] (size={crop_w})",
            flush=True,
        )
        print(
            "[COARSEN] block factors:",
            f"fh={fh}, fw={fw} -> output H={target_h}, W={target_w}",
            flush=True,
        )
        if ramp_angle_deg is not None:
            print(
                "[COARSEN] ramp angle:",
                f"{ramp_angle_deg:g} deg (source={ramp_angle_source})",
                flush=True,
            )
        else:
            print("[COARSEN] ramp angle: not found", flush=True)

        kwargs = {}
        if compression and str(compression).lower() != "none":
            kwargs["compression"] = str(compression).lower()
            kwargs["compression_opts"] = int(compression_level)

        with h5py.File(out_h5, "w") as fout:
            # Preserve file attrs when possible.
            for k, v in fin.attrs.items():
                try:
                    fout.attrs[k] = v
                except Exception:
                    pass
            if ramp_angle_deg is not None:
                fout.attrs["ramp_angle_deg"] = np.float64(float(ramp_angle_deg))
                fout.attrs["ramp_angle_rad"] = np.float64(float(np.deg2rad(ramp_angle_deg)))
                fout.attrs["ramp_angle_source"] = np.bytes_(str(ramp_angle_source or "unknown"))

            d_out = fout.create_dataset(
                "snapshots",
                shape=(t_count, int(target_h), int(target_w), n_chan),
                dtype=np.float32,
                chunks=(1, int(target_h), int(target_w), n_chan),
                **kwargs,
            )

            # Coarsen each snapshot independently to avoid large memory spikes.
            for t in range(t_count):
                snap = np.asarray(d_snap[t, r0:r1, c0:c1, :], dtype=np.float32)
                snap_out = _block_mean_2d(snap, int(target_h), int(target_w), fh, fw)
                d_out[t] = snap_out
                if (t + 1) % 5 == 0 or (t + 1) == t_count:
                    print(f"[COARSEN] processed snapshot {t + 1}/{t_count}", flush=True)

            # time is copied as-is.
            fout.create_dataset("time", data=np.asarray(d_time[()]))

            # Coarsen xy using the same crop/block averaging.
            xy_crop = np.asarray(d_xy[r0:r1, c0:c1, :2], dtype=np.float32)
            xy_out = _block_mean_2d(xy_crop, int(target_h), int(target_w), fh, fw)
            fout.create_dataset("xy", data=xy_out.astype(np.float32), **kwargs)

            # channel metadata pass-through.
            fout.create_dataset("channel_names", data=d_names[()])
            if d_units is not None:
                fout.create_dataset("channel_units", data=d_units[()])

            # Add explicit coarsening metadata.
            meta = fout.create_group("coarsen_meta")
            meta.attrs["source_h5"] = np.bytes_(in_h5)
            meta.attrs["input_H"] = np.int32(in_h)
            meta.attrs["input_W"] = np.int32(in_w)
            meta.attrs["output_H"] = np.int32(int(target_h))
            meta.attrs["output_W"] = np.int32(int(target_w))
            meta.attrs["crop_r0"] = np.int32(r0)
            meta.attrs["crop_r1"] = np.int32(r1)
            meta.attrs["crop_c0"] = np.int32(c0)
            meta.attrs["crop_c1"] = np.int32(c1)
            meta.attrs["factor_h"] = np.int32(fh)
            meta.attrs["factor_w"] = np.int32(fw)
            meta.attrs["method"] = np.bytes_("crop_then_block_mean")
            if ramp_angle_deg is not None:
                meta.attrs["ramp_angle_deg"] = np.float64(float(ramp_angle_deg))
                meta.attrs["ramp_angle_rad"] = np.float64(float(np.deg2rad(ramp_angle_deg)))
                meta.attrs["ramp_angle_source"] = np.bytes_(str(ramp_angle_source or "unknown"))

    print(f"[COARSEN] wrote: {out_h5}", flush=True)


def coarsen_uniform_h5_directory(
    in_dir: str,
    *,
    out_dir: str | None,
    target_h: int,
    target_w: int,
    overwrite: bool = False,
    compression: str = "gzip",
    compression_level: int = 4,
    recursive: bool = False,
) -> None:
    in_dir_p = Path(os.path.abspath(os.path.expanduser(str(in_dir))))
    if not in_dir_p.is_dir():
        raise NotADirectoryError(f"Input directory not found: {in_dir_p}")

    out_dir_p = (
        Path(os.path.abspath(os.path.expanduser(str(out_dir))))
        if out_dir is not None
        else in_dir_p
    )
    out_dir_p.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if bool(recursive) else "*"
    in_files = sorted(p for p in in_dir_p.glob(pattern) if _is_h5_file(p))
    if not in_files:
        raise RuntimeError(f"No .h5/.hdf5 files found in directory: {in_dir_p}")

    print(
        f"[COARSEN] discovered {len(in_files)} input files in {in_dir_p} "
        f"(recursive={bool(recursive)}).",
        flush=True,
    )

    n_ok = 0
    n_skip = 0
    for idx, in_path in enumerate(in_files, start=1):
        rel = in_path.relative_to(in_dir_p)
        out_parent = out_dir_p / rel.parent
        out_parent.mkdir(parents=True, exist_ok=True)
        out_path = _with_resolution_suffix(out_parent / in_path.name, int(target_h), int(target_w))

        # Avoid clobbering an input that is already at this output naming.
        try:
            same_path = in_path.resolve() == out_path.resolve()
        except Exception:
            same_path = (str(in_path) == str(out_path))
        if same_path and not overwrite:
            print(
                f"[COARSEN] ({idx}/{len(in_files)}) skip existing coarsened input: {in_path}",
                flush=True,
            )
            n_skip += 1
            continue

        print(
            f"[COARSEN] ({idx}/{len(in_files)}) {in_path} -> {out_path}",
            flush=True,
        )
        coarsen_uniform_h5(
            in_h5=str(in_path),
            out_h5=str(out_path),
            target_h=int(target_h),
            target_w=int(target_w),
            overwrite=bool(overwrite),
            compression=str(compression),
            compression_level=int(compression_level),
        )
        n_ok += 1

    print(
        f"[COARSEN] directory job done: written={n_ok}, skipped={n_skip}, total={len(in_files)}",
        flush=True,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="One-time coarsening utility for uniform-grid HDF5 snapshots."
    )
    ap.add_argument("--in_h5", type=str, default=None, help="Input HDF5 path (single-file mode).")
    ap.add_argument("--in_dir", type=str, default=None, help="Input directory of HDF5 files (batch mode).")
    ap.add_argument("--out_h5", type=str, default=None, help="Output HDF5 path (single-file mode).")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Output directory for batch mode (or single-file auto naming). "
            "Default: same directory as input."
        ),
    )
    ap.add_argument("--target_h", type=int, default=250, help="Target output height (default: 250).")
    ap.add_argument("--target_w", type=int, default=500, help="Target output width (default: 500).")
    ap.add_argument(
        "--compression",
        type=str,
        default="gzip",
        help="HDF5 compression: gzip or none (default: gzip).",
    )
    ap.add_argument(
        "--compression_level",
        type=int,
        default=4,
        help="Compression level for gzip (default: 4).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --in_dir for .h5/.hdf5 files.",
    )
    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()
    in_h5 = None if args.in_h5 is None else str(args.in_h5).strip()
    in_dir = None if args.in_dir is None else str(args.in_dir).strip()
    out_h5 = None if args.out_h5 is None else str(args.out_h5).strip()
    out_dir = None if args.out_dir is None else str(args.out_dir).strip()

    use_file = bool(in_h5)
    use_dir = bool(in_dir)
    if use_file == use_dir:
        raise ValueError("Provide exactly one of --in_h5 or --in_dir.")

    if use_file:
        in_path = Path(os.path.abspath(os.path.expanduser(str(in_h5))))
        if out_h5:
            out_path = Path(os.path.abspath(os.path.expanduser(str(out_h5))))
        else:
            base_dir = (
                Path(os.path.abspath(os.path.expanduser(str(out_dir))))
                if out_dir
                else in_path.parent
            )
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = _with_resolution_suffix(base_dir / in_path.name, int(args.target_h), int(args.target_w))

        coarsen_uniform_h5(
            in_h5=str(in_path),
            out_h5=str(out_path),
            target_h=int(args.target_h),
            target_w=int(args.target_w),
            overwrite=bool(args.overwrite),
            compression=str(args.compression),
            compression_level=int(args.compression_level),
        )
        return

    if out_h5:
        raise ValueError("--out_h5 is only valid with --in_h5. Use --out_dir for batch mode.")

    coarsen_uniform_h5_directory(
        in_dir=str(in_dir),
        out_dir=(None if not out_dir else str(out_dir)),
        target_h=int(args.target_h),
        target_w=int(args.target_w),
        overwrite=bool(args.overwrite),
        compression=str(args.compression),
        compression_level=int(args.compression_level),
        recursive=bool(args.recursive),
    )


if __name__ == "__main__":
    main()
