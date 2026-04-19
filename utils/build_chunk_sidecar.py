#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

try:
    from chunk_sidecar import build_chunk_sidecar
except ImportError:  # pragma: no cover
    from utils.chunk_sidecar import build_chunk_sidecar


def _to_abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build reusable chunk sidecar metadata from an existing precompute H5."
    )
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    ap.add_argument(
        "--precomp-path",
        type=str,
        default=None,
        help="Optional override for train.precomp_cache_path.",
    )
    ap.add_argument(
        "--sidecar-path",
        type=str,
        default=None,
        help="Optional override for chunk.builder.sidecar_path.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sidecar regardless of chunk.builder.overwrite.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logging regardless of chunk.builder.progress.",
    )
    ap.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Optional cap on number of timestep groups processed (debug/smoke test).",
    )
    args = ap.parse_args()

    cfg_path = _to_abs_path(args.config)
    cfg = _load_cfg(cfg_path)

    if args.precomp_path is not None:
        cfg.setdefault("train", {})["precomp_cache_path"] = _to_abs_path(args.precomp_path)

    chunk_cfg = cfg.setdefault("chunk", {})
    builder_cfg = chunk_cfg.setdefault("builder", {})

    if args.sidecar_path is not None:
        builder_cfg["sidecar_path"] = _to_abs_path(args.sidecar_path)
    if args.overwrite:
        builder_cfg["overwrite"] = True
    if args.quiet:
        builder_cfg["progress"] = False
    if args.max_timesteps is not None:
        builder_cfg["max_timesteps"] = int(args.max_timesteps)

    result = build_chunk_sidecar(cfg)
    print("[CHUNK-SIDECAR] complete")
    for k in sorted(result.keys()):
        print(f"  {k}: {result[k]}")


if __name__ == "__main__":
    main()
