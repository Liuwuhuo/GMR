#!/usr/bin/env python3
"""
Inspect custom motion .npz archives (e.g. datasets/txt/pufu.npz).

Expected keys (pufu-style):
  - posed_joints:     (T, J, 3) world-space joint positions
  - global_rot_mats:  (T, J, 3, 3) world rotation per joint
  - local_rot_mats:   (T, J, 3, 3) parent-local rotation per joint
  - root_positions:   (T, 3) root translation
  - foot_contacts:    (T, 4) bool contact flags
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _summarize_array(name: str, arr: np.ndarray) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "key": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if arr.size == 0:
        return info
    if np.issubdtype(arr.dtype, np.number):
        flat = arr.reshape(-1).astype(np.float64, copy=False)
        info["min"] = float(np.nanmin(flat))
        info["max"] = float(np.nanmax(flat))
        info["mean"] = float(np.nanmean(flat))
        info["has_nan"] = bool(np.isnan(flat).any())
        info["has_inf"] = bool(np.isinf(flat).any())
    if arr.dtype == bool or arr.dtype == np.bool_:
        info["true_ratio"] = float(np.mean(arr))
    return info


def _check_rotation_matrices(mats: np.ndarray, label: str, atol: float) -> None:
    if mats.ndim != 4 or mats.shape[-2:] != (3, 3):
        print(f"[skip] {label}: expected (T, J, 3, 3), got {mats.shape}")
        return
    r = mats.reshape(-1, 3, 3)
    dets = np.linalg.det(r)
    rr = np.matmul(r, np.swapaxes(r, -1, -2))
    eye = np.eye(3, dtype=r.dtype)
    ortho_err = np.linalg.norm(rr - eye, axis=(-2, -1))
    print(
        f"{label}: det in [{float(np.min(dets)):.4f}, {float(np.max(dets)):.4f}], "
        f"ortho_err max={float(np.max(ortho_err)):.2e}"
    )
    if float(np.max(np.abs(dets - 1.0))) > atol or float(np.max(ortho_err)) > atol:
        print(f"  warning: not orthogonal / det!=1 within atol={atol}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect motion .npz (pufu-style).")
    parser.add_argument(
        "--npz",
        type=str,
        default="datasets/txt/pufu.npz",
        help="Path to .npz file.",
    )
    parser.add_argument(
        "--check_rot",
        action="store_true",
        help="Check global/local rotation matrices (det, orthogonality).",
    )
    parser.add_argument(
        "--rot_atol",
        type=float,
        default=1e-2,
        help="Tolerance for rotation matrix checks.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="If >0, print first N frames of root_positions and foot_contacts.",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional path to write summary JSON.",
    )
    args = parser.parse_args()

    path = Path(args.npz).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    print(f"file: {path}")
    print(f"keys ({len(data.files)}): {data.files}")

    summary: Dict[str, Any] = {"file": str(path), "arrays": {}}
    for key in data.files:
        arr = np.asarray(data[key])
        s = _summarize_array(key, arr)
        summary["arrays"][key] = s
        print(json.dumps(s, indent=2, ensure_ascii=False))

    if "root_positions" in data and "posed_joints" in data:
        rp = np.asarray(data["root_positions"])
        pj = np.asarray(data["posed_joints"])
        if rp.ndim == 2 and rp.shape[1] == 3 and pj.ndim == 3 and pj.shape[2] == 3:
            if rp.shape[0] == pj.shape[0]:
                print(
                    f"inferred: T={rp.shape[0]} frames, J={pj.shape[1]} joints "
                    f"(posed_joints axis-1)."
                )

    if args.check_rot:
        if "global_rot_mats" in data:
            _check_rotation_matrices(
                np.asarray(data["global_rot_mats"]),
                "global_rot_mats",
                args.rot_atol,
            )
        if "local_rot_mats" in data:
            _check_rotation_matrices(
                np.asarray(data["local_rot_mats"]),
                "local_rot_mats",
                args.rot_atol,
            )

    if args.head > 0:
        n = int(args.head)
        if "root_positions" in data:
            rp = np.asarray(data["root_positions"])
            print(f"root_positions[:{n}]:\n{rp[:n]}")
        if "foot_contacts" in data:
            fc = np.asarray(data["foot_contacts"])
            print(f"foot_contacts[:{n}]:\n{fc[:n]}")

    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"wrote: {out}")


if __name__ == "__main__":
    main()
