#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def _resample_linear(arr: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    if arr.shape[0] <= 1 or abs(src_fps - tgt_fps) < 1e-9:
        return arr.copy()
    duration = (arr.shape[0] - 1) / src_fps
    tgt_len = int(round(duration * tgt_fps)) + 1
    t_src = np.arange(arr.shape[0], dtype=np.float64) / src_fps
    t_tgt = np.arange(tgt_len, dtype=np.float64) / tgt_fps
    out = np.empty((tgt_len, arr.shape[1]), dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(t_tgt, t_src, arr[:, j])
    return out


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _resample_angle_aware(arr: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """
    Resample joint angles with unwrap->interp->wrap to reduce +/-pi boundary jitter.
    """
    if arr.shape[0] <= 1 or abs(src_fps - tgt_fps) < 1e-9:
        return arr.copy()
    # Unwrap over time per dof to keep continuity.
    unwrapped = np.unwrap(arr, axis=0)
    out = _resample_linear(unwrapped, src_fps, tgt_fps)
    return _wrap_to_pi(out)


def _read_tail_from_template(template_path: Path) -> float:
    first_line = template_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    parts = first_line.split("\t")
    return float(parts[-1])


def _convert_one(
    npz_path: Path,
    out_path: Path,
    target_fps: float,
    tail_const: float,
    interp_mode: str,
    no_resample: bool,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    if "dof_pos" not in data.files or "fps" not in data.files:
        raise KeyError(f"npz missing required keys. found={data.files}")

    dof_pos = data["dof_pos"]  # (N, 29) for adam_sp
    if dof_pos.shape[1] < 29:
        raise ValueError(f"Expected dof_pos with >=29 dims, got {dof_pos.shape}")

    src_fps_arr = data["fps"]
    src_fps = float(src_fps_arr.reshape(-1)[0])
    effective_tgt_fps = src_fps if no_resample else target_fps
    if interp_mode == "angle_aware":
        dof_400 = _resample_angle_aware(
            dof_pos.astype(np.float64), src_fps, effective_tgt_fps
        )
    else:
        dof_400 = _resample_linear(
            dof_pos.astype(np.float64), src_fps, effective_tgt_fps
        )

    # adam_sp dof order in current repo:
    # 12..14 waist, 15..21 left arm, 22..28 right arm
    waist = dof_400[:, [12, 13, 14]]
    head = np.zeros((dof_400.shape[0], 2), dtype=np.float64)
    left_arm = dof_400[:, [15, 16, 17, 18, 19, 20, 21]]
    left_index_fixed = np.tile(np.array([0, 0, 0, 0, 400, 1000], dtype=np.float64), (dof_400.shape[0], 1))
    right_arm = dof_400[:, [22, 23, 24, 25, 26, 27, 28]]
    right_index_fixed = np.tile(np.array([0, 0, 0, 0, 400, 1000], dtype=np.float64), (dof_400.shape[0], 1))
    tail = np.full((dof_400.shape[0], 1), tail_const, dtype=np.float64)

    motion = np.concatenate(
        [waist, head, left_arm, left_index_fixed, right_arm, right_index_fixed, tail],
        axis=1,
    )  # (N, 32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in motion:
            f.write("\t".join(f"{x:.12g}" for x in row))
            f.write("\n")

    print(f"Saved motion txt: {out_path}")
    print(
        f"src_fps={src_fps}, target_fps={effective_tgt_fps}, "
        f"frames={motion.shape[0]}, cols={motion.shape[1]}, "
        f"no_resample={no_resample}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert GMR npz (adam_sp) to motion txt at 400Hz."
    )
    parser.add_argument("--npz_file", default=None, help="Input npz from fbx_offline_to_robot.py")
    parser.add_argument("--out_file", default=None, help="Output motion txt path")
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Enable batch conversion mode.",
    )
    parser.add_argument("--src_folder", default=None, help="Batch mode: folder containing .npz files")
    parser.add_argument(
        "--tgt_folder",
        default=None,
        help="Batch mode: output folder for .txt files",
    )
    parser.add_argument(
        "--keep_structure",
        action="store_true",
        default=False,
        help="Batch mode: preserve relative folder structure under tgt_folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Batch mode: overwrite existing txt files.",
    )
    parser.add_argument(
        "--template_file",
        default="/home/liuhongji/workspace/pnd_dds_deploy/src/motion_track/motion/Gentleman's Salute.txt",
        help="Template motion txt for tail constant (last column).",
    )
    parser.add_argument("--target_fps", type=float, default=400.0, help="Target output fps")
    parser.add_argument(
        "--no_resample",
        action="store_true",
        default=False,
        help="Export with original fps and frame count (no interpolation).",
    )
    parser.add_argument(
        "--interp_mode",
        choices=["angle_aware", "linear"],
        default="angle_aware",
        help="Interpolation mode for dof_pos resampling.",
    )
    args = parser.parse_args()

    template_path = Path(args.template_file).expanduser().resolve()
    tail_const = _read_tail_from_template(template_path) if template_path.exists() else 1.0

    if args.batch:
        if args.src_folder is None or args.tgt_folder is None:
            raise ValueError("Batch mode requires --src_folder and --tgt_folder")
        src_folder = Path(args.src_folder).expanduser().resolve()
        tgt_folder = Path(args.tgt_folder).expanduser().resolve()
        npz_files = sorted([p for p in src_folder.rglob("*.npz") if p.is_file()])
        if not npz_files:
            print(f"No .npz found under: {src_folder}")
            return
        done, skipped, failed = 0, 0, 0
        for npz_path in npz_files:
            rel = npz_path.relative_to(src_folder)
            out_path = (tgt_folder / rel).with_suffix(".txt") if args.keep_structure else (tgt_folder / f"{npz_path.stem}.txt")
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                _convert_one(
                    npz_path,
                    out_path,
                    args.target_fps,
                    tail_const,
                    args.interp_mode,
                    args.no_resample,
                )
                done += 1
            except Exception as e:
                failed += 1
                print(f"Failed: {npz_path} -> {e}")
        print(f"Batch done. success={done}, skipped={skipped}, failed={failed}")
    else:
        if args.npz_file is None or args.out_file is None:
            raise ValueError("Single mode requires --npz_file and --out_file")
        npz_path = Path(args.npz_file).expanduser().resolve()
        out_path = Path(args.out_file).expanduser().resolve()
        _convert_one(
            npz_path,
            out_path,
            args.target_fps,
            tail_const,
            args.interp_mode,
            args.no_resample,
        )


if __name__ == "__main__":
    main()
