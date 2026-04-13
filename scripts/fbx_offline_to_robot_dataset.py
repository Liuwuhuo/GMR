#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def collect_fbx_files(src_folder: Path):
    return sorted(
        [p for p in src_folder.rglob("*") if p.is_file() and p.suffix.lower() == ".fbx"]
    )


def main():
    parser = argparse.ArgumentParser(description="Batch retarget FBX files to robot NPZ")
    parser.add_argument("--src_folder", required=True, help="Folder containing .fbx files")
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "booster_t1",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "adam_sp",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--tgt_folder",
        default=None,
        help="Output root folder (default: retarget/<robot>/fbx)",
    )
    parser.add_argument("--fbx_root_joint", default="Hips", type=str)
    parser.add_argument("--fbx_fps", default=120, type=int)
    parser.add_argument("--target_fps", default=None, type=float)
    parser.add_argument("--compressed", action="store_true", default=False)
    parser.add_argument("--compute_local_body_pos", action="store_true", default=False)
    parser.add_argument("--height_adjust", action="store_true", default=False)
    parser.add_argument("--perframe_adjust", action="store_true", default=False)
    parser.add_argument(
        "--drop_first_frame",
        action="store_true",
        default=False,
        help="Drop the first retargeted frame for each clip.",
    )
    parser.add_argument(
        "--keep_structure",
        action="store_true",
        default=False,
        help="Preserve relative subfolders from src_folder under output root.",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    src_folder = Path(args.src_folder).expanduser().resolve()
    if not src_folder.exists():
        raise FileNotFoundError(f"src_folder not found: {src_folder}")

    tgt_folder = (
        Path(args.tgt_folder).expanduser().resolve()
        if args.tgt_folder is not None
        else Path("retarget") / args.robot / "fbx"
    )
    tgt_folder.mkdir(parents=True, exist_ok=True)

    fbx_files = collect_fbx_files(src_folder)
    if not fbx_files:
        print(f"No .fbx files found under: {src_folder}")
        return

    script_path = Path(__file__).parent / "fbx_offline_to_robot.py"
    success = 0
    failed = []
    skipped = 0

    for fbx_path in tqdm(fbx_files, desc="Batch FBX retarget"):
        rel = fbx_path.relative_to(src_folder)
        if args.keep_structure:
            out_path = tgt_folder / rel.with_suffix(".npz")
        else:
            out_path = tgt_folder / f"{fbx_path.stem}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--motion_file",
            str(fbx_path),
            "--robot",
            args.robot,
            "--save_path",
            str(out_path),
            "--fbx_root_joint",
            args.fbx_root_joint,
            "--fbx_fps",
            str(args.fbx_fps),
            "--no_viewer",
        ]
        # Keep output/playback fps aligned with parsing fps by default.
        effective_target_fps = args.target_fps if args.target_fps is not None else args.fbx_fps
        cmd += ["--target_fps", str(effective_target_fps)]
        if args.compressed:
            cmd += ["--compressed"]
        if args.compute_local_body_pos:
            cmd += ["--compute_local_body_pos"]
        if args.height_adjust:
            cmd += ["--height_adjust"]
        if args.perframe_adjust:
            cmd += ["--perframe_adjust"]
        if args.drop_first_frame:
            cmd += ["--drop_first_frame"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            success += 1
        else:
            failed.append((str(fbx_path), result.stderr or result.stdout))

    print(f"\nDone. success={success}, skipped={skipped}, failed={len(failed)}")
    if failed:
        print("\nFailed files:")
        for file_path, err in failed:
            print(f"- {file_path}")
            print(err.strip().splitlines()[-1] if err.strip() else "Unknown error")


if __name__ == "__main__":
    main()
