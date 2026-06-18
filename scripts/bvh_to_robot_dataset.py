#!/usr/bin/env python3
"""Batch version of bvh_to_robot.py.

Walks a folder of BVH files and retargets each one to a robot, reusing the exact
same core (formats, base_height_offset, ground/no_fly handling, pkl/npz output,
extra fields) as scripts/bvh_to_robot.py. No viewer/visualization.
"""
import argparse
import os

from tqdm import tqdm
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file

from bvh_to_robot import (
    infer_fps_from_bvh_frame_time,
    format_to_loader_and_src,
    filter_synthesized_keys,
    retarget_frames,
    build_motion_data,
    write_motion_file,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Batch retarget a folder of BVH files to a robot (no viewer)."
    )
    parser.add_argument("--src_folder", required=True, help="Folder of BVH files (searched recursively).")
    parser.add_argument(
        "--tgt_folder",
        default=None,
        help="Output folder. Default: retarget/{robot}/{format}.",
    )
    parser.add_argument(
        "--format",
        choices=[
            "lafan1", "nokov", "sfu", "noitom", "mocap", "mocap_hands",
            "opt_mocap", "jpg_lafan1", "smpl4d_bvh",
        ],
        default="lafan1",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy",
            "fourier_n1", "engineai_pm01", "pal_talos", "adam_sp_pro", "adam_sp", "adam_sp_box",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--output_format", choices=["auto", "pkl", "npz"], default="auto")
    parser.add_argument("--compressed", action="store_true", default=False)
    parser.add_argument(
        "--motion_fps",
        "--target_fps",
        dest="motion_fps",
        default=None,
        type=float,
        help="Output FPS. If omitted, inferred per file from BVH Frame Time.",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--drop_first_frame", action="store_true", default=False)
    parser.add_argument("--drop_first_n_frames", type=int, default=0)
    parser.add_argument(
        "--fly",
        action="store_true",
        default=False,
        help="Disable per-frame ground alignment (no_fly=False). Default keeps the root grounded.",
    )
    parser.add_argument("--base_height_offset", type=float, default=0.0)
    parser.add_argument("--compute_local_body_pos", action="store_true", default=False)
    parser.add_argument("--height_adjust", action="store_true", default=False)
    parser.add_argument("--perframe_adjust", action="store_true", default=False)
    parser.add_argument("--export_motion_fields", action="store_true", default=False)
    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Re-process files even if the output already exists.",
    )
    return parser


def retarget_one_file(bvh_file, args, out_ext):
    bvh_load_format, gmr_src_human = format_to_loader_and_src(args.format)
    frames, actual_human_height = load_bvh_file(bvh_file, format=bvh_load_format)

    start = max(0, args.start_frame)
    end = args.end_frame if args.end_frame is not None else len(frames)
    frames = frames[start:end]

    if args.motion_fps is None:
        motion_fps = infer_fps_from_bvh_frame_time(bvh_file)
    else:
        motion_fps = int(round(args.motion_fps))

    retargeter = GMR(
        src_human=gmr_src_human,
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        base_height_offset=args.base_height_offset,
    )
    frames = filter_synthesized_keys(retargeter, frames, args.format)

    qpos_list, qvel_list = retarget_frames(
        retargeter, frames, no_fly=not args.fly,
        drop_first_frame=args.drop_first_frame, show_progress=False,
    )
    return build_motion_data(
        retargeter,
        motion_fps,
        qpos_list,
        qvel_list,
        drop_first_n_frames=args.drop_first_n_frames,
        compute_local_body_pos=args.compute_local_body_pos,
        height_adjust=args.height_adjust,
        perframe_adjust=args.perframe_adjust,
        export_motion_fields=args.export_motion_fields,
    )


def main():
    args = build_arg_parser().parse_args()

    tgt_folder = args.tgt_folder or os.path.join("retarget", args.robot, args.format)
    out_ext = "npz" if args.output_format == "npz" else "pkl"
    print(f"输出目录: {tgt_folder} (.{out_ext})")

    bvh_files = []
    for dirpath, _, filenames in os.walk(args.src_folder):
        for filename in sorted(filenames):
            if filename.lower().endswith(".bvh"):
                bvh_files.append(os.path.join(dirpath, filename))

    if not bvh_files:
        print(f"在 {args.src_folder} 下没有找到 .bvh 文件")
        return

    success, skipped, failed = 0, 0, 0
    for bvh_file in tqdm(bvh_files, desc="Retargeting files"):
        rel = os.path.relpath(bvh_file, args.src_folder)
        tgt_file = os.path.join(tgt_folder, os.path.splitext(rel)[0] + f".{out_ext}")

        if os.path.exists(tgt_file) and not args.override:
            skipped += 1
            continue

        try:
            motion_data = retarget_one_file(bvh_file, args, out_ext)
            os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
            write_motion_file(motion_data, tgt_file, args.output_format, args.compressed)
            success += 1
        except Exception as e:
            failed += 1
            print(f"[red]失败[/red] {bvh_file}: {e}")

    print("=" * 50)
    print(f"完成: 成功 {success} | 跳过 {skipped} | 失败 {failed}")
    print(f"输出保存在: {tgt_folder}")


if __name__ == "__main__":
    main()
