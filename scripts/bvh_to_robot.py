#!/usr/bin/env python3
"""Unified BVH -> robot retargeting + visualization.

This single entry point replaces the previous bvh_to_robot.py / bvh_to_robot_npz.py
pair. Output format is chosen by the --save_path extension (.pkl / .npz) or the
explicit --output_format flag.
"""
import argparse
import os
import pickle
import re
import time

import numpy as np
import torch
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def infer_fps_from_bvh_frame_time(bvh_file):
    frame_time = None
    with open(bvh_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.match(r"\s*Frame Time:\s+([\d\.eE+-]+)", line)
            if match:
                frame_time = float(match.group(1))
                break

    if frame_time is None:
        raise ValueError(f"Cannot find 'Frame Time' in BVH file: {bvh_file}")
    if frame_time <= 0:
        raise ValueError(f"Invalid Frame Time ({frame_time}) in BVH file: {bvh_file}")

    return int(round(1.0 / frame_time))


def resolve_output_format(save_path, output_format):
    """Decide pkl vs npz from the explicit flag, falling back to the extension."""
    if output_format != "auto":
        return output_format
    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".npz":
        return "npz"
    return "pkl"


def format_to_loader_and_src(fmt):
    """Map a CLI --format to (bvh loader format, GMR src_human key).

    Some formats are loaded under their own name but reuse the lafan1 IK config.
    """
    bvh_load_format = fmt
    gmr_src_human = f"bvh_{fmt}"
    if fmt in ("jpg_lafan1", "smpl4d_bvh"):
        gmr_src_human = "bvh_lafan1"
    return bvh_load_format, gmr_src_human


def filter_synthesized_keys(retargeter, frames, fmt):
    """For synthesized formats, keep only the IK keys the retargeter expects."""
    if fmt not in ("jpg_lafan1", "smpl4d_bvh") or len(frames) == 0:
        return frames
    needed_keys = set(retargeter.pos_offsets1.keys())
    first_keys = set(frames[0].keys())
    missing = sorted(list(needed_keys - first_keys))
    if missing:
        raise KeyError(
            f"{fmt}: BVH loader did not synthesize required IK keys. "
            f"Missing (from first frame): {missing}."
        )
    return [{k: v for k, v in frame.items() if k in needed_keys} for frame in frames]


def retarget_frames(retargeter, frames, no_fly, drop_first_frame=False, show_progress=True):
    """Headless retarget of a list of frames. Returns (qpos_list, qvel_list)."""
    qpos_list, qvel_list = [], []
    iterator = enumerate(frames)
    if show_progress:
        iterator = enumerate(tqdm(frames, desc="Retargeting"))
    for i, frame in iterator:
        qpos, qvel = retargeter.retarget(frame, offset_to_ground=False, no_fly=no_fly)
        if drop_first_frame and i == 0:
            continue
        qpos_list.append(qpos.copy())
        qvel_list.append(qvel.copy())
    return qpos_list, qvel_list


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Retarget a single BVH file to a robot and visualize / save it."
    )
    parser.add_argument("--bvh_file", required=True, type=str, help="BVH motion file to load.")
    parser.add_argument(
        "--format",
        choices=[
            "lafan1",
            "nokov",
            "sfu",
            "noitom",
            "mocap",
            "mocap_hands",
            "opt_mocap",
            "jpg_lafan1",
            "smpl4d_bvh",
            "test_mocap",
        ],
        default="lafan1",
    )
    parser.add_argument(
        "--already_z_up",
        action="store_true",
        default=False,
        help=(
            "Skip Y-up→Z-up conversion in load_bvh_file. Only for legacy captures "
            "that are already z-up; typical Y-up BVH (e.g. PND) should leave this off."
        ),
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "booster_t1",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "pal_talos",
            "adam_sp_pro",
            "adam_sp_pro_with_hands",
            "adam_sp",
        ],
        default="unitree_g1",
    )

    # ---- output ----
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion. If omitted, only visualize (no save).",
    )
    parser.add_argument(
        "--output_format",
        choices=["auto", "pkl", "npz"],
        default="auto",
        help="Output format. 'auto' infers from --save_path extension (defaults to pkl).",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        default=False,
        help="Use compressed NPZ format (npz output only).",
    )

    # ---- fps ----
    parser.add_argument(
        "--motion_fps",
        "--target_fps",
        dest="motion_fps",
        default=None,
        type=float,
        help="Output motion FPS. If omitted, infer from BVH Frame Time and round.",
    )

    # ---- frame selection ----
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument(
        "--drop_first_frame",
        action="store_true",
        default=False,
        help="Drop the first retargeted frame (useful when frame 0 is unstable).",
    )
    parser.add_argument(
        "--drop_first_n_frames",
        type=int,
        default=0,
        help="Drop first N retargeted frames (applied before saving).",
    )

    # ---- ground / height ----
    parser.add_argument(
        "--fly",
        action="store_true",
        default=False,
        help=(
            "Disable per-frame ground alignment (no_fly=False). Default keeps the "
            "root grounded every frame (matches the old bvh_to_robot.py behavior)."
        ),
    )
    parser.add_argument(
        "--base_height_offset",
        type=float,
        default=0.0,
        help="Lift the whole motion up (meters) to keep robot soles from sinking into the ground.",
    )
    parser.add_argument(
        "--height_adjust",
        action="store_true",
        default=False,
        help="Adjust root height via FK to avoid ground penetration (requires --compute_local_body_pos).",
    )
    parser.add_argument(
        "--perframe_adjust",
        action="store_true",
        default=False,
        help="Adjust root height per frame (used with --height_adjust).",
    )

    # ---- extra exports ----
    parser.add_argument(
        "--compute_local_body_pos",
        action="store_true",
        default=False,
        help="Compute local body positions via FK and include them in the output.",
    )
    parser.add_argument(
        "--export_motion_fields",
        action="store_true",
        default=False,
        help=(
            "Also export motion fields: framerate, joint_names, joint_pos, "
            "base_pos_w, base_quat_w."
        ),
    )

    # ---- visualization ----
    parser.add_argument("--loop", action="store_true", default=False, help="Loop the motion.")
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")

    return parser


def main():
    args = build_arg_parser().parse_args()

    # Auto-generate a default save path when none is given (legacy npz behavior).
    if args.save_path is None:
        ext = "npz" if args.output_format == "npz" else "pkl"
        bvh_basename = os.path.splitext(os.path.basename(args.bvh_file))[0]
        args.save_path = os.path.join(
            "retarget", args.robot, args.format, f"{bvh_basename}.{ext}"
        )
        print(f"未指定保存路径，使用默认路径: {args.save_path}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Some formats load with one loader name but use the lafan1 IK config.
    bvh_load_format, gmr_src_human = format_to_loader_and_src(args.format)

    bvh_data_frames, actual_human_height = load_bvh_file(
        args.bvh_file,
        format=bvh_load_format,
        already_z_up=args.already_z_up,
    )

    # Frame slicing.
    start = max(0, args.start_frame)
    end = args.end_frame if args.end_frame is not None else len(bvh_data_frames)
    bvh_data_frames = bvh_data_frames[start:end]

    # Output fps.
    if args.motion_fps is None:
        motion_fps = infer_fps_from_bvh_frame_time(args.bvh_file)
        print(f"Auto motion_fps from Frame Time: {motion_fps}")
    else:
        motion_fps = int(round(args.motion_fps))
    print(f"mocap_frame_rate: {motion_fps}")

    retargeter = GMR(
        src_human=gmr_src_human,
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        base_height_offset=args.base_height_offset,
    )

    # For synthesized formats, keep only the IK keys the retargeter expects.
    bvh_data_frames = filter_synthesized_keys(retargeter, bvh_data_frames, args.format)

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    no_fly = not args.fly

    qpos_list = []
    qvel_list = []

    # FPS measurement.
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    pbar = tqdm(total=len(bvh_data_frames), desc="Retargeting")
    i = 0
    while True:
        # Block (do not advance retargeting) while the viewer is paused.
        robot_motion_viewer.wait_while_paused()
        if (not args.loop) and i >= len(bvh_data_frames):
            break

        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        pbar.update(1)

        smplx_data = bvh_data_frames[i]
        qpos, qvel = retargeter.retarget(smplx_data, offset_to_ground=True, no_fly=False)

        if args.drop_first_frame and i == 0:
            i = (i + 1) % len(bvh_data_frames) if args.loop else i + 1
            continue

        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )

        if args.save_path is not None:
            qpos_list.append(qpos.copy())
            qvel_list.append(qvel.copy())

        i = (i + 1) % len(bvh_data_frames) if args.loop else i + 1

    pbar.close()

    if args.save_path is not None:
        save_motion(args, retargeter, motion_fps, qpos_list, qvel_list)

    robot_motion_viewer.close()


def build_motion_data(
    retargeter,
    motion_fps,
    qpos_list,
    qvel_list,
    drop_first_n_frames=0,
    compute_local_body_pos=False,
    height_adjust=False,
    perframe_adjust=False,
    export_motion_fields=False,
):
    """Assemble the motion-data dict from collected qpos/qvel. Reusable by the
    interactive and the batch/dataset scripts so the output format stays identical."""
    qpos_arr = np.asarray(qpos_list)
    qvel_arr = np.asarray(qvel_list)

    drop_n = max(0, int(drop_first_n_frames))
    if drop_n > 0:
        if qpos_arr.shape[0] <= drop_n:
            raise ValueError(
                f"Cannot drop first {drop_n} frame(s): total saved frames = {qpos_arr.shape[0]}"
            )
        qpos_arr = qpos_arr[drop_n:]
        qvel_arr = qvel_arr[drop_n:]

    num_frames = qpos_arr.shape[0]
    root_pos = qpos_arr[:, :3]
    root_rot = qpos_arr[:, 3:7]  # wxyz
    dof_pos = qpos_arr[:, 7:]
    dof_vel = qvel_arr[:, 6:] if qvel_arr.ndim == 2 and qvel_arr.shape[1] >= 6 else None

    local_body_pos = None
    body_names = None
    if compute_local_body_pos:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kinematics_model = KinematicsModel(retargeter.xml_file, device=device)

        identity_root_pos = torch.zeros((num_frames, 3), device=device)
        identity_root_rot = torch.zeros((num_frames, 4), device=device)
        identity_root_rot[:, 0] = 1.0
        local_body_pos, _ = kinematics_model.forward_kinematics(
            identity_root_pos,
            identity_root_rot,
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
        )
        body_names = kinematics_model.body_names

        if height_adjust:
            body_pos, _ = kinematics_model.forward_kinematics(
                torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
            )
            ground_offset = 0.0
            if not perframe_adjust:
                lowest_height = torch.min(body_pos[..., 2]).item()
                root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
            else:
                for f in range(root_pos.shape[0]):
                    lowest_body_part = torch.min(body_pos[f, :, 2])
                    root_pos[f, 2] = root_pos[f, 2] - lowest_body_part + ground_offset

        local_body_pos = local_body_pos.detach().cpu().numpy()

    motion_data = {
        "fps": motion_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "local_body_pos": local_body_pos,
        "link_body_list": body_names,
    }

    if export_motion_fields:
        motor_name_by_id = sorted(retargeter.robot_motor_names.items(), key=lambda kv: kv[1])
        joint_names = [name for name, _ in motor_name_by_id]
        motion_data.update(
            {
                "framerate": np.array([motion_fps], dtype=np.float64),
                "joint_names": np.asarray(joint_names, dtype=object),
                "joint_pos": dof_pos,
                "base_pos_w": root_pos,
                "base_quat_w": root_rot,
            }
        )

    return motion_data


def write_motion_file(motion_data, save_path, output_format="auto", compressed=False):
    """Write motion_data to pkl or npz (decided by output_format / extension)."""
    out_format = resolve_output_format(save_path, output_format)
    if out_format == "npz":
        # np.savez cannot store None cleanly, so drop missing optional fields.
        npz_data = {k: v for k, v in motion_data.items() if v is not None}
        # Match the legacy npz exporter, which stored fps as a 1-D array.
        npz_data["fps"] = np.array([motion_data["fps"]])
        if compressed:
            np.savez_compressed(save_path, **npz_data)
        else:
            np.savez(save_path, **npz_data)
    else:
        with open(save_path, "wb") as f:
            pickle.dump(motion_data, f)
    print(f"Saved to {save_path}")


def save_motion(args, retargeter, motion_fps, qpos_list, qvel_list):
    motion_data = build_motion_data(
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
    write_motion_file(motion_data, args.save_path, args.output_format, args.compressed)


if __name__ == "__main__":
    main()
