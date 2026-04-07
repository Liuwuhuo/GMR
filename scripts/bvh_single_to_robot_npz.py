#!/usr/bin/env python3
import argparse
import os
import re

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

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
        raise ValueError(
            f"Invalid Frame Time ({frame_time}) in BVH file: {bvh_file}"
        )
    return int(round(1.0 / frame_time))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single-frame BVH -> static robot NPZ "
            "(same fields as bvh_to_robot_npz.py)"
        )
    )
    parser.add_argument("--bvh_file", required=True, help="BVH file path")
    parser.add_argument(
        "--format",
        choices=["smpl4d_bvh"],
        default="smpl4d_bvh",
    )
    parser.add_argument(
        "--robot",
        choices=["adam_sp"],
        default="adam_sp",
    )
    parser.add_argument(
        "--frame_idx", type=int, default=0, help="Which BVH frame to use (0-based)"
    )
    parser.add_argument(
        "--repeat_frames",
        type=int,
        default=250,
        help="Repeat this single retargeted frame to output length",
    )
    parser.add_argument("--save_path", default=None, help="Output NPZ path")
    parser.add_argument(
        "--target_fps", default=None, type=float, help="Output FPS"
    )
    parser.add_argument("--compressed", action="store_true", default=False)
    parser.add_argument("--no_vis", action="store_true", default=False, help="Disable visualization")
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument("--compute_local_body_pos", action="store_true", default=False)
    parser.add_argument("--height_adjust", action="store_true", default=False)
    parser.add_argument("--perframe_adjust", action="store_true", default=False)
    parser.add_argument(
        "--upright_root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force root quat to yaw-only (zero roll/pitch).",
    )
    parser.add_argument(
        "--ground_align_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align output root z so robot lowest point is on ground.",
    )
    args = parser.parse_args()

    if args.repeat_frames <= 0:
        raise ValueError(
            f"--repeat_frames must be positive, got {args.repeat_frames}"
        )

    if args.save_path is None:
        bvh_basename = os.path.splitext(os.path.basename(args.bvh_file))[0]
        args.save_path = os.path.join(
            "retarget",
            args.robot,
            args.format,
            f"{bvh_basename}_f{args.frame_idx}_static.npz",
        )
        print(f"未指定保存路径，使用默认路径: {args.save_path}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    bvh_load_format = "smpl4d_bvh"
    gmr_src_human = "bvh_smpl4d_bvh"

    bvh_frames, actual_human_height = load_bvh_file(args.bvh_file, format=bvh_load_format)
    if not bvh_frames:
        raise ValueError("No frames parsed from BVH")
    if args.frame_idx < 0 or args.frame_idx >= len(bvh_frames):
        raise IndexError(
            f"--frame_idx out of range [0, {len(bvh_frames)-1}], "
            f"got {args.frame_idx}"
        )

    output_fps = (
        infer_fps_from_bvh_frame_time(args.bvh_file)
        if args.target_fps is None
        else int(round(args.target_fps))
    )

    retargeter = GMR(
        src_human=gmr_src_human,
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    single_human_frame = bvh_frames[args.frame_idx]
    needed_keys = set(retargeter.pos_offsets1.keys())
    miss = sorted(list(needed_keys - set(single_human_frame.keys())))
    if miss:
        raise KeyError(f"{args.format} frame missing IK keys: {miss}")
    single_human_frame = {k: v for k, v in single_human_frame.items() if k in needed_keys}

    qpos_one, qvel_one = retargeter.retarget(
        single_human_frame, offset_to_ground=True, no_fly=True
    )
    qpos_one = np.asarray(qpos_one)
    qvel_one = np.asarray(qvel_one)

    if args.upright_root:
        root_quat = qpos_one[3:7]
        root_rot = R.from_quat(root_quat, scalar_first=True)
        roll, pitch, yaw = root_rot.as_euler("xyz", degrees=False)
        _ = (roll, pitch)  # explicit: we intentionally drop roll/pitch
        qpos_one[3:7] = R.from_euler(
            "xyz", [0.0, 0.0, yaw], degrees=False
        ).as_quat(scalar_first=True)

    qpos_arr = np.repeat(qpos_one[None, :], args.repeat_frames, axis=0)
    qvel_arr = np.repeat(qvel_one[None, :], args.repeat_frames, axis=0)

    if not args.no_vis:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=output_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=args.video_path,
        )
        i = 0
        while viewer.viewer.is_running():
            if i >= args.repeat_frames:
                break
            if viewer.paused:
                continue
            qpos = qpos_arr[i]
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=False,
            )
            i += 1
        viewer.close()

    root_pos = qpos_arr[:, :3]
    root_rot = qpos_arr[:, 3:7]  # wxyz
    dof_pos = qpos_arr[:, 7:]
    dof_vel = qvel_arr[:, 6:]

    if args.ground_align_output:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kin_tmp = KinematicsModel(retargeter.xml_file, device=device)
        body_pos_tmp, _ = kin_tmp.forward_kinematics(
            torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
            torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
        )
        lowest_height = torch.min(body_pos_tmp[..., 2]).item()
        root_pos[:, 2] = root_pos[:, 2] - lowest_height

    local_body_pos = None
    body_names = None
    if args.compute_local_body_pos:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kinematics_model = KinematicsModel(retargeter.xml_file, device=device)
        identity_root_pos = torch.zeros((args.repeat_frames, 3), device=device)
        identity_root_rot = torch.zeros((args.repeat_frames, 4), device=device)
        identity_root_rot[:, 0] = 1.0
        local_body_pos, _ = kinematics_model.forward_kinematics(
            identity_root_pos,
            identity_root_rot,
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
        )
        body_names = kinematics_model.body_names

        if args.height_adjust:
            body_pos, _ = kinematics_model.forward_kinematics(
                torch.from_numpy(root_pos).to(
                    device=device, dtype=torch.float
                ),
                torch.from_numpy(root_rot).to(
                    device=device, dtype=torch.float
                ),
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
            )
            if not args.perframe_adjust:
                lowest_height = torch.min(body_pos[..., 2]).item()
                root_pos[:, 2] = root_pos[:, 2] - lowest_height
            else:
                for i in range(root_pos.shape[0]):
                    lowest_body_part = torch.min(body_pos[i, :, 2])
                    root_pos[i, 2] = root_pos[i, 2] - lowest_body_part

        local_body_pos = local_body_pos.detach().cpu().numpy()

    save_dict = {
        "fps": np.array([output_fps]),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
    }
    if local_body_pos is not None:
        save_dict["local_body_pos"] = local_body_pos
    if body_names is not None:
        save_dict["link_body_list"] = body_names

    if args.compressed:
        np.savez_compressed(args.save_path, **save_dict)
    else:
        np.savez(args.save_path, **save_dict)

    print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
