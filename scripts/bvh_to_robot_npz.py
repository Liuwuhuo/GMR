#!/usr/bin/env python3
import argparse
import os
import re

import numpy as np
import torch
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


def main():
    parser = argparse.ArgumentParser(description="Single BVH -> NPZ (dataset format)")
    parser.add_argument("--bvh_file", required=True, help="BVH file path")
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "sfu", "noitom", "mocap", "opt_mocap"],
        default="lafan1",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "booster_t1",
            "stanford_toddy", "fourier_n1", "engineai_pm01",
            "pal_talos", "adam_sp_pro", "adam_sp",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--save_path", default=None, help="Output NPZ path")
    parser.add_argument(
        "--target_fps",
        default=None,
        type=float,
        help="Output FPS; if omitted, infer from BVH Frame Time and round.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        default=False,
        help="Use compressed NPZ format.",
    )
    parser.add_argument(
        "--compute_local_body_pos",
        action="store_true",
        default=False,
        help="Compute local body positions via FK.",
    )
    parser.add_argument(
        "--height_adjust",
        action="store_true",
        default=False,
        help="Adjust root height to avoid ground penetration.",
    )
    parser.add_argument(
        "--perframe_adjust",
        action="store_true",
        default=False,
        help="Adjust root height per frame (used with --height_adjust).",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
        help="Limit playback to output_fps during visualization.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame (inclusive)")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame (exclusive)")
    args = parser.parse_args()

    if args.save_path is None:
        bvh_basename = os.path.splitext(os.path.basename(args.bvh_file))[0]
        args.save_path = os.path.join(
            "retarget", args.robot, args.format, f"{bvh_basename}.npz"
        )
        print(f"未指定保存路径，使用默认路径: {args.save_path}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    bvh_data_frames, actual_human_height = load_bvh_file(
        args.bvh_file, format=args.format
    )
    start = max(0, args.start_frame)
    end = args.end_frame if args.end_frame is not None else len(bvh_data_frames)
    bvh_data_frames = bvh_data_frames[start:end]
    if args.target_fps is None:
        output_fps = infer_fps_from_bvh_frame_time(args.bvh_file)
    else:
        output_fps = int(round(args.target_fps))

    num_frames = len(bvh_data_frames)

    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=output_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    qpos_list = []
    qvel_list = []
    for i in tqdm(range(num_frames), desc="Retargeting"):
        qpos, qvel = retargeter.retarget(bvh_data_frames[i], offset_to_ground=True, no_fly=False)
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )
        qpos_list.append(qpos.copy())
        qvel_list.append(qvel.copy())

    qpos_arr = np.asarray(qpos_list)
    qvel_arr = np.asarray(qvel_list)
    root_pos = qpos_arr[:, :3]
    root_rot = qpos_arr[:, 3:7]  # wxyz
    dof_pos = qpos_arr[:, 7:]
    dof_vel = qvel_arr[:, 6:]

    local_body_pos = None
    body_names = None
    if args.compute_local_body_pos:
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

        if args.height_adjust:
            body_pos, _ = kinematics_model.forward_kinematics(
                torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
            )
            ground_offset = 0.0
            if not args.perframe_adjust:
                lowest_height = torch.min(body_pos[..., 2]).item()
                root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
            else:
                for i in range(root_pos.shape[0]):
                    lowest_body_part = torch.min(body_pos[i, :, 2])
                    root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset

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
    robot_motion_viewer.close()


if __name__ == "__main__":
    main()
