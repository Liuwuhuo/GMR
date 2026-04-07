#!/usr/bin/env python3
import argparse
import os
import pathlib
import time

import numpy as np
import torch
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import (
    get_gvhmr_data_offline_fast,
    load_gvhmr_pred_file,
)


def parse_qpos_offset_specs(offset_specs):
    """Parse repeated '--qpos_offset idx:delta' specs into list[(idx, delta)]."""
    if not offset_specs:
        return []
    parsed = []
    for spec in offset_specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid --qpos_offset '{spec}'. Expected format 'idx:delta'."
            )
        idx_str, delta_str = spec.split(":", 1)
        try:
            idx = int(idx_str)
            delta = float(delta_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --qpos_offset '{spec}'. "
                "idx must be int and delta must be float."
            ) from exc
        parsed.append((idx, delta))
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single GVHMR pt -> NPZ "
            "(same format as bvh_to_robot_npz.py)"
        )
    )
    parser.add_argument(
        "--gvhmr_pred_file",
        required=True,
        help="GVHMR prediction file path (e.g., hmr4d_results.pt)",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "unitree_h1",
            "unitree_h1_2",
            "booster_t1",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "kuavo_s45",
            "hightorque_hi",
            "galaxea_r1pro",
            "berkeley_humanoid_lite",
            "booster_k1",
            "pnd_adam_lite",
            "adam_sp",
            "openloong",
            "tienkung",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--save_path", default=None, help="Output NPZ path")
    parser.add_argument(
        "--target_fps",
        default=30,
        type=float,
        help="Output FPS after temporal alignment (default 30).",
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
        help="Limit playback to output_fps.",
    )
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame (inclusive)",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="End frame (exclusive)",
    )
    parser.add_argument(
        "--freeze_frame_idx",
        type=int,
        default=None,
        help=(
            "Use one frame as static template. "
            "If set, this frame will be repeated to build the whole sequence."
        ),
    )
    parser.add_argument(
        "--expand_to_frames",
        type=int,
        default=None,
        help=(
            "Target frame count when --freeze_frame_idx is set. "
            "If omitted, keep current sliced length."
        ),
    )
    parser.add_argument(
        "--force_feet_same_height",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force left/right foot target z to be equal each frame.",
    )
    parser.add_argument(
        "--force_feet_level",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force feet orientation level (keep yaw, zero roll/pitch).",
    )
    parser.add_argument(
        "--qpos_offset",
        action="append",
        default=[],
        help=(
            "Apply qpos offset as 'idx:delta'. "
            "Repeatable, e.g. --qpos_offset 11:-0.9 --qpos_offset 17:0.1"
        ),
    )
    parser.add_argument(
        "--verify_static",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When freezing is enabled, verify all saved frames are identical; "
            "raise error if not."
        ),
    )
    args = parser.parse_args()
    user_qpos_offsets = parse_qpos_offset_specs(args.qpos_offset)

    here = pathlib.Path(__file__).resolve().parent
    smplx_folder = here / ".." / "assets" / "body_models"

    if args.save_path is None:
        motion_basename = os.path.splitext(
            os.path.basename(args.gvhmr_pred_file)
        )[0]
        args.save_path = os.path.join(
            "retarget", args.robot, "gvhmr", f"{motion_basename}.npz"
        )
        print(f"未指定保存路径，使用默认路径: {args.save_path}")
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    (
        smplx_data,
        body_model,
        smplx_output,
        actual_human_height,
    ) = load_gvhmr_pred_file(args.gvhmr_pred_file, smplx_folder)
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=int(round(args.target_fps))
    )

    start = max(0, args.start_frame)
    end = args.end_frame if args.end_frame is not None else len(smplx_data_frames)
    smplx_data_frames = smplx_data_frames[start:end]

    base_num_frames = len(smplx_data_frames)
    if args.freeze_frame_idx is not None:
        if base_num_frames == 0:
            raise ValueError(
                "No frames available after start/end slicing; cannot freeze."
            )
        if args.freeze_frame_idx < 0 or args.freeze_frame_idx >= base_num_frames:
            raise IndexError(
                f"--freeze_frame_idx={args.freeze_frame_idx} out of range "
                f"[0, {base_num_frames - 1}]"
            )
        num_frames = (
            args.expand_to_frames
            if args.expand_to_frames is not None
            else base_num_frames
        )
        if num_frames <= 0:
            raise ValueError(
                f"--expand_to_frames must be positive, got {num_frames}"
            )
        if num_frames <= args.freeze_frame_idx:
            raise ValueError(
                "--expand_to_frames must be greater than freeze frame index. "
                f"Got expand_to_frames={num_frames}, "
                f"freeze_frame_idx={args.freeze_frame_idx}."
            )
        print(
            "启用冻结播放: "
            f"freeze_frame={args.freeze_frame_idx}, total_frames={num_frames}"
        )
    elif args.expand_to_frames is not None:
        raise ValueError(
            "--expand_to_frames requires --freeze_frame_idx. "
            "Please set both together."
        )
    else:
        num_frames = base_num_frames
    output_fps = int(
        round(
            args.target_fps if args.target_fps is not None else aligned_fps
        )
    )
    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        force_feet_same_height=args.force_feet_same_height,
        force_feet_level=args.force_feet_level,
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
    pbar = tqdm(total=num_frames, desc="Retargeting")
    i = 0
    static_qpos = None
    static_qvel = None
    freeze_enabled = args.freeze_frame_idx is not None
    freeze_idx = args.freeze_frame_idx if freeze_enabled else -1
    while robot_motion_viewer.viewer.is_running():
        if i >= num_frames:
            break

        if robot_motion_viewer.paused:
            # Space pause should pause retargeting progression too.
            time.sleep(0.05)
            continue

        is_replay_frame = (
            freeze_enabled
            and i > freeze_idx
            and static_qpos is not None
            and static_qvel is not None
        )
        if is_replay_frame:
            # After freeze frame, stop retargeting and replay frozen pose.
            qpos = static_qpos.copy()
            qvel = static_qvel.copy()
        else:
            qpos, qvel = retargeter.retarget(
                smplx_data_frames[i], offset_to_ground=True, no_fly=True
            )
            # Keep existing gvhmr_to_robot behavior and custom offsets.
            if qpos.shape[0] > 17:
                qpos[11] = qpos[17]
            # qpos[31] -= 0.6
            # qpos[32] -= 0.6
            # qpos[11] -= 0.9
            qpos[17] -= 0.3
            for idx, delta in user_qpos_offsets:
                if idx < 0 or idx >= qpos.shape[0]:
                    raise IndexError(
                        f"--qpos_offset index out of range: idx={idx}, "
                        f"qpos_dim={qpos.shape[0]}"
                    )
                qpos[idx] += delta

        if freeze_enabled and i == freeze_idx and static_qpos is None:
            # Lock the solved + offset-applied pose at freeze frame.
            static_qpos = qpos.copy()
            static_qvel = qvel.copy()
            # Make freeze frame output exactly identical to subsequent replay frames.
            qpos = static_qpos.copy()
            qvel = static_qvel.copy()

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
        i += 1
        pbar.update(1)

    pbar.close()

    qpos_arr = np.asarray(qpos_list)
    qvel_arr = np.asarray(qvel_list)
    if freeze_enabled and static_qpos is not None and static_qvel is not None:
        # Final hard guarantee: every saved frame is exactly identical.
        qpos_arr[:] = static_qpos
        qvel_arr[:] = static_qvel
        if args.verify_static and qpos_arr.shape[0] > 1:
            max_qpos_diff = float(np.max(np.abs(qpos_arr - qpos_arr[0])))
            max_qvel_diff = float(np.max(np.abs(qvel_arr - qvel_arr[0])))
            print(
                "Static verification: "
                f"max|qpos-qpos0|={max_qpos_diff:.12g}, "
                f"max|qvel-qvel0|={max_qvel_diff:.12g}"
            )
            # Hard fail if any difference remains.
            if max_qpos_diff != 0.0 or max_qvel_diff != 0.0:
                raise RuntimeError(
                    "Static verification failed: saved frames are not identical."
                )
    root_pos = qpos_arr[:, :3]
    root_rot = qpos_arr[:, 3:7]  # keep wxyz, same as bvh_to_robot_npz.py
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
                torch.from_numpy(root_pos).to(
                    device=device, dtype=torch.float
                ),
                torch.from_numpy(root_rot).to(
                    device=device, dtype=torch.float
                ),
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
