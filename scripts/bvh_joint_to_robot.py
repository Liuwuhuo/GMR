import argparse
import os
import pathlib
import re
import time

import numpy as np
from rich import print
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import general_motion_retargeting.utils.lafan_vendor.utils as lafan_utils
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


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


def load_bvh_frames_direct(bvh_file):
    """Direct BVH loader without lafan1.py post-processing."""
    data = read_bvh(bvh_file)
    global_quat, global_pos = lafan_utils.quat_fk(data.quats, data.pos, data.parents)

    # Keep the same axis conversion used by existing scripts.
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame_idx in range(global_pos.shape[0]):
        frame = {}
        for i, bone in enumerate(data.bones):
            orientation = lafan_utils.quat_mul(rotation_quat, global_quat[frame_idx, i])
            position = global_pos[frame_idx, i] @ rotation_matrix.T / 100.0  # cm -> m
            frame[bone] = [position, orientation]
        # Validation mode: keep FootMod semantics with ankle-roll pose.
        if "left_ankle_roll_joint" in frame:
            frame["LeftFootMod"] = [
                frame["left_ankle_roll_joint"][0],
                frame["left_ankle_roll_joint"][1],
            ]
        if "right_ankle_roll_joint" in frame:
            frame["RightFootMod"] = [
                frame["right_ankle_roll_joint"][0],
                frame["right_ankle_roll_joint"][1],
            ]
        frames.append(frame)

    return frames


def print_joint_rotations(frames, frame_idx, joint_names):
    if not frames:
        print("[rot-debug] empty frames")
        return
    frame_idx = max(0, min(frame_idx, len(frames) - 1))
    frame = frames[frame_idx]
    print(f"[rot-debug] frame={frame_idx}")
    for name in joint_names:
        if name not in frame:
            print(f"[rot-debug] {name}: NOT_FOUND")
            continue
        quat = np.asarray(frame[name][1], dtype=np.float64)  # wxyz
        print(
            f"[rot-debug] {name}: "
            f"[{quat[0]: .6f}, {quat[1]: .6f}, {quat[2]: .6f}, {quat[3]: .6f}] (wxyz)"
        )


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str, help="BVH motion file path.")
    parser.add_argument(
        "--src_human",
        default="bvh_joint_mocap",
        choices=["bvh_joint_mocap", "bvh_opt_mocap", "bvh_opt_mocap_footmod"],
        help="Source human config key in params.IK_CONFIG_DICT.",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "adam_sp_pro",
            "adam_sp",
        ],
        default="adam_sp",
    )
    parser.add_argument("--loop", action="store_true", default=False, help="Loop motion.")
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--save_path", default=None, help="Output .pkl path.")
    parser.add_argument(
        "--motion_fps",
        default=None,
        type=float,
        help="Output FPS. If omitted, infer from BVH Frame Time and round.",
    )
    parser.add_argument(
        "--actual_human_height",
        default=None,
        type=float,
        help="Optional actual human height in meters for config scaling.",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index (inclusive).")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index (exclusive).")
    parser.add_argument(
        "--print_rot_frame",
        type=int,
        default=None,
        help="Print world quat(wxyz) of selected joints at this frame index.",
    )
    parser.add_argument(
        "--print_rot_joints",
        type=str,
        default=(
            "left_hip_pitch_joint,left_hip_roll_joint,left_hip_yaw_joint,"
            "left_knee_joint,left_ankle_pitch_joint,left_ankle_roll_joint,"
            "right_hip_pitch_joint,right_hip_roll_joint,right_hip_yaw_joint,"
            "right_knee_joint,right_ankle_pitch_joint,right_ankle_roll_joint"
        ),
        help="Comma-separated joint names used with --print_rot_frame.",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        default=False,
        help="Only print joint rotations and exit.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        default=False,
        help="Use compressed NPZ format.",
    )
    args = parser.parse_args()

    if args.save_path is None:
        bvh_basename = os.path.splitext(os.path.basename(args.bvh_file))[0]
        dataset_dir = args.src_human.replace("bvh_", "")
        args.save_path = os.path.join("retarget", args.robot, dataset_dir, f"{bvh_basename}.npz")
        print(f"未指定保存路径，使用默认路径: {args.save_path}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    human_frames = load_bvh_frames_direct(args.bvh_file)
    start = max(0, args.start_frame)
    end = args.end_frame if args.end_frame is not None else len(human_frames)
    human_frames = human_frames[start:end]

    if args.print_rot_frame is not None:
        joint_names = [x.strip() for x in args.print_rot_joints.split(",") if x.strip()]
        print_joint_rotations(human_frames, args.print_rot_frame, joint_names)
        if args.print_only:
            raise SystemExit(0)

    retargeter = GMR(
        src_human=args.src_human,
        tgt_robot=args.robot,
        actual_human_height=args.actual_human_height,
    )

    # Early validation: fail fast if BVH frame misses IK-required joints.
    required_human_joints = set()
    if retargeter.use_ik_match_table1:
        required_human_joints.update(entry[0] for entry in retargeter.ik_match_table1.values())
    if retargeter.use_ik_match_table2:
        required_human_joints.update(entry[0] for entry in retargeter.ik_match_table2.values())
    if not human_frames:
        raise ValueError("No frames available after start/end frame slicing.")
    first_frame_keys = set(human_frames[0].keys())
    missing = sorted(required_human_joints - first_frame_keys)
    if missing:
        raise KeyError(
            "Missing required BVH joints for IK config "
            f"({args.src_human} -> {args.robot}): {missing}"
        )

    if args.motion_fps is None:
        motion_fps = infer_fps_from_bvh_frame_time(args.bvh_file)
        print(f"Auto motion_fps from Frame Time: {motion_fps}")
    else:
        motion_fps = int(round(args.motion_fps))

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0
    print(f"mocap_frame_rate: {motion_fps}")
    pbar = tqdm(total=len(human_frames), desc="Retargeting")

    qpos_list = []
    qvel_list = []
    i = 0
    while viewer.viewer.is_running():
        if i >= len(human_frames):
            if args.loop:
                i = 0
            else:
                break

        human_data = human_frames[i]
        qpos, qvel = retargeter.retarget(human_data, no_fly=False)

        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )

        if not viewer.paused:
            qpos_list.append(qpos)
            qvel_list.append(qvel)
            pbar.update(1)
            i += 1

            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

    if not qpos_list:
        raise RuntimeError("No retargeted frames were generated (viewer closed too early or all frames paused).")

    qpos_arr = np.asarray(qpos_list)
    qvel_arr = np.asarray(qvel_list)
    root_pos = qpos_arr[:, :3]
    root_rot = qpos_arr[:, 3:7]  # keep wxyz
    dof_pos = qpos_arr[:, 7:]
    dof_vel = qvel_arr[:, 6:]

    motion_data = {
        "fps": np.array([motion_fps]),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
    }
    if args.compressed:
        np.savez_compressed(args.save_path, **motion_data)
    else:
        np.savez(args.save_path, **motion_data)
    print(f"Saved to {args.save_path}")

    pbar.close()
    viewer.close()
