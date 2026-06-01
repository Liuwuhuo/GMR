#!/usr/bin/env python3
import argparse
import pathlib
import os
import re
import time
import gc
import mujoco as mj
import numpy as np
from tqdm import tqdm
import torch

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting import qvel_from_qpos_central
from rich import print


def infer_fps_from_bvh_frame_time(bvh_file):
    """从BVH文件中提取帧率"""
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


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        required=True,
        type=str,
    )

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
        ],
        default="lafan1",
        help="BVH format type.",
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
            "adam_sp",
        ],
        default="unitree_g1",
    )

    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
        help="Override existing files.",
    )

    parser.add_argument(
        "--target_fps",
        default=None,
        type=float,
        help="Target FPS for the retargeted motion. If omitted, infer from BVH Frame Time and round.",
    )

    parser.add_argument(
        "--compressed",
        action="store_true",
        default=False,
        help="Use compressed npz format to reduce file size.",
    )

    parser.add_argument(
        "--compute_local_body_pos",
        action="store_true",
        default=False,
        help="Compute local body positions using forward kinematics (requires GPU).",
    )

    parser.add_argument(
        "--height_adjust",
        action="store_true",
        default=False,
        help="Adjust root height to prevent ground penetration.",
    )

    parser.add_argument(
        "--perframe_adjust",
        action="store_true",
        default=False,
        help="Adjust height per frame (only works with --height_adjust).",
    )

    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Video output path. If None, videos will be saved in {tgt_folder}/videos/ with same filename as npz.",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
        help="Limit playback to output_fps during visualization.",
    )

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
        "--no_fly",
        action="store_true",
        default=False,
        help="Prevent flying (keep feet on ground).",
    )
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
    parser.add_argument(
        "--export_motion_fields",
        action="store_true",
        default=False,
        help=(
            "Also export motion fields: framerate, joint_names, joint_pos, "
            "base_pos_w, base_quat_w."
        ),
    )

    args = parser.parse_args()

    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    # 检查是否使用默认路径逻辑
    use_default_path_logic = False
    if tgt_folder == "default":
        use_default_path_logic = True
        print("使用默认路径逻辑")

    # Collect all BVH files
    bvh_files = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in sorted(filenames):
            if filename.endswith(".bvh"):
                bvh_file_path = os.path.join(dirpath, filename)
                bvh_files.append(bvh_file_path)

    print(f"Found {len(bvh_files)} BVH files to process")

    # Process each BVH file
    for bvh_file_path in tqdm(bvh_files, desc="Retargeting files"):
        # 根据是否使用默认路径逻辑来确定目标文件路径
        if use_default_path_logic:
            # 从 BVH 文件名中提取基本名称（不带扩展名）
            bvh_basename = os.path.splitext(os.path.basename(bvh_file_path))[0]
            # 创建默认保存路径
            default_dir = "retarget"
            datasets_dir = args.format
            robot_dir = args.robot
            os.makedirs(default_dir, exist_ok=True)
            tgt_file_path = os.path.join(default_dir, robot_dir, datasets_dir, f"{bvh_basename}.npz")
            print(f"使用默认路径: {tgt_file_path}")
        else:
            # 使用原有的相对路径逻辑
            relative_path = os.path.relpath(bvh_file_path, src_folder)
            tgt_file_path = os.path.join(tgt_folder, relative_path).replace(".bvh", ".npz")

        if os.path.exists(tgt_file_path) and not args.override:
            print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
            continue

        # 格式兼容性处理
        bvh_load_format = args.format
        gmr_src_human = f"bvh_{args.format}"
        if args.format == "jpg_lafan1":
            bvh_load_format = "jpg_lafan1"
            gmr_src_human = "bvh_lafan1"
        elif args.format == "smpl4d_bvh":
            bvh_load_format = "smpl4d_bvh"
            gmr_src_human = "bvh_lafan1"
        elif args.format == "mocap_hands":
            bvh_load_format = "mocap_hands"
            gmr_src_human = "bvh_mocap_hands"

        # Load BVH trajectory
        try:
            bvh_data_frames, actual_human_height = load_bvh_file(
                bvh_file_path, format=bvh_load_format
            )
            
            # 帧裁剪
            start = max(0, args.start_frame)
            end = args.end_frame if args.end_frame is not None else len(bvh_data_frames)
            bvh_data_frames = bvh_data_frames[start:end]
            
            if args.target_fps is None:
                src_fps = infer_fps_from_bvh_frame_time(bvh_file_path)
            else:
                src_fps = int(round(args.target_fps))
        except Exception as e:
            print(f"Error loading {bvh_file_path}: {e}")
            continue

        # Initialize the retargeting system
        try:
            retargeter = GMR(
                src_human=gmr_src_human,
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
                velocity_fps=src_fps,
            )
            model = mj.MjModel.from_xml_path(retargeter.xml_file)
            data = mj.MjData(model)
        except Exception as e:
            print(f"Error initializing retargeter for {bvh_file_path}: {e}")
            continue

        # 关键帧验证（针对特定格式）
        if args.format in ("jpg_lafan1", "smpl4d_bvh") and len(bvh_data_frames) > 0:
            needed_keys = set(retargeter.pos_offsets1.keys())
            first_keys = set(bvh_data_frames[0].keys())
            missing = sorted(list(needed_keys - first_keys))
            if missing:
                print(f"Warning: {args.format}: BVH loader did not synthesize required IK keys. "
                       f"Missing (from first frame): {missing}.")
                # 过滤只保留需要的键
                bvh_data_frames = [
                    {k: v for k, v in frame.items() if k in needed_keys}
                    for frame in bvh_data_frames
                ]

        # Retarget to get all qpos
        qpos_list = []
        try:
            for curr_frame in range(len(bvh_data_frames)):
                smplx_data = bvh_data_frames[curr_frame]

                # Retarget till convergence
                qpos, _ = retargeter.retarget(
                    smplx_data,
                    offset_to_ground=True,
                    no_fly=args.no_fly
                )

                qpos_list.append(qpos.copy())
        except Exception as e:
            print(f"Error retargeting {bvh_file_path}: {e}")
            continue

        qpos_list = np.array(qpos_list)
        drop_n = max(0, int(args.drop_first_n_frames))
        if args.drop_first_frame:
            drop_n = max(drop_n, 1)
        if drop_n > 0:
            if qpos_list.shape[0] <= drop_n:
                print(
                    f"Skipping {bvh_file_path}: cannot drop first {drop_n} frame(s) "
                    f"(total={qpos_list.shape[0]})."
                )
                continue
            qpos_list = qpos_list[drop_n:]
        # 基于最终保留的 qpos 序列统一重算速度，确保 drop 后速度与轨迹一致。
        qvel_list = qvel_from_qpos_central(qpos_list, src_fps)
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7]  # Keep wxyz format
        dof_pos = qpos_list[:, 7:]
        dof_vel = qvel_list[:, 6:]
        num_frames = root_pos.shape[0]

        # Compute local body positions if requested
        local_body_pos = None
        body_names = None

        if args.compute_local_body_pos:
            try:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                kinematics_model = KinematicsModel(retargeter.xml_file, device=device)

                # Obtain local body pos
                identity_root_pos = torch.zeros((num_frames, 3), device=device)
                identity_root_rot = torch.zeros((num_frames, 4), device=device)
                identity_root_rot[:, 0] = 1.0  # wxyz format: set w=1
                local_body_pos, _ = kinematics_model.forward_kinematics(
                    identity_root_pos,
                    identity_root_rot,
                    torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
                )
                body_names = kinematics_model.body_names

                # Height adjustment if requested
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
                            root_pos[i, 2] = (
                                root_pos[i, 2] - lowest_body_part + ground_offset
                            )

                # Convert to numpy
                local_body_pos = local_body_pos.detach().cpu().numpy()

            except Exception as e:
                print(
                    f"Warning: Error computing local body pos for {bvh_file_path}: {e}"
                )
                local_body_pos = None
                body_names = None

        # Prepare data dictionary for npz format
        save_dict = {
            "fps": np.array([src_fps]),  # Convert to array for npz
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
        }

        # Optional field names used by downstream motion pipelines.
        # - framerate: scalar-like array [fps]
        # - joint_names: actuator order (same order as joint_pos columns)
        # - joint_pos: robot joint positions, shape [T, DoF]
        # - base_pos_w: base position in world frame, shape [T, 3]
        # - base_quat_w: base quaternion (wxyz) in world frame, shape [T, 4]
        if args.export_motion_fields:
            motor_name_by_id = sorted(
                retargeter.robot_motor_names.items(), key=lambda kv: kv[1]
            )
            joint_names = [name for name, _ in motor_name_by_id]
            save_dict.update(
                {
                    "framerate": np.array([src_fps], dtype=np.float64),
                    "joint_names": np.asarray(joint_names, dtype=object),
                    "joint_pos": dof_pos,
                    "base_pos_w": root_pos,
                    "base_quat_w": root_rot,
                }
            )

        # Only add optional fields if they are not None
        if local_body_pos is not None:
            save_dict["local_body_pos"] = local_body_pos
        if body_names is not None:
            save_dict["link_body_list"] = body_names

        # Create target directory if needed
        os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)

        # Save as npz or compressed npz
        try:
            if args.compressed:
                np.savez_compressed(tgt_file_path, **save_dict)
            else:
                np.savez(tgt_file_path, **save_dict)
            print(f"Saved: {tgt_file_path}")
        except Exception as e:
            print(f"Error saving {tgt_file_path}: {e}")
            continue

        # Generate video if requested
        if args.record_video:
            try:
                # 根据是否使用默认路径逻辑确定视频路径
                if use_default_path_logic:
                    # 使用默认路径逻辑：在默认目录下创建videos子目录
                    default_dir = "retarget"
                    robot_dir = args.robot
                    datasets_dir = args.format
                    video_dir = os.path.join(default_dir, robot_dir, datasets_dir, "videos")
                    video_file_path = os.path.join(video_dir, f"{bvh_basename}.mp4")
                else:
                    # 原有的视频路径逻辑
                    if args.video_path is not None:
                        if os.path.isdir(args.video_path) or args.video_path.endswith("/") or args.video_path.endswith("\\"):
                            video_dir = args.video_path.rstrip("/\\")
                            rel_path = os.path.relpath(tgt_file_path, tgt_folder)
                            video_file_path = os.path.join(video_dir, rel_path).replace(".npz", ".mp4")
                        else:
                            video_dir = os.path.dirname(args.video_path)
                            if video_dir:
                                video_file_path = os.path.join(
                                    video_dir, os.path.basename(tgt_file_path).replace(".npz", ".mp4")
                                )
                            else:
                                video_file_path = args.video_path
                    else:
                        video_file_path = os.path.join(
                            tgt_folder, "videos", os.path.relpath(tgt_file_path, tgt_folder).replace(".npz", ".mp4")
                        )

                # Create video directory if needed
                os.makedirs(os.path.dirname(video_file_path), exist_ok=True)

                # Check if video already exists
                if os.path.exists(video_file_path) and not args.override:
                    print(f"Skipping video generation for {bvh_file_path} because {video_file_path} exists")
                else:
                    # Initialize RobotMotionViewer for this video
                    robot_motion_viewer = None
                    try:
                        robot_motion_viewer = RobotMotionViewer(
                            robot_type=args.robot,
                            motion_fps=src_fps,
                            transparent_robot=0,
                            record_video=True,
                            video_path=video_file_path,
                        )

                        # Render all frames
                        for frame_idx in range(num_frames):
                            robot_motion_viewer.step(
                                root_pos=root_pos[frame_idx],
                                root_rot=root_rot[frame_idx],
                                dof_pos=dof_pos[frame_idx],
                                rate_limit=args.rate_limit,
                                follow_camera=True,
                            )

                        # Close viewer to finalize video (this also closes mp4_writer)
                        robot_motion_viewer.close()
                        robot_motion_viewer = None
                        
                        # Force garbage collection to ensure resources are released
                        gc.collect()
                        
                        # Add delay to ensure resources are fully released before next video
                        time.sleep(0.3)
                        
                        print(f"Saved video to {video_file_path}")
                    except Exception as video_error:
                        # Ensure cleanup in case of exception
                        if robot_motion_viewer is not None:
                            try:
                                robot_motion_viewer.close()
                            except:
                                pass
                            robot_motion_viewer = None
                        raise video_error

            except Exception as e:
                print(f"Error generating video for {bvh_file_path}: {e}")
                continue

    print(f"Done. Processed {len(bvh_files)} files")