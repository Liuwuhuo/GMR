import argparse
import pathlib
import time
import torch
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np
import re


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

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "sfu", "noitom", "mocap", "mocap_hands", "opt_mocap", "sfu"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "adam_sp_pro", "adam_sp"],
        default="unitree_g1",
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

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=None,
        type=float,
        help="Output motion FPS. If omitted, infer from BVH Frame Time and round.",
    )

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
    

    if args.save_path is None:
        # 从 BVH 文件名中提取基本名称（不带扩展名）
        bvh_basename = os.path.splitext(os.path.basename(args.bvh_file))[0]
        # 创建默认保存路径
        default_dir = "retarget"
        datasets_dir = args.format
        robot_dir = args.robot
        os.makedirs(default_dir, exist_ok=True)
        args.save_path = os.path.join(default_dir, robot_dir, datasets_dir, f"{bvh_basename}.pkl")
        print(f"未指定保存路径，使用默认路径: {args.save_path}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:  # Only create directory if it's not empty
        os.makedirs(save_dir, exist_ok=True)
    qpos_list = []

    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)

    # >>>>> 新增：支持指定起止帧 <<<<<

    # args = parser.parse_args()  # 重新 parse，或把这两行提前到 load_bvh 之前

    # 截取所需帧段
    start = max(0, args.start_frame)
    end = args.end_frame if args.end_frame is not None else len(lafan1_data_frames)
    lafan1_data_frames = lafan1_data_frames[start:end]

    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )



    if args.motion_fps is None:
        motion_fps = infer_fps_from_bvh_frame_time(args.bvh_file)
        print(f"Auto motion_fps from Frame Time: {motion_fps}")
    else:
        motion_fps = int(round(args.motion_fps))
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    


    while True:
        # 与 bvh_to_robot_npz.py 保持一致：暂停时阻塞，不继续 retarget。
        robot_motion_viewer.wait_while_paused()
        if (not args.loop) and i >= len(lafan1_data_frames):
            break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos, qvel = retargeter.retarget(smplx_data, no_fly=False)

        if args.drop_first_frame and i == 0:
            # Skip unstable first frame for both visualization and saved output.
            if args.loop:
                i = (i + 1) % len(lafan1_data_frames)
            else:
                i += 1
            continue

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=False,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        
        if args.save_path is not None:
            qpos_list.append(qpos)

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
    
    if args.save_path is not None:
        import pickle

        local_body_pos = None
        body_names = None

        qpos_arr = np.asarray(qpos_list)
        drop_n = max(0, int(args.drop_first_n_frames))
        if drop_n > 0:
            if qpos_arr.shape[0] <= drop_n:
                raise ValueError(
                    f"Cannot drop first {drop_n} frame(s): total saved frames = {qpos_arr.shape[0]}"
                )
            qpos_arr = qpos_arr[drop_n:]

        root_pos = qpos_arr[:, :3]
        # save from wxyz to xyzw
        root_rot = qpos_arr[:, 3:7]
        dof_pos = qpos_arr[:, 7:]
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }

        if args.export_motion_fields:
            motor_name_by_id = sorted(
                retargeter.robot_motor_names.items(), key=lambda kv: kv[1]
            )
            joint_names = [name for name, _ in motor_name_by_id]
            motion_data.update(
                {
                    "framerate": np.array([motion_fps], dtype=np.float64),
                    "joint_names": np.asarray(joint_names, dtype=object),
                    "joint_pos": dof_pos,
                    "base_pos_w": root_pos,
                    # Keep field naming parity with npz exporter.
                    "base_quat_w": qpos_arr[:, 3:7],
                }
            )
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       