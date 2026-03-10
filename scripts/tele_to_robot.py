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
        choices=["lafan1", "nokov", "sfu", "noitom", "mocap"],
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
        default=30,
        type=int,
        help="Input motion FPS used for timestamping saved output.",
    )

    parser.add_argument(
        "--play_fps",
        default=None,
        type=float,
        help="Viewer playback FPS. Defaults to motion_fps.",
    )

    parser.add_argument(
        "--downsample_to_fps",
        default=None,
        type=float,
        help="Downsample input frames before retargeting to this FPS.",
    )

    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index (exclusive)")
    
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

    # 截取所需帧段
    start = max(0, args.start_frame)
    end = args.end_frame if args.end_frame is not None else len(lafan1_data_frames)
    lafan1_data_frames = lafan1_data_frames[start:end]

    input_motion_fps = float(args.motion_fps)
    play_fps = float(args.play_fps) if args.play_fps is not None else input_motion_fps
    output_motion_fps = input_motion_fps

    if args.downsample_to_fps is not None:
        target_fps = float(args.downsample_to_fps)
        if target_fps <= 0:
            raise ValueError("--downsample_to_fps must be positive")
        if target_fps < input_motion_fps:
            frame_stride = max(1, int(round(input_motion_fps / target_fps)))
            lafan1_data_frames = lafan1_data_frames[::frame_stride]
            output_motion_fps = input_motion_fps / frame_stride
            print(
                f"Downsample before retarget: input_fps={input_motion_fps:.2f}, "
                f"target_fps={target_fps:.2f}, stride={frame_stride}, "
                f"effective_fps={output_motion_fps:.2f}, frames={len(lafan1_data_frames)}"
            )
        else:
            print(
                f"Skip downsample: target_fps={target_fps:.2f} >= input_fps={input_motion_fps:.2f}"
            )
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )



    motion_fps = output_motion_fps
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=play_fps,
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
    
    print(f"mocap_frame_rate (output): {motion_fps:.2f}")
    print(f"viewer_play_fps: {play_fps:.2f}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    


    while True:

        if robot_motion_viewer.paused is False:
            if args.loop:
                i = (i + 1) % len(lafan1_data_frames)
            else:
                i += 1
                if i >= len(lafan1_data_frames):
                    # time.sleep(10000.0)  # 无限期休眠，不占用 CPU
                    # pass
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
        qpos, qvel = retargeter.retarget(smplx_data, no_fly=True)

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
    
    if args.save_path is not None:
        import pickle

        qpos_array = np.asarray(qpos_list)
        root_pos = qpos_array[:, :3]
        # save from wxyz to xyzw
        root_rot = qpos_array[:, 3:7][:, [1, 2, 3, 0]]
        dof_pos = qpos_array[:, 7:]

        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       