import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
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
        choices=["lafan1", "nokov", "sfu"],
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
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "adam_sp_pro", "adam_inspire"],
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
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    motion_fps = args.motion_fps
    
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
        qpos = retargeter.retarget(smplx_data)
        

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

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
   
        
        if args.save_path is not None:
            qpos_list.append(qpos)
    
    if args.save_path is not None:
        import pickle
        from scipy.spatial.transform import Rotation as R
        
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])  # 从 wxyz 转换为 xyzw
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        
        # 计算帧时长
        frame_duration = 1.0 / motion_fps
        
        # 获取关节名称
        dof_names = [name for name, index in sorted(retargeter.robot_dof_names.items(), key=lambda x: x[1]) if name != "floating_joint"]
        
        # 构建标签列表
        labels = []
        labels.extend(["root_pos/x", "root_pos/y", "root_pos/z"])
        labels.extend(["root_quat/x", "root_quat/y", "root_quat/z", "root_quat/w"])
        labels.extend([f"dof_pos/{name}" for name in dof_names])
        
        # 构建帧数据
        frames = []
        for i in range(len(root_pos)):
            frame_data = []
            # 添加根位置 (x, y, z)
            frame_data.extend(root_pos[i].tolist())
            # 添加根旋转四元数 (x, y, z, w)
            frame_data.extend(root_rot[i].tolist())
            # 添加关节位置
            frame_data.extend(dof_pos[i].tolist())
            frames.append(frame_data)
        
        # 构建最终输出结构
        motion_data = {
            "LoopMode": "Once",
            "LoopNum": 1,
            "FrameDuration": frame_duration,
            "EnableCycleOffsetPosition": True,
            "EnableCycleOffsetRotation": True,
            "MotionWeight": 1.0,
            "Labels": labels,
            "Frames": frames
        }
        
        # 保存为 pickle 文件
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       
