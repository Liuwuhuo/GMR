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
        choices=["lafan1", "nokov", "sfu", "noitom"],
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
        qpos, qvel = retargeter.retarget(smplx_data)

        # qpos_list = np.array(qpos_list)

        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])

        # num_frames = root_pos.shape[0]

        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # xml_file = "/home/liuhongji/RL/GMR/assets/adam_sp/adam_inspire.xml"
        # kinematics_model = KinematicsModel(xml_file, device=device)
        
        # # obtain local body pos
        # identity_root_pos = torch.zeros((num_frames, 3), device=device)
        # identity_root_rot = torch.zeros((num_frames, 4), device=device)
        # identity_root_rot[:, -1] = 1.0
        # local_body_pos, _ = kinematics_model.forward_kinematics(
        #     identity_root_pos, 
        #     identity_root_rot, 
        #     torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
        # )
        # body_names = kinematics_model.body_names

        # HEIGHT_ADJUST = False
        # PERFRAME_ADJUST = False
        # if HEIGHT_ADJUST:
        #     body_pos, _ = kinematics_model.forward_kinematics(
        #         torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
        #         torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
        #         torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
        #     )
        #     ground_offset = 0.00
        #     if not PERFRAME_ADJUST:
        #         lowest_height = torch.min(body_pos[..., 2]).item()
        #         root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
        #     else:
        #         for i in range(root_pos.shape[0]):
        #             lowest_body_part = torch.min(body_pos[i, :, 2])
        #             root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset

        
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
       