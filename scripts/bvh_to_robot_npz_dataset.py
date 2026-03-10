import argparse
import pathlib
import os
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
from rich import print


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
        choices=["lafan1", "nokov", "noitom", "sfu", "mocap"],
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
            "pnd_adam_lite",
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
        default=30,
        type=int,
        help="Target FPS for the retargeted motion.",
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

        # Load BVH trajectory
        try:
            bvh_data_frames, actual_human_height = load_bvh_file(
                bvh_file_path, format=args.format
            )
            src_fps = args.target_fps
        except Exception as e:
            print(f"Error loading {bvh_file_path}: {e}")
            continue

        # Initialize the retargeting system
        try:
            retargeter = GMR(
                src_human=f"bvh_{args.format}",
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )
            model = mj.MjModel.from_xml_path(retargeter.xml_file)
            data = mj.MjData(model)
        except Exception as e:
            print(f"Error initializing retargeter for {bvh_file_path}: {e}")
            continue

        # Retarget to get all qpos
        qpos_list = []
        qvel_list = []
        try:
            for curr_frame in range(len(bvh_data_frames)):
                smplx_data = bvh_data_frames[curr_frame]

                # Retarget till convergence
                qpos, qvel = retargeter.retarget(smplx_data,offset_to_ground=True)

                qpos_list.append(qpos.copy())
                qvel_list.append(qvel.copy())
        except Exception as e:
            print(f"Error retargeting {bvh_file_path}: {e}")
            continue

        qpos_list = np.array(qpos_list)
        qvel_list = np.array(qvel_list)
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
                    ground_offset = 0.00
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
                                rate_limit=False,
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

    print(f"Done. Saved {len(bvh_files)} files")