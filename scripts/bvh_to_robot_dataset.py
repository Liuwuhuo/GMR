import argparse
import pathlib
import os
import mujoco as mj
import numpy as np
from tqdm import tqdm
import torch
import pickle

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
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
        default="../../motion_data/LAFAN1_g1_gmr"
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "adam_sp_pro", "adam_sp"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
        help="Target FPS for output motion. Used in FrameDuration.",
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "sfu"],
        default="lafan1",
        help="BVH format; affects how load_bvh_file parses joint hierarchy.",
    )

    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder
    target_fps = args.target_fps

    # Walk over all files in src_folder
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in tqdm(sorted(filenames), desc="Retargeting files"):
            if not filename.endswith(".bvh"):
                continue
                
            # Get paths
            bvh_file_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(bvh_file_path, src_folder)
            tgt_file_path = os.path.join(tgt_folder, os.path.splitext(rel_path)[0] + ".pkl")

            if os.path.exists(tgt_file_path) and not args.override:
                print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
                continue
            
            # Load BVH
            try:
                lafan1_data_frames, actual_human_height = load_bvh_file(bvh_file_path, format=args.format)
            except Exception as e:
                print(f"Error loading {bvh_file_path}: {e}")
                continue

            # Initialize retargeter
            retarget = GMR(
                src_human=f"bvh_{args.format}",
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )

            # Retarget all frames
            qpos_list = []
            for curr_frame in range(len(lafan1_data_frames)):
                smplx_data = lafan1_data_frames[curr_frame]
                qpos = retarget.retarget(smplx_data)
                qpos_list.append(qpos.copy())
            qpos_list = np.array(qpos_list)

            # Extract components
            root_pos = qpos_list[:, :3]  # (N, 3)
            # MuJoCo uses [w, x, y, z]; convert to [x, y, z, w] for output
            root_rot_wxyz = qpos_list[:, 3:7]
            root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]  # wxyz → xyzw
            dof_pos = qpos_list[:, 7:]  # (N, D)

            num_frames = len(qpos_list)

            # === Optional: Height adjustment via FK ===
            try:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                kinematics_model = KinematicsModel(retarget.xml_file, device=device)

                # Forward kinematics for height adjustment
                body_pos, _ = kinematics_model.forward_kinematics(
                    torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                    torch.from_numpy(root_rot_xyzw).to(device=device, dtype=torch.float),
                    torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
                )

                HEIGHT_ADJUST = True
                PERFRAME_ADJUST = False
                ground_offset = 0.00
                if HEIGHT_ADJUST:
                    if not PERFRAME_ADJUST:
                        lowest_height = torch.min(body_pos[..., 2]).item()
                        root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
                    else:
                        for i in range(root_pos.shape[0]):
                            lowest_body_part = torch.min(body_pos[i, :, 2])
                            root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset
            except Exception as e:
                print(f"[Warning] Height adjustment failed for {filename}: {e}. Skipping adjustment.")

            # === Build standard output format (identical to single-file script) ===
            # Get joint DOF names (skip floating joint)
            dof_names = [name for name, idx in sorted(retarget.robot_dof_names.items(), key=lambda x: x[1])
                         if name != "floating_joint"]

            # Build labels
            labels = []
            labels.extend(["root_pos/x", "root_pos/y", "root_pos/z"])
            labels.extend(["root_quat/x", "root_quat/y", "root_quat/z", "root_quat/w"])
            labels.extend([f"dof_pos/{name}" for name in dof_names])

            # Build frames: each is [px, py, pz, qx, qy, qz, qw, j1, j2, ...]
            frames = []
            for i in range(num_frames):
                frame_data = []
                frame_data.extend(root_pos[i].tolist())        # 3
                frame_data.extend(root_rot_xyzw[i].tolist())   # 4
                frame_data.extend(dof_pos[i].tolist())         # D
                frames.append(frame_data)

            # Meta info
            frame_duration = 1.0 / target_fps
            motion_data = {
                "LoopMode": "Once",
                "LoopNum": 1,
                "FrameDuration": frame_duration,
                "EnableCycleOffsetPosition": True,
                "EnableCycleOffsetRotation": True,
                "MotionWeight": 1.0,
                "Labels": labels,
                "Frames": frames,
            }

            # Save
            os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
            with open(tgt_file_path, "wb") as f:
                pickle.dump(motion_data, f)

    print(f"✅ Done. All motions saved to: {tgt_folder}")
