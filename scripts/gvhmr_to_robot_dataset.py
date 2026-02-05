"""
Example usage:
python scripts/gvhmr_to_robot_dataset.py \
  --src_folder /path/to/gvhmr_outputs \
  --tgt_folder /path/to/save_robot_dataset \
  --robot unitree_g1 \
  --record_video \
  --offset_ground \
  --joint_vel_limit
"""
import argparse
import pathlib
import os
import time
import numpy as np
import torch
import pickle
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting import KinematicsModel
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print


def compute_local_body_pos(xml_file, dof_pos):
    """Compute local body positions with root at origin and identity rotation.

    Args:
        xml_file: Path to the robot MJCF file used by the retargeter.
        dof_pos: Numpy array of shape (T, dof_dim) with per-frame joint positions.

    Returns:
        local_body_pos: Numpy array (T, num_bodies, 3) of local body positions.
        body_names: List of body names corresponding to the second dimension.
    """
    device = torch.device("cpu")
    kinematics_model = KinematicsModel(xml_file, device=device)
    num_frames = dof_pos.shape[0]
    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0
    local_body_pos_t, _ = kinematics_model.forward_kinematics(
        fk_root_pos,
        fk_root_rot,
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
    )
    local_body_pos = local_body_pos_t.detach().cpu().numpy()
    body_names = kinematics_model.body_names
    return local_body_pos, body_names


def process_single_gvhmr_file(gvhmr_file_path, output_dir, robot_type, args):
    """Process a single GVHMR pt file and save the results.
    
    Args:
        gvhmr_file_path: Path to the GVHMR pt file
        output_dir: Directory to save the output files
        robot_type: Type of robot to retarget to
        args: Command line arguments
    """
    HERE = pathlib.Path(__file__).parent
    # Use GVHMR body models path (GMR assets doesn't have body_models)
    # smplx.create() expects model_path to be the parent directory, and will look for smplx/ subdirectory inside
    REPO_ROOT = HERE.parents[2]  # Go up to Switch4EAI root (scripts -> GMR -> third_party -> Switch4EAI)
    SMPLX_FOLDER = REPO_ROOT / "third_party" / "GVHMR" / "inputs" / "checkpoints" / "body_models"
    
    # Extract motion name from the file path
    motion_name = os.path.basename(os.path.dirname(gvhmr_file_path))
    
    # Create output directory for this motion
    motion_output_dir = os.path.join(output_dir, motion_name)
    os.makedirs(motion_output_dir, exist_ok=True)
    
    # Define output file paths
    pose_file_path = os.path.join(motion_output_dir, f"{motion_name}_poses.pkl")
    video_file_path = os.path.join(motion_output_dir, f"{motion_name}_{robot_type}.mp4")
    
    # Skip if files already exist and not overriding
    if os.path.exists(pose_file_path) and os.path.exists(video_file_path) and not args.override:
        print(f"Skipping {motion_name} - files already exist")
        return
    
    try:
        # Load SMPLX trajectory
        smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
            gvhmr_file_path, SMPLX_FOLDER
        )
        
        # align fps
        tgt_fps = 30
        smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
        
        # Initialize the retargeting system
        retarget = GMR(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=robot_type,
            # use_velocity_limit=args.joint_vel_limit,
            # use_collision_avoidance=args.collision_avoid,
        )

        # Initialize robot motion viewer
        robot_motion_viewer = RobotMotionViewer(
            robot_type=robot_type,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=video_file_path,
        )
        
        # Process all frames
        qpos_list = []
        qvel_list = []
        for i in range(len(smplx_data_frames)):
            smplx_data = smplx_data_frames[i]
            
            # retarget
            qpos, qvel = retarget.retarget(smplx_data)
            qpos_list.append(qpos)
            qvel_list.append(qvel)
            
            # visualize
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
            )
        
        # Close the viewer
        robot_motion_viewer.close()
        
        # Save pose data
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])

        qvel_array = np.array(qvel_list)
        # qvel layout: [root_vel(3), root_ang_vel(3), dof_vel(N)]
        root_vel = qvel_array[:, :3]
        root_ang_vel = qvel_array[:, 3:6]
        dof_vel = qvel_array[:, 6:]

        # Compute local body positions via helper
        local_body_pos, body_names = compute_local_body_pos(retarget.xml_file, dof_pos)

        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "root_vel": root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_vel": dof_vel,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        
        with open(pose_file_path, "wb") as f:
            pickle.dump(motion_data, f)
        
        print(f"✓ Processed {motion_name}: saved pose data to {pose_file_path} and video to {video_file_path}")
        
    except Exception as e:
        print(f"✗ Error processing {motion_name}: {e}")
        return


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing GVHMR pt files to process.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../outputs/gvhmr_retargeted",
        type=str,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "pnd_adam_inspire", "adam_inspire", "adam_sp", "openloong", "tienkung"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
        help="Override existing files if they exist.",
    )
    
    parser.add_argument(
        "--record_video",
        default=True,
        action="store_true",
        help="Record the video for each motion.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--joint_vel_limit",
        default=True,
        action="store_true",
        help="Give joint velocity limit filtering"
    )

    parser.add_argument(
        "--offset_ground",
        default=False,
        action="store_true",
        help="Give offset ground"
    )

    parser.add_argument(
        "--collision_avoid",
        default=True,
        action="store_true",
        help="Give collision avoidance"
    )

    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder
    
    # Create target folder
    os.makedirs(tgt_folder, exist_ok=True)
    
    # Find all hmr4d_results.pt files
    pt_files = []
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file == "hmr4d_results.pt":
                pt_files.append(os.path.join(root, file))
    
    if not pt_files:
        print(f"No hmr4d_results.pt files found in {src_folder}")
        exit(1)
    
    print(f"Found {len(pt_files)} GVHMR pt files to process:")
    for pt_file in pt_files:
        motion_name = os.path.basename(os.path.dirname(pt_file))
        print(f"  - {motion_name}: {pt_file}")
    
    print(f"\nProcessing files with robot: {args.robot}")
    print(f"Output directory: {tgt_folder}")
    print(f"Record video: {args.record_video}")
    print(f"Override existing: {args.override}")
    print("-" * 50)
    
    # Process each file
    for pt_file in tqdm(pt_files, desc="Processing GVHMR files"):
        process_single_gvhmr_file(pt_file, tgt_folder, args.robot, args)
    
    print("\n" + "=" * 50)
    print("Batch processing completed!")
    print(f"Results saved to: {tgt_folder}")
    
    # Print summary of created files
    print("\nCreated files:")
    for root, dirs, files in os.walk(tgt_folder):
        for file in files:
            if file.endswith(('.pkl', '.mp4')):
                rel_path = os.path.relpath(os.path.join(root, file), tgt_folder)
                print(f"  - {rel_path}")
