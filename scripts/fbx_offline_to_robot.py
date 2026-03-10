import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from rich import print
from tqdm import tqdm
import os
import numpy as np
import pickle
import sys


def _load_pkl_motion_file(motion_file):
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
    return motion_data


def _convert_skeleton_motion_to_retarget_frames(motion):
    import torch
    from poselib.core.rotation3d import quat_rotate, quat_mul_norm

    global_positions = quat_rotate(
        torch.tensor([0.70711, 0, 0, 0.70711]),
        motion.global_translation,
    ).detach().cpu().numpy() / 100.0  # cm -> m
    global_quaternions = quat_mul_norm(
        torch.tensor([0.70711, 0, 0, 0.70711]),
        motion.global_rotation,
    ).detach().cpu().numpy()  # y-up -> z-up
    joint_names = motion.skeleton_tree.node_names

    data = []
    num_frames = global_positions.shape[0]
    num_joints = len(joint_names)
    for frame in range(num_frames):
        frame_data = {}
        for i in range(num_joints):
            frame_data[joint_names[i].split("_")[1]] = [
                global_positions[frame, i].tolist(),
                global_quaternions[frame, i, [3, 0, 1, 2]].tolist(),  # xyzw -> wxyz
            ]
        data.append(frame_data)
    return data


def load_optitrack_motion_file(motion_file, fbx_root_joint="Hips", fbx_fps=120):
    ext = os.path.splitext(motion_file)[1].lower()
    if ext in [".pkl", ".pickle"]:
        return _load_pkl_motion_file(motion_file)
    if ext == ".fbx":
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        third_party_root = os.path.join(project_root, "third_party")
        if third_party_root not in sys.path:
            sys.path.insert(0, third_party_root)
        from poselib.skeleton.skeleton3d import SkeletonMotion

        motion = SkeletonMotion.from_fbx(
            fbx_file_path=motion_file,
            root_joint=fbx_root_joint,
            fps=fbx_fps,
        )
        return _convert_skeleton_motion_to_retarget_frames(motion)
    raise ValueError(
        f"Unsupported motion file extension: {ext}. "
        "Expected .pkl/.pickle or .fbx"
    )

def offset_to_ground(retargeter: GMR, motion_data):
    offset = np.inf
    for human_data in motion_data:
        human_data = retargeter.to_numpy(human_data)
        human_data = retargeter.scale_human_data(human_data, retargeter.human_root_name, retargeter.human_scale_table)
        human_data = retargeter.offset_human_data(human_data, retargeter.pos_offsets1, retargeter.rot_offsets1)
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < offset:
                offset = pos[2]

    return offset

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_file",
        help="OptiTrack motion file path (.pkl converted data or raw .fbx).",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--fbx_root_joint",
        default="Hips",
        type=str,
        help="Root joint name used when parsing raw .fbx.",
    )

    parser.add_argument(
        "--fbx_fps",
        default=120,
        type=int,
        help="FBX sampling fps used by PoseLib when parsing raw .fbx.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "adam_sp"],
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
        default="videos/optitrack_example.mp4",
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
    
    
    args = parser.parse_args()
    

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load OptiTrack motion trajectory (.pkl or .fbx)
    print(f"Loading OptiTrack motion file: {args.motion_file}")
    data_frames = load_optitrack_motion_file(
        args.motion_file,
        fbx_root_joint=args.fbx_root_joint,
        fbx_fps=args.fbx_fps,
    )
    print(f"Loaded {len(data_frames)} frames")
    
    
    # Initialize the retargeting system with fbx configuration
    retargeter = GMR(
        src_human="fbx_offline",  # Use the new fbx configuration
        tgt_robot=args.robot,
        actual_human_height=1.8,
    )

    height_offset = offset_to_ground(retargeter, data_frames)
    retargeter.set_ground_offset(height_offset)

    motion_fps = 120
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=1,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            camera_follow=False,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(data_frames), desc="Retargeting OptiTrack motion")
    
    # Start the viewer
    i = 0

    while i < len(data_frames):
        
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
        smplx_data = data_frames[i]

        # retarget
        qpos, _ = retargeter.retarget(smplx_data)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        i += 1

        if args.save_path is not None:
            qpos_list.append(qpos)

    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
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