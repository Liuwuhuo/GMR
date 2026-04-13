import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from rich import print
from tqdm import tqdm
import os
import numpy as np
import pickle
import sys
import torch


def _load_pkl_motion_file(motion_file):
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
    return motion_data


# Names expected by general_motion_retargeting/ik_configs/fbx_offline_to_adam_sp.json
_FBX_OFFLINE_ADAM_SP_CANONICAL_JOINTS = frozenset(
    {
        "Hips",
        "Spine1",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftLeg",
        "RightLeg",
        "LeftToeBase",
        "RightToeBase",
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
    }
)

# OptiTrack / other skeletons use different root names for the same bone
_FBX_JOINT_ALIASES = {
    "Pelvis": "Hips",
    "pelvis": "Hips",
    "Root": "Hips",
    "root": "Hips",
    "ROOT": "Hips",
}

# OptiTrack / Motive 常用腕部命名，与 fbx_offline_to_adam_sp.json 中的 LeftHand/RightHand 对齐
_FBX_SYNONYMS_TO_IK = {
    "LeftWrist": "LeftHand",
    "RightWrist": "RightHand",
    "Left_Wrist": "LeftHand",
    "Right_Wrist": "RightHand",
    "L_Wrist": "LeftHand",
    "R_Wrist": "RightHand",
    "LWrist": "LeftHand",
    "RWrist": "RightHand",
}


def _fbx_joint_name_to_canonical(raw: str) -> str:
    """
    Map FBX skeleton node names to keys used in fbx_offline_to_adam_sp.json.

    Handles e.g. OptiTrack_Skeleton_Hips -> Hips (not Skeleton_Hips), and
    mixamorig:LeftUpLeg style separators.
    """
    s = raw.replace(":", "_").replace(" ", "_").strip()
    parts = [p for p in s.split("_") if p]
    if not parts:
        return raw

    # Longest suffix that matches a canonical Mixamo-style name (handles multi-prefix).
    for length in range(len(parts), 0, -1):
        for start in range(len(parts) - length + 1):
            cand = "_".join(parts[start : start + length])
            if cand in _FBX_OFFLINE_ADAM_SP_CANONICAL_JOINTS:
                return cand

    last = parts[-1]
    if last in _FBX_OFFLINE_ADAM_SP_CANONICAL_JOINTS:
        return last

    # Backward compatible: strip first prefix once
    if "_" in s:
        legacy = s.split("_", 1)[1]
    else:
        legacy = s

    return _FBX_JOINT_ALIASES.get(legacy, legacy)


def _apply_fbx_synonyms(name: str) -> str:
    return _FBX_SYNONYMS_TO_IK.get(name, name)


def _load_fbx_skeleton_motion(motion_file: str, fbx_root_joint: str, fbx_fps: int):
    """Load raw SkeletonMotion from .fbx (poselib)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    third_party_root = os.path.join(project_root, "third_party")
    if third_party_root not in sys.path:
        sys.path.insert(0, third_party_root)
    from poselib.skeleton.skeleton3d import SkeletonMotion

    return SkeletonMotion.from_fbx(
        fbx_file_path=motion_file,
        root_joint=fbx_root_joint,
        fps=fbx_fps,
    )


def print_fbx_skeleton_joint_labels(motion_file: str, fbx_root_joint: str, fbx_fps: int) -> None:
    """Print FBX skeleton node names and mapping used for IK keys, then return."""
    motion = _load_fbx_skeleton_motion(motion_file, fbx_root_joint, fbx_fps)
    joint_names = motion.skeleton_tree.node_names
    print(
        "\n[FBX] poselib skeleton_tree.node_names（原始）与映射到 IK 的 key（canonical + 别名）:\n"
        f"    root_joint 过滤参数: {fbx_root_joint!r}, fps={fbx_fps}\n"
    )
    canon_counts = {}
    for i, raw in enumerate(joint_names):
        c0 = _fbx_joint_name_to_canonical(raw)
        c1 = _FBX_JOINT_ALIASES.get(c0, c0)
        c2 = _apply_fbx_synonyms(c1)
        canon_counts[c2] = canon_counts.get(c2, 0) + 1
        print(f"  [{i:3d}] {raw!r}\n        -> {c2!r}")
    dup = {k: v for k, v in canon_counts.items() if v > 1}
    if dup:
        print(
            "\n[警告] 多个原始关节映射到同一 IK key（后者会覆盖前者）: "
            f"{dup}"
        )
    keys0 = set()
    for i, raw in enumerate(joint_names):
        c0 = _fbx_joint_name_to_canonical(raw)
        c1 = _FBX_JOINT_ALIASES.get(c0, c0)
        c2 = _apply_fbx_synonyms(c1)
        keys0.add(c2)
    print(
        f"\n[FBX] 映射后第一帧可用的 body key 共 {len(keys0)} 个（去重）:\n    "
        + ", ".join(sorted(keys0))
    )


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
            joint_name = _fbx_joint_name_to_canonical(joint_names[i])
            joint_name = _FBX_JOINT_ALIASES.get(joint_name, joint_name)
            joint_name = _apply_fbx_synonyms(joint_name)
            frame_data[joint_name] = [
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
        motion = _load_fbx_skeleton_motion(motion_file, fbx_root_joint, fbx_fps)
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
        help="Path to save retargeted motion as NPZ.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        default=False,
        help="Use compressed NPZ format.",
    )
    parser.add_argument(
        "--target_fps",
        default=None,
        type=float,
        help="Output FPS for visualization and saving. Defaults to --fbx_fps.",
    )
    parser.add_argument(
        "--compute_local_body_pos",
        action="store_true",
        default=False,
        help="Compute local body positions via FK.",
    )
    parser.add_argument(
        "--height_adjust",
        action="store_true",
        default=False,
        help="Adjust root height to avoid ground penetration.",
    )
    parser.add_argument(
        "--perframe_adjust",
        action="store_true",
        default=False,
        help="Adjust root height per frame (used with --height_adjust).",
    )
    parser.add_argument(
        "--print_fbx_joint_names",
        action="store_true",
        default=False,
        help="仅对 .fbx：打印 poselib 解析出的关节名及映射后的 IK key，然后退出。",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        default=False,
        help="Disable MuJoCo viewer (useful for batch processing).",
    )
    parser.add_argument(
        "--drop_first_frame",
        action="store_true",
        default=False,
        help="Drop the first retargeted frame (useful when frame 0 is unstable).",
    )

    args = parser.parse_args()

    ext = os.path.splitext(args.motion_file)[1].lower()
    if args.print_fbx_joint_names:
        if ext != ".fbx":
            raise SystemExit("--print_fbx_joint_names 仅适用于 .fbx 文件")
        print_fbx_skeleton_joint_labels(
            args.motion_file,
            args.fbx_root_joint,
            args.fbx_fps,
        )
        raise SystemExit(0)

    if args.save_path is None:
        motion_basename = os.path.splitext(os.path.basename(args.motion_file))[0]
        args.save_path = os.path.join(
            "retarget", args.robot, "fbx", f"{motion_basename}.npz"
        )
        print(f"未指定保存路径，使用默认路径: {args.save_path}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    qpos_list = []
    qvel_list = []

    
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

    motion_fps = int(round(args.target_fps)) if args.target_fps is not None else args.fbx_fps
    
    robot_motion_viewer = None
    if not args.no_viewer:
        robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                                motion_fps=motion_fps,
                                                transparent_robot=0,
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
    start_idx = 1 if args.drop_first_frame else 0
    if start_idx >= len(data_frames):
        raise ValueError("Cannot drop first frame: sequence has <= 1 frame.")
    pbar = tqdm(total=len(data_frames) - start_idx, desc="Retargeting OptiTrack motion")
    
    # Start the viewer
    i = start_idx

    while i < len(data_frames):
        # 必须先等：Space 暂停时阻塞在此，本帧的 retarget / 进度条都不会前进
        if robot_motion_viewer is not None:
            robot_motion_viewer.wait_while_paused()

        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        # Update task targets.
        smplx_data = data_frames[i]

        # retarget（暂停时上面已阻塞，不会执行到这里）
        qpos, qvel = retargeter.retarget(smplx_data)

        # visualize
        if robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=False,
                # human_pos_offset=np.array([0.0, 0.0, 0.0])
            )

        qpos_list.append(qpos.copy())
        qvel_list.append(qvel.copy())

        pbar.update(1)
        i += 1

    qpos_arr = np.asarray(qpos_list)
    qvel_arr = np.asarray(qvel_list)
    root_pos = qpos_arr[:, :3]
    root_rot = qpos_arr[:, 3:7]  # wxyz
    dof_pos = qpos_arr[:, 7:]
    dof_vel = qvel_arr[:, 6:]

    local_body_pos = None
    body_names = None
    if args.compute_local_body_pos and len(qpos_arr) > 0:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kinematics_model = KinematicsModel(retargeter.xml_file, device=device)

        num_frames = qpos_arr.shape[0]
        identity_root_pos = torch.zeros((num_frames, 3), device=device)
        identity_root_rot = torch.zeros((num_frames, 4), device=device)
        identity_root_rot[:, 0] = 1.0
        local_body_pos, _ = kinematics_model.forward_kinematics(
            identity_root_pos,
            identity_root_rot,
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
        )
        body_names = kinematics_model.body_names

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
                for j in range(root_pos.shape[0]):
                    lowest_body_part = torch.min(body_pos[j, :, 2])
                    root_pos[j, 2] = root_pos[j, 2] - lowest_body_part + ground_offset

        local_body_pos = local_body_pos.detach().cpu().numpy()

    save_dict = {
        "fps": np.array([motion_fps]),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
    }
    if local_body_pos is not None:
        save_dict["local_body_pos"] = local_body_pos
    if body_names is not None:
        save_dict["link_body_list"] = body_names

    if args.compressed:
        np.savez_compressed(args.save_path, **save_dict)
    else:
        np.savez(args.save_path, **save_dict)
    print(f"Saved: {args.save_path}")

    # Close progress bar
    pbar.close()
    
    if robot_motion_viewer is not None:
        robot_motion_viewer.close()