#!/usr/bin/env python3
"""
基于 smplx_to_robot.py，支持「数据集 pkl」格式的 SMPL 人体数据并 retarget 到机器人。
pkl 格式：顶层 dict 下有一条或多条动捕（如 data["sidewalk"]），每条含 poses (T,66), trans (T,3), betas (10,), gender, mocap_framerate。
poses 拆为 root_orient (T,3) + pose_body (T,63)，IK 配置与 smplx_to_robot 一致（smplx -> robot）。
"""
import argparse
import pathlib
import os
import time
import pickle

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
import smplx
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast

from rich import print

# 与 datasets/g1_dof27_data/.../output/data.pt 一致的 .pt 字段；27 关节不含 waist_roll/waist_pitch
WAIST_ROLL_IDX, WAIST_PITCH_IDX = 13, 14
NUM_LINK_BODIES = 17
# unitree_g1 用于 link 的 17 个 body 名（按常见顺序）
G1_LINK_BODY_NAMES = [
    # 下半身（lower keyframes）
    "keyframe_pelvis_link",        # 0
    "keyframe_left_hip_link",      # 1
    "keyframe_left_knee_link",     # 2
    "keyframe_left_ankle_link",    # 3
    "keyframe_right_hip_link",     # 4
    "keyframe_right_knee_link",    # 5
    "keyframe_right_ankle_link",   # 6
    # 上半身
    "keyframe_head_link",          # 7
    "keyframe_torso_link",         # 8
    "keyframe_left_collar_link",   # 9
    "keyframe_left_shoulder_link", # 10
    "keyframe_left_elbow_link",    # 11
    "keyframe_left_wrist_link",    # 12
    "keyframe_right_collar_link",  # 13
    "keyframe_right_shoulder_link",# 14
    "keyframe_right_elbow_link",   # 15
    "keyframe_right_wrist_link",   # 16
]
ADAM_LINK_BODY_NAMES = [
    # 下半身（lower keyframes）
    "keyframe_pelvis_link",        # 0
    "keyframe_left_hip_link",      # 1
    "keyframe_left_knee_link",     # 2
    "keyframe_left_ankle_link",    # 3
    "keyframe_right_hip_link",     # 4
    "keyframe_right_knee_link",    # 5
    "keyframe_right_ankle_link",   # 6
    # 上半身
    "keyframe_head_link",          # 7
    "keyframe_torso_link",         # 8
    "keyframe_left_collar_link",   # 9
    "keyframe_left_shoulder_link", # 10
    "keyframe_left_elbow_link",    # 11
    "keyframe_left_wrist_link",    # 12
    "keyframe_right_collar_link",  # 13
    "keyframe_right_shoulder_link",# 14
    "keyframe_right_elbow_link",   # 15
    "keyframe_right_wrist_link",   # 16
]


def build_pt_motion(qpos_list, qvel_list, model, robot_type, fps):
    """
    从 qpos_list (T, 36)、qvel_list (T, 35) 构建与 output/data.pt 同结构的 dict，
    用于 torch.save(.pt)。含 base_*、joint_*（G1 与 adam_sp 均为 29 维，顺序与各自 XML 一致）、link_*（17 个 body）。
    速度统一用中心差分 + fps 计算（中间帧 (x[i+1]-x[i-1])/(2*dt)，首尾复制相邻），不直接使用 qvel。
    """
    qpos = np.array(qpos_list, dtype=np.float32)
    qvel = np.array(qvel_list, dtype=np.float32)  # 目前仅用于兼容接口，速度实际改用差分计算
    T = qpos.shape[0]
    link_names = G1_LINK_BODY_NAMES if "unitree" in robot_type else ADAM_LINK_BODY_NAMES
    body_ids = []
    for name in link_names[:NUM_LINK_BODIES]:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        body_ids.append(bid if bid >= 0 else None)

    base_position = qpos[:, :3].copy()
    # 记录首帧的世界坐标，用于统一平移 base 和 link
    base_xy0 = base_position[0, :2].copy()
    # XY 以首帧为原点平移（首帧 base_position 的 x,y = 0）
    base_position[:, 0] -= base_xy0[0]
    base_position[:, 1] -= base_xy0[1]
    quat = qpos[:, 3:7]
    base_pose = np.zeros((T, 3), dtype=np.float32)
    for i in range(T):
        w, x, y, z = quat[i]
        r = R.from_quat([x, y, z, w])
        base_pose[i] = r.as_euler("xyz")
    # === 中心差分 + fps 计算 base 线速度 / 角速度（中间帧 2*dt，首尾复制相邻）===
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    base_velocity = np.zeros((T, 3), dtype=np.float32)
    base_angular_velocity = np.zeros((T, 3), dtype=np.float32)
    if T > 1:
        # 中心差分
        base_velocity[1:-1] = (base_position[2:] - base_position[:-2]) / (2 * dt)
        base_velocity[0] = base_velocity[1]
        base_velocity[-1] = base_velocity[-2]

        # 欧拉角差分，注意 2π wrap
        dpose = base_pose[2:] - base_pose[:-2]
        dpose = (dpose + np.pi) % (2 * np.pi) - np.pi  # wrap 到 [-pi, pi]
        base_angular_velocity[1:-1] = dpose / (2 * dt)
        base_angular_velocity[0] = base_angular_velocity[1]
        base_angular_velocity[-1] = base_angular_velocity[-2]

    dof = qpos[:, 7:36]
    # G1 与 adam_sp 都按各自 XML 的 DoF 顺序完整保存 29 维关节，播放时可直接还原 retarget 结果
    joint_position = dof.astype(np.float32)
    # 关节速度：中心差分
    joint_velocity = np.zeros_like(joint_position, dtype=np.float32)
    if T > 1:
        joint_velocity[1:-1] = (joint_position[2:] - joint_position[:-2]) / (2 * dt)
        joint_velocity[0] = joint_velocity[1]
        joint_velocity[-1] = joint_velocity[-2]

    link_position = np.zeros((T, NUM_LINK_BODIES, 3), dtype=np.float32)
    link_orientation = np.zeros((T, NUM_LINK_BODIES, 3), dtype=np.float32)
    data = mj.MjData(model)
    for i in range(T):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mj.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            if bid is not None:
                link_position[i, j] = data.xpos[bid]
                xmat = np.array(data.xmat[bid]).reshape(3, 3)
                r = R.from_matrix(xmat)
                link_orientation[i, j] = r.as_euler("xyz")

    # link 位置也减去同样的 XY 偏移，使其与 base 处于同一坐标系（首帧 base 在 (0,0)）
    link_position[..., 0] -= base_xy0[0]
    link_position[..., 1] -= base_xy0[1]

    # link 线速度 / 角速度：中心差分，首尾复制相邻
    link_velocity = np.zeros((T, NUM_LINK_BODIES, 3), dtype=np.float32)
    link_angular_velocity = np.zeros((T, NUM_LINK_BODIES, 3), dtype=np.float32)
    for j in range(NUM_LINK_BODIES):
        for k in range(3):
            link_velocity[1:-1, j, k] = (link_position[2:, j, k] - link_position[:-2, j, k]) / (2 * dt)
            link_angular_velocity[1:-1, j, k] = (link_orientation[2:, j, k] - link_orientation[:-2, j, k]) / (2 * dt)
    if T > 1:
        link_velocity[0] = link_velocity[1]
        link_velocity[-1] = link_velocity[-2]
        link_angular_velocity[0] = link_angular_velocity[1]
        link_angular_velocity[-1] = link_angular_velocity[-2]

    # 为了与旧 g1 data.pt 结构统一，这里统一返回 torch.Tensor（dtype=float32）
    def t(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    return {
        "base_pose": t(base_pose),
        "base_position": t(base_position),
        "base_velocity": t(base_velocity),
        "base_angular_velocity": t(base_angular_velocity),
        "joint_position": t(joint_position),
        "joint_velocity": t(joint_velocity),
        "link_position": t(link_position),
        "link_orientation": t(link_orientation),
        "link_velocity": t(link_velocity),
        "link_angular_velocity": t(link_angular_velocity),
    }


def load_smplx_from_pkl(pkl_file, smplx_body_model_path, clip_key=None):
    """
    从数据集 pkl 加载 SMPL 格式并转为与 load_smplx_file 一致的 (smplx_data, body_model, smplx_output, human_height)。
    pkl 顶层为 dict，如 {"sidewalk": {"poses": (T,66), "trans": (T,3), "betas": (10,), "gender": str, "mocap_framerate": int}}。
    clip_key: 用哪一条动捕，None 则用第一个 key。
    """
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError("pkl 顶层需为非空 dict")
    if clip_key is None:
        clip_key = next(iter(data.keys()))
    clip = data[clip_key]
    if not isinstance(clip, dict):
        raise ValueError(f"pkl[{clip_key}] 需为 dict")
    for k in ["poses", "trans", "betas", "gender"]:
        if k not in clip:
            raise KeyError(f"pkl[{clip_key}] 缺少字段: {k}")

    poses = np.asarray(clip["poses"], dtype=np.float32)
    trans = np.asarray(clip["trans"], dtype=np.float32)
    betas = np.asarray(clip["betas"], dtype=np.float32)
    gender = str(clip["gender"])
    mocap_framerate = int(clip.get("mocap_framerate", clip.get("mocap_frame_rate", 30)))

    T = poses.shape[0]
    if poses.shape[1] != 66:
        raise ValueError(f"poses 需为 (T, 66)，当前为 {poses.shape}")
    root_orient = poses[:, :3].copy()
    pose_body = poses[:, 3:66].copy()

    if len(betas.shape) == 1:
        betas = betas.reshape(1, -1)
    betas = np.asarray(betas, dtype=np.float32)

    body_model = smplx.create(
        str(smplx_body_model_path),
        "smplx",
        gender=gender,
        use_pca=False,
    )
    num_betas = getattr(body_model, "num_betas", None)
    if num_betas is None and hasattr(body_model, "shapedirs") and body_model.shapedirs is not None:
        num_betas = body_model.shapedirs.shape[-1]
    if num_betas is None:
        num_betas = 10
    if betas.shape[1] < num_betas:
        betas = np.pad(betas, ((0, 0), (0, num_betas - betas.shape[1])), mode="constant", constant_values=0)
    betas = betas[:, :num_betas].astype(np.float32)
    betas_1d = betas.squeeze() if betas.shape[0] == 1 else betas

    smplx_data = {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "trans": trans,
        "betas": betas_1d,
        "gender": gender,
        "mocap_frame_rate": np.array(mocap_framerate, dtype=np.float64),
    }

    num_frames = T
    smplx_output = body_model(
        betas=torch.tensor(betas).float(),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )
    if len(smplx_data["betas"].shape) == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]
    return smplx_data, body_model, smplx_output, human_height


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="SMPL pkl -> robot retarget (same IK as smplx_to_robot)")
    parser.add_argument("--pkl_file", type=str, required=True, help="数据集 pkl 路径（含 poses/trans/betas/gender）")
    parser.add_argument("--clip", type=str, default=None, help="pkl 内动捕 key，默认用第一个")
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
            "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
            "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
            "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "adam_sp",
            "openloong", "tienkung", "unitree_g1_27dof",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="保存路径；不指定时默认保存到 pkl 同目录下 output/data.pkl 与 output/data.pt",
    )
    parser.add_argument("--loop", action="store_true", help="循环播放")
    parser.add_argument("--record_video", action="store_true", help="录屏")
    parser.add_argument("--rate_limit", action="store_true", help="按原帧率限速")
    parser.add_argument("--tgt_fps", type=float, default=50, help="对齐目标帧率（默认 50 Hz）")
    parser.add_argument(
        "--base_height_offset",
        type=float,
        default=0.0,
        help="贴地后整体上移的基座高度偏移（m），用于让 base/link_position 高度分布更接近旧数据集，例如 G1 可尝试 0.2~0.3",
    )
    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    pkl_path = pathlib.Path(args.pkl_file).resolve()
    if not pkl_path.is_file():
        raise FileNotFoundError(f"pkl 不存在: {pkl_path}")

    # 未指定 --save_path 时，默认保存到 pkl 同目录下 output/data.pkl（及 output/data.pt）
    save_path = args.save_path
    if save_path is None:
        save_path = str(pkl_path.parent / "output" / "data.pkl")

    print(f"加载 pkl: {pkl_path}, clip={args.clip or '首个 key'}")
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_from_pkl(
        str(pkl_path), SMPLX_FOLDER, clip_key=args.clip
    )
    print(f"帧数: {smplx_data['pose_body'].shape[0]}, 身高估计: {actual_human_height:.3f}")

    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps
    )

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        base_height_offset=args.base_height_offset,
    )
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{args.robot}_{pkl_path.stem}.mp4",
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    qpos_list = []
    qvel_list = []

    i = 0
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    while True:
        if not robot_motion_viewer.paused:
            if args.loop:
                i = (i + 1) % len(smplx_data_frames)
            else:
                i += 1
                if i >= len(smplx_data_frames):
                    break
        fps_counter += 1
        t = time.time()
        if t - fps_start_time >= fps_display_interval:
            print(f"Actual rendering FPS: {fps_counter / (t - fps_start_time):.2f}")
            fps_counter = 0
            fps_start_time = t

        smplx_frame = smplx_data_frames[i]
        qpos, qvel = retarget.retarget(smplx_frame, offset_to_ground=True)
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )
        qpos_list.append(qpos)
        qvel_list.append(qvel)

    # 为了与旧版数据集对齐，去掉首尾各一帧（若总帧数 > 2）
    if len(qpos_list) > 2:
        qpos_list = qpos_list[1:-1]
        qvel_list = qvel_list[1:-1]

    root_pos = np.array([q[:3] for q in qpos_list])
    root_rot = np.array([[q[4], q[5], q[6], q[3]] for q in qpos_list])
    dof_pos = np.array([q[7:] for q in qpos_list])
    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"Saved to {save_path}")

    # 保存与 datasets/g1_dof27_data/.../output/data.pt 同格式的 .pt（含 labels）
    pt_path = os.path.splitext(save_path)[0] + ".pt"
    dof_names_all = [
        name for name, _ in sorted(retarget.robot_dof_names.items(), key=lambda x: x[1])
        if name != "floating_joint"
    ]
    # G1 与 adam_sp 都保存完整 29 维关节名，顺序与 XML / MjModel 中一致
    dof_names_joint = dof_names_all
    labels = (
        ["base_position/x", "base_position/y", "base_position/z"]
        + ["base_pose/x", "base_pose/y", "base_pose/z"]
        + [f"joint_position/{n}" for n in dof_names_joint]
    )
    pt_dict = build_pt_motion(
        qpos_list, qvel_list, retarget.model, args.robot, aligned_fps
    )
    # 所有机器人统一与旧版 G1 data.pt 保持同样结构：
    # 仅包含 base_*, joint_*, link_*，且为 torch.Tensor，不额外存 fps/labels/robot
    torch.save(pt_dict, pt_path)
    print(f"Saved .pt to {pt_path}")

    robot_motion_viewer.close()
