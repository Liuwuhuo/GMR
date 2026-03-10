#!/usr/bin/env python3
"""
BVH -> Robot 重定向并导出为 NPZ，包含：
  fps: (1,)
  joint_pos: (T, 29)
  joint_vel: (T, 29)
  body_pos_w: (T, 30, 3)   # 固定 30 个 link（见 BODY_EXPORT_ORDER）
  body_quat_w: (T, 30, 4)  # wxyz
  body_lin_vel_w: (T, 30, 3)
  body_ang_vel_w: (T, 30, 3)
"""
import argparse
import os

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR, qvel_from_qpos_central
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.smpl import slerp

# 导出 body 的固定顺序（30 个），与 mesh/link 对应
# adam_sp 与 unitree_g1 的 body 名不同，按语义对齐
NUM_EXPORT_BODIES = 30
BODY_EXPORT_ORDER = {
    "adam_sp": [
        "pelvis", "hipPitchLeft", "hipRollLeft", "thighLeft", "shinLeft",
        "anklePitchLeft", "toeLeft",
        "hipPitchRight", "hipRollRight", "thighRight", "shinRight",
        "anklePitchRight", "toeRight",
        "waistRoll", "waistPitch", "torso",
        "shoulderPitchLeft", "shoulderRollLeft", "shoulderYawLeft",
        "elbowLeft", "wristYawLeft", "wristPitchLeft", "wristRollLeft",
        "shoulderPitchRight", "shoulderRollRight", "shoulderYawRight",
        "elbowRight", "wristYawRight", "wristPitchRight", "wristRollRight",
    ],
    "unitree_g1": [
        "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_toe_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_toe_link",
        "waist_yaw_link", "waist_roll_link", "torso_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
        "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
        "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ],
}
# 未在 BODY_EXPORT_ORDER 中的 robot 使用模型全部 body，数量可能不为 30


def _quat_xyzw_to_wxyz(q):
    """(..., 4) xyzw -> wxyz (导出与 MuJoCo 一致)."""
    q = np.asarray(q)
    if q.ndim == 2:
        return q[:, [3, 0, 1, 2]]
    return q[..., [3, 0, 1, 2]]


def _body_lin_vel_central(pos, dt):
    """pos (T, N, 3) -> lin_vel (T, N, 3), 中心差分，首尾单侧."""
    T = pos.shape[0]
    vel = np.zeros_like(pos)
    if T <= 1:
        return vel
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    if T > 2:
        vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    return vel


def _quat_mul_xyzw(a, b):
    """a, b: (..., 4) xyzw, 返回 a*b (..., 4) xyzw."""
    a = np.asarray(a)
    b = np.asarray(b)
    orig = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out = np.stack([x, y, z, w], axis=-1)
    return out.reshape(orig)


def _quat_conjugate_xyzw(q):
    """(..., 4) xyzw 单位四元数的逆."""
    q = np.asarray(q)
    c = np.empty_like(q)
    c[..., 0] = -q[..., 0]
    c[..., 1] = -q[..., 1]
    c[..., 2] = -q[..., 2]
    c[..., 3] = q[..., 3]
    return c


def _body_ang_vel_from_quat_xyzw(quat_xyzw, dt):
    """quat (T,N,4), dt -> body_ang_vel_w (T,N,3). omega: dq/dt=0.5*[0,omega]*q."""
    T, N, _ = quat_xyzw.shape
    dq = np.zeros_like(quat_xyzw)
    if T <= 1:
        return np.zeros((T, N, 3), dtype=quat_xyzw.dtype)
    dq[0] = (quat_xyzw[1] - quat_xyzw[0]) / dt
    dq[-1] = (quat_xyzw[-1] - quat_xyzw[-2]) / dt
    if T > 2:
        dq[1:-1] = (quat_xyzw[2:] - quat_xyzw[:-2]) / (2.0 * dt)
    # omega_quat = 2 * dq * q^{-1} (纯四元数，取 xyz 为角速度)
    q_inv = _quat_conjugate_xyzw(quat_xyzw)
    omega_quat = 2.0 * _quat_mul_xyzw(dq, q_inv)
    return omega_quat[..., :3]


def _resample_bvh_frames(frames, src_fps, tgt_fps):
    """按帧率重采样 BVH 人体帧（位置线性插值 + 旋转 SLERP）。

    frames: list[dict[bone_name -> (pos(3,), quat_wxyz(4,))]]
    返回: (new_frames, aligned_fps)
    """
    if not frames or src_fps <= 0 or tgt_fps <= 0:
        return frames, src_fps
    if abs(src_fps - tgt_fps) < 1e-6:
        return frames, src_fps

    num_frames = len(frames)
    bone_names = list(frames[0].keys())

    # 在 frame 索引空间 [0, num_frames-1] 上做重采样
    # 目标帧数按时长比例缩放
    duration = num_frames - 1
    new_num_frames = int(round(duration * tgt_fps / src_fps)) + 1
    new_num_frames = max(2, new_num_frames)

    target_idx = np.linspace(0.0, float(duration), new_num_frames)

    new_frames = []
    for t in target_idx:
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, num_frames - 1)
        alpha = float(t - i0)
        frame_new = {}
        for name in bone_names:
            pos0, quat0 = frames[i0][name]
            pos1, quat1 = frames[i1][name]
            pos0 = np.asarray(pos0, dtype=np.float64)
            pos1 = np.asarray(pos1, dtype=np.float64)
            # 位置线性插值
            pos = (1.0 - alpha) * pos0 + alpha * pos1

            # 四元数 wxyz -> SciPy 的 xyzw，然后 SLERP 再转回 wxyz
            q0 = np.asarray(quat0, dtype=np.float64)
            q1 = np.asarray(quat1, dtype=np.float64)
            rot0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
            rot1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
            rot_interp = slerp(rot0, rot1, alpha)
            q_xyzw = rot_interp.as_quat()
            quat = np.array(
                [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
            )  # 回到 wxyz

            frame_new[name] = [pos, quat]
        new_frames.append(frame_new)

    return new_frames, float(tgt_fps)


def main():
    parser = argparse.ArgumentParser(description="BVH -> Robot 重定向并导出 NPZ")
    parser.add_argument("--bvh_file", required=True, help="BVH 文件路径")
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "sfu", "noitom", "mocap"],
        default="lafan1",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "booster_t1",
            "stanford_toddy", "fourier_n1", "engineai_pm01",
            "pal_talos", "adam_sp_pro", "adam_sp",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--save_path", default=None, help="输出 NPZ 路径")
    parser.add_argument("--motion_fps", type=int, default=30,
                        help="目标输出 FPS（用于重采样 + 速度计算）")
    parser.add_argument(
        "--src_fps",
        type=float,
        default=30.0,
        help="BVH 原始帧率（lafan1 通常为 30），用于重采样",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="起始帧（含）")
    parser.add_argument("--end_frame", type=int, default=None, help="结束帧（不含）")
    args = parser.parse_args()

    if args.save_path is None:
        bvh_basename = os.path.splitext(os.path.basename(args.bvh_file))[0]
        default_dir = "retarget"
        robot_dir = args.robot
        os.makedirs(default_dir, exist_ok=True)
        args.save_path = os.path.join(
            default_dir, robot_dir, args.format, f"{bvh_basename}.npz"
        )  # noqa: E501
        print(f"未指定保存路径，使用默认: {args.save_path}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 加载 BVH（原始帧率假定为 args.src_fps）
    lafan1_data_frames, actual_human_height = load_bvh_file(
        args.bvh_file, format=args.format
    )
    # 按帧率重采样到 motion_fps
    lafan1_data_frames, aligned_fps = _resample_bvh_frames(
        lafan1_data_frames, src_fps=args.src_fps, tgt_fps=args.motion_fps
    )
    start = max(0, args.start_frame)
    end = args.end_frame if args.end_frame is not None else len(lafan1_data_frames)
    lafan1_data_frames = lafan1_data_frames[start:end]
    num_frames = len(lafan1_data_frames)

    # 重定向
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )
    qpos_list = []
    for i in tqdm(range(num_frames), desc="Retargeting"):
        qpos, _ = retargeter.retarget(lafan1_data_frames[i], offset_to_ground=True, no_fly=True)
        qpos_list.append(qpos.copy())

    qpos_arr = np.array(qpos_list, dtype=np.float32)
    dof_pos = qpos_arr[:, 7:]
    n_dof = dof_pos.shape[1]

    # 速度统一用中心差分（dt=1/aligned_fps），较单侧差分更平滑
    # base + joint：qvel 由 qpos 序列中心差分得到
    dt = 1.0 / aligned_fps
    qvel_arr = qvel_from_qpos_central(qpos_arr, aligned_fps)
    joint_vel = qvel_arr[:, 6 : 6 + n_dof]

    # 使用 MuJoCo mj_forward + data.xpos / data.xmat，与 retarget 同一 model
    model = retargeter.model
    data = mj.MjData(model)
    n_body_all = model.nbody
    body_pos_all = np.zeros((num_frames, n_body_all, 3), dtype=np.float32)
    body_quat_xyzw_all = np.zeros((num_frames, n_body_all, 4), dtype=np.float32)
    for i in tqdm(range(num_frames), desc="FK (xpos)"):
        data.qpos[:] = qpos_arr[i]
        mj.mj_forward(model, data)
        body_pos_all[i] = np.asarray(data.xpos).copy()
        for b in range(n_body_all):
            body_quat_xyzw_all[i, b] = R.from_matrix(
                np.asarray(data.xmat[b]).reshape(3, 3)
            ).as_quat()

    # 若该 robot 有固定 30 link 定义，则只导出这 30 个 body
    if args.robot in BODY_EXPORT_ORDER:
        export_names = BODY_EXPORT_ORDER[args.robot]
        body_ids = []
        for name in export_names:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise KeyError(
                    f"模型 {args.robot} 中未找到 body: {name}，请检查 BODY_EXPORT_ORDER"
                )
            body_ids.append(bid)
        body_ids = np.array(body_ids)
        body_pos = body_pos_all[:, body_ids, :]
        body_quat_xyzw = body_quat_xyzw_all[:, body_ids, :]
        export_body_names = export_names
    else:
        body_pos = body_pos_all
        body_quat_xyzw = body_quat_xyzw_all
        export_body_names = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b)
            for b in range(n_body_all)
        ]

    body_quat_w = _quat_xyzw_to_wxyz(body_quat_xyzw)
    # body 线速度 / 角速度：中心差分（_body_lin_vel_central、_body_ang_vel_from_quat_xyzw 内已用 2*dt）
    body_lin_vel_w = _body_lin_vel_central(body_pos, dt)
    body_ang_vel_w = _body_ang_vel_from_quat_xyzw(body_quat_xyzw, dt)

    fps_arr = np.array([aligned_fps], dtype=np.float64)

    np.savez(
        args.save_path,
        fps=fps_arr,
        joint_pos=dof_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"Saved: {args.save_path}")
    sh = (
        f"  fps {fps_arr.shape}, joint_pos {dof_pos.shape}, joint_vel {joint_vel.shape}, "
        f"body_pos_w {body_pos.shape}, body_quat_w {body_quat_w.shape}, "
        f"body_lin_vel_w {body_lin_vel_w.shape}, body_ang_vel_w {body_ang_vel_w.shape}"
    )
    print(sh)
    print(f"  body_names (n_bodies={len(export_body_names)}): {export_body_names}")


if __name__ == "__main__":
    main()
