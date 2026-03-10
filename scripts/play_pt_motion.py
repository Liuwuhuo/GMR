#!/usr/bin/env python3
"""
使用 MuJoCo 播放运动数据（.pt 或 .pkl）。
- .pt：
  - 支持老版 G1 data.pt：base_position, base_pose(欧拉), joint_position(27/29)；
  - 也支持新数据：base_position, base_quat(wxyz), joint_position(29)，无 base_pose。
- .pkl：root_pos, root_rot, dof_pos（或直接 qpos），与 retarget 输出格式一致。
支持 --robot unitree_g1（默认）或 adam_sp。
"""
import argparse
import os
import sys
import time
import pathlib

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# 宇树 G1：29 关节顺序中 waist_roll=13, waist_pitch=14（0-based），老数据只有 27 维不含这两项
WAIST_ROLL_IDX = 13
WAIST_PITCH_IDX = 14
G1_NQ = 7 + 29  # free(7) + joints(29)


def load_pt_motion(pt_file, robot="unitree_g1", base_quat_xyzw=False):
    """从 .pt 加载轨迹，返回 (qpos T×36, fps)。

    约定：
    - joint_position 为 29 维：直接视为与对应 XML 一致的关节顺序（不再额外重排）。
    - joint_position 为 27 维：按 G1 约定在 13/14 维插入 waist_roll/waist_pitch=0，以便填满 29 维。
    - base_quat：默认按 xyzw（PhysHSI/legged_gym 等常见）；若文件为 wxyz 则传 base_quat_xyzw=False（或 --quat-wxyz）。
    """
    import torch
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        data = {k: v.numpy() if hasattr(v, "numpy") else np.asarray(v) for k, v in data.items()}

    if "base_position" not in data or "joint_position" not in data:
        raise KeyError("pt 需至少包含 base_position 与 joint_position")

    base_pos = np.asarray(data["base_position"], dtype=np.float64)
    joint_pos = np.asarray(data["joint_position"], dtype=np.float64)  # (T, 27) 或 (T, 29)

    T = base_pos.shape[0]

    # 支持两种姿态表示：
    # 1) base_pose: 欧拉角 xyz（老数据）
    # 2) base_quat: 四元数，默认 xyzw（scipy 顺序，PhysHSI 常见）；若 base_quat_xyzw=False 则按 wxyz
    if "base_pose" in data:
        base_pose = np.asarray(data["base_pose"], dtype=np.float64)
    elif "base_quat" in data:
        base_quat = np.asarray(data["base_quat"], dtype=np.float64).reshape(T, 4)
        base_pose = np.zeros((T, 3), dtype=np.float64)
        for i in range(T):
            if base_quat_xyzw:
                # 文件里是 xyzw，直接给 scipy
                r = R.from_quat(base_quat[i])
            else:
                # 文件里是 wxyz，转成 xyzw 再给 scipy
                w, x, y, z = base_quat[i]
                r = R.from_quat([x, y, z, w])
            base_pose[i] = r.as_euler("xyz")
    else:
        raise KeyError("pt 需包含 base_pose(欧拉) 或 base_quat(四元数) 之一")

    # 接地：用 link_position 最低点或 base_position 最低点对齐地面 z=0，避免机器人浮空
    base_pos = np.asarray(base_pos, dtype=np.float64).copy()
    if "link_position" in data:
        link_pos = np.asarray(data["link_position"], dtype=np.float64)  # (T, N, 3)，N 可为 6 或 17
        if link_pos.ndim == 3 and link_pos.shape[1] > 0:
            min_z = 0
            base_pos[:, 2] -= min_z
    else:
        # 无 link_position 时用 base 自身最低点接地
        min_z = 0
        base_pos[:, 2] -= min_z
    # 若有 base_height 字段，则 base 的 z 直接使用 base_height（标量或每帧 (T,)）
    if "base_height" in data:
        h = np.asarray(data["base_height"], dtype=np.float64).ravel()
        base_pos[:, 2] = np.broadcast_to(h, (T,)).copy()

    # 欧拉 -> 四元数 wxyz（MuJoCo）
    quats = np.zeros((T, 4))
    for i in range(T):
        rot = R.from_euler("xyz", base_pose[i])
        q_xyzw = rot.as_quat()
        quats[i] = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])

    # 29 维：直接使用文件中的顺序（认为与 XML 一致），不做额外重排
    if joint_pos.shape[1] == 29:
        dof_full = joint_pos.copy()
    else:
        assert joint_pos.shape[1] == 27, f"joint_position 需为 (T, 27) 或 (T, 29)，当前 {joint_pos.shape}"
        dof_full = np.zeros((T, 29), dtype=np.float64)
        dof_full[:, :WAIST_ROLL_IDX] = joint_pos[:, :WAIST_ROLL_IDX]
        dof_full[:, WAIST_ROLL_IDX] = 0.0
        dof_full[:, WAIST_PITCH_IDX] = 0.0
        dof_full[:, WAIST_PITCH_IDX + 1 :] = joint_pos[:, WAIST_ROLL_IDX:]

    qpos_seq = np.concatenate([base_pos, quats, dof_full], axis=1)
    assert qpos_seq.shape == (T, G1_NQ), f"qpos shape {qpos_seq.shape} != (T, {G1_NQ})"
    fps = float(data.get("fps", 50.0))
    return qpos_seq, fps


def load_pkl_motion(pkl_file):
    """从 .pkl 加载轨迹（root_pos, root_rot, dof_pos 或 qpos），返回 (qpos_seq, fps)。"""
    import pickle
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        if "qpos" in data:
            qpos_seq = np.asarray(data["qpos"], dtype=np.float64)
            print(f"使用 qpos，形状: {qpos_seq.shape}")
        elif all(k in data for k in ["root_pos", "root_rot", "dof_pos"]):
            root_pos = np.asarray(data["root_pos"], dtype=np.float64)
            root_rot = np.asarray(data["root_rot"], dtype=np.float64)
            dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
            # 存盘多为 xyzw，MuJoCo 需要 wxyz
            if root_rot.shape[-1] == 4:
                root_rot = root_rot[:, [3, 0, 1, 2]]
            qpos_seq = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
            print(f"使用 root_pos + root_rot + dof_pos，形状: {qpos_seq.shape}")
        else:
            raise KeyError("pkl 需包含 qpos 或 root_pos/root_rot/dof_pos")
        fps = float(data.get("fps", data.get("frequency", 50.0)))
    elif isinstance(data, np.ndarray):
        qpos_seq = np.asarray(data, dtype=np.float64)
        fps = 50.0
        print(f"直接使用数组，形状: {qpos_seq.shape}")
    else:
        raise ValueError("无法解析 pkl 格式")
    return qpos_seq, fps


def main():
    HERE = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Play motion (.pt or .pkl) with Unitree G1 or adam_sp in MuJoCo")
    parser.add_argument("motion_file", type=str, help="Path to .pt or .pkl (e.g. datasets/g1_dof27_data/tennis_hit/data.pkl)", default="/home/liuhongji/workspace/AdaMimic/legged_gym/resources/dataset/g1_dof29_data/badminton_hit/output/data.pt")
    parser.add_argument("--robot", type=str, default="unitree_g1", choices=["unitree_g1", "adam_sp"],
                        help="Robot model to play: unitree_g1 (default) or adam_sp")
    parser.add_argument("--fps", type=float, default=None, help="Playback frequency (Hz); default: from file or 50")
    parser.add_argument("--no-loop", action="store_true", help="Stop at end instead of looping")
    parser.add_argument("--height-offset", type=float, default=0.0,
                        help="整体抬升 base_z (m)，默认 0.02；.pt 会先接地再加此值")
    parser.add_argument("--quat-wxyz", action="store_true",
                        help=".pt 中 base_quat 为 wxyz 顺序时使用；默认按 xyzw（PhysHSI/legged_gym 常见）")
    args = parser.parse_args()

    motion_path = pathlib.Path(args.motion_file)
    if not motion_path.is_file():
        print(f"错误: 文件不存在 {motion_path}")
        sys.exit(1)

    ext = motion_path.suffix.lower()
    if ext not in (".pt", ".pkl"):
        print(f"错误: 仅支持 .pt 或 .pkl，当前: {ext}")
        sys.exit(1)

    from general_motion_retargeting.params import ROBOT_XML_DICT
    if args.robot not in ROBOT_XML_DICT:
        print(f"错误: 不支持的 --robot {args.robot}，可选: unitree_g1, adam_sp")
        sys.exit(1)
    xml_path = pathlib.Path(ROBOT_XML_DICT[args.robot])
    if not xml_path.exists():
        xml_path = (HERE / ".." / xml_path).resolve()
    xml_path = str(xml_path.resolve())

    print(f"加载: {motion_path}")
    try:
        if ext == ".pt":
            # 默认 base_quat 按 xyzw；--quat-wxyz 时按 wxyz
            qpos_seq, frequency = load_pt_motion(
                str(motion_path), robot=args.robot, base_quat_xyzw=not args.quat_wxyz
            )
            if args.fps is not None:
                frequency = args.fps
        else:
            qpos_seq, frequency = load_pkl_motion(str(motion_path))
            if args.fps is not None:
                frequency = args.fps
    except Exception as e:
        print(f"加载失败: {e}")
        sys.exit(1)

    # qpos_seq[:, 2] += args.height_offset
    # if args.height_offset != 0:
    #     print(f"高度抬升: base_z += {args.height_offset}")

    T = qpos_seq.shape[0]
    print(f"轨迹: {T} 帧, 播放 {frequency} Hz, 机器人: {args.robot}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if model.nq != qpos_seq.shape[1]:
        print(f"警告: 模型 nq={model.nq}, 数据 nq={qpos_seq.shape[1]}")

    if qpos_seq.shape[1] != model.nq:
        if qpos_seq.shape[1] > model.nq:
            qpos_seq = qpos_seq[:, : model.nq]
        else:
            pad = np.zeros((T, model.nq - qpos_seq.shape[1]))
            qpos_seq = np.concatenate([qpos_seq, pad], axis=1)

    paused = True
    idx = 0
    base_frame_time = 1.0 / frequency
    speed_scale = 1.0
    frame_time = base_frame_time / speed_scale
    shift_pressed = False

    def key_callback(keycode):
        nonlocal paused, idx, speed_scale, frame_time, shift_pressed
        if keycode in [340, 344]:
            shift_pressed = True
            return
        if keycode == 32:
            paused = not paused
            print(f"\n{'暂停' if paused else '播放'}")
        elif keycode == 61:
            speed_scale = min(10.0, speed_scale * 1.5)
            frame_time = base_frame_time / speed_scale
            print(f"\n速度: {speed_scale:.1f}x")
        elif keycode == 45:
            speed_scale = max(0.1, speed_scale / 1.5)
            frame_time = base_frame_time / speed_scale
            print(f"\n速度: {speed_scale:.1f}x")
        elif keycode == 265:
            if idx < T - 1:
                idx += 1
                print(f"\n下一帧: {idx}")
        elif keycode == 264:
            if idx > 0:
                idx -= 1
                print(f"\n上一帧: {idx}")
        elif keycode == 262:
            jump = 5 if shift_pressed else 1
            n = max(1, int(T * jump / 100))
            idx = min(T - 1, idx + n)
            print(f"\n前进{jump}%: {idx}")
        elif keycode == 263:
            jump = 5 if shift_pressed else 1
            n = max(1, int(T * jump / 100))
            idx = max(0, idx - n)
            print(f"\n后退{jump}%: {idx}")
        elif keycode == 114:
            idx = 0
            print("\n重置到第一帧")
        shift_pressed = False

    def key_release_callback(keycode):
        nonlocal shift_pressed
        if keycode in [340, 344]:
            shift_pressed = False

    print("\n控制: 空格 暂停/播放  +/- 速度  上下箭头 单帧  左右箭头 跳转%  R 重置")
    print("按空格开始播放")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        last_frame_time = time.time()
        pbar = tqdm(
            total=T,
            desc="播放",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        while viewer.is_running:
            current_time = time.time()
            if not paused and (current_time - last_frame_time >= frame_time):
                last_frame_time = current_time
                idx += 1
                if args.no_loop and idx >= T:
                    idx = T - 1
                    paused = True
                    print("\n播放结束（--no-loop）")
            if not args.no_loop and idx >= T:
                idx = 0
            idx = max(0, min(T - 1, idx))

            data.qpos[:] = qpos_seq[idx]
            mujoco.mj_forward(model, data)

            if idx >= pbar.n:
                pbar.update(idx - pbar.n)
            else:
                pbar.n = idx
                pbar.refresh()
            viewer.sync()
            time.sleep(0.02)

        pbar.close()
    print("播放结束")


if __name__ == "__main__":
    main()
