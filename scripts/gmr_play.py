#!/usr/bin/env python3
import argparse
import re
import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os
from tqdm import tqdm
from general_motion_retargeting.params import ROBOT_XML_DICT


def to_wxyz(quat_array, quat_format):
    """将四元数数组转换为 wxyz。输入形状 (..., 4)。"""
    q = np.asarray(quat_array)
    if q.shape[-1] != 4:
        raise ValueError(f"四元数最后一维必须是4，当前: {q.shape}")
    if quat_format == "wxyz":
        return q
    if quat_format == "xyzw":
        return q[..., [3, 0, 1, 2]]
    raise ValueError(f"不支持的 quat_format: {quat_format}")


def load_motion_data(motion_file, quat_format="wxyz", fps_override=None):
    """加载运动数据，支持多种格式"""
    ext = os.path.splitext(motion_file)[1].lower()

    def infer_fps_from_name(path, default_fps=100.0):
        name = os.path.basename(path)
        m = re.search(r"(\d+(?:\.\d+)?)\s*fps", name, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
        return float(default_fps)
    
    if ext == '.npz':
        data = np.load(motion_file, allow_pickle=True)
        
        # 尝试不同的数据键名
        if 'qpos' in data:
            qpos = data['qpos']
            print(f"使用 qpos 数据，形状: {qpos.shape}")
        elif 'pos' in data:
            qpos = data['pos']
            print(f"使用 pos 数据，形状: {qpos.shape}")
        elif all(k in data for k in ['fps', 'joint_pos', 'body_pos_w', 'body_quat_w']):
            # bvh_to_robot_npz 格式: fps(1,) joint_pos(T,29) body_pos_w(T,N,3) body_quat_w(T,N,4) wxyz
            # 用第 0 号 body 作为 root，与 joint_pos 拼成 qpos
            print("使用 fps + joint_pos + body_pos_w + body_quat_w 组合")
            joint_pos = np.asarray(data['joint_pos'])
            body_pos_w = np.asarray(data['body_pos_w'])
            body_quat_w = np.asarray(data['body_quat_w'])
            root_pos = body_pos_w[:, 0, :]   # (T, 3)
            root_rot = to_wxyz(body_quat_w[:, 0, :], quat_format)  # (T, 4) -> wxyz
            qpos = np.concatenate([root_pos, root_rot, joint_pos], axis=1)
            print(f"  joint_pos {joint_pos.shape}, body_pos_w {body_pos_w.shape}, -> qpos {qpos.shape}")
        elif all(k in data for k in ['root_pos', 'root_rot', 'dof_pos']):
            print("使用 root_pos + root_rot + dof_pos 组合")
            root_pos = data['root_pos']
            root_rot = to_wxyz(data['root_rot'], quat_format)
            dof_pos = data['dof_pos']
            qpos = np.array([
                np.concatenate([root_pos[i], root_rot[i], dof_pos[i]])
                for i in range(len(root_pos))
            ])
        elif all(k in data for k in ['base_pos_w', 'base_quat_w', 'joint_pos']):
            # label 类型 npz:
            # base_pos_w(T,3), base_quat_w(T,4,wxyz), joint_pos(T,N)
            print("使用 base_pos_w + base_quat_w + joint_pos 组合")
            base_pos_w = np.asarray(data['base_pos_w'])
            base_quat_w = to_wxyz(np.asarray(data['base_quat_w']), quat_format)
            joint_pos = np.asarray(data['joint_pos'])
            qpos = np.concatenate([base_pos_w, base_quat_w, joint_pos], axis=1)
            print(
                f"  base_pos_w {base_pos_w.shape}, "
                f"base_quat_w {base_quat_w.shape}, "
                f"joint_pos {joint_pos.shape} -> qpos {qpos.shape}"
            )
        else:
            raise KeyError("找不到合适的位置数据")
        
        # 帧率：优先 fps(1,)，其次 framerate，否则 frequency / freq
        if 'fps' in data:
            frequency = float(np.asarray(data['fps']).flat[0])
        elif 'framerate' in data:
            frequency = float(np.asarray(data['framerate']).flat[0])
        else:
            frequency = float(data.get('frequency', data.get('freq', 100.0)))
        
    elif ext == '.pkl':
        import pickle
        with open(motion_file, 'rb') as f:
            data = pickle.load(f)
        
        # 处理PKL格式
        if isinstance(data, dict):
            if 'qpos' in data:
                qpos = data['qpos']
                print(f"使用 qpos 数据，形状: {qpos.shape}")
            elif all(k in data for k in ['root_pos', 'root_rot', 'dof_pos']):
                print("使用 root_pos + root_rot + dof_pos 组合")
                root_pos = data['root_pos']
                root_rot = to_wxyz(data['root_rot'], quat_format)
                dof_pos = data['dof_pos']
                qpos = np.array([
                    np.concatenate([root_pos[i], root_rot[i], dof_pos[i]])
                    for i in range(len(root_pos))
                ])
            else:
                raise KeyError("找不到合适的位置数据")
        elif isinstance(data, np.ndarray):
            qpos = data
            print(f"直接使用数组数据，形状: {qpos.shape}")
        else:
            raise ValueError("无法解析PKL文件格式")
        
        # 帧率：优先 fps(1,)，否则 frequency / freq
        if 'fps' in data:
            frequency = float(np.asarray(data['fps']).flat[0])
        else:
            frequency = float(data.get('frequency', data.get('freq', 100.0)))
    
    elif ext == '.npy':
        arr = np.load(motion_file, allow_pickle=True)
        arr = np.asarray(arr)
        # 特定 G1 格式: (T, 58) = 29关节位置 + 29关节速度
        if arr.ndim == 2 and arr.shape[1] == 58:
            print("使用 G1 特定 .npy 格式: [joint_pos(29), joint_vel(29)]")
            joint_pos = arr[:, :29]
            T = joint_pos.shape[0]
            root_pos = np.zeros((T, 3), dtype=joint_pos.dtype)
            root_rot = np.zeros((T, 4), dtype=joint_pos.dtype)
            root_rot[:, 0] = 1.0  # wxyz identity
            qpos = np.concatenate([root_pos, root_rot, joint_pos], axis=1)
            frequency = infer_fps_from_name(motion_file, default_fps=50.0)
            print(f"  joint_pos {joint_pos.shape} -> qpos {qpos.shape}")
            print(f"  fps: {frequency} (文件名推断/默认)")
        else:
            raise ValueError(
                f".npy 格式暂不支持，期望 (T,58)，当前 {arr.shape}"
            )
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    if fps_override is not None:
        frequency = float(fps_override)
        print(f"使用 --fps 覆盖帧率: {frequency}Hz")

    print(f"轨迹: {len(qpos)}帧, {frequency}Hz")
    return qpos, frequency

def main():
    parser = argparse.ArgumentParser(description="播放 GMR 运动文件（npz/pkl）")
    parser.add_argument("robot", help="机器人名（需在 ROBOT_XML_DICT 中）")
    parser.add_argument("motion_file", help="运动文件路径（.npz/.pkl）")
    parser.add_argument(
        "--quat_format",
        choices=["wxyz", "xyzw"],
        default="wxyz",
        help="输入文件里的根四元数格式（默认 wxyz）",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="手动指定播放帧率（Hz），会覆盖文件中的帧率信息",
    )
    parser.add_argument(
        "--box_pos",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="覆盖场景中 box body 的世界 xy（绝对坐标，仅当场景含 box 时生效）",
    )
    parser.add_argument(
        "--box_z",
        type=float,
        default=None,
        help="覆盖 box body 的世界 z（默认沿用 xml 中的值）",
    )
    parser.add_argument(
        "--box_step",
        type=float,
        default=0.05,
        help="实时微调 box 时每次移动的步长（米）",
    )
    args = parser.parse_args()

    robot = args.robot
    motion_file = args.motion_file
    if robot not in ROBOT_XML_DICT:
        print(f"错误: 未知机器人 '{robot}'，请检查 params.py 中 ROBOT_XML_DICT")
        return
    scene_path = str(ROBOT_XML_DICT[robot])
    if not os.path.exists(scene_path):
        print(f"错误: 找不到场景文件 {scene_path}")
        return
    
    print(f"场景: {scene_path}")
    
    try:
        # 加载运动数据
        qpos_seq, frequency = load_motion_data(
            motion_file,
            quat_format=args.quat_format,
            fps_override=args.fps,
        )
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # 可选的固定交互物体（box）。静态 body 没有自由度，直接改 model.body_pos 即可移动。
    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_step = args.box_step
    if box_bid != -1:
        if args.box_pos is not None:
            model.body_pos[box_bid, :2] = args.box_pos  # 覆盖（绝对坐标），非叠加
        if args.box_z is not None:
            model.body_pos[box_bid, 2] = args.box_z
        bp = model.body_pos[box_bid]
        print(f"检测到 box，初始位置: [{bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}]")
        print("  实时微调: I/K = ±x, L/J = ±y, U/O = ±z（步长 --box_step）")

    print(f"模型nq: {model.nq}, 数据nq: {qpos_seq.shape[1]}")
    
    # 调整维度
    if qpos_seq.shape[1] != model.nq:
        if qpos_seq.shape[1] > model.nq:
            print(f"数据维度({qpos_seq.shape[1]}) > 模型维度({model.nq})，进行裁剪")
            qpos_seq = qpos_seq[:, :model.nq]
        else:
            print(f"数据维度({qpos_seq.shape[1]}) < 模型维度({model.nq})，进行填充")
            padding = np.zeros((qpos_seq.shape[0], model.nq - qpos_seq.shape[1]))
            qpos_seq = np.concatenate([qpos_seq, padding], axis=1)
    
    # 调整高度
    # if qpos_seq[0, 2] < 0.3:
    #     height_diff = 0.95 - qpos_seq[0, 2]
    #     print(f"调整高度: +{height_diff:.3f}")
    #     qpos_seq[:, 2] += height_diff
    
    # 播放控制变量
    paused = True  # 初始为暂停状态
    idx = 0
    base_frame_time = 1.0 / frequency
    speed_scale = 1.0
    frame_time = base_frame_time / speed_scale
    # 与 idx 同步的“上一帧逻辑时间”，暂停时不应累积，否则恢复时会一次性追帧、tqdm 的 it/s 会飙高
    last_frame_time = time.time()
    
    # Shift键状态
    shift_pressed = False
    
    def key_callback(keycode):
        """按键回调函数"""
        nonlocal paused, idx, speed_scale, frame_time, shift_pressed, last_frame_time
        
        # 检测Shift键
        if keycode in [340, 344]:  # 左右Shift键码
            shift_pressed = True
            return
        
        # 空格键：暂停/继续
        if keycode == 32:  # 空格
            was_paused = paused
            paused = not paused
            if was_paused and not paused:
                last_frame_time = time.time()
            print(f"\n{'暂停' if paused else '播放'}")
        
        # +/- 键：调整速度
        elif keycode == 61:  # + (实际是=键)
            speed_scale = min(10.0, speed_scale * 1.5)
            frame_time = base_frame_time / speed_scale
            print(f"\n速度: {speed_scale:.1f}x")
        elif keycode == 45:  # - (减速)
            speed_scale = max(0.1, speed_scale / 1.5)
            frame_time = base_frame_time / speed_scale
            print(f"\n速度: {speed_scale:.1f}x")
        
        # 上下方向键：单帧跳转
        elif keycode == 265:  # 上箭头
            if idx < len(qpos_seq) - 1:
                idx += 1
                print(f"\n下一帧: {idx}")
        elif keycode == 264:  # 下箭头
            if idx > 0:
                idx -= 1
                print(f"\n上一帧: {idx}")
        
        # 左右方向键：百分比跳转
        elif keycode == 262:  # 右箭头
            jump_percent = 5 if shift_pressed else 1
            jump_frames = max(1, int(len(qpos_seq) * jump_percent / 100))
            idx = min(len(qpos_seq) - 1, idx + jump_frames)
            print(f"\n前进{jump_percent}%: {idx}帧 ({jump_frames}帧)")
            
        elif keycode == 263:  # 左箭头
            jump_percent = 5 if shift_pressed else 1
            jump_frames = max(1, int(len(qpos_seq) * jump_percent / 100))
            idx = max(0, idx - jump_frames)
            print(f"\n后退{jump_percent}%: {idx}帧 ({jump_frames}帧)")
        
        # R键：重置
        elif keycode == 114:  # R
            idx = 0
            print("\n重置到第一帧")

        # I/K/J/L/U/O：实时微调 box 位置（仅当场景含 box）
        elif box_bid != -1 and keycode in (73, 75, 74, 76, 85, 79):
            if keycode == 73:    # I: +x
                model.body_pos[box_bid, 0] += box_step
            elif keycode == 75:  # K: -x
                model.body_pos[box_bid, 0] -= box_step
            elif keycode == 76:  # L: +y
                model.body_pos[box_bid, 1] += box_step
            elif keycode == 74:  # J: -y
                model.body_pos[box_bid, 1] -= box_step
            elif keycode == 85:  # U: +z
                model.body_pos[box_bid, 2] += box_step
            elif keycode == 79:  # O: -z
                model.body_pos[box_bid, 2] -= box_step
            bp = model.body_pos[box_bid]
            print(f"\n箱子位置(绝对): [{bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}]")

        # 重置Shift状态（除非是Shift键本身）
        shift_pressed = False
    
    def key_release_callback(keycode):
        """按键释放回调"""
        nonlocal shift_pressed
        if keycode in [340, 344]:  # 左右Shift键码
            shift_pressed = False
    
    print("\n控制:")
    print("  空格: 暂停/继续")
    print("  +/-: 调整播放速度 (0.1x - 10x)")
    print("  上下箭头: 单帧跳转")
    print("  左右箭头: 跳转1% (先按Shift再按左右: 跳转5%)")
    print("  R: 重置到第一帧")
    if box_bid != -1:
        print("  I/K: 箱子 ±x   L/J: 箱子 ±y   U/O: 箱子 ±z")
    print("\n状态: 按空格开始播放")
    
    # 创建可视化
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        
        last_frame_time = time.time()
        pbar = tqdm(total=len(qpos_seq), 
                   desc="播放进度",
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                   postfix="速度:1.0x")
        
        while viewer.is_running and idx < len(qpos_seq):
            current_time = time.time()
            
            # 正常播放（当未暂停时）：支持追帧，尽量贴近目标频率
            if not paused:
                elapsed = current_time - last_frame_time
                if elapsed >= frame_time:
                    step_frames = int(elapsed / frame_time)
                    idx += max(1, step_frames)
                    last_frame_time += step_frames * frame_time
            
            # 确保idx在有效范围内
            idx = max(0, min(len(qpos_seq) - 1, idx))
            
            # 更新姿态
            data.qpos[:] = qpos_seq[idx]
            # data.qpos[2] += 0.04
            mujoco.mj_forward(model, data)
            
            # 更新进度条
            if idx >= pbar.n:
                pbar.update(idx - pbar.n)
            elif idx < pbar.n:
                # 如果后退了，需要手动设置进度
                pbar.n = idx
                pbar.refresh()
            
            # 更新进度条后缀显示速度
            if pbar.postfix != f"速度:{speed_scale:.1f}x":
                pbar.set_postfix_str(f"速度:{speed_scale:.1f}x")
            
            viewer.sync()
            
            # 避免CPU占用过高；不再固定 sleep 到一帧时长，否则高频会被额外限速
            time.sleep(0.001)
        
        pbar.close()
        print(f"\n播放完成！总帧数: {len(qpos_seq)}")

if __name__ == '__main__':
    main()