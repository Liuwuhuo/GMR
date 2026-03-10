#!/usr/bin/env python3
import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os
from tqdm import tqdm

def load_motion_data(motion_file):
    """加载运动数据，支持多种格式"""
    ext = os.path.splitext(motion_file)[1].lower()
    
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
            root_rot = body_quat_w[:, 0, :]  # (T, 4) 已是 wxyz
            qpos = np.concatenate([root_pos, root_rot, joint_pos], axis=1)
            print(f"  joint_pos {joint_pos.shape}, body_pos_w {body_pos_w.shape}, -> qpos {qpos.shape}")
        elif all(k in data for k in ['root_pos', 'root_rot', 'dof_pos']):
            print("使用 root_pos + root_rot + dof_pos 组合")
            root_pos = data['root_pos']
            root_rot = data['root_rot']
            dof_pos = data['dof_pos']
            qpos = np.array([
                np.concatenate([root_pos[i], root_rot[i], dof_pos[i]])
                for i in range(len(root_pos))
            ])
        else:
            raise KeyError("找不到合适的位置数据")
        
        # 帧率：优先 fps(1,)，否则 frequency / freq
        if 'fps' in data:
            frequency = float(np.asarray(data['fps']).flat[0])
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
                root_rot = data['root_rot']
                root_rot[:, [0, 1, 2, 3]] = root_rot[:, [3, 0, 1, 2]]
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
    
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    print(f"轨迹: {len(qpos)}帧, {frequency}Hz")
    return qpos, frequency

def main():
    if len(sys.argv) < 3:
        print("用法: python play.py <机器人名> <运动文件>")
        return
    
    robot, motion_file = sys.argv[1], sys.argv[2]
    scene_path = f"../assets/{robot}/scene.xml"
    
    if not os.path.exists(scene_path):
        # 尝试其他路径
        alt_path = f"assets/{robot}/scene.xml"
        if os.path.exists(alt_path):
            scene_path = alt_path
        else:
            print(f"错误: 找不到场景文件 {robot}/scene.xml")
            return
    
    print(f"场景: {scene_path}")
    
    try:
        # 加载运动数据
        qpos_seq, frequency = load_motion_data(motion_file)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
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
    if qpos_seq[0, 2] < 0.3:
        height_diff = 0.95 - qpos_seq[0, 2]
        print(f"调整高度: +{height_diff:.3f}")
        qpos_seq[:, 2] += height_diff
    
    # 播放控制变量
    paused = True  # 初始为暂停状态
    idx = 0
    base_frame_time = 1.0 / frequency
    speed_scale = 1.0
    frame_time = base_frame_time / speed_scale
    
    # Shift键状态
    shift_pressed = False
    
    def key_callback(keycode):
        """按键回调函数"""
        nonlocal paused, idx, speed_scale, frame_time, shift_pressed
        
        # 检测Shift键
        if keycode in [340, 344]:  # 左右Shift键码
            shift_pressed = True
            return
        
        # 空格键：暂停/继续
        if keycode == 32:  # 空格
            paused = not paused
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
            
            # 正常播放（当未暂停时）
            if not paused and (current_time - last_frame_time >= frame_time):
                last_frame_time = current_time
                idx += 1
            
            # 确保idx在有效范围内
            idx = max(0, min(len(qpos_seq) - 1, idx))
            
            # 更新姿态
            data.qpos[:] = qpos_seq[idx]
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
            
            # 避免CPU占用过高
            time.sleep(base_frame_time)
        
        pbar.close()
        print(f"\n播放完成！总帧数: {len(qpos_seq)}")

if __name__ == '__main__':
    main()