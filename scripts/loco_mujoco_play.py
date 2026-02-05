#!/usr/bin/env python3
import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os

def load_trajectory(motion_file):
    """加载轨迹数据"""
    ext = os.path.splitext(motion_file)[1].lower()
    
    if ext == '.npz':
        # 加载NPZ文件
        data = np.load(motion_file, allow_pickle=True)
        
        if 'qpos' in data:
            qpos = data['qpos']
        elif 'pos' in data:
            qpos = data['pos']
        else:
            raise KeyError("找不到位置数据")
        
        frequency = float(data.get('frequency', data.get('freq', 100.0)))
        
    elif ext == '.pkl':
        # 加载PKL文件
        import pickle
        with open(motion_file, 'rb') as f:
            data = pickle.load(f)
        
        # 统一格式：从PKL数据中提取qpos
        if isinstance(data, dict):
            # 如果是字典格式，按你的格式提取
            qpos = np.array([
                np.concatenate([data['root_pos'][i], data['root_rot'][i], data['dof_pos'][i]])
                for i in range(len(data['root_pos']))
            ])
        elif isinstance(data, np.ndarray):
            # 如果直接是numpy数组
            qpos = data
        else:
            # 尝试其他格式
            try:
                qpos = np.array(data)
            except:
                raise ValueError("无法解析PKL文件格式")
        
        frequency = 100.0
    
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    print(f"轨迹: {len(qpos)}帧, {frequency}Hz")
    return qpos, frequency

def main():
    if len(sys.argv) < 3:
        print("用法: python play.py <机器人> <运动文件>")
        return
    
    robot, motion_file = sys.argv[1], sys.argv[2]
    
    # 查找场景文件
    scene_path = None
    for path in [f"../assets/{robot}/scene.xml", f"assets/{robot}/scene.xml"]:
        if os.path.exists(path):
            scene_path = path
            break
    
    if not scene_path:
        print(f"找不到场景文件")
        return
    
    print(f"场景: {scene_path}")
    
    # 加载数据（现在统一使用load_trajectory）
    qpos_seq, frequency = load_trajectory(motion_file)
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    # 调整维度
    if qpos_seq.shape[1] != model.nq:
        if qpos_seq.shape[1] > model.nq:
            qpos_seq = qpos_seq[:, :model.nq]
        else:
            padding = np.zeros((qpos_seq.shape[0], model.nq - qpos_seq.shape[1]))
            qpos_seq = np.concatenate([qpos_seq, padding], axis=1)
    
    # 调整高度
    if qpos_seq[0, 2] < 0.3:
        height_diff = 0.95 - qpos_seq[0, 2]
        qpos_seq[:, 2] += height_diff

    qpos_seq[:, 26] *= 2
    qpos_seq[:, 27] *= 2
    qpos_seq[:, 28] *= 2
    qpos_seq[:, 33] *= 2
    qpos_seq[:, 34] *= 2
    qpos_seq[:, 35] *= 2
    
    print(f"模型nq: {model.nq}, 数据nq: {qpos_seq.shape[1]}")
    
    # 播放控制变量
    paused = True  # 初始为暂停状态
    idx = 0
    base_frame_time = 1.0 / frequency  # 基准帧时间
    speed_scale = 1.0  # 速度倍数
    frame_time = base_frame_time / speed_scale  # 实际帧时间
    
    # Shift键状态（简化处理，通过特殊键码检测）
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
            print(f" {'暂停' if paused else '播放'}")
        
        # +/- 键：调整速度
        elif keycode == 61:  # + (实际是=键，但通常用于加速)
            speed_scale = min(10.0, speed_scale * 1.5)
            frame_time = base_frame_time / speed_scale
            print(f"速度: {speed_scale:.1f}x")
        elif keycode == 45:  # - (减速)
            speed_scale = max(0.1, speed_scale / 1.5)
            frame_time = base_frame_time / speed_scale
            print(f"速度: {speed_scale:.1f}x")
        
        # 上下方向键：单帧跳转
        elif keycode == 265:  # 上箭头
            if idx < len(qpos_seq) - 1:
                idx += 1
                print(f"下一帧: {idx}")
        elif keycode == 264:  # 下箭头
            if idx > 0:
                idx -= 1
                print(f"上一帧: {idx}")
        
        # 左右方向键：百分比跳转
        elif keycode == 262:  # 右箭头
            jump_percent = 5 if shift_pressed else 1
            jump_frames = max(1, int(len(qpos_seq) * jump_percent / 100))
            idx = min(len(qpos_seq) - 1, idx + jump_frames)
            print(f"前进{jump_percent}%: {idx}帧 ({jump_frames}帧)")
            
        elif keycode == 263:  # 左箭头
            jump_percent = 5 if shift_pressed else 1
            jump_frames = max(1, int(len(qpos_seq) * jump_percent / 100))
            idx = max(0, idx - jump_frames)
            print(f"后退{jump_percent}%: {idx}帧 ({jump_frames}帧)")
        
        # R键：重置
        elif keycode == 114:  # R
            idx = 0
            print("重置到第一帧")
        
        # 重置Shift状态（除非是Shift键本身）
        shift_pressed = False
    
    def key_release_callback(keycode):
        """按键释放回调（如果Mujoco支持的话）"""
        nonlocal shift_pressed
        if keycode in [340, 344]:  # 左右Shift键码
            shift_pressed = False
    
    # 创建可视化
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        
        print("\n控制:")
        print("  空格: 暂停/继续")
        print("  +/-: 调整播放速度 (0.1x - 10x)")
        print("  上下箭头: 单帧跳转")
        print("  左右箭头: 跳转1% (先按Shift再按左右: 跳转5%)")
        print("  R: 重置到第一帧")
        print("\n状态: 按空格开始播放")
        print("提示: 要跳转5%，请先按Shift键，然后按左右箭头")
        
        last_frame_time = time.time()
        
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
            
            # 显示进度信息（包含速度倍数）
            percentage = idx / len(qpos_seq) * 100
            print(f"帧: {idx}/{len(qpos_seq)} ({percentage:.1f}%) | 速度: {speed_scale:.1f}x | {'播放中' if not paused else '暂停'}     ", end='\r')
            
            viewer.sync()
            
            # 避免CPU占用过高
            time.sleep(0.02)
        
        print(f"\n播放完成")

if __name__ == '__main__':
    main()