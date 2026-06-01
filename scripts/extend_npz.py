#!/usr/bin/env python3
"""
输入: fps, root_pos, root_rot, dof_pos, dof_vel
输出: 包含速度、加速度等扩展数据的 npz 文件
"""

import argparse
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_velocities(pos, rot, fps):
    """计算线速度和角速度"""
    dt = 1.0 / fps
    n = pos.shape[0]
    
    # 线速度：一阶差分
    linvel = np.diff(pos, axis=0) / dt
    
    # 角速度：四元数差分
    angvel = np.zeros((n - 1, 3))
    for i in range(n - 1):
        r0 = R.from_quat(rot[i])
        r1 = R.from_quat(rot[i + 1])
        rotvec = (r1 * r0.inv()).as_rotvec()
        angvel[i] = rotvec / dt
    
    return linvel, angvel


def calculate_accelerations(data, fps):
    """计算加速度（二阶差分）"""
    dt = 1.0 / fps
    acc = np.diff(data, n=2, axis=0) / (dt ** 2)
    # 补齐长度以匹配原始数据
    return np.concatenate([acc[[0]] * 2, acc, acc[[-1]] * 2], axis=0)[:len(data)]


def extend_trajectory(qpos, qvel, fps):
    """扩展轨迹数据：计算加速度、速度、位置等"""
    n_frames = qpos.shape[0]
    
    # 1. 计算加速度（二阶差分）
    qacc = calculate_accelerations(qvel, fps)
    
    # 2. 提取各部分
    root_pos = qpos[:, :3]
    root_rot = qpos[:, 3:7]
    dof_pos = qpos[:, 7:]
    
    root_linvel = qvel[:, :3]
    root_angvel = qvel[:, 3:6]
    dof_vel = qvel[:, 6:]
    
    return {
        'qpos': qpos,
        'qvel': qvel,
        'qacc': qacc,
        'root_pos': root_pos,
        'root_rot': root_rot,
        'dof_pos': dof_pos,
        'root_linvel': root_linvel,
        'root_angvel': root_angvel,
        'dof_vel': dof_vel,
        'fps': fps,
        'n_frames': n_frames,
        'duration': n_frames / fps,
    }


def load_data(input_file):
    """加载数据文件，支持 .npz 和 .pkl 格式"""
    if input_file.endswith('.npz'):
        data = np.load(input_file)
    elif input_file.endswith('.pkl'):
        import pickle
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {input_file}，只支持 .npz 或 .pkl")
    
    # 提取数据
    fps = int(data['fps']) if isinstance(data['fps'], np.ndarray) else int(data['fps'])
    root_pos = np.array(data['root_pos'])
    root_rot = np.array(data['root_rot'])
    dof_pos = np.array(data['dof_pos'])
    dof_vel = np.array(data['dof_vel'])
    
    return fps, root_pos, root_rot, dof_pos, dof_vel


def convert_single_file(input_file, output_file, output_fps=None, verbose=True):
    """转换单个文件"""
    
    if verbose:
        print(f"处理: {os.path.basename(input_file)}")
    
    # 1. 加载数据
    fps, root_pos, root_rot, dof_pos, dof_vel = load_data(input_file)
    
    if verbose:
        print(f"  FPS: {fps}, 帧数: {len(root_pos)}, DOF: {dof_pos.shape[1]}")
    
    # 2. 构建 qpos 和 qvel
    qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
    
    # 计算根部速度
    root_linvel, root_angvel = calculate_velocities(root_pos, root_rot, fps)
    
    # 确保长度一致（速度会少一帧）
    n = min(len(qpos) - 1, len(root_linvel), len(dof_vel))
    qpos_trimmed = qpos[:n]
    qvel = np.concatenate([root_linvel[:n], root_angvel[:n], dof_vel[:n]], axis=1)
    
    if verbose:
        print(f"  构建: qpos {qpos_trimmed.shape}, qvel {qvel.shape}")
    
    # 3. 插值到目标频率（如果需要）
    target_fps = output_fps if output_fps else fps
    if target_fps != fps:
        if verbose:
            print(f"  插值: {fps}Hz -> {target_fps}Hz")
        from scipy import interpolate
        
        old_t = np.arange(len(qpos_trimmed)) / fps
        new_t = np.arange(0, old_t[-1], 1/target_fps)
        
        qpos_interp = interpolate.interp1d(old_t, qpos_trimmed, axis=0)(new_t)
        qvel_interp = interpolate.interp1d(old_t, qvel, axis=0)(new_t)
        
        qpos_trimmed, qvel = qpos_interp, qvel_interp
        fps = target_fps
    
    # 4. 扩展数据
    extended = extend_trajectory(qpos_trimmed, qvel, fps)
    
    # 5. 保存
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    np.savez_compressed(output_file, **extended)
    
    if verbose:
        print(f"  保存: {os.path.basename(output_file)}")
        print(f"  完成! 帧数: {extended['n_frames']}, 时长: {extended['duration']:.2f}s\n")
    
    return extended


def batch_convert(input_dir, output_dir, pattern="*.npz", output_fps=None, recursive=False):
    """批量转换目录中的所有文件"""
    
    # 查找所有匹配的文件
    if recursive:
        # 递归查找
        files = []
        for ext in ['npz', 'pkl']:
            files.extend(glob.glob(os.path.join(input_dir, f"**/*.{ext}"), recursive=True))
    else:
        # 非递归查找
        files = []
        if 'npz' in pattern:
            files.extend(glob.glob(os.path.join(input_dir, "*.npz")))
        if 'pkl' in pattern:
            files.extend(glob.glob(os.path.join(input_dir, "*.pkl")))
    
    if not files:
        print(f"错误: 在 {input_dir} 中未找到匹配的文件")
        return
    
    files = sorted(files)
    
    print("=" * 80)
    print(f"批量转换模式")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(files)} 个文件")
    if output_fps:
        print(f"目标帧率: {output_fps} Hz")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    success = 0
    failed = []
    
    for i, input_file in enumerate(files, 1):
        # 生成输出文件名
        rel_path = os.path.relpath(input_file, input_dir)
        base_name = os.path.splitext(rel_path)[0]
        output_file = os.path.join(output_dir, f"{base_name}_extended.npz")
        
        # 确保输出子目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"[{i}/{len(files)}] ", end="")
        try:
            convert_single_file(input_file, output_file, output_fps, verbose=True)
            success += 1
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            failed.append((input_file, str(e)))
    
    # 总结
    print("=" * 80)
    print(f"批量转换完成: 成功 {success}/{len(files)}")
    if failed:
        print(f"\n失败的文件:")
        for f, err in failed:
            print(f"  - {os.path.basename(f)}: {err}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='最简版运动数据转换和扩展（支持批量）')
    
    # 输入输出
    parser.add_argument('-i', '--input', required=True, help='输入文件或目录')
    parser.add_argument('-o', '--output', help='输出文件或目录')
    parser.add_argument('-of', '--output-fps', type=float, help='输出帧率 (默认: 使用输入帧率)')
    
    # 批量模式
    parser.add_argument('-b', '--batch', action='store_true', help='批量处理模式')
    parser.add_argument('-r', '--recursive', action='store_true', help='递归处理子目录')
    parser.add_argument('-p', '--pattern', default='*.npz', help='文件匹配模式 (默认: *.npz)')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量模式
        input_dir = args.input
        output_dir = args.output if args.output else os.path.join(input_dir, "extended")
        batch_convert(input_dir, output_dir, args.pattern, args.output_fps, args.recursive)
    else:
        # 单文件模式
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_extended.npz"
        convert_single_file(args.input, args.output, args.output_fps)


if __name__ == "__main__":
    main()