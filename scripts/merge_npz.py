#!/usr/bin/env python3
"""
NPZ文件合并工具 - 修复版
支持合并两个NPZ文件，包含所有与帧相关的数据
"""
import numpy as np
import argparse
import os
import sys
import tempfile
import shutil
from scipy.interpolate import interp1d

def parse_frame_spec(spec_str, total_frames):
    """解析帧范围规范"""
    if spec_str == '' or spec_str == ':':
        return 0, total_frames
    
    if '-' not in spec_str:
        # 单个帧
        frame = int(spec_str)
        if frame < 0:
            frame = total_frames + frame
        return frame, frame + 1
    
    parts = spec_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"无效的帧范围格式: {spec_str}")
    
    start_str, end_str = parts
    
    # 解析起始帧
    if start_str == '':
        start = 0
    else:
        start = int(start_str)
        if start < 0:
            start = total_frames + start
    
    # 解析结束帧
    if end_str == '':
        end = total_frames
    else:
        end = int(end_str)
        if end < 0:
            end = total_frames + end
    
    return start, end

def extract_npz_to_temp(npz_path, frame_spec):
    """提取NPZ文件到临时目录并切片 - 修复版"""
    print(f"处理文件: {npz_path}")
    
    # 加载NPZ文件（允许pickle）
    npz_data = np.load(npz_path, allow_pickle=True)
    keys = list(npz_data.keys())
    print(f"  包含{len(keys)}个数据项")
    
    # 查找帧数（基于qpos或pos）
    total_frames = None
    frame_based_keys_to_check = ['qpos', 'pos', 'poses', 'body_pos', 'xpos', 'qvel']
    
    for key in frame_based_keys_to_check:
        if key in npz_data:
            arr = npz_data[key]
            if hasattr(arr, 'shape') and len(arr.shape) >= 1:
                total_frames = arr.shape[0]
                print(f"  使用 {key} 确定总帧数: {total_frames}")
                break
    
    if total_frames is None:
        # 如果没有找到标准键，找第一个2维数组
        for key in keys:
            arr = npz_data[key]
            if hasattr(arr, 'shape') and len(arr.shape) == 2:
                total_frames = arr.shape[0]
                print(f"  使用 {key} 确定总帧数: {total_frames}")
                break
    
    if total_frames is None:
        raise ValueError(f"无法确定文件 {npz_path} 的总帧数")
    
    # 解析帧范围
    start_frame, end_frame = parse_frame_spec(frame_spec, total_frames)
    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    frame_count = end_frame - start_frame
    
    print(f"  切片范围: [{start_frame}:{end_frame}] (共{frame_count}帧)")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='npz_merge_')
    
    # 处理每个数据项
    frame_based_keys = []
    static_keys = []
    
    # 常见帧相关数据键（根据你提供的文件列表）
    frame_based_patterns = [
        'qpos', 'pos', 'poses',
        'qvel', 'cvel',
        'xpos', 'xquat',
        'body_pos', 'body_quat',
        'site_pos', 'site_quat', 'site_xpos', 'site_xmat',
        'subtree_com'
    ]
    
    static_patterns = [
        '_names',  # 所有名称数据
        'type',    # 类型数据
        'id',      # ID数据
        'points',  # 点数据
        'frequency',
        '_bodyid'
    ]
    
    for key in keys:
        arr = npz_data[key]
        
        # 跳过空数据
        if arr is None:
            continue
            
        if not hasattr(arr, 'shape'):
            # 非数组数据（标量、字符串等）
            np.save(os.path.join(temp_dir, f"{key}.npy"), arr, allow_pickle=True)
            static_keys.append(key)
            
        elif len(arr.shape) == 0:
            # 0维数组
            np.save(os.path.join(temp_dir, f"{key}.npy"), arr, allow_pickle=True)
            static_keys.append(key)
            
        else:
            # 判断是否是帧相关数据
            is_frame_based = False
            
            # 检查是否是明确的静态数据（通过键名模式）
            is_static_by_name = False
            for pattern in static_patterns:
                if pattern in key:
                    is_static_by_name = True
                    break
            
            if is_static_by_name:
                # 确定为静态数据
                is_frame_based = False
            else:
                # 方法1：键名匹配
                for pattern in frame_based_patterns:
                    if pattern in key:
                        is_frame_based = True
                        break
                
                # 方法2：检查第一维是否匹配总帧数
                if not is_frame_based and arr.shape[0] == total_frames:
                    is_frame_based = True
                
                # 方法3：检查形状是否是典型的帧数据形状
                if not is_frame_based:
                    # 3维数组通常是帧数据（如xpos: [frames, bodies, 3]）
                    if len(arr.shape) == 3 and arr.shape[0] > 100:  # 增加阈值判断
                        is_frame_based = True
                    # 2维数组且第一维较大、第二维较小（不是静态参数）
                    elif len(arr.shape) == 2 and arr.shape[0] > 100 and arr.shape[1] < 100:
                        is_frame_based = True
            
            if is_frame_based:
                # 确保第一维足够大，避免负索引问题
                if arr.shape[0] <= start_frame:
                    print(f"  警告: {key} 形状 {arr.shape} 小于起始帧 {start_frame}，保存为静态数据")
                    np.save(os.path.join(temp_dir, f"{key}.npy"), arr, allow_pickle=True)
                    static_keys.append(key)
                    continue
                    
                try:
                    # 帧相关数据：切片
                    end_frame_adj = min(end_frame, arr.shape[0])
                    if end_frame_adj <= start_frame:
                        print(f"  警告: {key} 有效切片范围为空，保存为静态数据")
                        np.save(os.path.join(temp_dir, f"{key}.npy"), arr, allow_pickle=True)
                        static_keys.append(key)
                        continue
                        
                    if len(arr.shape) == 1:
                        sliced = arr[start_frame:end_frame_adj]
                    elif len(arr.shape) == 2:
                        sliced = arr[start_frame:end_frame_adj, :]
                    elif len(arr.shape) == 3:
                        sliced = arr[start_frame:end_frame_adj, :, :]
                    else:
                        # 更高维度，只切片第一维
                        slices = [slice(None)] * len(arr.shape)
                        slices[0] = slice(start_frame, end_frame_adj)
                        sliced = arr[tuple(slices)]
                    
                    # 检查切片是否有效
                    if sliced.shape[0] > 0:
                        np.save(os.path.join(temp_dir, f"{key}.npy"), sliced, allow_pickle=True)
                        frame_based_keys.append(key)
                        print(f"  切片 {len(arr.shape)}D: {key} [{arr.shape} -> {sliced.shape}]")
                    else:
                        print(f"  警告: {key} 切片后为空 [{arr.shape} -> {sliced.shape}]，保存为静态数据")
                        np.save(os.path.join(temp_dir, f"{key}.npy"), arr, allow_pickle=True)
                        static_keys.append(key)
                        
                except Exception as e:
                    print(f"  警告: 切片 {key} 失败: {e}，保存为静态数据")
                    np.save(os.path.join(temp_dir, f"{key}.npy"), arr, allow_pickle=True)
                    static_keys.append(key)
                    
            else:
                # 静态数据：完整保存
                np.save(os.path.join(temp_dir, f"{key}.npy"), arr, allow_pickle=True)
                static_keys.append(key)
                print(f"  保存静态 {len(arr.shape)}D: {key} ({arr.shape})")
    
    print(f"  帧相关数据: {len(frame_based_keys)}个")
    print(f"  静态数据: {len(static_keys)}个")
    
    # 验证qpos是否成功切片
    qpos_file = os.path.join(temp_dir, "qpos.npy")
    if os.path.exists(qpos_file):
        qpos_sliced = np.load(qpos_file, allow_pickle=True)
        print(f"  验证: qpos切片后形状 = {qpos_sliced.shape}")
    
    return temp_dir, frame_based_keys, static_keys, frame_count

def safe_load_npy(file_path):
    """安全加载npy文件"""
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"  警告: 无法加载 {file_path}: {e}")
        return None

def merge_arrays(arr1, arr2, interpolate_frames_count, key):
    """合并两个数组，支持插值"""
    if arr1 is None or arr2 is None:
        print(f"    {key}: 数组为空，跳过")
        return None
    
    # 检查数组是否为空
    if arr1.shape[0] == 0 or arr2.shape[0] == 0:
        print(f"    {key}: 数组有空的维度，跳过")
        return None
    
    # 检查形状兼容性
    if len(arr1.shape) != len(arr2.shape):
        print(f"    {key}: 维度不匹配 {len(arr1.shape)}D != {len(arr2.shape)}D")
        return None
    
    print(f"    {key}: 合并 {arr1.shape} + {arr2.shape}", end='')
    
    # 检查除第一维外的其他维度
    shape_mismatch = False
    for i in range(1, min(len(arr1.shape), len(arr2.shape))):
        if arr1.shape[i] != arr2.shape[i]:
            print(f"\n      警告: 第{i+1}维不匹配 {arr1.shape[i]} != {arr2.shape[i]}")
            shape_mismatch = True
    
    # 如果有形状不匹配，尝试修复
    if shape_mismatch:
        if len(arr1.shape) == 2 and len(arr2.shape) == 2:
            min_dim = min(arr1.shape[1], arr2.shape[1])
            arr1 = arr1[:, :min_dim]
            arr2 = arr2[:, :min_dim]
            print(f"\n      调整为 {arr1.shape}")
        elif len(arr1.shape) == 3 and len(arr2.shape) == 3:
            min_dim1 = min(arr1.shape[1], arr2.shape[1])
            min_dim2 = min(arr1.shape[2], arr2.shape[2])
            arr1 = arr1[:, :min_dim1, :min_dim2]
            arr2 = arr2[:, :min_dim1, :min_dim2]
            print(f"\n      调整为 {arr1.shape}")
        else:
            print(f"\n      无法调整形状，跳过")
            return None
    
    # 插值处理
    if interpolate_frames_count > 0:
        try:
            if len(arr1.shape) == 1:
                # 1维数组
                t = np.linspace(0, 1, interpolate_frames_count + 2)[1:-1]
                interp_func = interp1d([0, 1], [arr1[-1], arr2[0]], kind='linear')
                interpolated = interp_func(t)
                merged = np.concatenate([arr1, interpolated, arr2])
                print(f" + {interpolated.shape[0]}插值帧")
                
            elif len(arr1.shape) == 2:
                # 2维数组
                t = np.linspace(0, 1, interpolate_frames_count + 2)[1:-1]
                interpolated = np.zeros((interpolate_frames_count, arr1.shape[1]), dtype=arr1.dtype)
                
                for i in range(arr1.shape[1]):
                    interp_func = interp1d([0, 1], [arr1[-1, i], arr2[0, i]], kind='linear')
                    interpolated[:, i] = interp_func(t)
                
                merged = np.vstack([arr1, interpolated, arr2])
                print(f" + {interpolated.shape[0]}插值帧")
                
            elif len(arr1.shape) == 3:
                # 3维数组
                interpolated = np.zeros((interpolate_frames_count,) + arr1.shape[1:], dtype=arr1.dtype)
                
                for i in range(interpolate_frames_count):
                    alpha = (i + 1) / (interpolate_frames_count + 1)
                    interpolated[i] = (1 - alpha) * arr1[-1] + alpha * arr2[0]
                
                merged = np.concatenate([arr1, interpolated, arr2], axis=0)
                print(f" + {interpolated.shape[0]}插值帧")
                
            else:
                # 高维数组 - 直接拼接
                merged = np.concatenate([arr1, arr2], axis=0)
                print(f" (高维，跳过插值)")
                
        except Exception as e:
            print(f"\n      插值失败: {e}，直接拼接")
            merged = np.concatenate([arr1, arr2], axis=0)
    else:
        # 直接拼接
        merged = np.concatenate([arr1, arr2], axis=0)
        print(" (直接拼接)")
    
    print(f"      -> {merged.shape}")
    return merged

def main():
    parser = argparse.ArgumentParser(description='合并两个NPZ运动文件（修复版）')
    parser.add_argument('--file1', required=True, 
                       help='第一个文件，格式: 路径[:帧范围]，如: file.npz:100-200 或 file.npz:-50')
    parser.add_argument('--file2', required=True,
                       help='第二个文件，格式: 路径[:帧范围]')
    parser.add_argument('--output', required=True,
                       help='输出文件路径')
    parser.add_argument('--interpolate', type=int, default=0,
                       help='中间插值帧数 (默认: 0)')
    parser.add_argument('--frequency', type=float, default=None,
                       help='输出文件的频率 (默认: 使用第一个文件的频率)')
    parser.add_argument('--compress', action='store_true',
                       help='使用压缩存储 (减小文件大小)')
    parser.add_argument('--debug', action='store_true',
                       help='显示调试信息')
    
    args = parser.parse_args()
    
    # 解析文件路径和帧范围
    file1_parts = args.file1.split(':')
    file1_path = file1_parts[0]
    file1_spec = file1_parts[1] if len(file1_parts) > 1 else ''
    
    file2_parts = args.file2.split(':')
    file2_path = file2_parts[0]
    file2_spec = file2_parts[1] if len(file2_parts) > 1 else ''
    
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        print(f"错误: 文件不存在 - {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"错误: 文件不存在 - {file2_path}")
        return
    
    print("=" * 70)
    print("NPZ文件合并工具（修复版）")
    print("=" * 70)
    
    temp_dirs = []
    try:
        # 步骤1: 提取并切片两个NPZ文件
        print("\n[步骤1] 提取并切片文件1...")
        temp_dir1, frame_keys1, static_keys1, frames1 = extract_npz_to_temp(file1_path, file1_spec)
        temp_dirs.append(temp_dir1)
        
        print("\n[步骤2] 提取并切片文件2...")
        temp_dir2, frame_keys2, static_keys2, frames2 = extract_npz_to_temp(file2_path, file2_spec)
        temp_dirs.append(temp_dir2)
        
        # 获取所有帧相关键的交集
        all_frame_keys = list(set(frame_keys1) & set(frame_keys2))
        print(f"\n两个文件共有的帧相关数据键 ({len(all_frame_keys)}个):")
        for key in sorted(all_frame_keys):
            print(f"  - {key}")
        
        # 获取所有静态键的并集
        all_static_keys = list(set(static_keys1) | set(static_keys2))
        print(f"\n静态数据键 ({len(all_static_keys)}个):")
        for key in sorted(all_static_keys):
            print(f"  - {key}")
        
        # 步骤2: 合并帧相关数据
        print(f"\n[步骤3] 合并帧相关数据 (插值: {args.interpolate}帧)...")
        merged_data = {}
        
        # 先合并qpos，作为基准
        qpos1_path = os.path.join(temp_dir1, "qpos.npy")
        qpos2_path = os.path.join(temp_dir2, "qpos.npy")
        
        qpos1 = safe_load_npy(qpos1_path) if os.path.exists(qpos1_path) else None
        qpos2 = safe_load_npy(qpos2_path) if os.path.exists(qpos2_path) else None
        
        if qpos1 is not None and qpos2 is not None:
            merged_qpos = merge_arrays(qpos1, qpos2, args.interpolate, "qpos")
            if merged_qpos is not None:
                merged_data['qpos'] = merged_qpos
                qpos_total_frames = merged_qpos.shape[0]
        
        # 合并其他帧相关数据
        successful_keys = []
        for key in sorted(all_frame_keys):
            if key == 'qpos':
                continue
                
            file1 = os.path.join(temp_dir1, f"{key}.npy")
            file2 = os.path.join(temp_dir2, f"{key}.npy")
            
            if os.path.exists(file1) and os.path.exists(file2):
                arr1 = safe_load_npy(file1)
                arr2 = safe_load_npy(file2)
                
                if arr1 is not None and arr2 is not None:
                    # 检查数组是否有效
                    if arr1.shape[0] == 0 or arr2.shape[0] == 0:
                        print(f"  {key}: 数组有空的维度，跳过")
                        continue
                    
                    merged_arr = merge_arrays(arr1, arr2, args.interpolate, key)
                    if merged_arr is not None:
                        # 检查帧数是否一致
                        if 'qpos' in merged_data and merged_arr.shape[0] == merged_data['qpos'].shape[0]:
                            merged_data[key] = merged_arr
                            successful_keys.append(key)
                        elif 'qpos' not in merged_data:
                            merged_data[key] = merged_arr
                            successful_keys.append(key)
                        else:
                            print(f"  {key}: 帧数不匹配 ({merged_arr.shape[0]} != {merged_data['qpos'].shape[0]})，跳过")
            else:
                print(f"  {key}: 文件不存在，跳过")
        
        print(f"\n成功合并的帧相关数据: {len(successful_keys)}个")
        for key in sorted(successful_keys):
            print(f"  - {key}: {merged_data[key].shape}")
        
        # 步骤3: 合并静态数据
        print(f"\n[步骤4] 合并静态数据...")
        static_data = {}
        
        for key in sorted(all_static_keys):
            file1 = os.path.join(temp_dir1, f"{key}.npy")
            file2 = os.path.join(temp_dir2, f"{key}.npy")
            
            data = None
            
            # 特殊处理frequency
            if key == 'frequency':
                if args.frequency is not None:
                    data = args.frequency
                    print(f"  frequency: 使用指定值 {data}Hz")
                elif os.path.exists(file1):
                    freq_data = safe_load_npy(file1)
                    if freq_data is not None:
                        data = float(freq_data) if hasattr(freq_data, '__float__') else float(freq_data.item())
                        print(f"  frequency: 使用文件1的 {data}Hz")
                elif os.path.exists(file2):
                    freq_data = safe_load_npy(file2)
                    if freq_data is not None:
                        data = float(freq_data) if hasattr(freq_data, '__float__') else float(freq_data.item())
                        print(f"  frequency: 使用文件2的 {data}Hz")
                else:
                    data = 100.0
                    print(f"  frequency: 使用默认值 100Hz")
                    
            # 其他静态数据：优先使用文件1的
            elif os.path.exists(file1):
                data = safe_load_npy(file1)
                if data is not None:
                    print(f"  {key}: 使用文件1的数据")
            elif os.path.exists(file2):
                data = safe_load_npy(file2)
                if data is not None:
                    print(f"  {key}: 使用文件2的数据")
            
            if data is not None:
                static_data[key] = data
        
        # 合并所有数据
        all_merged_data = {**merged_data, **static_data}
        
        # 步骤4: 保存结果
        print(f"\n[步骤5] 保存到: {args.output}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存为NPZ文件
        if args.compress:
            np.savez_compressed(args.output, **all_merged_data)
            print("  使用压缩存储")
        else:
            np.savez(args.output, **all_merged_data)
        
        # 步骤5: 验证结果
        print(f"\n[步骤6] 验证输出文件...")
        orig_size1 = os.path.getsize(file1_path) / 1024 / 1024
        orig_size2 = os.path.getsize(file2_path) / 1024 / 1024
        new_size = os.path.getsize(args.output) / 1024 / 1024
        
        print(f"\n文件大小对比:")
        print(f"  文件1: {orig_size1:.2f} MB")
        print(f"  文件2: {orig_size2:.2f} MB")
        print(f"  合并后: {new_size:.2f} MB")
        print(f"  原始总大小: {orig_size1 + orig_size2:.2f} MB")
        print(f"  大小比例: {new_size/(orig_size1 + orig_size2):.2f}x")
        
        if 'qpos' in all_merged_data:
            qpos = all_merged_data['qpos']
            print(f"\n关键数据信息:")
            print(f"  qpos: {qpos.shape[0]} 帧, {qpos.shape[1]} 个关节")
            print(f"  插值帧数: {args.interpolate}")
            print(f"  总帧数: {qpos.shape[0]}")
            print(f"  理论最小帧数: {frames1 + frames2 + args.interpolate}")
        
        print(f"\n✅ 合并完成!")
        print(f"   输出文件: {args.output}")
        print(f"   包含数据项: {len(all_merged_data)}个")
        
    except Exception as e:
        print(f"\n❌ 合并过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理临时文件
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

if __name__ == '__main__':
    main()