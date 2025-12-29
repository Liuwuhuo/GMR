import pickle
import json
import numpy as np
import argparse
import os
import glob
from pathlib import Path

def convert_to_json(pkl_path, json_path):
    """将单个 PKL 文件转换为 JSON 格式"""
    with open(pkl_path, 'rb') as f:
        motion_data = pickle.load(f)
    
    # 自定义 JSON 编码器
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif obj is None:
                return None
            return super().default(obj)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(motion_data, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)

def convert_to_readable_json(pkl_path, json_path):
    """将单个 PKL 文件转换为可读的 JSON 格式"""
    with open(pkl_path, 'rb') as f:
        motion_data = pickle.load(f)
    
    # 创建可读的数据结构
    readable_data = {
        "metadata": {
            "source_file": os.path.basename(pkl_path),
            "loop_mode": motion_data.get("LoopMode", "Unknown"),
            "loop_num": motion_data.get("LoopNum", 1),
            "frame_duration": motion_data.get("FrameDuration", 0),
            "frame_rate": 1.0 / motion_data.get("FrameDuration", 1.0) if motion_data.get("FrameDuration", 0) > 0 else 0,
            "total_frames": len(motion_data.get("Frames", [])),
            "total_duration": len(motion_data.get("Frames", [])) * motion_data.get("FrameDuration", 0),
            "enable_cycle_offset_position": motion_data.get("EnableCycleOffsetPosition", False),
            "enable_cycle_offset_rotation": motion_data.get("EnableCycleOffsetRotation", False),
            "motion_weight": motion_data.get("MotionWeight", 1.0),
            "data_channels": len(motion_data.get("Labels", [])),
        },
        "channel_labels": motion_data.get("Labels", []),
        "frames": []
    }
    
    # 处理帧数据
    frames = motion_data.get("Frames", [])
    labels = motion_data.get("Labels", [])
    
    for i, frame in enumerate(frames):
        frame_data = {
            "frame_index": i,
            "timestamp": i * motion_data.get("FrameDuration", 0),
            "root_position": {
                "x": frame[0],
                "y": frame[1], 
                "z": frame[2]
            },
            "root_rotation": {
                "x": frame[3],
                "y": frame[4],
                "z": frame[5],
                "w": frame[6]
            },
            "joint_positions": {}
        }
        
        # 添加关节位置
        for j in range(7, len(frame)):
            if j < len(labels):
                label = labels[j]
                joint_name = label.replace("dof_pos/", "")
                frame_data["joint_positions"][joint_name] = frame[j]
        
        readable_data["frames"].append(frame_data)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(readable_data, f, ensure_ascii=False, indent=2)

def batch_pkl_to_json(input_path, output_dir=None, readable=False, overwrite=False):
    """
    批量将 PKL 文件转换为 JSON 格式
    
    Args:
        input_path: 输入路径（可以是文件或文件夹）
        output_dir: 输出目录（可选，默认为输入文件同目录）
        readable: 是否生成可读格式
        overwrite: 是否覆盖已存在的文件
    """
    
    # 确定输入文件列表
    if os.path.isfile(input_path):
        pkl_files = [input_path]
    elif os.path.isdir(input_path):
        # 获取文件夹中所有的 .pkl 文件（去重）
        pkl_files = list(set(glob.glob(os.path.join(input_path, "**/*.pkl"), recursive=True)))
        # 按文件名排序
        pkl_files.sort()
    else:
        print(f"错误: 路径 {input_path} 不存在")
        return
    
    if not pkl_files:
        print(f"在 {input_path} 中没有找到 PKL 文件")
        return
    
    print(f"找到 {len(pkl_files)} 个 PKL 文件")
    
    # 显示找到的文件
    print("找到的文件:")
    for i, file in enumerate(pkl_files):
        print(f"  {i+1}: {os.path.basename(file)}")
    
    # 处理每个文件
    success_count = 0
    for i, pkl_file in enumerate(pkl_files):
        print(f"\n处理文件 {i+1}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
        
        try:
            # 确定输出路径
            if output_dir:
                # 保持原始目录结构
                rel_path = os.path.relpath(pkl_file, input_path) if os.path.isdir(input_path) else os.path.basename(pkl_file)
                json_file = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ('.json' if not readable else '_readable.json'))
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(json_file), exist_ok=True)
            else:
                # 在原始文件同目录生成
                json_file = pkl_file.replace('.pkl', '.json' if not readable else '_readable.json')
            
            # 检查是否已存在
            if os.path.exists(json_file) and not overwrite:
                print(f"跳过已存在的文件: {json_file}")
                continue
            
            # 转换文件
            if readable:
                convert_to_readable_json(pkl_file, json_file)
            else:
                convert_to_json(pkl_file, json_file)
            
            print(f"✓ 已保存: {json_file}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ 转换文件 {pkl_file} 时出错: {str(e)}")
    
    print(f"\n批量转换完成! 成功转换 {success_count}/{len(pkl_files)} 个文件")

def main():
    parser = argparse.ArgumentParser(description='批量将 PKL 文件转换为 JSON 格式')
    parser.add_argument('input_path', type=str, help='输入的 .pkl 文件路径或包含pkl文件的文件夹路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出的目录路径（可选）')
    parser.add_argument('--readable', action='store_true', default=False, help='生成更可读的结构化 JSON 格式')
    parser.add_argument('--overwrite', action='store_true', default=False, help='覆盖已存在的文件')
    
    args = parser.parse_args()
    
    batch_pkl_to_json(
        input_path=args.input_path,
        output_dir=args.output_dir,
        readable=args.readable,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()