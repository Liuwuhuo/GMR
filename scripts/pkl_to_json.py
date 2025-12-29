import pickle
import json
import numpy as np
import argparse

def pkl_to_json(pkl_path, json_path=None):
    """
    将 retargeting 生成的 pkl 文件转换为可读的 JSON 格式
    
    Args:
        pkl_path: 输入的 .pkl 文件路径
        json_path: 输出的 .json 文件路径（可选，默认为同目录同文件名）
    """
    
    # 如果没有指定输出路径，使用相同的文件名但扩展名为 .json
    if json_path is None:
        json_path = pkl_path.replace('.pkl', '.json')
    
    # 1. 加载 pkl 文件
    print(f"正在加载 {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        motion_data = pickle.load(f)
    
    print(f"数据键值: {list(motion_data.keys())}")
    print(f"FPS: {motion_data.get('fps', '未知')}")
    print(f"总帧数: {len(motion_data.get('root_pos', []))}")
    
    # 2. 自定义 JSON 编码器，处理 NumPy 数组和其他特殊类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  # 将数组转换为列表
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif obj is None:
                return None
            # 添加其他需要处理的类型...
            return super().default(obj)
    
    # 3. 转换为 JSON 并保存
    print(f"正在转换为 JSON 并保存到 {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(motion_data, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    
    print("转换完成！")
    
    # 4. 显示一些统计信息
    print("\n数据统计:")
    for key, value in motion_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: 形状 {value.shape}, 数据类型 {value.dtype}")
        else:
            print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description='将 retargeting 的 pkl 文件转换为 JSON 格式')
    parser.add_argument('--pkl_file', type=str, required=True, help='输入的 .pkl 文件路径')
    parser.add_argument('--json_file', type=str, default=None, help='输出的 .json 文件路径（可选）')
    
    args = parser.parse_args()
    
    pkl_to_json(args.pkl_file, args.json_file)

if __name__ == "__main__":
    # 如果直接运行这个脚本，使用示例
    # python pkl_to_json.py --pkl_file your_motion.pkl --json_file output.json
    
    # 或者可以硬编码文件路径进行测试
    # pkl_to_json("your_motion_data.pkl", "converted_motion.json")
    
    main()