import numpy as np
import json

def np_to_serializable(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                         np.int16, np.int32, np.int64, np.uint8,
                         np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, 
                         np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.void):
        return None
    return obj

# 读取并转换
data = np.load('assets/body_models/smplx/SMPLX_MALE.npz', allow_pickle=True)
data_dict = {key: np_to_serializable(data[key]) for key in data.files}

# 保存为JSON
with open('output.json', 'w') as f:
    json.dump(data_dict, f, indent=2)