#!/usr/bin/env python3
"""
解析 .npz/.npy 动作文件：打印 keys / shape / dtype，并输出常见字段摘要。
用法:
  python scripts/parse_npz.py /path/to/motion.npz
  python scripts/parse_npz.py /path/to/motion.npy
  python scripts/parse_npz.py /path/to/motion.npz --out_json /tmp/meta.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np


def _as_float(v):
    arr = np.asarray(v, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.flat[0])


def _infer_fps(data):
    fps_candidates = (
        "fps",
        "framerate",
        "frame_rate",
        "frame_rate_hz",
        "frequency",
        "freq",
        "hz",
        "mocap_framerate",
        "mocap_frame_rate",
        "sampling_rate",
        "dt",
    )
    for key in fps_candidates:
        if key not in data:
            continue
        try:
            val = _as_float(data[key])
        except (TypeError, ValueError):
            continue
        if val is None:
            continue
        if key == "dt":
            if val <= 0:
                continue
            return 1.0 / val, key
        return val, key
    return None, None


def _load_arrays(input_path):
    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        with np.load(input_path, allow_pickle=True) as data:
            keys = sorted(data.files)
            arrays = {k: data[k] for k in keys}
        return keys, arrays

    if suffix == ".npy":
        data = np.load(input_path, allow_pickle=True)
        # npy 可能保存为 dict（0-d object），也可能是单个 ndarray
        if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
            obj = data.item()
            if isinstance(obj, dict):
                keys = sorted(obj.keys())
                arrays = {k: np.asarray(obj[k]) for k in keys}
                return keys, arrays
        keys = ["array"]
        arrays = {"array": np.asarray(data)}
        return keys, arrays

    raise ValueError(f"不支持的文件格式: {suffix}（仅支持 .npz/.npy）")


def parse_motion_file(input_file, out_json=None):
    input_path = pathlib.Path(input_file)
    if not input_path.is_file():
        print(f"错误: 文件不存在 {input_path}", file=sys.stderr)
        return None

    keys, arrays = _load_arrays(input_path)

    print("=== keys / shapes / dtypes ===")
    for k in keys:
        v = arrays[k]
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", None)
        print(f"  {k}: shape={shape}, dtype={dtype}")

    # 常见字段摘要
    print("\n=== 常见字段摘要 ===")
    common_keys = (
        "root_pos",
        "root_rot",
        "dof_pos",
        "base_position",
        "base_quat",
        "joint_position",
        "link_position",
    )
    for k in common_keys:
        if k not in arrays:
            continue
        v = arrays[k]
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > 0:
            print(f"  {k}: T={v.shape[0]}, tail_shape={v.shape[1:]}")
            sample = v[0]
            if np.asarray(sample).size <= 12:
                print(f"    first_frame={np.asarray(sample).tolist()}")
            else:
                flat = np.asarray(sample).reshape(-1)
                print(f"    first_frame_sample={flat[:6].tolist()}")

    fps, fps_key = _infer_fps(arrays)
    print("\n=== 帧率 (frame rate) ===")
    if fps is not None:
        print(f"  找到字段: '{fps_key}' = {fps} Hz")
    else:
        print("  未找到 fps/dt 等常见字段")

    result = {
        "path": str(input_path),
        "format": input_path.suffix.lower(),
        "keys": keys,
        "shapes": {
            k: list(arrays[k].shape)
            for k in keys
            if hasattr(arrays[k], "shape")
        },
        "dtypes": {k: str(arrays[k].dtype) for k in keys if hasattr(arrays[k], "dtype")},
        "fps": fps,
        "fps_key": fps_key,
    }

    if out_json:
        out_path = pathlib.Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n已写入: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="解析 .npz/.npy 文件结构并打印关键信息"
    )
    parser.add_argument("input_file", type=str, help=".npz 或 .npy 文件路径")
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="可选：将解析信息保存为 JSON",
    )
    args = parser.parse_args()
    parse_motion_file(args.input_file, out_json=args.out_json)


if __name__ == "__main__":
    main()
