#!/usr/bin/env python3
"""
简单检查 .pt 数据：打印前若干帧（默认前 5 帧）的内容。

用法：
  python scripts/print_pt_first_frames.py path/to/data.pt [--num_frames 5]
"""

import argparse
import os
from typing import Any

import numpy as np
import torch


def to_numpy(x: Any) -> np.ndarray | None:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="打印 .pt 文件前若干帧的数据")
    parser.add_argument("pt_file", type=str, help="输入的 .pt 路径")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="打印前多少帧（默认 5）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pt_file):
        raise FileNotFoundError(args.pt_file)

    data = torch.load(args.pt_file, map_location="cpu", weights_only=False)
    print(f"加载 .pt: {args.pt_file}")
    print(f"类型: {type(data)}")

    if not isinstance(data, dict):
        print("顶层不是 dict，直接打印：")
        print(data)
        return

    print("\n=== 顶层键 ===")
    for k in sorted(data.keys()):
        v = data[k]
        arr = to_numpy(v)
        if arr is None:
            print(f"- {k}: type={type(v)}")
        else:
            print(f"- {k}: shape={arr.shape}, dtype={arr.dtype}")

    print(f"\n=== 前 {args.num_frames} 帧详情 ===")
    for k in sorted(data.keys()):
        v = data[k]
        arr = to_numpy(v)
        print(f"\n键: {k}")
        if arr is None:
            print(f"  非数组类型: {type(v)} -> {v}")
            continue

        if arr.ndim == 0:
            print(f"  标量: {arr}")
        elif arr.ndim == 1:
            # (T,) 或 (D,)
            n = min(args.num_frames, arr.shape[0])
            print(f"  前 {n} 个元素:\n{arr[:n]}")
        else:
            # 认为第一维是时间维，切前 num_frames 帧
            n = min(args.num_frames, arr.shape[0])
            print(f"  前 {n} 帧切片 (沿第 0 维):")
            print(arr[:n])


if __name__ == "__main__":
    main()

