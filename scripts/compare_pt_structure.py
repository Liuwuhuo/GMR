#!/usr/bin/env python3
"""
对比两个 .pt 文件的“数据结构”，检查：
- 顶层类型（dict / 非 dict）
- 若为 dict：键集合是否一致
- 各字段的类型、shape、dtype 是否一致

不比较具体数值，只比较结构。
"""

import sys
import os
from typing import Any

import numpy as np

import torch


def describe_value(v: Any):
    """返回 (kind, shape, dtype, extra) 方便比较和打印。"""
    # torch.Tensor
    if isinstance(v, torch.Tensor):
        return ("tensor", tuple(v.shape), str(v.dtype), None)

    # numpy array
    if isinstance(v, np.ndarray):
        return ("ndarray", tuple(v.shape), str(v.dtype), None)

    # list / tuple
    if isinstance(v, (list, tuple)):
        length = len(v)
        subtype = type(v[0]).__name__ if length > 0 else "empty"
        return ("sequence", (length,), subtype, None)

    # 标量类型
    if isinstance(v, (int, float, str, bool)):
        return ("scalar", (), type(v).__name__, None)

    # 其他（例如 dict、None 等）
    return (type(v).__name__, None, None, None)


def compare_dicts(d1: dict, d2: dict):
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())

    only_in_1 = sorted(keys1 - keys2)
    only_in_2 = sorted(keys2 - keys1)
    common = sorted(keys1 & keys2)

    print("=== 顶层键比较 ===")
    print("A 中独有键:", only_in_1 if only_in_1 else "无")
    print("B 中独有键:", only_in_2 if only_in_2 else "无")

    print("\n=== 公共键的结构比较 ===")
    for k in common:
        v1 = d1[k]
        v2 = d2[k]
        desc1 = describe_value(v1)
        desc2 = describe_value(v2)
        same = desc1 == desc2

        status = "✅ 一致" if same else "❌ 不一致"
        print(f"\n[{status}] 键: {k}")
        print("  A:", desc1)
        print("  B:", desc2)


def main(path_a: str, path_b: str):
    if not os.path.isfile(path_a):
        print(f"错误: A 文件不存在: {path_a}")
        return
    if not os.path.isfile(path_b):
        print(f"错误: B 文件不存在: {path_b}")
        return

    print(f"加载 A: {path_a}")
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    print(f"加载 B: {path_b}")
    b = torch.load(path_b, map_location="cpu", weights_only=False)

    print("\n=== 顶层类型 ===")
    print("A 顶层类型:", type(a).__name__)
    print("B 顶层类型:", type(b).__name__)

    if isinstance(a, dict) and isinstance(b, dict):
        compare_dicts(a, b)
    else:
        print("至少一方不是 dict，直接比较顶层描述：")
        print("A 描述:", describe_value(a))
        print("B 描述:", describe_value(b))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python compare_pt_structure.py A.pt B.pt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])