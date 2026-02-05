#!/usr/bin/env python3
"""
对比两个 .pt 文件的数值分布：
- 顶层键相同的字段，打印 mean/std/min/max
- 对 joint_position / joint_velocity，额外打印每一列的 mean/std（可用于检查各关节分布差异）
"""

import sys
import os
from typing import Any, Dict

import numpy as np
import torch


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(f"不支持的类型: {type(x)}")


def print_basic_stats(name: str, arr: np.ndarray):
    flat = arr.reshape(-1)
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"    mean={flat.mean():.6f}, std={flat.std():.6f}, "
          f"min={flat.min():.6f}, max={flat.max():.6f}")


def print_joint_stats(name: str, arr: np.ndarray, max_dims: int = 10):
    """
    对 (T, D) 的关节数据，按维度打印前 max_dims 个的 mean/std。
    """
    if arr.ndim != 2:
        print_basic_stats(name, arr)
        return
    T, D = arr.shape
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    print("    per-dim (前 {}/{} 维):".format(min(max_dims, D), D))
    for i in range(min(max_dims, D)):
        print(f"      dim[{i:2d}]: mean={means[i]:.6f}, std={stds[i]:.6f}")


def load_pt(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(d, dict):
        raise TypeError(f"{path} 顶层不是 dict，而是 {type(d)}")
    return d


def main(a_path: str, b_path: str):
    print(f"加载 A: {a_path}")
    A = load_pt(a_path)
    print(f"加载 B: {b_path}")
    B = load_pt(b_path)

    keys_a = set(A.keys())
    keys_b = set(B.keys())
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print("\n=== 顶层键 ===")
    print("A 仅有:", only_a if only_a else "无")
    print("B 仅有:", only_b if only_b else "无")
    print("公共键:", common)

    print("\n=== 公共键数值对比 ===")
    for k in common:
        va, vb = A[k], B[k]
        try:
            na, nb = to_numpy(va), to_numpy(vb)
        except TypeError:
            print(f"\n[跳过] 键 {k}: 类型分别为 {type(va)} / {type(vb)}")
            continue

        print(f"\n键: {k}")
        # 基本统计
        print(" A:")
        if k in ("joint_position", "joint_velocity"):
            print_joint_stats(k, na)
        else:
            print_basic_stats(k, na)

        print(" B:")
        if k in ("joint_position", "joint_velocity"):
            print_joint_stats(k, nb)
        else:
            print_basic_stats(k, nb)

        # 简单差值统计（在 shape 一致时）
        if na.shape == nb.shape:
            diff = (na - nb).reshape(-1)
            print(f" diff: mean={diff.mean():.6f}, std={diff.std():.6f}, "
                  f"min={diff.min():.6f}, max={diff.max():.6f}")
        else:
            print(f" diff: shape 不同 A{na.shape} vs B{nb.shape}，不做差值统计")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python scripts/compare_pt_stats.py A.pt B.pt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])