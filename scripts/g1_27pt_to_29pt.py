#!/usr/bin/env python3
"""
将 G1 27 DoF 的 .pt「扩展」为 29 DoF：
- 在 joint_position / joint_velocity 中间插入 waist_roll、waist_pitch 两维
- 默认这两维填 0（中立位），其余关节按 27DoF 的顺序嵌回 29DoF

注意：
- 27DoF 顺序可以看作 29DoF 去掉 index 13(waist_roll)、14(waist_pitch) 后的结果
  因此 27DoF 的 dim 映射为：
    27[0..12]   -> 29[0..12]
    27[13..26]  -> 29[15..28]

用法:
  python g1_27pt_to_29pt.py <input_27dof.pt> [output_29dof.pt]
"""
import argparse
import sys
from pathlib import Path

import torch


def expand_27_to_29(arr_27: torch.Tensor) -> torch.Tensor:
    """
    给定 (..., 27) 的张量，返回 (..., 29)：
    - index 13,14（waist_roll, waist_pitch）填 0
    - 其它索引按 27DoF -> 29DoF 映射复制。
    """
    if arr_27.shape[-1] != 27:
        # 维度不对时直接返回原始张量，避免 silent 错误
        return arr_27

    *prefix, _ = arr_27.shape
    out = arr_27.new_zeros(*prefix, 29)

    # 0..12 直接拷贝
    out[..., 0:13] = arr_27[..., 0:13]
    # 13,14 腰部 roll / pitch 置 0（已在 new_zeros 中完成）
    # 剩余 13..26 -> 15..28
    out[..., 15:29] = arr_27[..., 13:27]
    return out


def convert_27_to_29(data: dict) -> dict:
    """对字典中的 joint_position / joint_velocity 做 27->29 扩展，其它键原样拷贝。"""
    out = {}
    for k, v in data.items():
        if k in ("joint_position", "joint_velocity"):
            arr = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            out[k] = expand_27_to_29(arr).clone()
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="G1 27 DoF .pt -> 29 DoF .pt（补 waist_roll / waist_pitch）")
    parser.add_argument("input_pt", type=str, help="输入的 27DoF data.pt 路径（joint_position 维度=27）")
    parser.add_argument(
        "output_pt",
        type=str,
        nargs="?",
        default=None,
        help="输出路径；不填则默认为输入同目录下的 xxx_29dof.pt",
    )
    args = parser.parse_args()

    input_path = Path(args.input_pt).resolve()
    if not input_path.is_file():
        print(f"错误: 文件不存在 {input_path}", file=sys.stderr)
        sys.exit(1)

    data = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        print("错误: .pt 内容不是 dict", file=sys.stderr)
        sys.exit(1)

    if "joint_position" not in data:
        print("错误: 未找到 joint_position", file=sys.stderr)
        sys.exit(1)

    n = data["joint_position"].shape[-1]
    if n != 27:
        print(f"警告: joint_position 维度为 {n}，非 27，将保持原维度不做扩展", file=sys.stderr)

    out = convert_27_to_29(data)

    if args.output_pt:
        output_path = Path(args.output_pt).resolve()
    else:
        output_path = input_path.parent / f"{input_path.stem}_29dof.pt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    print(f"已保存 29 DoF（waist_roll/pitch=0）: {output_path}")


if __name__ == "__main__":
    main()

