#!/usr/bin/env python3
"""
将 G1 29 DoF 的 .pt 转为 27 DoF：去掉 waist_roll、waist_pitch（对应 29 维中的索引 13、14）。
用法: python g1_29pt_to_27pt.py <input.pt> [output.pt]
"""
import argparse
import sys
from pathlib import Path

import torch


# G1 29 DoF 顺序中，去掉索引 13(waist_roll)、14(waist_pitch) 得到 27 DoF
G1_29_TO_27_JOINT_INDICES = [i for i in range(29) if i not in (13, 14)]


def convert_29_to_27(data):
    """把 joint_position / joint_velocity 从 29 维转为 27 维（复制一份，不写回原 dict）。"""
    out = {}
    for k, v in data.items():
        if k in ("joint_position", "joint_velocity"):
            arr = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            if arr.shape[-1] == 29:
                out[k] = arr[..., G1_29_TO_27_JOINT_INDICES].clone()
            else:
                out[k] = arr
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="G1 29 DoF .pt -> 27 DoF .pt")
    parser.add_argument("input_pt", type=str, help="输入的 data.pt 路径（29 维 joint）")
    parser.add_argument(
        "output_pt",
        type=str,
        nargs="?",
        default=None,
        help="输出路径；不填则默认为输入同目录下的 xxx_27dof.pt",
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
    if n != 29:
        print(f"警告: joint_position 维度为 {n}，非 29，已原样复制", file=sys.stderr)

    out = convert_29_to_27(data)
    if args.output_pt:
        output_path = Path(args.output_pt).resolve()
    else:
        output_path = input_path.parent / f"{input_path.stem}_27dof.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    print(f"已保存 27 DoF: {output_path}")


if __name__ == "__main__":
    main()
