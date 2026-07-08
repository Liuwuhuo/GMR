#!/usr/bin/env python3
"""
G1 29-DOF 运动数据的关节列顺序重排：在 DFS(MuJoCo) 与 BFS(Isaac Lab) 之间互转。

背景：
  - DFS / MuJoCo / 硬件顺序（g1_29dof xml、GMR、RoboNaldo csv_to_npz 的 joint_names）：
        左腿6 -> 右腿6 -> 腰(yaw,roll,pitch) -> 左臂7 -> 右臂7
  - BFS / Isaac Lab 顺序（RoboNaldo_Deploy/FreeKick.py 的 ISAAC_JOINT_NAMES）：
        按关节树深度交错：l_hip_p, r_hip_p, waist_y, l_hip_r, r_hip_r, waist_r, ...

输入支持：
  - .csv：每行 = root pos(3) + root quat(4) + 29 关节；只重排后 29 列，root 原样保留。
  - .npz：含 joint_pos[/joint_vel] (T,29) 时按列重排；其余键原样保留。

示例：
  # csv：BFS -> DFS（按你的要求）
  python scripts/reorder_g1_joints.py in.csv --src_order bfs --dst_order dfs -o out.csv
  # csv：DFS -> BFS（喂给期望 Isaac 顺序的 deploy）
  python scripts/reorder_g1_joints.py in.csv --src_order dfs --dst_order bfs -o out.csv
  # npz 同理
  python scripts/reorder_g1_joints.py motion.npz --src_order dfs --dst_order bfs -o motion_bfs.npz
"""
import argparse
import csv as csv_mod
import pathlib
import sys

import numpy as np

# DFS / MuJoCo / 硬件顺序
DFS_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# BFS / Isaac Lab 顺序（来自 RoboNaldo_Deploy/policy/robonaldo/FreeKick.py）
BFS_NAMES = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw",
    "left_hip_roll", "right_hip_roll", "waist_roll",
    "left_hip_yaw", "right_hip_yaw", "waist_pitch",
    "left_knee", "right_knee",
    "left_shoulder_pitch", "right_shoulder_pitch",
    "left_ankle_pitch", "right_ankle_pitch",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_ankle_roll", "right_ankle_roll",
    "left_shoulder_yaw", "right_shoulder_yaw",
    "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll",
    "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_yaw", "right_wrist_yaw",
]

ORDERS = {"dfs": DFS_NAMES, "bfs": BFS_NAMES}


def build_perm(src_order: str, dst_order: str) -> np.ndarray:
    """返回 perm，使得 out[:, k] = in[:, perm[k]]，k 为目标(dst)列索引。"""
    src = ORDERS[src_order]
    dst = ORDERS[dst_order]
    assert sorted(src) == sorted(dst), "DFS/BFS 名单关节集合不一致"
    src_index = {name: i for i, name in enumerate(src)}
    return np.array([src_index[name] for name in dst], dtype=np.int64)


def reorder_columns(joints: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """joints: (T, 29) 源顺序 -> (T, 29) 目标顺序。"""
    if joints.shape[1] != 29:
        raise ValueError(f"期望 29 个关节列, 实际 {joints.shape[1]}")
    return joints[:, perm]


def process_csv(in_path, out_path, perm, has_header, write_header):
    rows = []
    header = None
    with open(in_path, "r", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv_mod.reader(f)):
            if not row or all(c.strip() == "" for c in row):
                continue
            if i == 0:
                is_header = has_header
                if not has_header:
                    try:
                        float(row[0])
                    except ValueError:
                        is_header = True
                if is_header:
                    header = row
                    continue
            rows.append([float(x) for x in row[:36]])

    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[1] < 36:
        raise ValueError(f"csv 列数 {arr.shape[1]} < 36 (root7 + joints29)")
    root = arr[:, :7]
    joints = arr[:, 7:36]
    out_joints = reorder_columns(joints, perm)
    out = np.concatenate([root, out_joints], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.writer(f)
        if write_header:
            if header is not None and len(header) >= 36:
                new_header = header[:7] + [header[7 + p] for p in perm]
                w.writerow(new_header)
            else:
                w.writerow(["root_x", "root_y", "root_z",
                            "root_qx", "root_qy", "root_qz", "root_qw"]
                           + ORDERS_dst_names)
        w.writerows(out)
    print(f"已保存 csv: {out_path}  帧数={out.shape[0]}  列数={out.shape[1]}")


def process_npz(in_path, out_path, perm):
    data = np.load(in_path, allow_pickle=True)
    out = {}
    changed = []
    for k in data.files:
        v = data[k]
        if k in ("joint_pos", "joint_vel") and getattr(v, "ndim", 0) == 2 \
                and v.shape[1] == 29:
            out[k] = reorder_columns(np.asarray(v), perm).astype(v.dtype)
            changed.append(k)
        else:
            out[k] = v
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"已保存 npz: {out_path}  已重排键={changed}  其余键原样保留")


def main():
    p = argparse.ArgumentParser(description="G1 29DOF 关节列 DFS<->BFS 重排")
    p.add_argument("input", help="输入 .csv 或 .npz")
    p.add_argument("--src_order", choices=["dfs", "bfs"], default="bfs",
                   help="输入数据的关节顺序（默认 bfs）")
    p.add_argument("--dst_order", choices=["dfs", "bfs"], default="dfs",
                   help="输出数据的关节顺序（默认 dfs）")
    p.add_argument("-o", "--output", default=None, help="输出路径")
    p.add_argument("--has_header", action="store_true", help="输入 csv 含表头")
    p.add_argument("--no_header", action="store_true", help="输出 csv 不写表头")
    args = p.parse_args()

    if args.src_order == args.dst_order:
        print("src_order == dst_order，无需重排。", file=sys.stderr)
        sys.exit(1)

    in_path = pathlib.Path(args.input)
    if not in_path.is_file():
        print(f"文件不存在: {in_path}", file=sys.stderr)
        sys.exit(1)

    perm = build_perm(args.src_order, args.dst_order)
    global ORDERS_dst_names
    ORDERS_dst_names = ORDERS[args.dst_order]

    out_path = pathlib.Path(args.output) if args.output else \
        in_path.with_name(f"{in_path.stem}_{args.dst_order}{in_path.suffix}")

    print(f"重排: {args.src_order.upper()} -> {args.dst_order.upper()}  perm={perm.tolist()}")
    if in_path.suffix.lower() == ".csv":
        process_csv(in_path, out_path, perm,
                    has_header=args.has_header, write_header=(not args.no_header))
    elif in_path.suffix.lower() == ".npz":
        process_npz(in_path, out_path, perm)
    else:
        print(f"不支持的格式: {in_path.suffix}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
