#!/usr/bin/env python3
"""
解析 G1 retarget 导出的 .pt 文件：打印所有 key 的 shape，并推断与 smplx_pkl_to_robot 一致的 labels。
.pt 内不含 fps/labels，labels 按 G1 XML 的 DoF 顺序推断。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

# 从仓库根加载 G1 模型以获取 dof 顺序
HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def get_g1_dof_labels():
    """从 G1 MuJoCo 模型获取与 .pt joint_position 顺序一致的 29 个关节名（不含 free joint）。"""
    import mujoco as mj
    xml_path = REPO_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    if not xml_path.exists():
        xml_path = (REPO_ROOT / "general_motion_retargeting" / ".." / "assets" / "unitree_g1" / "g1_mocap_29dof.xml").resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"G1 XML not found: {xml_path}")
    model = mj.MjModel.from_xml_path(str(xml_path))
    # qpos: 3 pos + 4 quat + 29 joint => joint_position 对应 dof 索引 6..34（跳过 free joint 的 6 个 dof）
    dof_names = []
    for i in range(6, model.nv):
        jnt_id = model.dof_jntid[i]
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        dof_names.append(name or f"dof_{i}")
    return dof_names


def get_pt_labels(dof_names):
    """与 smplx_pkl_to_robot 保存时的 labels 一致。"""
    return (
        ["base_position/x", "base_position/y", "base_position/z"]
        + ["base_pose/x", "base_pose/y", "base_pose/z"]
        + [f"joint_position/{n}" for n in dof_names]
    )


def parse_pt(pt_path, robot="unitree_g1", out_json=None):
    import torch
    pt_path = pathlib.Path(pt_path)
    if not pt_path.is_file():
        print(f"错误: 文件不存在 {pt_path}", file=sys.stderr)
        return None

    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        data_np = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in data.items()}
    else:
        print("错误: .pt 根节点不是 dict", file=sys.stderr)
        return None

    print("=== .pt keys & shapes ===")
    for k in sorted(data_np.keys()):
        v = data_np[k]
        print(f"  {k}: {getattr(v, 'shape', type(v))}")

    # base_position z 最后 10 帧
    if "base_position" in data_np:
        bp = np.asarray(data_np["base_position"], dtype=np.float64)
        if bp.ndim >= 2 and bp.shape[0] >= 1:
            n_show = min(10, bp.shape[0])
            z_vals = bp[-n_show:]  # ← 修改：取最后 n_show 帧
            print("\n=== base_position (最后 10 帧) ===")
            for i in range(n_show):
                print(f"  帧 {bp.shape[0] - n_show + i}: z = {z_vals[i]}")  # ← 显示实际帧索引
            if bp.shape[0] > 10:
                print(f"  ... 共 {bp.shape[0]} 帧")

    if "chair_quat" in data_np:
        kk = np.asarray(data_np["chair_quat"], dtype=np.float64)
        print("\nchair_quat:", kk)

    if "base_height_bias" in data_np:
        kk = np.asarray(data_np["base_height_bias"], dtype=np.float64)
        print("base_height_bias:", kk)

    if "chair_pos" in data_np:
        cp = np.asarray(data_np["chair_pos"], dtype=np.float64)
        if cp.ndim >= 2 and cp.shape[0] >= 1:
            n_show_a = min(10, cp.shape[0])
            z_vals_a = cp[-n_show_a:]  # ← 修改：取最后 n_show_a 帧
            print("\n=== chair_pos z (最后 10 帧) ===")
            for i in range(n_show_a):
                print(f"  帧 {cp.shape[0] - n_show_a + i}: z = {z_vals_a[i]}")  # ← 显示实际帧索引
            if cp.shape[0] > 10:
                print(f"  ... 共 {cp.shape[0]} 帧")

    if "link_angular_velocity" in data_np:
        cho = np.asarray(data_np["link_angular_velocity"], dtype=np.float64)
        print("\nlink_angular_velocity:", cho)

    # 打印最后 10 帧箱子的位置（如果存在 box_pos_local）
    if "box_pos_local" in data:
        box_pos = np.asarray(data["box_pos_local"])
        if box_pos.ndim == 2 and box_pos.shape[1] >= 3:
            print("\n最后 10 帧 box_pos_local (x, y, z):")
            T_box = min(10, box_pos.shape[0])
            start_idx = box_pos.shape[0] - T_box  # ← 计算起始帧索引
            for i in range(T_box):
                x, y, z = box_pos[start_idx + i, :3]
                print(f"  frame {start_idx + i:3d}: x={x:.4f}, y={y:.4f}, z={z:.4f}")  # ← 显示实际帧索引
        else:
            print("box_pos_local 形状不是 (T,3/...)，暂不打印。")

    if "box_height_global" in data:
        box_height = np.asarray(data["box_height_global"])
        if box_height.ndim == 1:
            print("\n最后 10 帧 box_height_global:")
            T_box = min(10, box_height.shape[0])
            start_idx = box_height.shape[0] - T_box  # ← 计算起始帧索引
            for i in range(T_box):
                print(f"  frame {start_idx + i:3d}: {box_height[start_idx + i]:.4f}")  # ← 显示实际帧索引
        else:
            print("box_height_global 形状不是 (T,)，暂不打印。")

    # 推断 labels（仅 G1 有固定 29 dof 顺序）
    labels = None
    if robot == "unitree_g1":
        dof_names = get_g1_dof_labels()
        labels = get_pt_labels(dof_names)
        print("\n=== inferred labels (G1, 3 + 3 + 29) ===")
        for i, lab in enumerate(labels):
            print(f"  {i}: {lab}")
        nj = data_np.get("joint_position")
        if nj is not None and getattr(nj, "shape", None) is not None and len(nj.shape) >= 2:
            n_joint_dim = nj.shape[1]
            if n_joint_dim != len(dof_names):
                print(f"\n注意: joint_position 维度 {n_joint_dim} 与 G1 dof 数 {len(dof_names)} 不一致")

    # 帧率：检查多种常见字段名（fps / frequency / frame_rate 等）
    fps_candidates = (
        "fps", "frame_rate", "frame_rate_hz", "frequency", "freq", "hz",
        "mocap_framerate", "mocap_frame_rate", "sampling_rate", "dt",
    )
    fps = None
    fps_key_used = None
    for key in fps_candidates:
        if key in data_np:
            val = data_np[key]
            try:
                v = np.asarray(val, dtype=np.float64)
                fps_val = float(v.flat[0]) if v.size > 0 else None
                if fps_val is not None and (key != "dt" or fps_val > 0):
                    if key == "dt":
                        fps_val = 1.0 / fps_val  # dt -> Hz
                    fps = fps_val
                    fps_key_used = key
                    break
            except (TypeError, ValueError):
                pass
    print("\n=== 帧率 (frame rate) ===")
    if fps is not None and fps_key_used is not None:
        print(f"  找到字段: '{fps_key_used}' = {fps} Hz")
    else:
        print("  未找到以下任一字段: " + ", ".join(fps_candidates))
        print("  (请从 pkl 或数据来源确认帧率，例如 50)")

    out = {
        "path": str(pt_path),
        "keys": list(data_np.keys()),
        "shapes": {k: getattr(v, "shape", None) for k, v in data_np.items() if hasattr(v, "shape")},
        "labels": labels,
        "fps": fps,
        "fps_key": fps_key_used,
    }
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({**out, "shapes": {k: list(v) if v is not None else None for k, v in out["shapes"].items()}}, f, indent=2)
        print(f"\n已写入: {out_json}")
    return out


def main():
    parser = argparse.ArgumentParser(description="解析 G1 retarget 的 .pt，打印 keys/shapes 并推断 labels")
    parser.add_argument("pt_file", type=str, help=".pt 文件路径")
    parser.add_argument("--robot", type=str, default="unitree_g1", choices=["unitree_g1"], help="用于推断 joint 顺序的机器人")
    parser.add_argument("--out_json", type=str, default=None, help="将解析结果（含 labels）写入 JSON")
    args = parser.parse_args()
    parse_pt(args.pt_file, robot=args.robot, out_json=args.out_json)


if __name__ == "__main__":
    main()
