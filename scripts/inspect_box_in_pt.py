#!/usr/bin/env python3
"""
解析 AdaMimic 动作数据 .pt 文件中的字段，打印结构、形状和样例值。
用法:
  python legged_gym/scripts/inspect_pt_dataset.py [path/to/output]
  python legged_gym/scripts/inspect_pt_dataset.py legged_gym/resources/dataset/g1_dof27_data/badminton_hit/output
若不传路径，默认使用 badminton_hit/output。
"""

import os
import sys
import argparse
import torch
import numpy as np


def _describe(obj, indent=0, max_sample=3):
    """递归描述对象：类型、形状、前几个值（若为数组）。"""
    prefix = "  " * indent
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        arr = obj.numpy() if isinstance(obj, torch.Tensor) else obj
        shape = arr.shape
        dtype = str(arr.dtype)
        print(f"{prefix}shape={shape}, dtype={dtype}")
        if arr.size > 0 and arr.size <= 12:
            print(f"{prefix}  value = {arr.tolist()}")
        elif arr.size > 12:
            flat = arr.flatten()
            sample = flat[:max_sample].tolist()
            print(f"{prefix}  sample (first {max_sample}) = {sample}")
        return
    if isinstance(obj, dict):
        print(f"{prefix}dict with {len(obj)} keys: {list(obj.keys())[:10]}{'...' if len(obj) > 10 else ''}")
        for k, v in list(obj.items())[:15]:
            print(f"{prefix}  [{k!r}]")
            _describe(v, indent + 2, max_sample)
        if len(obj) > 15:
            print(f"{prefix}  ... and {len(obj) - 15} more keys")
        return
    if isinstance(obj, (list, tuple)):
        print(f"{prefix}len={len(obj)}")
        if len(obj) > 0 and len(obj) <= 5:
            for i, x in enumerate(obj):
                print(f"{prefix}  [{i}]")
                _describe(x, indent + 2, max_sample)
        elif len(obj) > 5:
            print(f"{prefix}  [0]")
            _describe(obj[0], indent + 2, max_sample)
            print(f"{prefix}  ... and {len(obj) - 1} more items")
        return
    print(f"{prefix}{type(obj).__name__} = {obj!r}")


def inspect_pt_file(pt_path):
    """加载单个 .pt 文件并打印所有字段描述。"""
    print(f"\n{'='*60}")
    print(f"File: {pt_path}")
    print("=" * 60)
    data = torch.load(pt_path, map_location="cpu")
    if not isinstance(data, dict):
        print(f"Top-level type: {type(data)}")
        _describe(data, indent=0)
        return
    print(f"Top-level keys: {list(data.keys())}")
    for key in sorted(data.keys()):
        print(f"\n--- {key} ---")
        _describe(data[key], indent=0)
    # 若有 framerate，顺带算一下时长
    if "framerate" in data and "base_position" in data:
        base = data["base_position"]
        n_frames = base.shape[0] if hasattr(base, "shape") else len(base)
        fps = data["framerate"]
        if isinstance(fps, (torch.Tensor, np.ndarray)):
            fps = float(fps.flat[0])
        duration = (n_frames - 1) / fps if fps > 0 else 0
        print(f"\n--- summary ---")
        print(f"  num_frames = {n_frames}, framerate = {fps}, duration ≈ {duration:.2f} s")

    # 若有 link_position，打印第一帧全部 link 的 pos（17 个或实际数量）
    if "link_position" in data:
        lp = data["link_position"]
        if isinstance(lp, torch.Tensor):
            lp = lp.cpu().numpy()
        if lp.ndim == 3:
            # shape (T, N_links, 3)
            first_frame = lp[0]
            n_links = first_frame.shape[0]
            link_names = data.get("link_names", data.get("body_names", None))
            if link_names is not None and hasattr(link_names, "__len__"):
                if isinstance(link_names, (torch.Tensor, np.ndarray)) and link_names.ndim == 1:
                    link_names = [str(link_names[i]) for i in range(min(len(link_names), n_links))]
                elif isinstance(link_names, (list, tuple)):
                    link_names = list(link_names)[:n_links]
                else:
                    link_names = None
            else:
                link_names = None

            # 若没有名称，根据 link 数量给出默认命名
            if link_names is None and n_links == 17:
                # G1 / adam_sp 17 keyframes 的默认顺序
                link_names = [
                    "pelvis",        # 0
                    "left_hip",      # 1
                    "left_knee",     # 2
                    "left_ankle",    # 3
                    "right_hip",     # 4
                    "right_knee",    # 5
                    "right_ankle",   # 6
                    "head",          # 7
                    "torso",         # 8
                    "left_collar",   # 9
                    "left_shoulder", # 10
                    "left_elbow",    # 11
                    "left_wrist",    # 12
                    "right_collar",  # 13
                    "right_shoulder",# 14
                    "right_elbow",   # 15
                    "right_wrist",   # 16
                ]
            elif link_names is None and n_links == 6:
                # carrybox/adam_sp 6-link 下采样的约定顺序：
                # 0 left_wrist, 1 right_wrist, 2 left_ankle, 3 right_ankle, 4 head, 5 head(dup)
                link_names = [
                    "left_wrist",     # 0
                    "right_wrist",    # 1
                    "left_ankle",     # 2
                    "right_ankle",    # 3
                    "head",           # 4
                    "head_dup",       # 5
                ]

            print(f"\n--- link_position (first frame, all {n_links} links) ---")
            for j in range(n_links):
                pos = first_frame[j]
                name_str = f" {link_names[j]}" if link_names and j < len(link_names) else ""
                print(f"  [{j:2d}]{name_str}: pos = [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
        else:
            print(f"\n--- link_position (first frame) ---")
            print(f"  unexpected shape: {lp.shape}, skipping per-link print")


def visualize_pt_file(pt_path):
    """交互式逐帧可视化 base / link_position / box（或 chair）。"""
    print(f"\n{'='*60}")
    print(f"Visualize: {pt_path}")
    print("=" * 60)
    data = torch.load(pt_path, map_location="cpu")
    if not isinstance(data, dict):
        print("Top-level 不是 dict，无法可视化 link_position。")
        return
    if "link_position" not in data:
        print("数据中没有 'link_position' 字段，无法绘制。")
        return

    lp = data["link_position"]
    if isinstance(lp, torch.Tensor):
        lp = lp.cpu().numpy()
    if lp.ndim != 3 or lp.shape[2] != 3:
        print(f"'link_position' 形状异常: {lp.shape}，期望为 (T, N_links, 3)")
        return

    T, N_links, _ = lp.shape

    # base
    base = data.get("base_position", None)
    if isinstance(base, torch.Tensor):
        base = base.cpu().numpy()

    # box / chair 相关字段
    box_local = data.get("box_pos_local", None)
    box_h = data.get("box_height_global", None)
    chair_pos = data.get("chair_pos", None)
    if isinstance(box_local, torch.Tensor):
        box_local = box_local.cpu().numpy()
    if isinstance(box_h, torch.Tensor):
        box_h = box_h.cpu().numpy()
    if isinstance(chair_pos, torch.Tensor):
        chair_pos = chair_pos.cpu().numpy()

    # 尝试读取 link 名称（若数据中有），否则对 G1 17 keyframes 使用固定顺序映射
    link_names = data.get("link_names", data.get("body_names", None))
    if link_names is not None and hasattr(link_names, "__len__"):
        if isinstance(link_names, (torch.Tensor, np.ndarray)) and link_names.ndim == 1:
            link_names = [str(link_names[i]) for i in range(min(len(link_names), N_links))]
        elif isinstance(link_names, (list, tuple)):
            link_names = list(link_names)[:N_links]
        else:
            link_names = None
    else:
        link_names = None

    # 若数据中没有名称，根据 link 数量给出默认命名
    if link_names is None and N_links == 17:
        # G1 / adam_sp 17 keyframes
        link_names = [
            "pelvis",        # 0
            "left_hip",      # 1
            "left_knee",     # 2
            "left_ankle",    # 3
            "right_hip",     # 4
            "right_knee",    # 5
            "right_ankle",   # 6
            "head",          # 7
            "torso",         # 8
            "left_collar",   # 9
            "left_shoulder", # 10
            "left_elbow",    # 11
            "left_wrist",    # 12
            "right_collar",  # 13
            "right_shoulder",# 14
            "right_elbow",   # 15
            "right_wrist",   # 16
        ]
    elif link_names is None and N_links == 6:
        # carrybox/adam_sp 6-link 下采样约定：
        # 0 left_wrist, 1 right_wrist, 2 left_ankle, 3 right_ankle, 4 head, 5 head(dup)
        link_names = [
            "left_wrist",     # 0
            "right_wrist",    # 1
            "left_ankle",     # 2
            "right_ankle",    # 3
            "head",           # 4
            "head_dup",       # 5
        ]

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("未安装 matplotlib，无法绘制可视化图像。请先安装: pip install matplotlib")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    frame = 0

    def draw_frame():
        pts = lp[frame]  # (N_links, 3)
        ax.clear()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="b", label="links")

        # 在点旁边标注 keyframe/link 名称（若可用）
        if link_names:
            for j in range(min(N_links, len(link_names))):
                name = str(link_names[j])
                x, y, z = pts[j]
                ax.text(x, y, z, name, fontsize=7)

        # base
        if base is not None and hasattr(base, "shape") and base.shape[0] > frame:
            bp = base[frame]
            ax.scatter([bp[0]], [bp[1]], [bp[2]], c="r", s=60, label="base")

        # box / chair（world 坐标）
        box_world = None
        if box_local is not None and box_local.shape[0] > frame and base is not None:
            bl = np.asarray(box_local[frame])
            bp = np.asarray(base[frame])
            if box_h is not None and box_h.shape[0] > frame:
                h0 = float(np.asarray(box_h[frame]))
                box_world = np.array([bp[0] + bl[0], bp[1] + bl[1], h0], dtype=float)
            else:
                box_world = bp + bl
        elif chair_pos is not None and chair_pos.shape[0] > frame:
            box_world = np.asarray(chair_pos[frame], dtype=float)

        if box_world is not None:
            ax.scatter(
                [box_world[0]], [box_world[1]], [box_world[2]],
                c="g", s=60, label="box/chair"
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"frame {frame+1}/{T}")

        # 设置近似等比例坐标，方便看整体姿态
        xyz_min = pts.min(axis=0)
        xyz_max = pts.max(axis=0)
        if box_world is not None:
            xyz_min = np.minimum(xyz_min, box_world)
            xyz_max = np.maximum(xyz_max, box_world)
        if base is not None and hasattr(base, "shape") and base.shape[0] > frame:
            xyz_min = np.minimum(xyz_min, base[frame])
            xyz_max = np.maximum(xyz_max, base[frame])

        max_range = float((xyz_max - xyz_min).max())
        if max_range <= 0:
            max_range = 1.0
        center = (xyz_max + xyz_min) / 2.0
        for (set_lim, c) in ((ax.set_xlim, center[0]), (ax.set_ylim, center[1]), (ax.set_zlim, center[2])):
            set_lim(c - max_range / 2.0, c + max_range / 2.0)

        ax.legend(loc="upper right")
        plt.draw()
        plt.pause(0.001)

    draw_frame()
    print("\n交互说明：")
    print("  回车 / n: 下一帧")
    print("  p       : 上一帧")
    print("  数字    : 跳转到该帧（1-based）")
    print("  q       : 退出")

    while True:
        try:
            cmd = input(f"[{frame+1}/{T}] 输入指令: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "" or cmd.lower() == "n":
            if frame < T - 1:
                frame += 1
            else:
                print("已到最后一帧。")
                continue
        elif cmd.lower() == "p":
            if frame > 0:
                frame -= 1
            else:
                print("已到第一帧。")
                continue
        elif cmd.lower() == "q":
            break
        else:
            try:
                idx = int(cmd) - 1
                if 0 <= idx < T:
                    frame = idx
                else:
                    print(f"超出帧范围，合法范围为 1~{T}。")
                    continue
            except ValueError:
                print("无效输入，请使用 回车/n/p/数字/q。")
                continue

        draw_frame()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inspect single .pt file: fields / link / box / chair.")
    parser.add_argument(
        "pt_file",
        type=str,
        help=".pt 文件路径（绝对路径或相对路径均可）",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="交互式逐帧可视化 base/link/box（一次绘制一帧）。",
    )
    args = parser.parse_args()

    pt_path = args.pt_file
    # 若给的是相对路径，则以当前工作目录为基准解析
    if not os.path.isabs(pt_path):
        pt_path = os.path.normpath(os.path.join(os.getcwd(), pt_path))

    if not os.path.isfile(pt_path):
        print(f"Error: pt file not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    if args.viz:
        visualize_pt_file(pt_path)
    else:
        inspect_pt_file(pt_path)


if __name__ == "__main__":
    main()
