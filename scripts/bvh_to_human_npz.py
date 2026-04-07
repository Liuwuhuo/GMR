#!/usr/bin/env python3
"""
Load BVH as human joints only (no robot retarget),
apply configurable axis conversion, and export to NPZ for debugging.
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as lafan_utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def build_axis_matrix(mode: str) -> np.ndarray:
    """
    Return rotation matrix that maps BVH coordinates
    to target debug coordinates.
    """
    if mode == "lafan":
        # Existing project convention
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    if mode == "identity":
        return np.eye(3, dtype=np.float64)
    if mode == "x180":
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    raise ValueError(f"Unknown axis_mode: {mode}")


def infer_fps_from_bvh(path: str) -> float:
    frame_time = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Frame Time:"):
                frame_time = float(line.split(":", 1)[1].strip())
                break
    if frame_time is None or frame_time <= 0:
        return 30.0
    return 1.0 / frame_time


def main():
    parser = argparse.ArgumentParser(
        description="Convert BVH to human-only NPZ for axis/debug checks"
    )
    parser.add_argument("--bvh_file", required=True, help="Input BVH")
    parser.add_argument(
        "--axis_mode",
        choices=["lafan", "identity", "x180"],
        default="lafan",
        help="Axis conversion mode before export",
    )
    parser.add_argument(
        "--auto_upright",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If head z is below pelvis z after conversion, "
            "apply 180 deg rotation around x for all joints."
        ),
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Optional output NPZ path (debug mode defaults to no file output)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Visualize frame-0 human skeleton in a matplotlib 3D window",
    )
    args = parser.parse_args()

    data = read_bvh(args.bvh_file)
    global_quat, global_pos = lafan_utils.quat_fk(data.quats, data.pos, data.parents)

    rot_m = build_axis_matrix(args.axis_mode)
    rot_q_wxyz = R.from_matrix(rot_m).as_quat(
        scalar_first=True
    )

    # Apply axis transform
    T = global_pos.shape[0]
    J = global_pos.shape[1]
    pos = np.zeros((T, J, 3), dtype=np.float64)
    quat = np.zeros((T, J, 4), dtype=np.float64)  # wxyz
    for t in range(T):
        for j in range(J):
            q = lafan_utils.quat_mul(rot_q_wxyz, global_quat[t, j])
            p = global_pos[t, j] @ rot_m.T
            pos[t, j] = p
            quat[t, j] = q

    # cm -> m (BVH legacy)
    pos = pos

    # Optional upright auto-fix
    if args.auto_upright:
        try:
            pelvis_idx = data.bones.index("Pelvis")
        except ValueError:
            pelvis_idx = 0
        head_idx = None
        for name in ("Head", "head", "Neck", "neck"):
            if name in data.bones:
                head_idx = data.bones.index(name)
                break
        if head_idx is not None and pos[0, head_idx, 2] < pos[0, pelvis_idx, 2]:
            fix_m = build_axis_matrix("x180")
            fix_q_wxyz = R.from_matrix(fix_m).as_quat(
                scalar_first=True
            )
            for t in range(T):
                for j in range(J):
                    quat[t, j] = lafan_utils.quat_mul(fix_q_wxyz, quat[t, j])
                    pos[t, j] = pos[t, j] @ fix_m.T
            print("[auto_upright] applied x180 because head was below pelvis")

    fps = infer_fps_from_bvh(args.bvh_file)

    print(f"frames={T}, joints={J}, fps={fps:.3f}")
    try:
        pelvis_idx = data.bones.index("Pelvis")
    except ValueError:
        pelvis_idx = 0
    sample_names = ["Pelvis", "Head", "L_Foot", "R_Foot"]
    print("sample joint positions at frame0:")
    for n in sample_names:
        if n in data.bones:
            j = data.bones.index(n)
            p = pos[0, j]
            print(f"  {n}: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
    z_all = pos[0, :, 2]
    print(
        f"frame0 z-range: min={np.min(z_all):.4f}, max={np.max(z_all):.4f}, "
        f"pelvis_z={pos[0, pelvis_idx, 2]:.4f}"
    )

    if args.visualize:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "matplotlib is required for --visualize"
            ) from exc

        p0 = pos[0]  # (J, 3)
        fig = plt.figure("human_debug_frame0")
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], s=16, c="tab:blue")

        # Draw kinematic edges using BVH parent relation.
        for j in range(J):
            parent = int(data.parents[j])
            if parent < 0:
                continue
            xs = [p0[parent, 0], p0[j, 0]]
            ys = [p0[parent, 1], p0[j, 1]]
            zs = [p0[parent, 2], p0[j, 2]]
            ax.plot(xs, ys, zs, c="tab:gray", linewidth=1.2)

        # Mark key joints for orientation sanity check.
        for name, color in [
            ("Pelvis", "tab:red"),
            ("Head", "tab:green"),
            ("L_Foot", "tab:orange"),
            ("R_Foot", "tab:purple"),
        ]:
            if name in data.bones:
                j = data.bones.index(name)
                ax.scatter(
                    [p0[j, 0]], [p0[j, 1]], [p0[j, 2]], s=42, c=color, label=name
                )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{os.path.basename(args.bvh_file)} | axis={args.axis_mode}")
        ax.legend(loc="best")
        ax.set_box_aspect((1, 1, 1))
        plt.tight_layout()
        plt.show()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        np.savez(
            args.save_path,
            fps=np.array([fps], dtype=np.float64),
            joint_names=np.array(data.bones, dtype=object),
            joint_pos=pos,
            joint_quat_wxyz=quat,
            axis_mode=np.array(args.axis_mode),
            auto_upright=np.array(bool(args.auto_upright)),
        )
        print(f"Saved human debug npz: {args.save_path}")


if __name__ == "__main__":
    main()
