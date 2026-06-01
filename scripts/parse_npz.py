#!/usr/bin/env python3
"""
解析 .npz/.npy 动作文件：打印 keys / shape / dtype，并输出常见字段摘要。

适用任意 npz（含 posed_joints 等非 SMPL-X 轨迹）；SMPL-X 入口需 pose_body 等字段。

用法:
  python scripts/parse_npz.py /path/to/motion.npz
  python scripts/parse_npz.py /path/to/motion.npz --verbose
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


_NAME_HINT_SUBSTR = (
    "joint_name",
    "joint_names",
    "bone_name",
    "bone_names",
    "skeleton",
    "hier",
    "parent",
    "topology",
    "offsets",
    "kin_tree",
    "link_name",
    "body_name",
)


def _is_name_like_array(arr):
    if not isinstance(arr, np.ndarray):
        return False
    if arr.dtype == object:
        return True
    kind = getattr(arr.dtype, "kind", "")
    return kind in ("U", "S", "O")


def _find_skeleton_hint_keys(keys):
    hinted = [k for k in keys if any(h in k.lower() for h in _NAME_HINT_SUBSTR)]
    return sorted(hinted)


def _print_array_stats(label, arr):
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        print(f"  {label}: (empty or non-array)")
        return
    print(f"  {label}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.dtype == bool or arr.dtype == np.bool_:
        print(f"    true_ratio={float(np.mean(arr)):.4f}")
        return
    if np.issubdtype(arr.dtype, np.number):
        flat = arr.astype(np.float64, copy=False).ravel()
        print(
            f"    min={float(np.min(flat)):.6g}, max={float(np.max(flat)):.6g}, "
            f"mean={float(np.mean(flat)):.6g}"
        )


def _print_verbose_frame0(arrays):
    print("\n=== --verbose: 第 0 帧采样（仍不含骨骼名时只能看索引）===")
    if "root_positions" in arrays:
        rp = np.asarray(arrays["root_positions"])
        if rp.ndim == 2 and rp.shape[0] > 0:
            print(f"  root_positions[0] = {rp[0].tolist()}")
    if "foot_contacts" in arrays:
        fc = np.asarray(arrays["foot_contacts"])
        if fc.ndim == 2 and fc.shape[0] > 0:
            print(f"  foot_contacts[0] = {fc[0].tolist()}")
    if "posed_joints" in arrays:
        pj = np.asarray(arrays["posed_joints"])
        if pj.ndim == 3 and pj.shape[0] > 0:
            jn = pj.shape[1]
            print(f"  posed_joints[0, j] 世界坐标 (j=0..{jn - 1}):")
            for j in range(jn):
                print(f"    j={j:2d}  {pj[0, j].tolist()}")
    if "global_rot_mats" in arrays:
        g = np.asarray(arrays["global_rot_mats"])
        if g.ndim == 4 and g.shape[0] > 0:
            r0 = g[0, 0]
            det = float(np.linalg.det(r0))
            print(f"  global_rot_mats[0,0] det={det:.6f} (接近 1 为正常旋转)")


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


def parse_motion_file(input_file, out_json=None, verbose=False):
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

    print("\n=== 骨骼/关节命名（从 npz 里能直接读到的）===")
    hint_keys = _find_skeleton_hint_keys(keys)
    name_like = [k for k in keys if _is_name_like_array(arrays[k])]
    if hint_keys:
        print(f"  名字/层级相关键名（启发式匹配）: {hint_keys}")
    else:
        print("  启发式未匹配到 joint_names / parents 等常见键名。")
    if name_like:
        print(f"  疑似「字符串或 object」数组的键: {name_like}")
        for k in name_like:
            arr = arrays[k]
            preview = arr
            if isinstance(arr, np.ndarray) and arr.size > 0:
                flat = arr.reshape(-1)
                n = min(8, flat.size)
                preview = [flat[i] for i in range(n)]
            print(f"    {k} 前若干项预览: {preview}")
    else:
        print("  未发现 U/S/O 类型的关节名数组。")
    if "posed_joints" in arrays and not any(
        "joint" in k.lower() and "name" in k.lower() for k in keys
    ):
        pj = arrays["posed_joints"]
        if isinstance(pj, np.ndarray) and pj.ndim == 3:
            print(
                f"\n  说明: 当前仅有 J={pj.shape[1]} 的**索引关节**，"
                "npz 内若没有 joint_names / skeleton 等字段，"
                "无法从文件本身推断每根骨骼的人类可读名字；需查导出工具文档或源码。"
            )

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

    # 关节轨迹类 npz（如 posed_joints + root_positions），非 AMASS/SMPL-X 参数格式
    if "posed_joints" in arrays:
        pj = arrays["posed_joints"]
        if isinstance(pj, np.ndarray) and pj.ndim == 3:
            print("\n=== 关节轨迹摘要（非 SMPL-X pose_body 格式）===")
            print(f"  posed_joints: T={pj.shape[0]}, J={pj.shape[1]}, xyz")
    if "root_positions" in arrays:
        rp = arrays["root_positions"]
        if isinstance(rp, np.ndarray) and rp.ndim == 2 and rp.shape[1] == 3:
            print(f"  root_positions: T={rp.shape[0]}, (3,)")
    if "foot_contacts" in arrays:
        fc = arrays["foot_contacts"]
        if isinstance(fc, np.ndarray) and fc.ndim == 2:
            print(f"  foot_contacts: T={fc.shape[0]}, flags={fc.shape[1]}")
    smplx_need = ("pose_body", "root_orient", "trans", "betas")
    if all(k not in arrays for k in smplx_need) and "posed_joints" in arrays:
        print(
            "\n提示: smplx_to_robot.py 的 load_smplx_file 需要 SMPL-X 参数字段 "
            f"{smplx_need} 等；当前文件为关节/旋转矩阵轨迹，不能直接走该入口。"
        )

    if verbose:
        print("\n=== --verbose: 各数组数值范围 ===")
        for k in keys:
            _print_array_stats(k, arrays[k])
        _print_verbose_frame0(arrays)

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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="打印数值范围与第 0 帧逐关节位置（仍无法凭空生成骨骼名）",
    )
    args = parser.parse_args()
    parse_motion_file(
        args.input_file, out_json=args.out_json, verbose=args.verbose
    )


if __name__ == "__main__":
    main()
