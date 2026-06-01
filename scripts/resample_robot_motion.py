#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

_MOTION_EXTS = (".pkl", ".npz")


def _to_float_fps(value) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("Empty fps/framerate field.")
    return float(arr.reshape(-1)[0])


def _resample_linear(
    data: np.ndarray, t_old: np.ndarray, t_new: np.ndarray
) -> np.ndarray:
    if data.ndim < 1 or data.shape[0] != t_old.shape[0]:
        return data
    if data.shape[0] == 1:
        reps = [t_new.shape[0]] + [1] * (data.ndim - 1)
        return np.tile(data, reps)

    flat = data.reshape(data.shape[0], -1)
    out = np.empty((t_new.shape[0], flat.shape[1]), dtype=np.float64)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(t_new, t_old, flat[:, i])
    return out.reshape((t_new.shape[0],) + data.shape[1:]).astype(
        data.dtype, copy=False
    )


def _resample_quat(
    quat: np.ndarray, t_old: np.ndarray, t_new: np.ndarray, fmt: str
) -> np.ndarray:
    if quat.ndim != 2 or quat.shape[1] != 4 or quat.shape[0] != t_old.shape[0]:
        return quat
    if quat.shape[0] == 1:
        return np.repeat(quat, t_new.shape[0], axis=0)

    if fmt == "wxyz":
        quat_xyzw = quat[:, [1, 2, 3, 0]]
    elif fmt == "xyzw":
        quat_xyzw = quat
    else:
        raise ValueError(f"Unsupported quaternion format: {fmt}")

    rot = R.from_quat(quat_xyzw)
    slerp = Slerp(t_old, rot)
    rot_new = slerp(t_new)
    quat_xyzw_new = rot_new.as_quat()

    if fmt == "wxyz":
        return quat_xyzw_new[:, [3, 0, 1, 2]].astype(quat.dtype, copy=False)
    return quat_xyzw_new.astype(quat.dtype, copy=False)


def _build_output_path(
    in_file: Path,
    input_path: Path,
    output_path: Path,
    is_dir_mode: bool,
    suffix: str,
) -> Path:
    if is_dir_mode:
        rel = in_file.relative_to(input_path)
        return output_path / rel
    if output_path.suffix.lower() in _MOTION_EXTS:
        return output_path
    return output_path / f"{in_file.stem}{suffix}{in_file.suffix}"


def _iter_motion_files(input_path: Path, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in _MOTION_EXTS:
            raise ValueError(
                f"Input must be {_MOTION_EXTS} file, got: {input_path}"
            )
        yield input_path
        return

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    patterns = (
        [f"**/*{e}" for e in _MOTION_EXTS]
        if recursive
        else [f"*{e}" for e in _MOTION_EXTS]
    )
    seen = set()
    for pat in patterns:
        for p in sorted(input_path.glob(pat)):
            if p.is_file() and p not in seen:
                seen.add(p)
                yield p


def _load_motion(path: Path) -> Dict:
    suf = path.suffix.lower()
    if suf == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif suf == ".npz":
        with np.load(path, allow_pickle=True) as z:
            data = {k: np.array(z[k]) for k in z.files}
    else:
        raise ValueError(f"Unsupported format: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict-like motion data, got {type(data)}")
    return data


def _save_motion(path: Path, data: Dict, compressed: bool) -> None:
    suf = path.suffix.lower()
    if suf == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(data, f)
    elif suf == ".npz":
        if compressed:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def resample_motion_dict(
    motion_data: Dict,
    target_fps: float,
    *,
    root_rot_format: str = "wxyz",
) -> Tuple[Dict, int, int, float]:
    src_fps = None
    if "fps" in motion_data:
        src_fps = _to_float_fps(motion_data["fps"])
    elif "framerate" in motion_data:
        src_fps = _to_float_fps(motion_data["framerate"])
    else:
        raise KeyError("Cannot find fps/framerate in motion data.")

    if src_fps <= 0 or target_fps <= 0:
        raise ValueError(
            f"Invalid fps values: src={src_fps}, target={target_fps}"
        )

    if "root_pos" not in motion_data:
        raise KeyError("Cannot find root_pos in motion data.")

    root_pos = np.asarray(motion_data["root_pos"])
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Unexpected root_pos shape: {root_pos.shape}")
    if root_pos.shape[0] < 1:
        raise ValueError("Empty sequence in root_pos.")

    old_len = root_pos.shape[0]
    if old_len == 1:
        new_len = 1
        duration = 0.0
    else:
        duration = (old_len - 1) / src_fps
        new_len = max(2, int(round(duration * target_fps)) + 1)

    t_old = np.linspace(0.0, duration, old_len, dtype=np.float64)
    t_new = np.linspace(0.0, duration, new_len, dtype=np.float64)

    out = dict(motion_data)
    quat_key_format = {"root_rot": root_rot_format, "base_quat_w": "wxyz"}

    for key, value in motion_data.items():
        if key in ("fps", "framerate", "joint_names", "link_body_list"):
            continue

        arr = np.asarray(value)
        if (
            arr.ndim >= 1
            and arr.shape[0] == old_len
            and np.issubdtype(arr.dtype, np.number)
        ):
            if key in quat_key_format and arr.ndim == 2 and arr.shape[1] == 4:
                out[key] = _resample_quat(
                    arr, t_old, t_new, quat_key_format[key]
                )
            else:
                out[key] = _resample_linear(arr, t_old, t_new)

    if "fps" in out:
        if np.isscalar(out["fps"]):
            out["fps"] = float(target_fps)
        else:
            out["fps"] = np.array([target_fps], dtype=np.float64)
    else:
        out["fps"] = float(target_fps)

    if "framerate" in out:
        if np.isscalar(out["framerate"]):
            out["framerate"] = float(target_fps)
        else:
            out["framerate"] = np.array([target_fps], dtype=np.float64)

    return out, old_len, new_len, src_fps


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resample retargeted robot motion .pkl / .npz to target FPS "
            "(linear + quaternion SLERP)."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input .pkl / .npz file or folder.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Output file (single file mode) or folder (folder mode). "
            "Default: same folder with suffix; extension matches input."
        ),
    )
    parser.add_argument(
        "--target_fps", type=float, required=True, help="Target FPS."
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix for output filename in default mode, e.g. _30hz.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursive folder scan.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        default=False,
        help="For .npz output, use np.savez_compressed (.pkl ignores this).",
    )
    parser.add_argument(
        "--root_rot_format",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help=(
            "Quaternion layout of root_rot in the file. "
            "smplx_to_robot / bvh_to_robot_npz use wxyz; "
            "some legacy pkl used xyzw."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    is_dir_mode = input_path.is_dir()

    if args.suffix is None:
        suffix = f"_{int(round(args.target_fps))}hz"
    else:
        suffix = args.suffix

    if args.output_path is None:
        if is_dir_mode:
            output_path = input_path.parent / f"{input_path.name}{suffix}"
        else:
            output_path = (
                input_path.parent
                / f"{input_path.stem}{suffix}{input_path.suffix}"
            )
    else:
        output_path = Path(args.output_path).expanduser().resolve()

    files = list(_iter_motion_files(input_path, args.recursive))
    if not files:
        raise ValueError(
            f"No .pkl/.npz files found under: {input_path}"
        )

    if is_dir_mode:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    for in_file in files:
        out_file = _build_output_path(
            in_file, input_path, output_path, is_dir_mode, suffix
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists() and not args.overwrite:
            print(f"Skip existing: {out_file}")
            continue

        try:
            motion_data = _load_motion(in_file)
        except (TypeError, ValueError, pickle.UnpicklingError) as e:
            print(f"Failed load: {in_file} ({e})")
            continue

        try:
            resampled, old_len, new_len, src_fps = resample_motion_dict(
                motion_data,
                args.target_fps,
                root_rot_format=args.root_rot_format,
            )
        except (KeyError, ValueError, TypeError) as e:
            print(f"Failed: {in_file} ({e})")
            continue

        try:
            _save_motion(out_file, resampled, compressed=args.compressed)
        except (TypeError, ValueError) as e:
            print(f"Failed save: {out_file} ({e})")
            continue

        print(
            f"Saved: {out_file} | fps {src_fps:.3f} -> {args.target_fps:.3f}, "
            f"frames {old_len} -> {new_len}"
        )
        ok_count += 1

    print(f"Done. Resampled {ok_count}/{len(files)} file(s).")


if __name__ == "__main__":
    main()
