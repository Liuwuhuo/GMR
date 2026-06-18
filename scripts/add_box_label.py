#!/usr/bin/env python3
"""Append a fixed-box label to an already-retargeted motion file.

Pure post-process: reads the robot base trajectory from an existing .npz/.pkl
and writes two extra fields (no re-retargeting, no mujoco/torch needed):

  - box_pos_local      (T, 3) = box_center_world - base_position   (per frame)
  - box_height_global  (T,)   = box top surface height (world z, constant)

The box is fixed, so its world position is a single value. By default it is read
from the box scene XML (the same `pos` you tune for playback in gmr_play), keeping
one source of truth. You can override it with --box_pos / --box_size.

Examples
--------
# Read box pose from assets/adam_sp/scene_box.xml (robot adam_sp_box), in-place:
python scripts/add_box_label.py --motion_path retarget/adam_sp/opt_mocap/man_pufu_002.npz

# Override the box xy and save to a new file:
python scripts/add_box_label.py --motion_path in.npz --save_path out.npz \
    --box_pos 1.2 -0.3 --box_size 1.0
"""
import argparse
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

from general_motion_retargeting.params import ROBOT_XML_DICT


BASE_POS_KEYS = ("root_pos", "base_pos_w", "base_position")


def load_motion(motion_path):
    """Return (data_dict, ext). For npz, materialize into a plain dict."""
    ext = os.path.splitext(motion_path)[1].lower()
    if ext == ".npz":
        with np.load(motion_path, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
        return data, ext
    if ext == ".pkl":
        with open(motion_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"PKL must contain a dict, got {type(data)}")
        return data, ext
    raise ValueError(f"Unsupported motion format: {ext} (expected .npz or .pkl)")


def get_base_position(data):
    """Extract the (T, 3) base/root world position from a motion dict."""
    for key in BASE_POS_KEYS:
        if key in data:
            arr = np.asarray(data[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3], key
    raise KeyError(
        f"None of the base-position keys {BASE_POS_KEYS} found in motion data. "
        f"Available keys: {list(data.keys())}"
    )


def read_box_pose_from_xml(robot):
    """Parse the box body `pos` from the robot's scene XML.

    Returns (center_xyz, ) as a length-3 numpy array. The box size is not encoded
    in the XML body pos, so the caller supplies it (default 1.0).
    """
    if robot not in ROBOT_XML_DICT:
        raise KeyError(
            f"Unknown robot '{robot}'. Known: {sorted(ROBOT_XML_DICT.keys())}"
        )
    scene_path = str(ROBOT_XML_DICT[robot])
    tree = ET.parse(scene_path)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.attrib.get("name") == "box":
            pos = body.attrib.get("pos", "0 0 0").split()
            return np.array([float(v) for v in pos], dtype=float), scene_path
    raise ValueError(
        f"No <body name=\"box\"> found in {scene_path}. "
        "Pass --box_pos explicitly instead."
    )


def add_box_to_file(motion_path, save_path, box_center, box_top_z, compressed):
    """Read one motion file, append the box label, and write it out."""
    data, _ = load_motion(motion_path)
    base_pos, base_key = get_base_position(data)
    num_frames = base_pos.shape[0]

    # box_pos_local is the per-frame vector from the robot base to the box center;
    # box_height_global is the (constant) world height of the climb surface (box top).
    data["box_pos_local"] = box_center[None, :] - base_pos
    data["box_height_global"] = np.full((num_frames,), box_top_z, dtype=float)

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_ext = os.path.splitext(save_path)[1].lower()
    if out_ext == ".npz":
        npz_data = {k: v for k, v in data.items() if v is not None}
        if compressed:
            np.savez_compressed(save_path, **npz_data)
        else:
            np.savez(save_path, **npz_data)
    elif out_ext == ".pkl":
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported output format: {out_ext} (expected .npz or .pkl)")
    print(f"  {motion_path} ({num_frames} frames, base='{base_key}') -> {save_path}")


def find_motion_files(src_folder):
    files = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in sorted(filenames):
            if filename.lower().endswith((".npz", ".pkl")):
                files.append(os.path.join(dirpath, filename))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Append a fixed-box label (box_pos_local + box_height_global) to motion file(s)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--motion_path", help="Single .npz/.pkl motion file.")
    src.add_argument(
        "--src_folder",
        help="Folder of .npz/.pkl files (searched recursively). Edited in place.",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Output path (single-file mode only). Defaults to overwriting in place.",
    )
    parser.add_argument(
        "--robot",
        default="adam_sp_box",
        help="Robot whose scene XML holds the box pose (used when --box_pos is omitted).",
    )
    parser.add_argument(
        "--box_pos",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Box center world xy. Overrides the value read from the scene XML.",
    )
    parser.add_argument(
        "--box_size",
        type=float,
        default=1.0,
        help="Box edge length in meters (default 1.0). Top surface = center_z + size/2.",
    )
    parser.add_argument(
        "--box_center_z",
        type=float,
        default=None,
        help="Box center z. Defaults to size/2 (a ground-resting cube) or the XML pos z.",
    )
    parser.add_argument("--compressed", action="store_true", help="Use compressed NPZ output.")
    args = parser.parse_args()

    if args.save_path is not None and args.src_folder is not None:
        parser.error("--save_path only applies to --motion_path (single-file) mode.")

    # Resolve the box world center once (the box is fixed).
    if args.box_pos is not None:
        center_z = args.box_center_z if args.box_center_z is not None else args.box_size / 2.0
        box_center = np.array([args.box_pos[0], args.box_pos[1], center_z], dtype=float)
        print(f"Box center from --box_pos: {box_center.tolist()}")
    else:
        box_center, scene_path = read_box_pose_from_xml(args.robot)
        if args.box_center_z is not None:
            box_center[2] = args.box_center_z
        print(f"Box center from {scene_path}: {box_center.tolist()}")

    box_top_z = box_center[2] + args.box_size / 2.0
    print(f"Box top (climb height) z = {box_top_z:.3f}")

    if args.motion_path is not None:
        save_path = args.save_path or args.motion_path
        add_box_to_file(args.motion_path, save_path, box_center, box_top_z, args.compressed)
        return

    motion_files = find_motion_files(args.src_folder)
    if not motion_files:
        print(f"No .npz/.pkl files found under {args.src_folder}")
        return
    print(f"Found {len(motion_files)} file(s); adding box label in place.")
    ok, failed = 0, 0
    for motion_path in tqdm(motion_files, desc="Adding box label"):
        try:
            add_box_to_file(motion_path, motion_path, box_center, box_top_z, args.compressed)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED {motion_path}: {e}")
    print(f"Done: success {ok} | failed {failed}")


if __name__ == "__main__":
    main()
