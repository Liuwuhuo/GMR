#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np

# adam_sp dof_pos order used by GMR retarget output.
# Names correspond to joints/actuators in adam_sp XML.
ADAM_SP_SOURCE_JOINT_ORDER = [
    "hipPitch_Left",
    "hipRoll_Left",
    "hipYaw_Left",
    "kneePitch_Left",
    "anklePitch_Left",
    "ankleRoll_Left",
    "hipPitch_Right",
    "hipRoll_Right",
    "hipYaw_Right",
    "kneePitch_Right",
    "anklePitch_Right",
    "ankleRoll_Right",
    "waistYaw",
    "waistRoll",
    "waistPitch",
    "shoulderPitch_Left",
    "shoulderRoll_Left",
    "shoulderYaw_Left",
    "elbow_Left",
    "wristYaw_Left",
    "wristPitch_Left",
    "wristRoll_Left",
    "shoulderPitch_Right",
    "shoulderRoll_Right",
    "shoulderYaw_Right",
    "elbow_Right",
    "wristYaw_Right",
    "wristPitch_Right",
    "wristRoll_Right",
]


def build_root_header_cols() -> list[str]:
    return [
        "root_joint_x",
        "root_joint_y",
        "root_joint_z",
        "root_joint_qx",
        "root_joint_qy",
        "root_joint_qz",
        "root_joint_qw",
    ]


def convert_one(
    npz_path: Path,
    csv_path: Path,
    source_joint_order: list[str],
    write_header: bool,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    files = set(data.files)

    # Format A (GMR-style): root_pos/root_rot/dof_pos
    if {"root_pos", "root_rot", "dof_pos"}.issubset(files):
        root_pos = np.asarray(data["root_pos"], dtype=np.float64)  # [T,3]
        root_rot_wxyz = np.asarray(data["root_rot"], dtype=np.float64)  # [T,4]
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)  # [T,N]
        actual_source_joint_order = source_joint_order
    # Format B (rollout-style): qpos + optional joint_names
    elif "qpos" in files:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] < 8:
            raise ValueError(f"{npz_path}: invalid qpos shape {qpos.shape}, expected [T, >=8]")
        root_pos = qpos[:, :3]
        root_rot_wxyz = qpos[:, 3:7]
        dof_pos = qpos[:, 7:]
        if "joint_names" in files:
            jn = [str(x) for x in np.asarray(data["joint_names"]).tolist()]
            # Usually ["root", <dof names...>]
            if len(jn) == dof_pos.shape[1] + 1:
                actual_source_joint_order = jn[1:]
            elif len(jn) == dof_pos.shape[1]:
                actual_source_joint_order = jn
            else:
                actual_source_joint_order = source_joint_order
        else:
            actual_source_joint_order = source_joint_order
    else:
        raise KeyError(
            f"{npz_path}: unsupported npz keys. "
            f"need either {{root_pos, root_rot, dof_pos}} or {{qpos}}, found={data.files}"
        )

    if root_pos.shape[0] != root_rot_wxyz.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError(f"{npz_path}: frame count mismatch")
    if dof_pos.shape[1] != len(actual_source_joint_order):
        raise ValueError(
            f"{npz_path}: dof cols={dof_pos.shape[1]} but source joint order has {len(actual_source_joint_order)}"
        )

    # root quat: npz is wxyz -> target CSV expects qx qy qz qw
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

    # Export joint columns in native source order, no remapping.
    out = np.concatenate([root_pos, root_rot_xyzw, dof_pos], axis=1)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(build_root_header_cols() + actual_source_joint_order)
        writer.writerows(out)


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert adam_sp npz to requested CSV joint schema."
    )
    parser.add_argument("--src_folder", required=True, help="Input folder containing .npz")
    parser.add_argument("--tgt_folder", required=True, help="Output folder for .csv")
    parser.add_argument(
        "--keep_structure",
        action="store_true",
        default=False,
        help="Preserve relative subfolder structure from src_folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing csv files.",
    )
    parser.add_argument(
        "--no_header",
        action="store_true",
        default=False,
        help="Do not write CSV header row.",
    )
    args = parser.parse_args()

    src_folder = Path(args.src_folder).expanduser().resolve()
    tgt_folder = Path(args.tgt_folder).expanduser().resolve()
    if not src_folder.exists():
        raise FileNotFoundError(f"src_folder not found: {src_folder}")
    tgt_folder.mkdir(parents=True, exist_ok=True)

    npz_files = sorted([p for p in src_folder.rglob("*.npz") if p.is_file()])
    if not npz_files:
        print(f"No npz files found under: {src_folder}")
        return

    source_joint_order = ADAM_SP_SOURCE_JOINT_ORDER
    ok = 0
    skipped = 0
    failed = 0
    for npz_path in npz_files:
        rel = npz_path.relative_to(src_folder)
        csv_path = (tgt_folder / rel).with_suffix(".csv") if args.keep_structure else (tgt_folder / f"{npz_path.stem}.csv")

        if csv_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            convert_one(
                npz_path=npz_path,
                csv_path=csv_path,
                source_joint_order=source_joint_order,
                write_header=(not args.no_header),
            )
            ok += 1
        except Exception as e:
            failed += 1
            print(f"Failed: {npz_path} -> {e}")

    print(f"Done. success={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
