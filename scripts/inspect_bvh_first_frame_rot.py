#!/usr/bin/env python3
import argparse

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def main():
    parser = argparse.ArgumentParser(
        description="Inspect post-IK-config human target rotations for one BVH frame."
    )
    parser.add_argument("--bvh_file", required=True, help="Input BVH file path")
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "sfu", "noitom", "mocap", "opt_mocap_footmod"],
        required=True,
        help="BVH loader format.",
    )
    parser.add_argument(
        "--robot",
        required=True,
        help="Target robot name used to select IK config.",
    )
    parser.add_argument(
        "--actual_human_height",
        default=None,
        type=float,
        help="Optional human height for IK config scaling.",
    )
    parser.add_argument(
        "--joints",
        default=None,
        help="Optional comma-separated joint names. Default: joints from IK config.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to inspect (default: 0)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Visualize this one adjusted frame on robot.",
    )
    args = parser.parse_args()

    src_human = f"bvh_{args.format}"
    frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    frame_idx = max(0, min(args.frame, len(frames) - 1))

    retargeter = GMR(
        src_human=src_human,
        tgt_robot=args.robot,
        actual_human_height=(
            args.actual_human_height if args.actual_human_height is not None else actual_human_height
        ),
    )

    # Run target update once to get post-config adjusted human targets.
    retargeter.update_targets(frames[frame_idx], offset_to_ground=True, no_fly=True)
    adjusted = retargeter.scaled_human_data
    if args.joints:
        joint_names = [x.strip() for x in args.joints.split(",") if x.strip()]
    else:
        # Default to all human targets referenced by IK config.
        joint_names = []
        for frame_name, entry in retargeter.ik_match_table1.items():
            _ = frame_name
            human_name = entry[0]
            if human_name not in joint_names:
                joint_names.append(human_name)
        for frame_name, entry in retargeter.ik_match_table2.items():
            _ = frame_name
            human_name = entry[0]
            if human_name not in joint_names:
                joint_names.append(human_name)
        if retargeter.human_root_name not in joint_names:
            joint_names.insert(0, retargeter.human_root_name)

    print(f"bvh_file: {args.bvh_file}")
    print(f"src_human: {src_human}, robot: {args.robot}")
    print(f"frame: {frame_idx}")
    print("format: adjusted target quat (wxyz), adjusted target pos (m)")
    print("-" * 90)

    for name in joint_names:
        if name not in adjusted:
            print(f"{name}: NOT_FOUND")
            continue
        p_m, q_wxyz = adjusted[name]
        print(
            f"{name}: quat=[{q_wxyz[0]: .6f}, {q_wxyz[1]: .6f}, {q_wxyz[2]: .6f}, {q_wxyz[3]: .6f}], "
            f"pos=[{p_m[0]: .6f}, {p_m[1]: .6f}, {p_m[2]: .6f}]"
        )

    if args.visualize:
        qpos, _ = retargeter.retarget(frames[frame_idx], no_fly=True)
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=30,
            transparent_robot=0,
            record_video=False,
            video_path=None,
        )
        while viewer.viewer.is_running():
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=True,
                follow_camera=False,
            )
        viewer.close()


if __name__ == "__main__":
    main()
