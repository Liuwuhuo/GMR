#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from general_motion_retargeting import RobotMotionViewer


def load_motion_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter="\t")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 32:
        raise ValueError(f"Expected >=32 columns, got {data.shape[1]}")
    return data


def txt_to_adam_sp_qpos_frames(motion_txt: np.ndarray) -> np.ndarray:
    """
    motion txt column layout (32 cols):
      waist3, head2, left_arm7, left_index6, right_arm7, right_index6, tail1
    Map to adam_sp qpos:
      [root_pos(3), root_rot_wxyz(4), dof_pos(29)]
    We only fill:
      dof[12:15]=waist3, dof[15:22]=left_arm7, dof[22:29]=right_arm7
    Other DoFs remain 0.
    """
    n = motion_txt.shape[0]
    qpos = np.zeros((n, 3 + 4 + 29), dtype=np.float64)
    qpos[:, 3] = 1.0  # root quat w=1

    waist = motion_txt[:, 0:3]
    left_arm = motion_txt[:, 5:12]
    right_arm = motion_txt[:, 18:25]

    dof = qpos[:, 7:]
    dof[:, 12:15] = waist
    dof[:, 15:22] = left_arm
    dof[:, 22:29] = right_arm
    return qpos


def main():
    parser = argparse.ArgumentParser(description="Visualize custom motion txt in MuJoCo")
    parser.add_argument("--motion_txt", required=True, help="Path to motion txt")
    parser.add_argument("--robot", default="adam_sp", choices=["adam_sp"])
    parser.add_argument("--fps", type=float, default=400.0, help="Playback fps")
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", default="videos/motion_txt_vis.mp4")
    parser.add_argument("--rate_limit", action="store_true", default=True)
    args = parser.parse_args()

    motion_path = Path(args.motion_txt).expanduser().resolve()
    if not motion_path.exists():
        raise FileNotFoundError(f"motion txt not found: {motion_path}")

    motion_txt = load_motion_txt(motion_path)
    qpos_frames = txt_to_adam_sp_qpos_frames(motion_txt)

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=int(round(args.fps)),
        transparent_robot=0,
        camera_follow=False,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    frame_idx = 0
    total = qpos_frames.shape[0]
    while True:
        viewer.wait_while_paused()
        q = qpos_frames[frame_idx]
        viewer.step(
            root_pos=q[:3],
            root_rot=q[3:7],
            dof_pos=q[7:],
            rate_limit=args.rate_limit,
            follow_camera=False,
        )
        frame_idx += 1
        if frame_idx >= total:
            frame_idx = 0


if __name__ == "__main__":
    main()
