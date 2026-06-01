import argparse
import os
import pathlib
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)

from rich import print

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "unitree_h1",
            "unitree_h1_2",
            "booster_t1",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "kuavo_s45",
            "hightorque_hi",
            "galaxea_r1pro",
            "berkeley_humanoid_lite",
            "booster_k1",
            "pnd_adam_lite",
            "adam_sp",
            "openloong",
            "tienkung",
        ],
        default="adam_sp",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Output .npz path. Default: retarget/<robot>/smplx/<stem>.npz (.pkl is rewritten to .npz).",
    )

    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--drop_first_frame",
        action="store_true",
        default=False,
        help="Drop the first retargeted frame (useful when frame 0 is unstable).",
    )
    parser.add_argument(
        "--drop_first_n_frames",
        type=int,
        default=0,
        help="Drop first N retargeted frames (applied before saving).",
    )
    parser.add_argument(
        "--export_motion_fields",
        action="store_true",
        default=False,
        help=(
            "Also export motion fields: framerate, joint_names, joint_pos, "
            "base_pos_w, base_quat_w."
        ),
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        default=False,
        help="Use compressed npz (np.savez_compressed).",
    )

    args = parser.parse_args()

    if args.save_path is None:
        smplx_basename = os.path.splitext(os.path.basename(args.smplx_file))[0]
        default_dir = "retarget"
        robot_dir = args.robot
        subdir = "smplx"
        os.makedirs(default_dir, exist_ok=True)
        args.save_path = os.path.join(
            default_dir, robot_dir, subdir, f"{smplx_basename}.npz"
        )
        print(f"未指定保存路径，使用默认路径: {args.save_path}")
    else:
        sp = pathlib.Path(args.save_path)
        if sp.suffix.lower() == ".pkl":
            args.save_path = str(sp.with_suffix(".npz"))
            print(f"保存格式为 npz，已更新路径: {args.save_path}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )

    # 与数据 mocap_frame_rate 一致传入，避免 get_smplx_data_offline_fast 内降采样。
    native_fps = float(np.asarray(smplx_data["mocap_frame_rate"]).reshape(-1)[0])
    smplx_data_frames, _ = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=native_fps
    )

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=native_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=(
            f"videos/{args.robot}_"
            f"{args.smplx_file.split('/')[-1].split('.')[0]}.mp4"
        ),
    )

    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    qpos_list = []
    qvel_list = []
    i = 0

    while True:
        robot_motion_viewer.wait_while_paused()
        if (not args.loop) and i >= len(smplx_data_frames):
            break

        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        smplx_frame = smplx_data_frames[i]

        offset_to_ground = True
        qpos, qvel = retarget.retarget(
            smplx_frame, offset_to_ground=offset_to_ground, no_fly=True
        )

        if args.drop_first_frame and i == 0:
            if args.loop:
                i = (i + 1) % len(smplx_data_frames)
            else:
                i += 1
            continue

        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )

        qpos_list.append(qpos.copy())
        qvel_list.append(qvel.copy())

        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1

    local_body_pos = None
    body_names = None

    qpos_arr = np.asarray(qpos_list)
    qvel_arr = np.asarray(qvel_list)
    drop_n = max(0, int(args.drop_first_n_frames))
    if drop_n > 0:
        if qpos_arr.shape[0] <= drop_n:
            raise ValueError(
                f"Cannot drop first {drop_n} frame(s): total saved frames = "
                f"{qpos_arr.shape[0]}"
            )
        qpos_arr = qpos_arr[drop_n:]
        qvel_arr = qvel_arr[drop_n:]

    motion_fps_save = float(native_fps)
    root_pos = qpos_arr[:, :3]
    root_rot = qpos_arr[:, 3:7]  # wxyz, same as bvh_to_robot_npz.py
    dof_pos = qpos_arr[:, 7:]
    dof_vel = qvel_arr[:, 6:]

    if args.export_motion_fields:
        motor_name_by_id = sorted(
            retarget.robot_motor_names.items(), key=lambda kv: kv[1]
        )
        joint_names = [name for name, _ in motor_name_by_id]
        save_dict = {
            "framerate": np.array([motion_fps_save], dtype=np.float64),
            "joint_names": np.asarray(joint_names, dtype=object),
            "joint_pos": dof_pos,
            "base_pos_w": root_pos,
            "base_quat_w": root_rot,
        }
    else:
        save_dict = {
            "fps": np.array([motion_fps_save], dtype=np.float64),
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
        }
        if local_body_pos is not None:
            save_dict["local_body_pos"] = local_body_pos
        if body_names is not None:
            save_dict["link_body_list"] = body_names

    if args.compressed:
        np.savez_compressed(args.save_path, **save_dict)
    else:
        np.savez(args.save_path, **save_dict)
    print(f"Saved: {args.save_path}")

    robot_motion_viewer.close()
