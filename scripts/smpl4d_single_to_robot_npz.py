#!/usr/bin/env python3
import argparse
import os
import pathlib

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import get_smplx_data


def _to_rotvec(arr):
    """Convert rotation matrix (..., 3, 3) to rotvec (..., 3)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[-2:] != (3, 3):
        raise ValueError(f"Expect rotation matrices (...,3,3), got {arr.shape}")
    flat = arr.reshape(-1, 3, 3)
    out = R.from_matrix(flat).as_rotvec()
    return out.reshape(*arr.shape[:-2], 3)


def load_4d_single_as_smplx(npz_path, smplx_body_model_path):
    """
    Parse 4D Human single-frame npz:
      keys: global_orient(1,3,3), body_pose(23,3,3), betas(10,), cam_t(3,)
    Convert to SMPL-X format expected by GMR pipeline.
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    required = {"global_orient", "body_pose", "betas"}
    missing = required - keys
    if missing:
        raise KeyError(f"Missing required keys: {sorted(missing)}")

    global_orient_m = np.asarray(data["global_orient"], dtype=np.float64)
    body_pose_m = np.asarray(data["body_pose"], dtype=np.float64)
    betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)
    cam_t = np.asarray(data["cam_t"], dtype=np.float64).reshape(-1) if "cam_t" in data else np.zeros(3)

    # Normalize to one-frame batch.
    # global_orient: (1,3,3) or (3,3)
    if global_orient_m.shape == (3, 3):
        global_orient_m = global_orient_m[None, ...]
    if global_orient_m.shape != (1, 3, 3):
        raise ValueError(
            "global_orient expected (1,3,3) or (3,3), "
            f"got {global_orient_m.shape}"
        )

    # body_pose: (23,3,3) or (1,23,3,3)
    if body_pose_m.shape == (23, 3, 3):
        body_pose_m = body_pose_m[None, ...]
    if body_pose_m.shape != (1, 23, 3, 3):
        raise ValueError(
            "body_pose expected (23,3,3) or (1,23,3,3), "
            f"got {body_pose_m.shape}"
        )

    root_orient = _to_rotvec(global_orient_m)  # (1,3)
    pose_body = _to_rotvec(body_pose_m).reshape(1, 23 * 3)  # (1,69)
    # GMR uses first 63 dims (21 body joints) for SMPL-X body_pose.
    pose_body = pose_body[:, :63]

    betas16 = np.zeros(16, dtype=np.float64)
    betas16[: min(16, betas.shape[0])] = betas[:16]

    # For single-frame static use-case, set translation to zero by default.
    # If you want keep camera translation influence, pass --use_cam_t.
    trans = np.zeros((1, 3), dtype=np.float64)

    smplx_data = {
        "pose_body": pose_body.astype(np.float32),
        "root_orient": root_orient.astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": betas16.astype(np.float32),
        "gender": np.array("neutral"),
        "mocap_frame_rate": np.array(30.0, dtype=np.float32),
    }

    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender="neutral",
        use_pca=False,
    )
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(1, 45).float(),
        right_hand_pose=torch.zeros(1, 45).float(),
        jaw_pose=torch.zeros(1, 3).float(),
        leye_pose=torch.zeros(1, 3).float(),
        reye_pose=torch.zeros(1, 3).float(),
        return_full_pose=True,
    )

    human_height = 1.66 + 0.1 * float(smplx_data["betas"][0])
    return smplx_data, body_model, smplx_output, human_height, cam_t


def main():
    parser = argparse.ArgumentParser(
        description="4D-Human single-frame SMPLX NPZ -> robot NPZ"
    )
    parser.add_argument(
        "--smpl4d_npz", required=True, help="Input 4D Human SMPLX npz"
    )
    parser.add_argument("--robot", default="adam_sp", help="Target robot")
    parser.add_argument("--save_path", default=None, help="Output NPZ path")
    parser.add_argument(
        "--repeat_frames",
        type=int,
        default=250,
        help="Repeat the single frame to this length",
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Output FPS")
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/smpl4d_single.mp4")
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--compressed", action="store_true", default=False)
    parser.add_argument(
        "--compute_local_body_pos", action="store_true", default=False
    )
    parser.add_argument("--height_adjust", action="store_true", default=False)
    parser.add_argument("--perframe_adjust", action="store_true", default=False)
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    smplx_folder = here / ".." / "assets" / "body_models"

    if args.save_path is None:
        stem = pathlib.Path(args.smpl4d_npz).stem
        args.save_path = str(
            pathlib.Path("retarget") / args.robot / "smpl4d" / f"{stem}.npz"
        )
        print(f"未指定保存路径，使用默认路径: {args.save_path}")
    out_dir = os.path.dirname(args.save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    (
        smplx_data,
        body_model,
        smplx_output,
        human_height,
        _cam_t,
    ) = load_4d_single_as_smplx(args.smpl4d_npz, smplx_folder)

    # Build one human frame and repeat.
    single_frame = get_smplx_data(smplx_data, body_model, smplx_output, 0)
    num_frames = int(args.repeat_frames)
    if num_frames <= 0:
        raise ValueError(f"--repeat_frames must be positive, got {num_frames}")
    smplx_data_frames = [single_frame for _ in range(num_frames)]

    output_fps = int(round(args.fps))
    retargeter = GMR(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=output_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    qpos_list = []
    qvel_list = []
    for i in tqdm(range(num_frames), desc="Retargeting"):
        qpos, qvel = retargeter.retarget(
            smplx_data_frames[i], offset_to_ground=True, no_fly=True
        )
        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )
        qpos_list.append(qpos.copy())
        qvel_list.append(qvel.copy())

    qpos_arr = np.asarray(qpos_list)
    qvel_arr = np.asarray(qvel_list)
    # Hard static guarantee for single-frame repeat use-case.
    qpos_arr[:] = qpos_arr[0]
    qvel_arr[:] = qvel_arr[0]

    root_pos = qpos_arr[:, :3]
    root_rot = qpos_arr[:, 3:7]
    dof_pos = qpos_arr[:, 7:]
    dof_vel = qvel_arr[:, 6:]

    local_body_pos = None
    body_names = None
    if args.compute_local_body_pos:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kin = KinematicsModel(retargeter.xml_file, device=device)
        identity_root_pos = torch.zeros((num_frames, 3), device=device)
        identity_root_rot = torch.zeros((num_frames, 4), device=device)
        identity_root_rot[:, 0] = 1.0
        local_body_pos, _ = kin.forward_kinematics(
            identity_root_pos,
            identity_root_rot,
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
        )
        body_names = kin.body_names

        if args.height_adjust:
            body_pos, _ = kin.forward_kinematics(
                torch.from_numpy(root_pos).to(
                    device=device, dtype=torch.float
                ),
                torch.from_numpy(root_rot).to(
                    device=device, dtype=torch.float
                ),
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
            )
            if not args.perframe_adjust:
                lowest_height = torch.min(body_pos[..., 2]).item()
                root_pos[:, 2] = root_pos[:, 2] - lowest_height
            else:
                for i in range(root_pos.shape[0]):
                    lowest_body_part = torch.min(body_pos[i, :, 2])
                    root_pos[i, 2] = root_pos[i, 2] - lowest_body_part

        local_body_pos = local_body_pos.detach().cpu().numpy()

    save_dict = {
        "fps": np.array([output_fps]),
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
    viewer.close()


if __name__ == "__main__":
    main()
