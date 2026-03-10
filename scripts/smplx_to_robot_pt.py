import argparse
import pathlib
import os
import time

import numpy as np
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from scipy.spatial.transform import Rotation as R

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "adam_sp", "openloong", "tienkung"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
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
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,  # 默认改为 50 FPS
        help="Target FPS for retargeting and saving (default: 50)",
    )
    # 手动指定 box_pos_local / box_height_global，统一填充到每一帧
    parser.add_argument(
        "--box-pos-local",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.3138, 12.7614, 0.0000],
        help="可选：手动指定 box_pos_local 常量 (x, y, z)，会在 .pt 中每一帧都填同一个值",
    )
    parser.add_argument(
        "--box-height-global",
        type=float,
        default=0,
        help="可选：手动指定 box_height_global 常量（标量），会在 .pt 中每一帧都填同一个值",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="可选：参考的 .pt 文件（如 G1 数据集），从中读取 box_pos_local / box_height_global 第一帧并扩展到所有帧",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # 若未显式指定 --label_file，则默认在 smplx_file 同目录下寻找同名 .pt 作为参考
    if args.label_file is None and args.smplx_file is not None:
        smplx_dir = os.path.dirname(args.smplx_file)
        smplx_stem = os.path.splitext(os.path.basename(args.smplx_file))[0]
        default_label = os.path.join(smplx_dir, smplx_stem + ".pt")
        if os.path.isfile(default_label):
            args.label_file = default_label
            print(f"[smplx_to_robot_pt] 使用同目录下的参考 label_file: {args.label_file}")
        
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 50
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is None:
        # 从 BVH 文件名中提取基本名称（不带扩展名）
        file_basename = os.path.splitext(os.path.basename(args.smplx_file))[0]
        # 创建默认保存路径
        default_dir = "retarget"
        datasets_dir = "ACCAD"
        robot_dir = args.robot
        os.makedirs(default_dir, exist_ok=True)
        args.save_path = os.path.join(default_dir, robot_dir, datasets_dir, f"{file_basename}.pt")
        print(f"未指定保存路径，使用默认路径: {args.save_path}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:  # Only create directory if it's not empty
        os.makedirs(save_dir, exist_ok=True)
    qpos_list = []
    
    # Start the viewer
    i = 0

    while True:
        if robot_motion_viewer.paused is False:
            if args.loop:
                i = (i + 1) % len(smplx_data_frames)
            else:
                i += 1
                if i >= len(smplx_data_frames):
                    # time.sleep(10000.0)  # 无限期休眠，不占用 CPU
                    # pass
                    break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            # time.sleep(0.1)
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]

        offset_to_ground = True
        # retarget
        qpos, qvel = retarget.retarget(smplx_data, offset_to_ground)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )
        
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        import pickle
        from smplx_pkl_to_robot import build_pt_motion

        # 统一约定：以 --save_path 的 basename 作为前缀，保存一份 .pkl（旧格式）和一份 .pt（标准 pt 格式）
        base, _ = os.path.splitext(args.save_path)
        pkl_path = base + ".pkl"
        pt_path = base + ".pt"

        # ===== 1) 保存旧的 pickle 结构（root_pos/root_rot/dof_pos）=====
        root_pos = np.array([q[:3] for q in qpos_list])
        # 存 xyzw：从 MuJoCo wxyz 转为 xyzw
        root_rot = np.array([[q[4], q[5], q[6], q[3]] for q in qpos_list])
        dof_pos = np.array([q[7:] for q in qpos_list])
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved pickle motion to {pkl_path}")

        # ===== 2) 保存与其他脚本一致的 .pt（含标准 labels）=====
        # build_pt_motion 需要 qpos_list 和 qvel_list；这里 qvel 仅用于占位，实际内部用中心差分
        qpos_np_list = [np.asarray(q, dtype=np.float32) for q in qpos_list]
        if len(qpos_np_list) == 0:
            print("[smplx_to_robot_pt] qpos_list 为空，跳过 .pt 保存")
        else:
            nq = qpos_np_list[0].shape[0]
            qvel_list = [np.zeros(nq - 1, dtype=np.float32) for _ in qpos_np_list]

            pt_dict = build_pt_motion(qpos_np_list, qvel_list, retarget.model, args.robot, aligned_fps)

            # === 1. 字段转换：将 build_pt_motion 的输出转换为目标格式 ===
            
            # base_height: 从 base_position 的 z 轴提取
            if "base_position" in pt_dict:
                pt_dict["base_height"] = pt_dict["base_position"][:, 2].clone()
            
            # base_quat: 从 base_pose (欧拉角 xyz) 转换回四元数 xyzw
            if "base_pose" in pt_dict:
                base_pose = pt_dict["base_pose"].numpy()
                quat_xyzw = np.zeros((base_pose.shape[0], 4), dtype=np.float32)
                for i in range(base_pose.shape[0]):
                    r = R.from_euler("xyz", base_pose[i])
                    q = r.as_quat()  # 返回 xyzw 格式
                    quat_xyzw[i] = q
                pt_dict["base_quat"] = torch.from_numpy(quat_xyzw)
            
            # base_linear_velocity: 直接使用 base_velocity
            if "base_velocity" in pt_dict:
                pt_dict["base_linear_velocity"] = pt_dict["base_velocity"].clone()

            # === 2. 对 adam_sp 系列，裁剪 link_position 到 6 个 ===
            if args.robot.startswith("adam_sp") and "link_position" in pt_dict:
                idx_order = [12, 16, 3, 6, 7, 7]
                if pt_dict["link_position"].shape[1] >= 8:
                    pt_dict["link_position"] = pt_dict["link_position"][:, idx_order, :].contiguous()

            # === 3. 目标帧数 ===
            T = pt_dict["base_position"].shape[0] if "base_position" in pt_dict else len(qpos_np_list)

            # === 4. 注入 box 字段 (从 label_file 或命令行) ===
            if args.label_file is not None and os.path.isfile(args.label_file):
                label_src = torch.load(args.label_file, map_location="cpu", weights_only=False)
                if isinstance(label_src, dict):
                    label_np = {
                        k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
                        for k, v in label_src.items()
                    }
                    if "box_pos_local" in label_np:
                        arr = np.asarray(label_np["box_pos_local"], dtype=np.float32)
                        if arr.ndim >= 2 and arr.shape[1] >= 3:
                            first = arr[0, :3]
                        elif arr.ndim == 1 and arr.size >= 3:
                            first = arr[:3]
                        else:
                            first = None
                        if first is not None:
                            box_pos = np.tile(first.reshape(1, 3), (T, 1))
                            pt_dict["box_pos_local"] = torch.from_numpy(box_pos.astype(np.float32))
                    if "box_height_global" in label_np:
                        arr_h = np.asarray(label_np["box_height_global"], dtype=np.float32)
                        if arr_h.size > 0:
                            h0 = float(arr_h.ravel()[0])
                            box_h = np.full((T,), h0, dtype=np.float32)
                            pt_dict["box_height_global"] = torch.from_numpy(box_h)

            if args.box_pos_local is not None and "box_pos_local" not in pt_dict:
                box_pos = np.tile(
                    np.asarray(args.box_pos_local, dtype=np.float32).reshape(1, 3),
                    (T, 1),
                )
                pt_dict["box_pos_local"] = torch.from_numpy(box_pos)
            if args.box_height_global is not None and "box_height_global" not in pt_dict:
                box_h = np.full((T,), float(args.box_height_global), dtype=np.float32)
                pt_dict["box_height_global"] = torch.from_numpy(box_h)

            # === 5. 定义目标白名单字段 ===
            desired_keys = {
                "base_position",
                "base_quat",
                "base_height",
                "base_linear_velocity",
                "base_angular_velocity",
                "joint_position",
                "joint_velocity",
                "link_position",
                "box_pos_local",
                "box_height_global"
            }

            # === 6. 删除多余字段 ===
            keys_to_remove = [k for k in list(pt_dict.keys()) if k not in desired_keys]
            for k in keys_to_remove:
                del pt_dict[k]

            # === 7. 保存并确认 ===
            final_keys = list(pt_dict.keys())
            missing_keys = desired_keys - set(final_keys)
            
            torch.save(pt_dict, pt_path)
            print(f"Saved .pt to {pt_path}")
            print(f"  >>> 保存帧数：{T} 帧 (FPS={aligned_fps}) <<<")
            print(f"  最终字段 ({len(final_keys)}): {final_keys}")
            if missing_keys:
                print(f"  [Info] 以下字段未生成（取决于机器人类型）：{missing_keys}")
            
      
    
    robot_motion_viewer.close()
