#!/usr/bin/env python3
"""将 G1 的 .npz 动作直接 retarget 为目标机器人 .npz（默认 adam_sp）。"""

import argparse
import json
import pathlib
import sys
import time

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.params import IK_CONFIG_DICT

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

WAIST_ROLL_IDX, WAIST_PITCH_IDX = 13, 14
G1_NQ = 7 + 29

HUMAN_TO_G1_BODY = {
    "pelvis": "keyframe_pelvis_link",
    "left_hip": "keyframe_left_hip_link",
    "left_knee": "keyframe_left_knee_link",
    "left_foot": "keyframe_left_ankle_link",
    "right_hip": "keyframe_right_hip_link",
    "right_knee": "keyframe_right_knee_link",
    "right_foot": "keyframe_right_ankle_link",
    "spine2": "torso_link",
    "left_shoulder": "keyframe_left_shoulder_link",
    "left_elbow": "keyframe_left_elbow_link",
    "left_wrist": "left_palm_link",
    "right_shoulder": "keyframe_right_shoulder_link",
    "right_elbow": "keyframe_right_elbow_link",
    "right_wrist": "right_palm_link",
}


def _pick_key(data, candidates):
    for k in candidates:
        if k in data:
            return k
    return None


def _infer_fps(data, fallback=50.0):
    keys = (
        "framerate",
        "fps",
        "frame_rate",
        "frame_rate_hz",
        "frequency",
        "freq",
        "hz",
        "sampling_rate",
        "dt",
    )
    for k in keys:
        if k not in data:
            continue
        v = np.asarray(data[k], dtype=np.float64)
        if v.size == 0:
            continue
        x = float(v.flat[0])
        if k == "dt":
            if x > 0:
                return 1.0 / x
        else:
            return x
    return fallback


def load_g1_npz_to_qpos(npz_path, fps_fallback=50.0, quat_xyzw=False):
    data = np.load(str(npz_path), allow_pickle=True)
    keys = set(data.files)

    k_pos = _pick_key(keys, ("base_pos_w", "base_position", "root_pos"))
    k_quat = _pick_key(keys, ("base_quat_w", "base_quat", "root_rot"))
    k_joint = _pick_key(keys, ("joint_pos", "joint_position", "dof_pos"))
    if k_pos is None or k_quat is None or k_joint is None:
        raise KeyError(
            f"缺少必要字段，当前 keys={sorted(keys)}，"
            "至少需要 base_pos/base_quat/joint_pos（或同义字段）"
        )

    base_pos = np.asarray(data[k_pos], dtype=np.float64).copy()
    base_quat = np.asarray(
        data[k_quat], dtype=np.float64
    ).reshape(base_pos.shape[0], 4)
    joint_pos = np.asarray(data[k_joint], dtype=np.float64)

    T = base_pos.shape[0]
    if joint_pos.shape[0] != T:
        raise ValueError(f"帧数不一致: base={T}, joint={joint_pos.shape[0]}")

    quats_wxyz = np.zeros((T, 4), dtype=np.float64)
    for i in range(T):
        if quat_xyzw:
            q_xyzw = base_quat[i]
        else:
            q_wxyz = base_quat[i]
            q_xyzw = np.array(
                [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]],
                dtype=np.float64,
            )
        q_xyzw = q_xyzw / max(np.linalg.norm(q_xyzw), 1e-12)
        quats_wxyz[i] = np.array(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
            dtype=np.float64,
        )

    if joint_pos.shape[1] == 29:
        dof_full = joint_pos
    elif joint_pos.shape[1] == 27:
        dof_full = np.zeros((T, 29), dtype=np.float64)
        dof_full[:, :WAIST_ROLL_IDX] = joint_pos[:, :WAIST_ROLL_IDX]
        dof_full[:, WAIST_ROLL_IDX] = 0.0
        dof_full[:, WAIST_PITCH_IDX] = 0.0
        dof_full[:, WAIST_PITCH_IDX + 1:] = joint_pos[:, WAIST_ROLL_IDX:]
    else:
        raise ValueError(f"joint 维度不支持: {joint_pos.shape[1]} (期望 27 或 29)")

    qpos = np.concatenate([base_pos, quats_wxyz, dof_full], axis=1)
    assert qpos.shape == (T, G1_NQ)
    fps = _infer_fps(data, fallback=fps_fallback)
    return qpos, fps


def g1_qpos_to_human_data(qpos_one, model_g1, data_g1, base_xy0):
    data_g1.qpos[:] = qpos_one
    data_g1.qvel[:] = 0
    mj.mj_forward(model_g1, data_g1)

    human_data = {}
    for human_name, body_name in HUMAN_TO_G1_BODY.items():
        bid = mj.mj_name2id(model_g1, mj.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            continue
        pos = data_g1.xpos[bid].copy()
        pos[0] -= base_xy0[0]
        pos[1] -= base_xy0[1]
        xmat = np.array(data_g1.xmat[bid]).reshape(3, 3)
        quat_xyzw = R.from_matrix(xmat).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )
        human_data[human_name] = [pos, quat_wxyz]
    return human_data


def get_target_joint_names(model):
    names = []
    for i in range(model.nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        names.append(name if name is not None else f"actuator_{i}")
    return names


def main():
    parser = argparse.ArgumentParser(
        description="G1 .npz 直接 retarget 到目标机器人 .npz（保持 labels 顺序一致）"
    )
    parser.add_argument("npz_file", type=str, help="源 G1 .npz 路径")
    parser.add_argument(
        "--target_robot",
        type=str,
        default="adam_sp",
        help="目标机器人",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="输出 .npz 路径",
    )
    parser.add_argument("--fps", type=float, default=None, help="覆盖输出 fps")
    parser.add_argument(
        "--quat-xyzw",
        action="store_true",
        help="源 base_quat 是 xyzw 时启用",
    )
    parser.add_argument("--no-vis", action="store_true", help="关闭 retarget 可视化")
    parser.add_argument("--rate-limit", action="store_true", help="可视化按 fps 限速")
    parser.add_argument("--no-human-scale", action="store_true", help="不做人高缩放")
    parser.add_argument("--scale", type=float, default=None, help="直接指定缩放 ratio")
    parser.add_argument(
        "--align-ground",
        action="store_true",
        help="启用地面对齐（默认关闭，保留输入的绝对高度轨迹）",
    )
    parser.add_argument(
        "--no-fly",
        action="store_true",
        help="与 --align-ground 搭配，使用逐帧贴地模式",
    )
    args = parser.parse_args()

    npz_path = pathlib.Path(args.npz_file)
    if not npz_path.is_file():
        print(f"错误: 文件不存在 {npz_path}", file=sys.stderr)
        sys.exit(1)

    fps_fallback = args.fps if args.fps is not None else 50.0
    qpos_g1, fps = load_g1_npz_to_qpos(
        npz_path,
        fps_fallback=fps_fallback,
        quat_xyzw=args.quat_xyzw,
    )
    if args.fps is not None:
        fps = args.fps

    T = qpos_g1.shape[0]
    base_xy0 = qpos_g1[0, :2].copy()

    g1_xml = REPO_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    model_g1 = mj.MjModel.from_xml_path(str(g1_xml))
    data_g1 = mj.MjData(model_g1)

    src_human = (
        "pt"
    )
    with open(
        IK_CONFIG_DICT[src_human][args.target_robot],
        "r",
        encoding="utf-8",
    ) as f:
        ik_cfg = json.load(f)
    assumption = ik_cfg["human_height_assumption"]
    if args.scale is not None:
        actual_human_height = assumption * args.scale
    elif args.no_human_scale:
        actual_human_height = assumption
    else:
        actual_human_height = 1.65

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human=src_human,
        tgt_robot=args.target_robot,
    )
    target_model = retarget.model

    qpos_list = []
    do_vis = not args.no_vis
    if do_vis:
        viewer = RobotMotionViewer(
            robot_type=args.target_robot,
            motion_fps=fps,
            record_video=False,
            video_path=None,
        )
        i = 0
        while viewer.viewer.is_running():
            if not viewer.paused:
                if i >= T:
                    break
                human_data = g1_qpos_to_human_data(
                    qpos_g1[i], model_g1, data_g1, base_xy0
                )
                qpos, _ = retarget.retarget(
                    human_data,
                    offset_to_ground=args.align_ground,
                    no_fly=args.no_fly,
                    apply_ground_alignment=args.align_ground,
                )
                qpos = np.asarray(qpos, dtype=np.float32)
                qpos_list.append(qpos)
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data,
                    rate_limit=args.rate_limit,
                    follow_camera=True,
                )
                i += 1
            else:
                time.sleep(0.05)
        viewer.close()
    else:
        for i in range(T):
            human_data = g1_qpos_to_human_data(
                qpos_g1[i], model_g1, data_g1, base_xy0
            )
            qpos, _ = retarget.retarget(
                human_data,
                offset_to_ground=args.align_ground,
                no_fly=args.no_fly,
                apply_ground_alignment=args.align_ground,
            )
            qpos_list.append(np.asarray(qpos, dtype=np.float32))

    qpos_arr = np.asarray(qpos_list, dtype=np.float32)
    base_pos_w = qpos_arr[:, :3]
    base_quat_w = qpos_arr[:, 3:7]
    joint_pos = qpos_arr[:, 7:]
    joint_names = np.asarray(get_target_joint_names(target_model))

    out_path = args.save_path
    if out_path is None:
        out_path = str(
            npz_path.parent / f"{npz_path.stem}_{args.target_robot}.npz"
        )
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        base_pos_w=base_pos_w,
        base_quat_w=base_quat_w,
        joint_pos=joint_pos,
        joint_names=joint_names,
        labels=joint_names,
        framerate=np.array(fps, dtype=np.float64),
    )
    print(f"已保存: {out_path} (frames={qpos_arr.shape[0]}, fps={fps})")
    print(f"labels(joint_names) 数量: {len(joint_names)}")


if __name__ == "__main__":
    main()

