#!/usr/bin/env python3
"""
将 G1 的 .csv 动作直接 retarget 到目标机器人（默认 adam_sp）。

输入 csv 格式（与 RoboNaldo motions/*.csv 一致，每行 36 列）：
    root_x, root_y, root_z,
    root_qx, root_qy, root_qz, root_qw,          # 注意：xyzw 顺序
    29 个 Unitree G1 关节角                        # 标准 G1 29DOF 顺序（与 g1_mocap_29dof.xml actuator 顺序一致）

重定向原理（复用 scripts/pt_to_robot_pt.py）：
    1) 从 csv 重建 G1 qpos (T, 36)：pos(3) + quat_wxyz(4) + joint(29)，用 G1 MuJoCo 模型做 FK；
    2) 把 G1 的 link 位姿映射成 IK 用的 human 关键点（pelvis/left_hip/...）；
    3) 逐帧调用 GMR(pt -> 目标机器人).retarget()，得到目标机器人 qpos；
    4) 输出：默认写 adam 的 csv（root pos 3 + root quat xyzw 4 + 目标机器人关节），也可 --output pt。

示例：
    python scripts/g1_csv_to_robot.py motions/right_kick_reference.csv \
        --target_robot adam_sp --input_fps 50 --no-vis \
        --save_path out/right_kick_adam.csv
"""
import argparse
import csv as csv_mod
import pathlib
import sys
import time

import numpy as np
import torch  # noqa: F401  (pt 输出/与 pt_to_robot_pt 复用时需要)
import mujoco as mj

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.params import IK_CONFIG_DICT
import json

# 复用 pt_to_robot_pt 里已经标定好的 G1 link -> human 映射 与 FK->human_data、检查、pt 构建逻辑
from scripts.pt_to_robot_pt import (
    g1_qpos_to_human_data,
    check_human_data,
    build_pt_motion,
)

G1_NQ = 7 + 29  # base(7) + 29 joints


def load_g1_csv_to_qpos(csv_path: pathlib.Path, has_header: bool):
    """读取 G1 csv -> qpos (T, 36)。csv 四元数为 xyzw，转为 MuJoCo wxyz。"""
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv_mod.reader(f)
        for i, row in enumerate(reader):
            if not row or all(c.strip() == "" for c in row):
                continue
            if i == 0 and has_header:
                continue
            # 自动跳过首行表头（首列非数字时）
            if i == 0 and not has_header:
                try:
                    float(row[0])
                except ValueError:
                    continue
            rows.append([float(x) for x in row[:36]])

    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 36:
        raise ValueError(f"{csv_path}: 期望每行 >=36 列, 实际 shape={arr.shape}")
    T = arr.shape[0]

    pos = arr[:, 0:3].copy()
    quat_xyzw = arr[:, 3:7].copy()
    joints = arr[:, 7:36].copy()  # 29

    # xyzw -> wxyz, 顺便归一化
    quat_wxyz = np.zeros((T, 4), dtype=np.float64)
    for i in range(T):
        q = quat_xyzw[i]
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        quat_wxyz[i] = [q[3], q[0], q[1], q[2]]

    qpos = np.concatenate([pos, quat_wxyz, joints], axis=1)
    assert qpos.shape == (T, G1_NQ), qpos.shape
    return qpos


def get_target_joint_names(model) -> list[str]:
    """返回目标机器人 qpos 中 free-root 之后的关节顺序名（与 qpos[7:] 对应）。"""
    names = []
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype == mj.mjtJoint.mjJNT_FREE:
            continue
        nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        names.append(nm if nm is not None else f"joint_{jid}")
    return names


def write_adam_csv(out_path: pathlib.Path, qpos_seq: np.ndarray, joint_names: list[str], write_header: bool):
    """写 csv：root pos(3) + root quat xyzw(4) + 关节角。qpos_seq 的 quat 为 wxyz。"""
    root_pos = qpos_seq[:, 0:3]
    root_quat_wxyz = qpos_seq[:, 3:7]
    dof = qpos_seq[:, 7:]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    out = np.concatenate([root_pos, root_quat_xyzw, dof], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "root_joint_x", "root_joint_y", "root_joint_z",
        "root_joint_qx", "root_joint_qy", "root_joint_qz", "root_joint_qw",
    ] + list(joint_names)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_mod.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerows(out)


def main():
    parser = argparse.ArgumentParser(description="G1 .csv 直接 retarget 到目标机器人（默认 adam_sp）")
    parser.add_argument("csv_file", type=str, help="源 G1 .csv 路径")
    parser.add_argument("--target_robot", type=str, default="adam_sp", help="目标机器人（如 adam_sp）")
    parser.add_argument("--input_fps", type=float, default=50.0, help="输入帧率")
    parser.add_argument("--save_path", type=str, default=None,
                        help="输出路径，默认源同目录 <stem>_<target_robot>.<csv|pt>")
    parser.add_argument("--output", type=str, default="npz",
                        choices=["npz", "csv", "pt"], help="输出格式")
    parser.add_argument("--compress", action="store_true",
                        help="npz 输出使用 np.savez_compressed")
    parser.add_argument("--has_header", action="store_true", help="输入 csv 含表头行")
    parser.add_argument("--no_header", action="store_true", help="输出 csv 不写表头")
    parser.add_argument("--no-vis", dest="no_vis", action="store_true", help="关闭可视化")
    parser.add_argument("--record-video", dest="record_video", action="store_true", help="录屏（需开可视化）")
    parser.add_argument("--video-path", dest="video_path", type=str, default=None, help="录屏输出路径")
    parser.add_argument("--rate-limit", dest="rate_limit", action="store_true", help="按帧率限速播放")
    parser.add_argument("--no-human-scale", dest="no_human_scale", action="store_true",
                        help="缩放比例=1（actual_human_height=config 的 human_height_assumption）")
    parser.add_argument("--scale", type=float, default=None,
                        help="直接指定缩放比例 ratio（覆盖 no-human-scale/默认身高）")
    parser.add_argument("--check-human-data", dest="check_human_data", action="store_true",
                        help="打印首帧 human_data 的 key 与简单统计")
    # ---- 贴地（推荐）：直接用库自带的 human 层逐帧贴地 ----
    parser.add_argument("--fly", action="store_true",
                        help="关闭逐帧贴地，退回库的 _fly 模式（仅第0帧算一次偏移并整段复用）")
    parser.add_argument("--ground_align_scope", choices=["full_body", "feet_hint"],
                        default="full_body",
                        help="逐帧贴地取最低点的范围（默认 full_body）")
    parser.add_argument("--base_height_offset", type=float, default=0.0,
                        help="贴地后整体抬高，补偿脚底碰撞体低于被跟踪脚 body 的量（米）")
    parser.add_argument("--force_feet_same_height", action="store_true",
                        help="强制左右脚目标 z 相同（治两脚不等高）")
    parser.add_argument("--force_feet_level", action="store_true",
                        help="强制脚掌水平（去掉 roll/pitch，只保留 yaw）")
    # ---- 可选：IK 之后再做一次机器人 FK 刚性贴地（一般不需要，配合上面的库内贴地即可）----
    parser.add_argument("--auto_ground", action="store_true",
                        help="[可选] IK 后用目标模型 FK 把机器人最低 body 压到 --ground_z")
    parser.add_argument("--ground_mode", choices=["per_frame", "global"],
                        default="per_frame",
                        help="per_frame=逐帧贴地(默认); global=整段同一平移")
    parser.add_argument("--ground_z", type=float, default=0.0,
                        help="auto_ground 的目标接地高度（默认 0.0）")
    parser.add_argument("--z_offset", type=float, default=0.0,
                        help="整体 z 平移量（负值下移），在 auto_ground 之后再叠加")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_file)
    if not csv_path.is_file():
        print(f"错误: 文件不存在 {csv_path}", file=sys.stderr)
        sys.exit(1)

    qpos_g1 = load_g1_csv_to_qpos(csv_path, has_header=args.has_header)
    T = qpos_g1.shape[0]
    fps = args.input_fps
    base_xy0 = qpos_g1[0, :2].copy()
    print(f"[g1_csv_to_robot] 读取 {csv_path.name}: 帧数={T}, fps={fps}")

    # G1 模型用于 FK
    g1_xml = REPO_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    model_g1 = mj.MjModel.from_xml_path(str(g1_xml))
    data_g1 = mj.MjData(model_g1)

    # 目标机器人 retarget config：优先 pt 专用，否则 smplx
    src_human = "pt" if ("pt" in IK_CONFIG_DICT and args.target_robot in IK_CONFIG_DICT["pt"]) else "smplx"
    with open(IK_CONFIG_DICT[src_human][args.target_robot]) as f:
        ik_cfg = json.load(f)
    assumption = ik_cfg["human_height_assumption"]
    if args.scale is not None:
        actual_human_height = assumption * args.scale
    elif args.no_human_scale:
        actual_human_height = assumption
    else:
        actual_human_height = 1.65
    ratio = actual_human_height / assumption
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human=src_human,
        tgt_robot=args.target_robot,
        base_height_offset=args.base_height_offset,
        force_feet_same_height=args.force_feet_same_height,
        force_feet_level=args.force_feet_level,
    )
    target_model = retarget.model
    # 默认走逐帧贴地（offset_to_ground=True 且 no_fly=True）；--fly 时退回 _fly 模式
    no_fly = not args.fly
    print(f"[g1_csv_to_robot] src_human={src_human}, target={args.target_robot}, "
          f"ratio={actual_human_height:.4f}/{assumption:.4f}={ratio:.4f}")

    if args.check_human_data:
        hd0 = g1_qpos_to_human_data(qpos_g1[0], model_g1, data_g1, base_xy0)
        check_human_data(hd0, args.target_robot, base_xy0, src_human=src_human)

    do_vis = not args.no_vis
    viewer = None
    if do_vis:
        video_path = args.video_path
        if video_path is None and args.record_video:
            video_path = f"videos/{args.target_robot}_{csv_path.stem}.mp4"
        if args.record_video and video_path:
            pathlib.Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        viewer = RobotMotionViewer(
            robot_type=args.target_robot,
            motion_fps=fps,
            record_video=args.record_video,
            video_path=video_path,
        )

    qpos_list = []
    if do_vis:
        i = 0
        while True:
            if not viewer.paused:
                if i >= T:
                    break
                human_data = g1_qpos_to_human_data(qpos_g1[i], model_g1, data_g1, base_xy0)
                qpos, _ = retarget.retarget(
                    human_data, offset_to_ground=True, no_fly=no_fly,
                    ground_align_scope=args.ground_align_scope,
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
        for t in range(T):
            human_data = g1_qpos_to_human_data(qpos_g1[t], model_g1, data_g1, base_xy0)
            qpos, _ = retarget.retarget(
                human_data, offset_to_ground=True, no_fly=no_fly,
                ground_align_scope=args.ground_align_scope,
            )
            qpos_list.append(np.asarray(qpos, dtype=np.float32))

    qpos_seq = np.asarray(qpos_list, dtype=np.float32)

    # 贴地：用目标模型 FK 求机器人本体最低 body z，把最低点压到 ground_z。
    #   per_frame（默认）：逐帧单独压地，支撑脚每帧都贴地、base 不会整体抬高；
    #   global：整段用同一平移量（保留竖直起伏，但非最低帧会离地）。
    if args.auto_ground:
        tmp_data = mj.MjData(target_model)
        min_z_per_frame = np.empty(qpos_seq.shape[0], dtype=np.float64)
        for t in range(qpos_seq.shape[0]):
            tmp_data.qpos[:] = qpos_seq[t]
            mj.mj_forward(target_model, tmp_data)
            min_z_per_frame[t] = float(tmp_data.xpos[1:, 2].min())
        if args.ground_mode == "per_frame":
            qpos_seq[:, 2] += (args.ground_z - min_z_per_frame).astype(np.float32)
            print(f"[g1_csv_to_robot] auto_ground(per_frame): 逐帧最低body z "
                  f"{min_z_per_frame.min():.4f}~{min_z_per_frame.max():.4f} -> {args.ground_z:.4f}")
        else:
            shift = args.ground_z - float(min_z_per_frame.min())
            qpos_seq[:, 2] += shift
            print(f"[g1_csv_to_robot] auto_ground(global): 最低body z="
                  f"{min_z_per_frame.min():.4f} -> {args.ground_z:.4f}, 整体平移 {shift:+.4f}")
    if args.z_offset != 0.0:
        qpos_seq[:, 2] += args.z_offset
        print(f"[g1_csv_to_robot] z_offset: 整体 z {args.z_offset:+.4f}")

    out_path = args.save_path
    if not out_path:
        out_path = str(csv_path.parent /
                       f"{csv_path.stem}_{args.target_robot}.{args.output}")
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output == "npz":
        root_pos = qpos_seq[:, 0:3].astype(np.float64)
        root_rot = qpos_seq[:, 3:7].astype(np.float64)  # wxyz
        dof_pos = qpos_seq[:, 7:].astype(np.float64)
        # dof_vel：有限差分（与帧率相关），末帧补零
        dof_vel = np.zeros_like(dof_pos)
        if T > 1:
            dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) * float(fps)
            dof_vel[-1] = dof_vel[-2]
        save_dict = {
            "fps": np.array([int(round(fps))], dtype=np.int64),
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
        }
        if args.compress:
            np.savez_compressed(str(out_path), **save_dict)
        else:
            np.savez(str(out_path), **save_dict)
        print(f"已保存 npz: {out_path} (帧数={T}, dof={dof_pos.shape[1]})")
    elif args.output == "csv":
        joint_names = get_target_joint_names(target_model)
        if len(joint_names) != qpos_seq.shape[1] - 7:
            print(f"[warn] 关节名数量({len(joint_names)}) 与 qpos 关节维度"
                  f"({qpos_seq.shape[1]-7}) 不一致，仍按 qpos 顺序导出。")
        write_adam_csv(out_path, qpos_seq, joint_names,
                       write_header=(not args.no_header))
        print(f"已保存 csv: {out_path} (帧数={T}, 列数={qpos_seq.shape[1]})")
    else:
        pt_dict = build_pt_motion(qpos_seq, target_model, args.target_robot, fps)
        torch.save(pt_dict, str(out_path))
        print(f"已保存 pt: {out_path} (帧数={T})")


if __name__ == "__main__":
    main()
