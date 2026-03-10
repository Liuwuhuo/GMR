#!/usr/bin/env python3
"""
将 G1 格式的 .pt 直接 retarget 到目标机器人的 .pt，无需经过 SMPL。
用 G1 的 link 姿态作为「人体」关键点，走现有 smplx->robot 的 IK 得到目标机器人 qpos，
再写出目标 .pt，并原样保留源 .pt 中的额外字段（如 box_height_global、box_pos_local、base_height 等）。

Retarget 过程（简要）：
  1) 加载源 G1 .pt：base_position、base_quat(或 base_pose)、joint_position(27/29)。
     base_quat 默认按 xyzw 解析（PhysHSI 常见）；有 base_height 时 base z 用 base_height，否则用 link/base 最低点接地。
  2) 重建 G1 qpos (T, 36)：pos(3) + quat_wxyz(4) + joint(29)，并用 G1 的 MuJoCo 模型做 FK。
  3) 将 G1 的 body 姿态映射到 IK 用的 human 名（pelvis, left_hip, left_knee, left_foot, ...）：
     body 名直接从 smplx_to_g1.json 的 ik_match_table1/2 读取，以保证与 SMPL-X→G1 标定时用的 G1 body 完全一致。
  4) 对每一帧调用 GMR(pt/smplx -> 目标机器人).retarget(human_data)，得到目标机器人的 qpos。
  5) 用 build_pt_motion 从 qpos 序列生成目标 .pt（base_*、joint_*、link_*）；若目标为 adam_sp 则 link 只保留 6 个。
  6) 写入 labels（与目标机器人 dof 顺序一致），并原样拷贝源 .pt 中的额外键（如 box_*、base_height）。

注意（human_data 与 smplx config 的差异）：
  - 当前使用 smplx/pt→目标机器人的 IK config，但人体输入来自 G1 FK 的 link 位姿，不是 SMPL-X。
  - SMPL-X 提供的是「关节」位置与朝向；G1 提供的是「连杆」位置与朝向，语义不同，直接套用 smplx 的
    pos/rot_offset 可能导致错位。建议加上 --no-human-scale 避免按人体身高再次缩放（G1 已是机器人尺度）。
  - 若需完全正确的 retarget，应使用专为 G1→目标机器人标定的 IK config（如 g1_to_*），当前仓库暂无。

Retarget 中的缩放（motion_retarget.py）：
  - ratio = actual_human_height / human_height_assumption（config 里的值）
  - human_scale_table 中每个 body 的系数 = 原表值 * ratio
  - scale_human_data：root 位置 *= scale_table[root]；其余 body 的 (pos - root_pos) *= scale_table[body]，再加到 scaled_root 上
  - 等价于所有位置做「以 root 为参考的均匀缩放」：pos' = root_pos*ratio + (pos - root_pos)*ratio = ratio * pos（相对原点）
  - 故最终位置整体乘 ratio。用 --scale 可指定 ratio（例如 --scale 1.05 表示放大 5%）。
"""
import argparse
import pathlib
import sys
import time
import json

import numpy as np
import torch
import mujoco as mj
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# 与 datasets/g1_dof27_data/.../output/data.pt 一致的 .pt 字段；27 关节不含 waist_roll/waist_pitch
WAIST_ROLL_IDX, WAIST_PITCH_IDX = 13, 14
G1_NQ = 7 + 29


# ====================== G1 human_data 映射：从 smplx_to_g1.json 读取 ======================

HUMAN_TO_G1_BODY = {
    # 下半身
    "pelvis":        "keyframe_pelvis_link",

    # 这几个你可以按自己的关节/连杆选，比如 hipRollLeft、left_hip_pitch_link 等
    "left_hip":      "keyframe_left_hip_link",
    "left_knee":     "keyframe_left_knee_link",
    "left_foot":     "keyframe_left_ankle_link",     # TODO: 换成你在 XML 里希望对齐的 G1 脚部 body 名

    "right_hip":     "keyframe_right_hip_link",
    "right_knee":    "keyframe_right_knee_link",
    "right_foot":    "keyframe_right_ankle_link",    # TODO: 同上，右脚 body 名

    # 躯干 / 头部
    "spine2":        "torso_link",              # 或者你在 smplx_to_g1.json 里用到的躯干 body

    # 上肢（同样可以按你在配置里用的 body 名改）
    "left_shoulder": "keyframe_left_shoulder_link",
    "left_elbow":    "keyframe_left_elbow_link",
    "left_wrist":    "left_palm_link",

    "right_shoulder": "keyframe_right_shoulder_link",
    "right_elbow":    "keyframe_right_elbow_link",
    "right_wrist":    "right_palm_link",
}


def load_g1_pt_to_qpos(pt_path, fps_fallback=50.0, base_quat_xyzw=True):
    """从 G1 .pt 加载为 qpos 序列 (T, 36) 与 fps。支持 base_quat 或 base_pose、joint 27/29 维。
    base_quat_xyzw: 源文件 base_quat 为 xyzw 时为 True（PhysHSI 常见），为 wxyz 时为 False。
    """
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        data = {k: v.numpy() if hasattr(v, "numpy") else np.asarray(v) for k, v in data.items()}

    base_pos = np.asarray(data["base_position"], dtype=np.float64).copy()
    T = base_pos.shape[0]

    if "base_quat" in data:
        base_quat = np.asarray(data["base_quat"], dtype=np.float64).reshape(T, 4)
        quats = np.zeros((T, 4), dtype=np.float64)
        for i in range(T):
            if base_quat_xyzw:
                r = R.from_quat(base_quat[i])  # 文件 xyzw
            else:
                w, x, y, z = base_quat[i]
                r = R.from_quat([x, y, z, w])  # 文件 wxyz -> scipy xyzw
            q_xyzw = r.as_quat()
            quats[i] = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])  # MuJoCo wxyz
    else:
        base_pose = np.asarray(data["base_pose"], dtype=np.float64)
        quats = np.zeros((T, 4))
        for i in range(T):
            r = R.from_euler("xyz", base_pose[i])
            q = r.as_quat()
            quats[i] = (q[3], q[0], q[1], q[2])

    joint_pos = np.asarray(data["joint_position"], dtype=np.float64)
    if joint_pos.shape[1] == 29:
        dof_full = joint_pos
    else:
        assert joint_pos.shape[1] == 27
        dof_full = np.zeros((T, 29), dtype=np.float64)
        dof_full[:, :WAIST_ROLL_IDX] = joint_pos[:, :WAIST_ROLL_IDX]
        dof_full[:, WAIST_ROLL_IDX] = 0.0
        dof_full[:, WAIST_PITCH_IDX] = 0.0
        dof_full[:, WAIST_PITCH_IDX + 1:] = joint_pos[:, WAIST_ROLL_IDX:]

    # base z：优先 base_height；否则用 link 或 base 最低点接地
    if "base_height" in data:
        h = np.asarray(data["base_height"], dtype=np.float64).ravel()
        base_pos[:, 2] = np.broadcast_to(h, (T,)).copy()
    elif "link_position" in data:
        link_pos = np.asarray(data["link_position"], dtype=np.float64)
        if link_pos.ndim == 3 and link_pos.shape[1] >= 4:
            ankle_idx = [2, 3]
            min_z = float(link_pos[:, ankle_idx, 2].min())
            base_pos[:, 2] -= min_z
    else:
        min_z = float(base_pos[:, 2].min())
        base_pos[:, 2] -= min_z

    qpos = np.concatenate([base_pos, quats, dof_full], axis=1)
    assert qpos.shape == (T, G1_NQ)
    fps = float(np.asarray(data.get("fps", fps_fallback)).flat[0]) if data.get("fps") is not None else fps_fallback
    return qpos, fps, data


def g1_qpos_to_human_data(qpos_one, model_g1, data_g1, base_xy0):
    """
    单帧 G1 qpos -> human_data dict: human_body_name -> [pos (3,), quat (4,) wxyz].

    注意：
      - 这里的人体关键点，完全由上面的 HUMAN_TO_G1_BODY 映射决定；
        你可以修改 HUMAN_TO_G1_BODY 来换不同的 G1 body / link。
      - 不再依赖 smplx_to_g1.json，也不再使用 XML 里的 keyframe_* body。
    """
    data_g1.qpos[:] = qpos_one
    data_g1.qvel[:] = 0
    mj.mj_forward(model_g1, data_g1)

    human_data = {}

    for human_name, body_name in HUMAN_TO_G1_BODY.items():
        bid = mj.mj_name2id(model_g1, mj.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            # body 名没找到时跳过，方便你试验/增删 body
            continue

        # 位置：用 G1 body 的 xpos，并减去首帧 base_xy0 做平移归一
        pos = data_g1.xpos[bid].copy()
        pos[0] -= base_xy0[0]
        pos[1] -= base_xy0[1]

        # 方向：用 body 的 xmat，转成四元数 xyzw，再转为 MuJoCo wxyz
        xmat = np.array(data_g1.xmat[bid]).reshape(3, 3)
        r = R.from_matrix(xmat)
        quat_xyzw = r.as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )

        human_data[human_name] = [pos, quat_wxyz]

    return human_data


def get_ik_required_human_keys(target_robot, src_human="pt"):
    """返回 IK config 中需要的 human body 名集合（用于检查 human_data 是否完整）。"""
    if src_human not in IK_CONFIG_DICT or target_robot not in IK_CONFIG_DICT[src_human]:
        src_human = "pt"
    with open(IK_CONFIG_DICT[src_human][target_robot]) as f:
        cfg = json.load(f)
    return set(cfg.get("human_scale_table", {}).keys())


def check_human_data(human_data, target_robot, base_xy0, src_human="pt"):
    """检查 human_data 是否包含 IK 所需 key，并打印首帧简单统计（pelvis 高度、近似肢长）。"""
    required = get_ik_required_human_keys(target_robot, src_human)
    missing = required - set(human_data.keys())
    extra = set(human_data.keys()) - required
    print("[check-human-data] required keys (from IK config):", sorted(required))
    print("[check-human-data] human_data keys:", sorted(human_data.keys()))
    if missing:
        print("[check-human-data] 缺少 key（IK 会报错）:", sorted(missing))
    if extra:
        print("[check-human-data] 多余 key（IK 会忽略）:", sorted(extra))
    # 首帧统计（未减 base_xy0 的 pos 已在 g1_qpos_to_human_data 里减过 xy）
    pelvis_pos = human_data.get("pelvis", [np.zeros(3), None])[0]
    left_foot_pos = human_data.get("left_foot", [np.zeros(3), None])[0]
    right_foot_pos = human_data.get("right_foot", [np.zeros(3), None])[0]
    left_hip_pos = human_data.get("left_hip", [np.zeros(3), None])[0]
    right_hip_pos = human_data.get("right_hip", [np.zeros(3), None])[0]
    print("[check-human-data] 首帧: pelvis_z={:.4f}, left_foot_z={:.4f}, right_foot_z={:.4f}".format(
        float(pelvis_pos[2]), float(left_foot_pos[2]), float(right_foot_pos[2])))
    left_leg = float(np.linalg.norm(pelvis_pos - left_hip_pos) + np.linalg.norm(left_hip_pos - left_foot_pos))
    right_leg = float(np.linalg.norm(pelvis_pos - right_hip_pos) + np.linalg.norm(right_hip_pos - right_foot_pos))
    print("[check-human-data] 近似腿长: left={:.4f}, right={:.4f}".format(left_leg, right_leg))


def build_pt_motion(qpos_list, model, robot_type, fps):
    """从 qpos_list (T, 36 或目标机器人 nq) 构建 .pt 的 base/joint/link 部分（numpy then torch）。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.smplx_pkl_to_robot import build_pt_motion as _build
    qvel_list = [np.zeros(qpos_list[0].shape[0] - 1, dtype=np.float32) for _ in qpos_list]
    return _build(qpos_list, qvel_list, model, robot_type, fps)


def main():
    parser = argparse.ArgumentParser(description="G1 .pt 直接 retarget 到目标机器人 .pt，保留 box 等额外字段")
    parser.add_argument("pt_file", type=str, help="源 G1 .pt 路径")
    parser.add_argument("--target_robot", type=str, default="adam_sp", help="目标机器人（如 adam_sp, unitree_g1 仅做拷贝）")
    parser.add_argument("--save_path", type=str, default=None, help="输出 .pt 路径，默认源同目录下 <stem>_<target_robot>.pt")
    parser.add_argument("--fps", type=float, default=None, help="覆盖帧率（源 .pt 无 fps 时必填或默认 50）")
    parser.add_argument("--quat-wxyz", action="store_true", help="源 G1 .pt 的 base_quat 为 wxyz 时使用；默认按 xyzw（PhysHSI 常见）")
    parser.add_argument("--no-vis", action="store_true", help="关闭 retarget 过程可视化")
    parser.add_argument("--record-video", action="store_true", help="录屏（需开启可视化）")
    parser.add_argument("--video-path", type=str, default=None, help="录屏输出路径，默认 videos/<target_robot>_<pt_stem>.mp4")
    parser.add_argument("--rate-limit", action="store_true", help="按原帧率限速播放")
    parser.add_argument("--no-human-scale", action="store_true", help="缩放比例=1（actual_human_height=config 的 human_height_assumption）")
    parser.add_argument("--scale", type=float, default=None, help="直接指定缩放比例 ratio（例如 1.05 表示放大 5%%），覆盖 no-human-scale/默认身高")
    parser.add_argument("--check-human-data", action="store_true", help="打印首帧 human_data 的 key 与简单统计，用于检查是否符合 IK 期望")
    args = parser.parse_args()

    pt_path = pathlib.Path(args.pt_file)
    if not pt_path.is_file():
        print(f"错误: 文件不存在 {pt_path}", file=sys.stderr)
        sys.exit(1)

    fps_fallback = args.fps if args.fps is not None else 50.0
    qpos_g1, fps, source_data = load_g1_pt_to_qpos(
        pt_path, fps_fallback=fps_fallback, base_quat_xyzw=not args.quat_wxyz
    )
    if args.fps is not None:
        fps = args.fps
    T = qpos_g1.shape[0]
    base_xy0 = qpos_g1[0, :2].copy()

    # G1 模型用于 FK
    g1_xml = REPO_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    if not g1_xml.exists():
        g1_xml = (REPO_ROOT / "general_motion_retargeting" / ".." / "assets" / "unitree_g1" / "g1_mocap_29dof.xml").resolve()
    model_g1 = mj.MjModel.from_xml_path(str(g1_xml))
    data_g1 = mj.MjData(model_g1)

    # 目标机器人 retarget：有 pt 专用 config 时用 pt（G1 link 语义、rot 已按恒等修正），否则用 smplx。
    src_human = "pt" if "pt" in IK_CONFIG_DICT and args.target_robot in IK_CONFIG_DICT["pt"] else "smplx"
    with open(IK_CONFIG_DICT[src_human][args.target_robot]) as f:
        ik_cfg = json.load(f)
    assumption = ik_cfg["human_height_assumption"]
    if args.scale is not None:
        actual_human_height = assumption * args.scale
    elif args.no_human_scale:
        actual_human_height = assumption
    else:
        actual_human_height = 1.65
    retarget = GMR(actual_human_height=actual_human_height, src_human=src_human, tgt_robot=args.target_robot)
    target_model = retarget.model
    ratio = actual_human_height / assumption
    print(f"[pt_to_robot_pt] 缩放: ratio = actual_human_height / human_height_assumption = {actual_human_height:.4f} / {assumption:.4f} = {ratio:.4f}")

    if args.check_human_data:
        human_data_0 = g1_qpos_to_human_data(qpos_g1[0], model_g1, data_g1, base_xy0)
        check_human_data(human_data_0, args.target_robot, base_xy0, src_human=src_human)

    do_vis = not args.no_vis
    if do_vis:
        video_path = args.video_path
        if video_path is None and args.record_video:
            video_path = f"videos/{args.target_robot}_{pt_path.stem}.mp4"
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
                qpos, _qvel = retarget.retarget(human_data, offset_to_ground=True)
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
            qpos, _qvel = retarget.retarget(human_data, offset_to_ground=True)
            qpos_list.append(np.asarray(qpos, dtype=np.float32))

    qpos_list = np.array(qpos_list, dtype=np.float32)
    pt_dict = build_pt_motion(qpos_list, target_model, args.target_robot, fps)

    # === 定义核心字段白名单（最终需要保留的字段）===
    core_keys = {
        "base_angular_velocity",
        "base_height",
        "base_linear_velocity",
        "base_position",
        "base_quat",
        "joint_position",
        "joint_velocity",
        "link_position"
    }

    # === 字段转换：将 build_pt_motion 的输出转换为目标格式 ===
    
    # base_height: 从 base_position 的 z 轴提取
    if "base_position" in pt_dict:
        pt_dict["base_height"] = pt_dict["base_position"][:, 2].clone()
    
    # base_quat: 从 base_pose (欧拉角 xyz) 转换回四元数 xyzw
    if "base_pose" in pt_dict:
        base_pose = pt_dict["base_pose"].numpy()
        quat_xyzw = np.zeros((base_pose.shape[0], 4), dtype=np.float32)
        for i in range(base_pose.shape[0]):
            r = R.from_euler("xyz", base_pose[i])
            q = r.as_quat()  # scipy 返回 xyzw 格式
            quat_xyzw[i] = q
        pt_dict["base_quat"] = torch.from_numpy(quat_xyzw)
    
    # base_linear_velocity: 直接使用 base_velocity
    if "base_velocity" in pt_dict:
        pt_dict["base_linear_velocity"] = pt_dict["base_velocity"].clone()

    # === 对 adam_sp 系列，只保留 6 个 link，用于数据集中的 link_position(T,6,3) 结构 ===
    if args.target_robot.startswith("adam_sp") and "link_position" in pt_dict:
        idx_order = [12, 16, 3, 6, 7, 7]
        key = "link_position"
        if key in pt_dict and pt_dict[key].ndim == 3 and pt_dict[key].shape[1] >= 8:
            pt_dict[key] = pt_dict[key][:, idx_order, :].contiguous()

    # === 目标帧数 ===
    T = pt_dict["base_position"].shape[0] if "base_position" in pt_dict else len(qpos_list)

    # === 从 source_data 中获取 chair 相关的 key 名称 ===
    source_chair_keys = [k for k in source_data.keys() if k.startswith("chair_")]
    print(f"  源数据集中的 chair 字段：{source_chair_keys}")

    # === 获取最后一帧的 base 和 link 信息（用于 chair 字段填充）===
    last_base_pos = pt_dict["base_position"][-1].numpy() if "base_position" in pt_dict else None
    last_base_quat = pt_dict["base_quat"][-1].numpy() if "base_quat" in pt_dict else None
    last_pelvis_z = None
    if "link_position" in pt_dict and pt_dict["link_position"].shape[1] > 0:
        # idx0 = pelvis，取最后一帧的 z 值
        last_pelvis_z = float(pt_dict["link_position"][-1, 0, 2].numpy())

    # === 生成 chair 相关字段（只填充源数据集中存在的 key）===
    chair_keys_added = []
    
    for k in source_chair_keys:
        if k in ["chair_pos", "chair_pos_local", "chair_position"]:
            # chair_pos: xy 用最后一帧 base_position，z 用最后一帧 pelvis (link_position idx0)
            if last_base_pos is not None and last_pelvis_z is not None:
                chair_pos = np.zeros((T, 3), dtype=np.float32)
                chair_pos[:, 0] = last_base_pos[0]
                chair_pos[:, 1] = last_base_pos[1]
                chair_pos[:, 2] = last_base_pos[2] - 0.18
                pt_dict[k] = torch.from_numpy(chair_pos)
                chair_keys_added.append(k)
                print(f"  [生成] {k}: ({T}, 3) - 最后一帧 base_xy + pelvis_z")
        
        elif k in ["chair_quat", "chair_orientation"]:
            # chair_quat: 用最后一帧 base_quat，保持 (4,) 单值形状
            if last_base_quat is not None:
                pt_dict[k] = torch.from_numpy(last_base_quat.astype(np.float32))
                chair_keys_added.append(k)
                print(f"  [生成] {k}: (4,) - 最后一帧 base_quat")
        
        elif k in ["chair_height_offset", "base_height_bias"]:
            # chair_height_offset: 标量，填充 0，形状 (T,)
            pt_dict[k] = torch.from_numpy(np.zeros((T,), dtype=np.float32))
            chair_keys_added.append(k)
            print(f"  [生成] {k}: ({T},) - 填充 0")
        
        else:
            # 其他 chair_* 字段：填充 0，形状 (T,)
            pt_dict[k] = torch.from_numpy(np.zeros((T,), dtype=np.float32))
            chair_keys_added.append(k)
            print(f"  [生成] {k}: ({T},) - 填充 0 (未知类型)")

    # === 最终清理：删除 pt_dict 中所有不在核心字段 + chair 字段中的键 ===
    allowed_keys = core_keys | set(chair_keys_added)
    keys_to_remove = [k for k in list(pt_dict.keys()) if k not in allowed_keys]
    for k in keys_to_remove:
        del pt_dict[k]

    out_path = args.save_path
    if not out_path:
        out_path = str(pt_path.parent / f"{pt_path.stem}_{args.target_robot}.pt")
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存前再次确认 keys
    final_keys = list(pt_dict.keys())
    torch.save(pt_dict, out_path)
    print(f"\n已保存：{out_path} (帧数={T}, fps={fps})")
    print(f"  核心字段 ({len(core_keys)}): {sorted(core_keys)}")
    print(f"  chair 字段 ({len(chair_keys_added)}): {sorted(chair_keys_added)}")
    print(f"  最终总字段 ({len(final_keys)}): {sorted(final_keys)}")


if __name__ == "__main__":
    main()
