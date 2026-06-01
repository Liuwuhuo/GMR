#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

KIT21_NAMES = [
    "pelvis",        # 0
    "spine1",        # 1
    "spine2",        # 2
    "neck",          # 3
    "head",          # 4
    "left_shoulder", # 5
    "left_elbow",    # 6
    "left_wrist",    # 7
    "right_shoulder",# 8
    "right_elbow",   # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "left_knee",     # 12
    "left_ankle",    # 13
    "left_foot",     # 14
    "left_toe",      # 15
    "right_hip",     # 16
    "right_knee",    # 17
    "right_ankle",   # 18
    "right_foot",    # 19
    "right_toe",     # 20
]

HML22_NAMES = [
    "pelvis",        # 0
    "left_hip",      # 1
    "right_hip",     # 2
    "spine1",        # 3
    "left_knee",     # 4
    "right_knee",    # 5
    "spine2",        # 6
    "left_ankle",    # 7
    "right_ankle",   # 8
    "spine3",        # 9
    "left_foot",     # 10
    "right_foot",    # 11
    "neck",          # 12
    "left_collar",   # 13
    "right_collar",  # 14
    "head",          # 15
    "left_shoulder", # 16
    "right_shoulder",# 17
    "left_elbow",    # 18
    "right_elbow",   # 19
    "left_wrist",    # 20
    "right_wrist",   # 21
]

IK_TO_COMMON_NAME = {
    "Hips": "pelvis",
    "Spine1": "spine1",
    "LeftUpLeg": "left_hip",
    "RightUpLeg": "right_hip",
    "LeftLeg": "left_knee",
    "RightLeg": "right_knee",
    "LeftFootMod": "left_foot",
    "RightFootMod": "right_foot",
    "LeftArm": "left_shoulder",
    "RightArm": "right_shoulder",
    "LeftForeArm": "left_elbow",
    "RightForeArm": "right_elbow",
    "LeftHand": "left_wrist",
    "RightHand": "right_wrist",
}

COMMON_CHILDREN = {
    "pelvis": ["spine1", "left_hip", "right_hip"],
    "spine1": ["spine2", "neck"],
    "spine2": ["spine3", "neck", "head"],
    "spine3": ["neck", "head"],
    "neck": ["head"],
    "left_shoulder": ["left_elbow"],
    "left_elbow": ["left_wrist"],
    "left_hip": ["left_knee"],
    "left_knee": ["left_ankle"],
    "left_ankle": ["left_foot", "left_toe"],
    "left_foot": ["left_toe"],
    "right_shoulder": ["right_elbow"],
    "right_elbow": ["right_wrist"],
    "right_hip": ["right_knee"],
    "right_knee": ["right_ankle"],
    "right_ankle": ["right_foot", "right_toe"],
    "right_foot": ["right_toe"],
}

COMMON_PARENTS = {
    "spine1": "pelvis",
    "spine2": "spine1",
    "spine3": "spine2",
    "neck": "spine3",
    "head": "neck",
    "left_shoulder": "neck",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "right_shoulder": "neck",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_hip": "pelvis",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "left_foot": "left_ankle",
    "left_toe": "left_foot",
    "right_hip": "pelvis",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "right_foot": "right_ankle",
    "right_toe": "right_foot",
}


def load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, allow_pickle=True)
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if "new_joints" in data:
            return np.asarray(data["new_joints"])
        if "arr_0" in data:
            return np.asarray(data["arr_0"])
        raise KeyError(f"{path} 中未找到 'new_joints' 或 'arr_0'")
    raise ValueError(f"不支持的数组文件格式: {path}")


def load_joint_names(path: str | None) -> List[str]:
    if path is None:
        return []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        data = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("joint_names json 需为字符串列表")
    if ext in (".txt", ".csv"):
        text = open(path, "r", encoding="utf-8").read().strip()
        if not text:
            return []
        if ext == ".csv":
            return [x.strip() for x in text.split(",") if x.strip()]
        return [x.strip() for x in text.splitlines() if x.strip()]
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        return [str(x) for x in arr.tolist()]
    raise ValueError(f"不支持的 joint_names 文件格式: {path}")


def build_index_map(
    map_path: str | None,
    joint_names: List[str],
    required_human_joints: List[str],
) -> Dict[str, int]:
    if map_path is None:
        if joint_names and all(
            name in joint_names for name in required_human_joints
        ):
            return {
                name: joint_names.index(name)
                for name in required_human_joints
            }
        raise ValueError(
            "缺少映射信息。请提供 --index_map（推荐）或提供可直接匹配 IK 名称的 --joint_names。"
        )

    raw = json.load(open(map_path, "r", encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--index_map 必须是 dict")

    index_map: Dict[str, int] = {}
    for target_name in required_human_joints:
        if target_name not in raw:
            raise KeyError(f"--index_map 缺少目标关节: {target_name}")
        value = raw[target_name]
        if isinstance(value, int):
            index_map[target_name] = value
        elif isinstance(value, str):
            if value not in joint_names:
                raise KeyError(
                    f"index_map[{target_name}]={value} 不在 --joint_names 中"
                )
            index_map[target_name] = joint_names.index(value)
        else:
            raise ValueError(f"index_map[{target_name}] 仅支持 int 或 str")
    return index_map


def build_preset_index_map(
    preset: str,
    required_human_joints: List[str],
) -> Dict[str, int]:
    if preset == "kit21":
        names = KIT21_NAMES
    elif preset == "hml22":
        names = HML22_NAMES
    else:
        raise ValueError(f"不支持的 preset: {preset}")

    index_map: Dict[str, int] = {}
    for ik_name in required_human_joints:
        if ik_name not in IK_TO_COMMON_NAME:
            raise KeyError(
                f"IK关节 {ik_name} 没有内置映射，请改用 --index_map"
            )
        src_name = IK_TO_COMMON_NAME[ik_name]
        if src_name not in names:
            raise KeyError(
                f"{preset} 中找不到关节 {src_name}（来自 {ik_name}）"
            )
        index_map[ik_name] = names.index(src_name)
    return index_map


def convert_y_up_to_z_up(pos_xyz: np.ndarray) -> np.ndarray:
    # 与现有 BVH 管线一致: [x, y, z] -> [x, -z, y]
    x = pos_xyz[..., 0]
    y = pos_xyz[..., 1]
    z = pos_xyz[..., 2]
    return np.stack([x, -z, y], axis=-1)


def normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < 1e-8:
        return vec * 0.0
    return vec / n


def quat_from_forward_up(forward: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    f = normalize(forward)
    if np.linalg.norm(f) < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    u = normalize(up_hint)
    if np.linalg.norm(u) < 1e-8:
        u = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Remove up component parallel to forward
    u = u - np.dot(u, f) * f
    if np.linalg.norm(u) < 1e-8:
        # choose a robust fallback
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, f)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = normalize(tmp - np.dot(tmp, f) * f)
    else:
        u = normalize(u)

    r = normalize(np.cross(f, u))
    u = normalize(np.cross(r, f))

    # Local axes: x->right, y->forward, z->up
    rot_mat = np.column_stack([r, f, u])
    return R.from_matrix(rot_mat).as_quat(scalar_first=True)


def build_common_quats(
    frame_joints: np.ndarray,
    name_to_index: Dict[str, int],
) -> Dict[str, np.ndarray]:
    quats: Dict[str, np.ndarray] = {}
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for joint_name, joint_idx in name_to_index.items():
        origin = frame_joints[joint_idx]
        forward = None

        for child in COMMON_CHILDREN.get(joint_name, []):
            if child in name_to_index:
                child_pos = frame_joints[name_to_index[child]]
                cand = child_pos - origin
                if np.linalg.norm(cand) > 1e-8:
                    forward = cand
                    break

        if forward is None and joint_name in COMMON_PARENTS:
            parent = COMMON_PARENTS[joint_name]
            if parent in name_to_index:
                parent_pos = frame_joints[name_to_index[parent]]
                cand = origin - parent_pos
                if np.linalg.norm(cand) > 1e-8:
                    forward = cand

        if forward is None:
            quats[joint_name] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
        else:
            quats[joint_name] = quat_from_forward_up(forward, world_up)

    return quats


def main():
    parser = argparse.ArgumentParser(
        description="HumanML3D new_joints -> robot npz"
    )
    parser.add_argument(
        "--new_joints",
        required=True,
        help="new_joints 路径 (.npy/.npz), shape=(T,J,3)",
    )
    parser.add_argument(
        "--joint_names",
        default=None,
        help="关节名文件 (.json/.txt/.csv/.npy)",
    )
    parser.add_argument(
        "--index_map",
        default=None,
        help=(
            "目标IK名到源关节索引/名称的映射 json，"
            "例如 {'Hips':0,'LeftArm':'left_shoulder'}"
        ),
    )
    parser.add_argument(
        "--joint_preset",
        default=None,
        choices=["kit21", "hml22"],
        help="使用内置关节顺序自动构建 index_map",
    )
    parser.add_argument(
        "--texts",
        default=None,
        help="texts 文件路径（可选，保存到输出）",
    )
    parser.add_argument(
        "--new_joint_vecs",
        default=None,
        help="new_joint_vecs 路径（可选，保存到输出）",
    )
    parser.add_argument(
        "--src_human",
        default="humanml3d",
        help="IK 源类型（需在 IK_CONFIG_DICT 中存在）",
    )
    parser.add_argument("--robot", default="adam_sp", help="机器人名")
    parser.add_argument("--actual_human_height", type=float, default=None)
    parser.add_argument(
        "--default_human_height",
        type=float,
        default=1.65,
        help=(
            "当未显式提供 --actual_human_height 时使用的默认身高（米），"
            "仅影响本脚本。"
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="HumanML3D 常见为 20fps",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--save_path", default=None, help="输出 npz")
    parser.add_argument("--compressed", action="store_true", default=False)
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument(
        "--visualize_raw_joints",
        action="store_true",
        default=False,
        help=(
            "额外用数据集 new_joints 画完整骨架（调试用）。"
            "默认关闭：与 scripts/bvh_to_robot_npz.py 一样只画 retargeter.scaled_human_data，"
            "即与 IK 任务目标一致。"
        ),
    )
    parser.add_argument(
        "--raw_vis_auto_scale",
        action="store_true",
        default=True,
        help="原始 joints 可视化时自动按量纲缩放到米级（仅影响显示）",
    )
    parser.add_argument(
        "--raw_vis_align_to_robot",
        action="store_true",
        default=True,
        help="原始 joints 可视化时以 pelvis 对齐到机器人根部（仅影响显示）",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        default=False,
        help="不打开可视化窗口",
    )
    parser.add_argument(
        "--coord_scale",
        type=float,
        default=1.0,
        help="对 new_joints 整体坐标做缩放（例如 mm->m 可设 0.001）",
    )
    parser.add_argument(
        "--auto_coord_scale",
        action="store_true",
        default=True,
        help=(
            "当 --coord_scale 为 1 且关节包围盒尺度很大（疑似毫米）时，"
            "自动乘 0.001，使 retarget 与 scaled_human_data 可视化与机器人为同一米制。"
        ),
    )
    parser.add_argument(
        "--no_auto_coord_scale",
        action="store_false",
        dest="auto_coord_scale",
    )
    parser.add_argument("--y_up_to_z_up", action="store_true", default=True)
    parser.add_argument(
        "--no_y_up_to_z_up",
        action="store_false",
        dest="y_up_to_z_up",
    )
    args = parser.parse_args()

    if args.save_path is None:
        base = os.path.splitext(os.path.basename(args.new_joints))[0]
        args.save_path = os.path.join(
            "retarget",
            args.robot,
            "humanml3d",
            f"{base}.npz",
        )
        print(f"未指定保存路径，使用默认路径: {args.save_path}")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    joints = np.asarray(load_array(args.new_joints), dtype=np.float64)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"new_joints 形状需为 (T,J,3)，当前: {joints.shape}")

    if args.y_up_to_z_up:
        joints = convert_y_up_to_z_up(joints)

    # 与「仅显示乘 0.001」不同：这里写入 joints，retarget 与 scaled_human_data 同源，对齐 bvh_to_robot_npz。
    auto_mm_scale = 1.0
    if args.auto_coord_scale and float(args.coord_scale) == 1.0:
        extent = np.max(joints, axis=(0, 1)) - np.min(joints, axis=(0, 1))
        extent_norm = float(np.linalg.norm(extent))
        if extent_norm > 10.0:
            auto_mm_scale = 0.001
            joints = joints * auto_mm_scale
            print(
                f"[humanml3d] auto_coord_scale: joints *= {auto_mm_scale} "
                f"(extent_norm={extent_norm:.3f}, 疑似 mm→m；可用 --no_auto_coord_scale 关闭)"
            )

    joints = joints * float(args.coord_scale)

    preset_expected_j = {"kit21": 21, "hml22": 22}
    if args.joint_preset in preset_expected_j:
        expected_j = preset_expected_j[args.joint_preset]
        if joints.shape[1] != expected_j:
            raise ValueError(
                f"--joint_preset={args.joint_preset} 期望 J={expected_j}，"
                f"但当前数据 J={joints.shape[1]}。"
                f"{' 你可能应该使用 --joint_preset kit21。' if joints.shape[1] == 21 else ''}"
            )

    start = max(0, args.start_frame)
    end = args.end_frame if args.end_frame is not None else joints.shape[0]
    joints = joints[start:end]
    if joints.shape[0] == 0:
        raise ValueError("切片后没有可用帧")

    effective_human_height = (
        args.actual_human_height
        if args.actual_human_height is not None
        else float(args.default_human_height)
    )
    if args.actual_human_height is None:
        print(
            f"[humanml3d] --actual_human_height 未提供，使用默认值 "
            f"--default_human_height={effective_human_height:.3f}"
        )

    retargeter = GMR(
        src_human=args.src_human,
        tgt_robot=args.robot,
        actual_human_height=effective_human_height,
    )
    required_human_joints = []
    if retargeter.use_ik_match_table1:
        required_human_joints += [
            v[0] for v in retargeter.ik_match_table1.values()
        ]
    if retargeter.use_ik_match_table2:
        required_human_joints += [
            v[0] for v in retargeter.ik_match_table2.values()
        ]
    required_human_joints = sorted(set(required_human_joints))

    joint_names = load_joint_names(args.joint_names)
    if args.index_map is not None:
        index_map = build_index_map(
            args.index_map,
            joint_names,
            required_human_joints,
        )
    elif args.joint_preset is not None:
        index_map = build_preset_index_map(
            args.joint_preset,
            required_human_joints,
        )
    else:
        index_map = build_index_map(
            args.index_map,
            joint_names,
            required_human_joints,
        )
    max_idx = max(index_map.values())
    if max_idx >= joints.shape[1]:
        raise IndexError(f"映射索引越界: 最大索引 {max_idx}, 但 J={joints.shape[1]}")

    preset_common_map = None
    if args.joint_preset == "kit21":
        preset_common_map = {name: i for i, name in enumerate(KIT21_NAMES)}
        vis_joint_names = KIT21_NAMES
    elif args.joint_preset == "hml22":
        preset_common_map = {name: i for i, name in enumerate(HML22_NAMES)}
        vis_joint_names = HML22_NAMES
    elif joint_names:
        vis_joint_names = joint_names
    else:
        vis_joint_names = None

    viewer = None
    if not args.no_viewer:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=int(round(args.fps)),
            transparent_robot=0,
        )

    qpos_list = []
    qvel_list = []
    raw_vis_scale = 1.0
    raw_vis_scale_logged = False
    for i in tqdm(range(joints.shape[0]), desc="Retargeting(HumanML3D)"):
        if viewer is not None:
            viewer.wait_while_paused()
            if not viewer.viewer.is_running():
                break

        frame = {}
        frame_joints = joints[i]
        common_quats = None
        if preset_common_map is not None:
            common_quats = build_common_quats(frame_joints, preset_common_map)
        for name in required_human_joints:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if common_quats is not None and name in IK_TO_COMMON_NAME:
                common_name = IK_TO_COMMON_NAME[name]
                if common_name in common_quats:
                    quat = common_quats[common_name]
            frame[name] = [frame_joints[index_map[name]], quat]

        qpos, qvel = retargeter.retarget(
            frame,
            offset_to_ground=True,
            no_fly=False,
        )
        qpos_list.append(qpos.copy())
        qvel_list.append(qvel.copy())

        if viewer is not None:
            # 与 bvh_to_robot_npz.py 一致：默认同屏显示 IK 使用的目标位姿（update_targets 之后的值）。
            vis_human_data = retargeter.scaled_human_data
            if args.visualize_raw_joints and vis_joint_names is not None:
                vis_human_data = {}
                vis_points = frame_joints.copy()

                # Only for visualization: auto scale large-value datasets (e.g., mm)
                if args.raw_vis_auto_scale:
                    extent = np.max(vis_points, axis=0) - np.min(vis_points, axis=0)
                    extent_norm = float(np.linalg.norm(extent))
                    raw_vis_scale = 0.001 if extent_norm > 10.0 else 1.0
                    vis_points = vis_points * raw_vis_scale
                    if not raw_vis_scale_logged:
                        print(
                            f"[raw-vis] extent_norm={extent_norm:.3f}, "
                            f"scale={raw_vis_scale}"
                        )
                        raw_vis_scale_logged = True

                # Only for visualization: center pelvis at robot root for camera visibility
                if args.raw_vis_align_to_robot and vis_points.shape[0] > 0:
                    vis_points = vis_points - vis_points[0] + qpos[:3]

                for j_idx, j_name in enumerate(vis_joint_names):
                    if j_idx >= vis_points.shape[0]:
                        break
                    # raw joints visualization: use estimated quat if available
                    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                    if common_quats is not None and j_name in common_quats:
                        quat = common_quats[j_name]
                    vis_human_data[j_name] = [vis_points[j_idx], quat]

            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=vis_human_data,
                rate_limit=args.rate_limit,
                follow_camera=False,
            )

    if viewer is not None:
        viewer.close()

    if not qpos_list:
        raise RuntimeError("没有生成任何 retarget 结果（可能窗口过早关闭）")

    qpos_arr = np.asarray(qpos_list)
    qvel_arr = np.asarray(qvel_list)
    save_dict = {
        "fps": np.array([float(args.fps)]),
        "root_pos": qpos_arr[:, :3],
        "root_rot": qpos_arr[:, 3:7],  # wxyz
        "dof_pos": qpos_arr[:, 7:],
        "dof_vel": qvel_arr[:, 6:],
    }

    if args.texts is not None:
        save_dict["texts"] = np.array(
            open(args.texts, "r", encoding="utf-8").read(),
            dtype=object,
        )
    if args.new_joint_vecs is not None:
        save_dict["new_joint_vecs"] = np.asarray(
            load_array(args.new_joint_vecs)
        )

    if args.compressed:
        np.savez_compressed(args.save_path, **save_dict)
    else:
        np.savez(args.save_path, **save_dict)
    print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
