#!/usr/bin/env python3
import json
import numpy as np
import argparse
import os

# 机器人关节配置（与 bvh_to_robot_npz.py 兼容）
ROBOT_JOINT_CONFIGS = {
    "adam_lite": {
        "dof_names": [
            "hipPitch_Left", "hipRoll_Left", "hipYaw_Left",
            "kneePitch_Left", "anklePitch_Left", "ankleRoll_Left",
            "hipPitch_Right", "hipRoll_Right", "hipYaw_Right",
            "kneePitch_Right", "anklePitch_Right", "ankleRoll_Right",
            "waistRoll", "waistPitch", "waistYaw",
            "shoulderPitch_Left", "shoulderRoll_Left", "shoulderYaw_Left",
            "elbow_Left",
            "shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right",
            "elbow_Right"
        ]
    },
    "adam_sp": {
        "dof_names": [
            "hipPitch_Left", "hipRoll_Left", "hipYaw_Left",
            "kneePitch_Left", "anklePitch_Left", "ankleRoll_Left",
            "hipPitch_Right", "hipRoll_Right", "hipYaw_Right",
            "kneePitch_Right", "anklePitch_Right", "ankleRoll_Right",
            "waistRoll", "waistPitch", "waistYaw",
            "shoulderPitch_Left", "shoulderRoll_Left", "shoulderYaw_Left",
            "elbow_Left", "wristYaw_Left", "wristPitch_Left", "wristRoll_Left",
            "shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right",
            "elbow_Right", "wristYaw_Right", "wristPitch_Right", "wristRoll_Right"
        ]
    }
}


def load_joint_config(config_path=None, robot_type="adam_sp"):
    """加载关节配置"""
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
        print(f"使用自定义关节配置：{config_path}")
        return custom_config
    
    print(f"使用默认关节配置：{robot_type}")
    config = ROBOT_JOINT_CONFIGS.get(robot_type, ROBOT_JOINT_CONFIGS["adam_sp"])
    # 确保 dof_pos 和 dof_vel 使用相同的关节列表
    return {
        "dof_pos": config["dof_names"],
        "dof_vel": config["dof_names"].copy()
    }


def json_to_npz(json_file, save_path=None, compressed=False, 
                joint_config=None, robot_type="adam_sp",
                fill_missing_with_zero=True):
    """
    将 JSON 格式转换为与 bvh_to_robot_npz.py 完全兼容的 NPZ 格式
    """
    # 读取 JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取基本信息
    frame_duration = data.get("FrameDuration", 0.004166666666666667)
    fps = int(round(1.0 / frame_duration))
    frames_data = data.get("Frames", [])
    labels = data.get("Labels", [])
    
    num_frames = len(frames_data)
    print(f"帧数：{num_frames}")
    print(f"FPS: {fps}")
    print(f"JSON 标签数：{len(labels)}")
    
    # 创建标签到索引的映射
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # 加载关节配置
    if joint_config is None:
        joint_config = load_joint_config(None, robot_type)
    
    dof_pos_joints = joint_config.get("dof_pos", [])
    dof_vel_joints = joint_config.get("dof_vel", [])
    
    num_dofs = len(dof_pos_joints)
    
    # 验证 dof_pos 和 dof_vel 关节数量一致
    if len(dof_vel_joints) != num_dofs:
        print(f"⚠️  警告：dof_pos({num_dofs}) 和 dof_vel({len(dof_vel_joints)}) 关节数量不一致!")
        print(f"   将使用 dof_pos 的关节数量：{num_dofs}")
        dof_vel_joints = dof_pos_joints.copy()
    
    print(f"机器人类型：{robot_type}")
    print(f"DOF 数量：{num_dofs}")
    
    # 初始化数据数组 (与 bvh_to_robot_npz.py 格式一致)
    root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    root_rot = np.zeros((num_frames, 4), dtype=np.float32)  # wxyz
    dof_pos = np.zeros((num_frames, num_dofs), dtype=np.float32)
    dof_vel = np.zeros((num_frames, num_dofs), dtype=np.float32)
    
    # 定义标签
    root_pos_labels = ["root_pos/x", "root_pos/y", "root_pos/z"]
    root_quat_labels = ["root_quat/x", "root_quat/y", "root_quat/z", "root_quat/w"]
    
    # 构建完整标签名
    dof_pos_labels = [f"dof_pos/{name}" for name in dof_pos_joints]
    dof_vel_labels = [f"dof_vel/{name}" for name in dof_vel_joints]
    
    # 统计缺失关节
    missing_joints_pos = []
    missing_joints_vel = []
    
    # 提取每一帧数据
    for frame_idx, frame_data in enumerate(frames_data):
        # root_pos
        for i, label in enumerate(root_pos_labels):
            if label in label_to_idx:
                root_pos[frame_idx, i] = frame_data[label_to_idx[label]]
        
        # root_rot (xyzw -> wxyz)
        quat_xyzw = []
        for label in root_quat_labels:
            if label in label_to_idx:
                quat_xyzw.append(frame_data[label_to_idx[label]])
            else:
                quat_xyzw.append(0.0)
        root_rot[frame_idx] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        
        # dof_pos - 按配置顺序填充，缺失填 0
        for i, label in enumerate(dof_pos_labels):
            if label in label_to_idx:
                dof_pos[frame_idx, i] = frame_data[label_to_idx[label]]
            elif fill_missing_with_zero:
                dof_pos[frame_idx, i] = 0.0
                if frame_idx == 0:
                    missing_joints_pos.append(dof_pos_joints[i])
        
        # dof_vel - 按配置顺序填充，缺失填 0
        for i, label in enumerate(dof_vel_labels):
            if label in label_to_idx:
                dof_vel[frame_idx, i] = frame_data[label_to_idx[label]]
            elif fill_missing_with_zero:
                dof_vel[frame_idx, i] = 0.0
                if frame_idx == 0:
                    missing_joints_vel.append(dof_vel_joints[i])
    
    # 打印缺失关节警告
    if missing_joints_pos:
        print(f"\n⚠️  以下 dof_pos 关节在 JSON 中未找到，已填充为 0:")
        for joint in missing_joints_pos:
            print(f"   - {joint}")
    
    if missing_joints_vel:
        print(f"\n⚠️  以下 dof_vel 关节在 JSON 中未找到，已填充为 0:")
        for joint in missing_joints_vel:
            print(f"   - {joint}")
    
    # 构建保存字典 (与 bvh_to_robot_npz.py 完全一致)
    save_dict = {
        "fps": np.array([fps]),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
    }
    
    # 保存关节名称
    save_dict["dof_names"] = np.array(dof_pos_joints, dtype=object)
    
    # 创建保存目录
    if save_path is None:
        json_basename = os.path.splitext(os.path.basename(json_file))[0]
        save_path = json_basename + ".npz"
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 保存 NPZ
    if compressed:
        np.savez_compressed(save_path, **save_dict)
    else:
        np.savez(save_path, **save_dict)
    
    print(f"\n✅ 已保存：{save_path}")
    print(f"  - fps: {fps}")
    print(f"  - root_pos 形状：{root_pos.shape}")
    print(f"  - root_rot 形状：{root_rot.shape} (wxyz)")
    print(f"  - dof_pos 形状：{dof_pos.shape}")
    print(f"  - dof_vel 形状：{dof_vel.shape}")
    
    return save_path


def main():
    parser = argparse.ArgumentParser(description="JSON -> NPZ (兼容 bvh_to_robot_npz.py 格式)")
    parser.add_argument("--json_file", required=True, help="JSON file path")
    parser.add_argument("--save_path", default=None, help="Output NPZ path")
    parser.add_argument("--compressed", action="store_true", default=False, 
                        help="Use compressed NPZ format")
    parser.add_argument("--robot", type=str, default="adam_sp",
                        choices=["adam_lite", "adam_sp", "custom"],
                        help="Robot type")
    parser.add_argument("--joint_config", type=str, default=None,
                        help="Custom joint configuration JSON file path")
    parser.add_argument("--fill_zero", action="store_true", default=True,
                        help="Fill missing joints with 0")
    args = parser.parse_args()
    
    # 加载自定义关节配置
    joint_config = None
    if args.joint_config is not None:
        joint_config = load_joint_config(args.joint_config, args.robot)
    
    json_to_npz(
        args.json_file,
        args.save_path,
        args.compressed,
        joint_config=joint_config,
        robot_type=args.robot,
        fill_missing_with_zero=args.fill_zero
    )


if __name__ == "__main__":
    main()