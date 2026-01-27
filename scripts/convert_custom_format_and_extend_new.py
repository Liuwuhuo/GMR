#!/usr/bin/env python3
"""
转换自定义格式 (root_pos, root_rot, dof_pos, dof_vel) 为 LocoMuJoCo 格式并扩展

输入格式:
  - fps: 频率
  - root_pos: (n_frames, 3) - 根位置
  - root_rot: (n_frames, 4) - 根旋转（四元数）
  - dof_pos: (n_frames, n_dof) - 关节角度
  - dof_vel: (n_frames, n_dof) - 关节速度 ← 新增，直接使用

输出格式:
  - 完整的 LocoMuJoCo Trajectory（包含扩展的物理数据）

使用方法:
    python convert_custom_format_and_extend_new.py \
        -i motion_data/input/dance1_subject2.npz \
        -o motion_data/output/dance1_extended.npz \
        --output-frequency 50
"""

import argparse
import os
import numpy as np
import jax.numpy as jnp
import mujoco
from scipy.spatial.transform import Rotation as R

from loco_mujoco.trajectory import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryData,
    TrajectoryModel,
)
from loco_mujoco.datasets.data_generation import ExtendTrajData
from loco_mujoco.environments import LocoEnv


def calculate_root_velocity(root_pos, root_rot, frequency):
    """
    计算根部的线速度和角速度

    Args:
        root_pos: (n_frames, 3) - xyz 位置
        root_rot: (n_frames, 4) - 四元数旋转 (xyzw)
        frequency: 频率 (Hz)

    Returns:
        root_linvel: (n_frames-1, 3) - 线速度
        root_angvel: (n_frames-1, 3) - 角速度
    """
    dt = 1.0 / frequency
    n_frames = root_pos.shape[0]

    # 线速度：位置差分
    root_linvel = np.diff(root_pos, axis=0) / dt

    # 角速度：四元数差分
    root_angvel = np.zeros((n_frames - 1, 3))

    for i in range(n_frames - 1):
        # 当前和下一帧的旋转
        q0 = root_rot[i]  # xyzw
        q1 = root_rot[i + 1]

        # 转换为 scipy 格式 (xyzw)
        r0 = R.from_quat(q0)
        r1 = R.from_quat(q1)

        # 计算相对旋转
        r_delta = r1 * r0.inv()

        # 转换为旋转向量（角速度 * dt）
        rotvec = r_delta.as_rotvec()

        # 除以 dt 得到角速度
        root_angvel[i] = rotvec / dt

    return root_linvel, root_angvel


def convert_custom_to_qpos_qvel_new(
    root_pos, root_rot, dof_pos, dof_vel, frequency, env_model
):
    """
    转换自定义格式为 qpos/qvel（使用提供的 dof_vel）

    Args:
        root_pos: (n_frames, 3) - xyz 位置
        root_rot: (n_frames, 4) - 四元数旋转
        dof_pos: (n_frames, n_dof) - 关节角度
        dof_vel: (n_frames, n_dof) - 关节速度
        frequency: 频率 (Hz)
        env_model: MuJoCo model

    Returns:
        qpos, qvel
    """

    n_frames = root_pos.shape[0]

    # 构建 qpos: [root_pos(3), root_rot(4), joints(n_dof)]
    qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

    print(f"  构建 qpos: {qpos.shape}")
    print(f"    - root_pos: {root_pos.shape}")
    print(f"    - root_rot: {root_rot.shape}")
    print(f"    - dof_pos: {dof_pos.shape}")

    # 验证维度
    expected_nq = env_model.nq
    if qpos.shape[1] != expected_nq:
        raise ValueError(
            f"qpos 维度不匹配! 构建的: {qpos.shape[1]}, 环境期望: {expected_nq}"
        )

    # 计算根部速度
    print(f"\n  计算根部速度...")
    root_linvel, root_angvel = calculate_root_velocity(root_pos, root_rot, frequency)

    # 构建 qvel: [root_linvel(3), root_angvel(3), joint_vel(n_dof)]
    # 注意：dof_vel 也需要裁剪以匹配根部速度的帧数
    qvel = np.concatenate(
        [root_linvel, root_angvel, dof_vel[:-1]], axis=1  # 去掉最后一帧
    )

    # qpos 也需要裁剪以匹配 qvel
    qpos_trimmed = qpos[:-1]

    print(f"  ✓ qpos: {qpos_trimmed.shape}")
    print(f"  ✓ qvel: {qvel.shape}")
    print(f"    - root_linvel: {root_linvel.shape}")
    print(f"    - root_angvel: {root_angvel.shape}")
    print(f"    - dof_vel: {dof_vel[:-1].shape}")
    print(f"  注意: 丢失了 1 帧（最后一帧，用于计算速度）")

    # 验证 qvel 维度
    expected_nv = env_model.nv
    if qvel.shape[1] != expected_nv:
        raise ValueError(
            f"qvel 维度不匹配! 构建的: {qvel.shape[1]}, 环境期望: {expected_nv}"
        )

    return qpos_trimmed, qvel


def convert_and_extend(
    input_file,
    output_file,
    env_name="PndAdamSp",
    output_frequency=None,
    visualize=False,
):
    """
    转换自定义格式并扩展为完整轨迹

    Args:
        input_file: 输入文件 (包含 fps, root_pos, root_rot, dof_pos, dof_vel)
        output_file: 输出文件 (LocoMuJoCo Trajectory 格式)
        env_name: 环境名称
        output_frequency: 输出频率 (Hz)，如果为 None 使用环境默认
        visualize: 是否可视化
    """

    print("=" * 80)
    print("转换自定义格式并扩展轨迹数据（包含 dof_vel）")
    print("=" * 80)
    print(f"\nInput: {input_file}")
    print(f"Output: {output_file}")
    print(f"Environment: {env_name}")
    if output_frequency is not None:
        print(f"Output Frequency: {output_frequency} Hz (用户指定)")
    print()

    # 1. 加载原始数据
    print("[1/5] Loading custom format data...")
    data = np.load(input_file, allow_pickle=True)

    print(f"  文件中的键: {list(data.keys())}")

    required_keys = ["fps", "root_pos", "root_rot", "dof_pos", "dof_vel"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(
            f"文件缺少必需的键: {missing_keys}. "
            f"需要: {required_keys}, 实际: {list(data.keys())}"
        )

    fps = int(data["fps"][0]) if data["fps"].shape else int(data["fps"])
    root_pos = data["root_pos"]
    root_rot = data["root_rot"]
    dof_pos = data["dof_pos"]
    dof_vel = data["dof_vel"]  # 新增

    print(f"  ✓ Loaded custom format:")
    print(f"    - fps: {fps} Hz")
    print(f"    - root_pos: {root_pos.shape}")
    print(f"    - root_rot: {root_rot.shape}")
    print(f"    - dof_pos: {dof_pos.shape}")
    print(f"    - dof_vel: {dof_vel.shape}  ← 直接使用")
    print(f"    - 总帧数: {root_pos.shape[0]}")
    print(f"    - 时长: {root_pos.shape[0] / fps:.2f} 秒")

    # 2. 创建环境
    print(f"\n[2/5] Creating {env_name} environment...")
    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls(th_params=dict(random_start=False, fixed_start_conf=(0, 0)))

    print(f"  ✓ Environment created:")
    print(f"    - Expected qpos dim: {env._model.nq}")
    print(f"    - Expected qvel dim: {env._model.nv}")
    print(f"    - Number of joints: {env._model.njnt}")

    # 3. 转换为 qpos/qvel
    print(f"\n[3/5] Converting to qpos/qvel format...")
    qpos, qvel = convert_custom_to_qpos_qvel_new(
        root_pos, root_rot, dof_pos, dof_vel, fps, env._model
    )

    n_samples = qpos.shape[0]

    # 创建 Trajectory 对象
    print(f"\n  Creating Trajectory object...")

    # 获取关节名称
    joint_names = [
        mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(env._model.njnt)
    ]

    traj_info = TrajectoryInfo(
        joint_names=joint_names,
        model=TrajectoryModel(
            njnt=env._model.njnt, jnt_type=jnp.array(env._model.jnt_type)
        ),
        frequency=float(fps),
    )

    traj_data = TrajectoryData(
        qpos=jnp.array(qpos),
        qvel=jnp.array(qvel),
        split_points=jnp.array([0, n_samples]),
    )

    traj = Trajectory(info=traj_info, data=traj_data)

    print(f"  ✓ Trajectory created:")
    print(f"    - Samples: {traj.data.n_samples}")
    print(f"    - Frequency: {traj.info.frequency} Hz")
    print(f"    - Duration: {n_samples / fps:.2f} seconds")
    print(f"    - Is complete: {traj.data.is_complete}")

    # 4. 插值到目标频率并扩展
    print(f"\n[4/5] Interpolating and extending...")

    from loco_mujoco.trajectory import interpolate_trajectories

    # 确定目标频率
    if output_frequency is None:
        target_freq = 1.0 / env.dt
        print(f"  Using environment default frequency: {target_freq} Hz")
    else:
        target_freq = output_frequency
        print(f"  Using user-specified frequency: {target_freq} Hz")

    # 如果频率已经匹配，跳过插值
    if abs(fps - target_freq) < 0.01:
        print(f"  ✓ Frequency already matches, skipping interpolation")
        traj_interp = traj
    else:
        print(f"  Interpolating from {fps}Hz to {target_freq}Hz...")
        traj_data_interp, traj_info_interp = interpolate_trajectories(
            traj.data, traj.info, target_freq
        )
        traj_interp = Trajectory(info=traj_info_interp, data=traj_data_interp)
        print(f"  ✓ Interpolated: {n_samples} → {traj_interp.data.n_samples} samples")

    # 加载到环境
    env.load_trajectory(traj_interp, warn=False)

    # 使用实际的样本数
    actual_n_samples = env.th.traj.data.n_samples
    print(f"  Actual samples after loading: {actual_n_samples}")

    # 扩展数据
    print(f"\n  Extending with physics data...")
    callback = ExtendTrajData(env, model=env._model, n_samples=actual_n_samples)

    env.play_trajectory(n_episodes=1, render=visualize, callback_class=callback)

    # 获取扩展数据
    traj_data_ext, traj_info_ext = callback.extend_trajectory_data(
        env.th.traj.data, env.th.traj.info
    )

    traj_extended = Trajectory(info=traj_info_ext, data=traj_data_ext)

    print(f"\n  ✓ Extension completed!")
    print(f"    - Extended samples: {traj_extended.data.n_samples}")
    print(f"    - Extended frequency: {traj_extended.info.frequency} Hz")
    print(f"    - Is complete: {traj_extended.data.is_complete}")
    print(f"    - xpos shape: {traj_extended.data.xpos.shape}")
    print(f"    - site_xpos shape: {traj_extended.data.site_xpos.shape}")

    # 如果用户指定了输出频率且与当前频率不同，重新插值
    if (
        output_frequency is not None
        and abs(traj_extended.info.frequency - output_frequency) > 0.01
    ):
        print(
            f"\n  重新插值到目标频率: {traj_extended.info.frequency} Hz → {output_frequency} Hz..."
        )
        traj_data_final, traj_info_final = interpolate_trajectories(
            traj_extended.data, traj_extended.info, output_frequency
        )
        traj_extended = Trajectory(info=traj_info_final, data=traj_data_final)
        print(f"  ✓ 最终样本数: {traj_extended.data.n_samples}")
        print(f"  ✓ 最终频率: {traj_extended.info.frequency} Hz")

    # 5. 保存
    print(f"\n[5/5] Saving...")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    traj_extended.save(output_file)
    print(f"  ✓ Saved to: {output_file}")

    # 文件大小
    input_size = os.path.getsize(input_file) / 1024 / 1024
    output_size = os.path.getsize(output_file) / 1024 / 1024

    print(f"\n  File sizes:")
    print(f"    - Input:  {input_size:.2f} MB")
    print(f"    - Output: {output_size:.2f} MB")
    print(f"    - Ratio:  {output_size/input_size:.1f}x")

    print("\n" + "=" * 80)
    print("✓ Conversion and extension completed!")
    print("=" * 80)

    return traj_extended


def batch_convert_and_extend(
    input_dir,
    output_dir,
    env_name="PndAdamSp",
    output_frequency=None,
    pattern="*.npz",
    visualize=False,
):
    """
    批量转换目录中的所有文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        env_name: 环境名称
        output_frequency: 输出频率
        pattern: 文件匹配模式
        visualize: 是否可视化
    """
    import glob

    # 查找所有匹配的文件
    search_pattern = os.path.join(input_dir, pattern)
    input_files = glob.glob(search_pattern)

    if not input_files:
        print(f"错误: 在 {input_dir} 中未找到匹配 {pattern} 的文件")
        return

    print("=" * 80)
    print(f"批量转换模式")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"匹配模式: {pattern}")
    print(f"找到 {len(input_files)} 个文件")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 批量处理
    success_count = 0
    failed_count = 0
    failed_files = []

    for i, input_file in enumerate(input_files, 1):
        filename = os.path.basename(input_file)
        output_file = os.path.join(
            output_dir, filename.replace(".npz", "_extended.npz")
        )

        print(f"\n[{i}/{len(input_files)}] 处理: {filename}")
        print("-" * 80)

        try:
            convert_and_extend(
                input_file,
                output_file,
                env_name,
                output_frequency,
                visualize,
            )
            success_count += 1
            print(f"✓ 成功: {filename}")
        except Exception as e:
            failed_count += 1
            failed_files.append(filename)
            print(f"✗ 失败: {filename}")
            print(f"  错误: {e}")
            import traceback

            traceback.print_exc()

    # 总结
    print("\n" + "=" * 80)
    print("批量转换完成")
    print("=" * 80)
    print(f"总文件数: {len(input_files)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")

    if failed_files:
        print(f"\n失败的文件:")
        for f in failed_files:
            print(f"  - {f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom format (fps, root_pos, root_rot, dof_pos, dof_vel) to extended LocoMuJoCo Trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单文件转换
  python convert_custom_format_and_extend_new.py -i input.npz -o output.npz -of 50
  
  # 批量转换
  python convert_custom_format_and_extend_new.py --batch -i input_dir/ -o output_dir/ -of 50
  
  # 批量转换指定模式
  python convert_custom_format_and_extend_new.py --batch -i input_dir/ -o output_dir/ --pattern "dance*.npz"

注意:
  此版本需要输入数据包含 dof_vel 字段，直接使用而不是通过有限差分计算。
        """,
    )

    # 批量模式
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="批量转换模式（处理整个目录）",
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入文件（单文件模式）或输入目录（批量模式）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="输出文件（单文件）或输出目录（批量）。默认: {input}_extended.npz 或 output/",
    )
    parser.add_argument(
        "--output-frequency",
        "-of",
        type=float,
        default=None,
        help="输出轨迹频率 (Hz)。如果未指定，使用环境默认频率 (100Hz)",
    )
    parser.add_argument(
        "--env",
        default="PndAdamSp",
        help="环境名称 (default: PndAdamSp)",
    )
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="批量模式下的文件匹配模式 (default: *.npz)",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="可视化扩展过程",
    )

    args = parser.parse_args()

    if args.batch:
        # 批量模式
        input_dir = args.input
        output_dir = args.output if args.output else "output"

        if not os.path.isdir(input_dir):
            print(f"错误: 输入目录不存在: {input_dir}")
            return

        batch_convert_and_extend(
            input_dir,
            output_dir,
            args.env,
            args.output_frequency,
            args.pattern,
            args.visualize,
        )
    else:
        # 单文件模式
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_extended.npz"

        convert_and_extend(
            args.input,
            args.output,
            args.env,
            args.output_frequency,
            args.visualize,
        )


if __name__ == "__main__":
    main()
