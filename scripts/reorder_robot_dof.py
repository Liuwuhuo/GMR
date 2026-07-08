#!/usr/bin/env python3
"""
通用机器人 dof 列顺序重排：在 DFS(MuJoCo qpos 顺序) 与 BFS(Isaac Lab 顺序) 之间互转。

与 reorder_g1_joints.py 的区别：
  - 不硬编码关节名，直接从目标机器人的 MuJoCo 模型推导两种顺序：
      * DFS：遍历 njnt，跳过 free root 后的 hinge/slide 关节顺序（= qpos[7:] 的顺序，也是本仓库 npz 的列顺序）。
      * BFS：从运动树 world 的子节点做广度优先遍历，按 body 层级交错收集关节（= Isaac Lab 顺序）。
  - 支持本仓库 retarget 输出的 npz（键：root_pos / root_rot / dof_pos / dof_vel），按列重排 dof_pos/dof_vel。

示例：
  # adam_sp 的 retarget npz：DFS -> BFS（喂给期望 Isaac 顺序的 deploy/policy）
  python scripts/reorder_robot_dof.py reorder_out/right_kick_adam_v2.npz \
      --robot adam_sp --src_order dfs --dst_order bfs -o reorder_out/right_kick_adam_v2_bfs.npz

  # 只想看顺序、不转换
  python scripts/reorder_robot_dof.py x.npz --robot adam_sp --print_only
"""
import argparse
import collections
import pathlib
import sys

import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.params import ROBOT_XML_DICT

# 本仓库 npz 中按 dof 列存储、需要同步重排的键
DOF_KEYS = ("dof_pos", "dof_vel", "joint_pos", "joint_vel")


def get_actuated_joint_ids(model) -> list[int]:
    """返回去掉 free root 之后的关节 id 列表（hinge/slide），顺序即 qpos[7:] 的列顺序。"""
    ids = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        ids.append(j)
    return ids


def dfs_order(model) -> list[str]:
    """DFS / MuJoCo qpos 顺序的关节名。"""
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in get_actuated_joint_ids(model)
    ]


def bfs_order(model) -> list[str]:
    """BFS / Isaac Lab 顺序：从运动树 world 子节点广度优先遍历，按层级交错收集关节。"""
    body_joints = collections.defaultdict(list)
    for j in get_actuated_joint_ids(model):
        body_joints[model.jnt_bodyid[j]].append(j)
    children = collections.defaultdict(list)
    for b in range(1, model.nbody):
        children[model.body_parentid[b]].append(b)
    order = []
    queue = collections.deque(children[0])
    while queue:
        b = queue.popleft()
        for j in body_joints.get(b, []):
            order.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
        for c in children[b]:
            queue.append(c)
    return order


def build_perm(src_names: list[str], dst_names: list[str]) -> np.ndarray:
    """返回 perm，使 out[:, k] = in[:, perm[k]]（k 为目标 dst 列索引）。"""
    assert sorted(src_names) == sorted(dst_names), "DFS/BFS 关节集合不一致"
    src_index = {name: i for i, name in enumerate(src_names)}
    return np.array([src_index[name] for name in dst_names], dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description="通用机器人 dof 列 DFS<->BFS 重排（按模型运动树自动推导顺序）")
    p.add_argument("input", help="输入 .npz（含 dof_pos/dof_vel 或 joint_pos/joint_vel）")
    p.add_argument("--robot", required=True, help="目标机器人名（用于查 MuJoCo 模型，如 adam_sp）")
    p.add_argument("--src_order", choices=["dfs", "bfs"], default="dfs",
                   help="输入数据关节顺序（默认 dfs，本仓库 retarget 输出即 dfs）")
    p.add_argument("--dst_order", choices=["dfs", "bfs"], default="bfs",
                   help="输出数据关节顺序（默认 bfs）")
    p.add_argument("-o", "--output", default=None, help="输出路径")
    p.add_argument("--compress", action="store_true", help="np.savez_compressed")
    p.add_argument("--print_only", action="store_true", help="只打印 DFS/BFS 顺序与 perm，不转换")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    orders = {"dfs": dfs_order(model), "bfs": bfs_order(model)}
    ndof = len(orders["dfs"])
    print(f"[reorder] robot={args.robot} 关节数={ndof}")
    if args.print_only:
        for k in ("dfs", "bfs"):
            print(f"--- {k.upper()} ---")
            for i, n in enumerate(orders[k]):
                print(f"  {i:2d} {n}")
        return

    if args.src_order == args.dst_order:
        print("src_order == dst_order，无需重排。", file=sys.stderr)
        sys.exit(1)

    perm = build_perm(orders[args.src_order], orders[args.dst_order])
    print(f"[reorder] {args.src_order.upper()} -> {args.dst_order.upper()} perm={perm.tolist()}")

    in_path = pathlib.Path(args.input)
    if not in_path.is_file():
        print(f"文件不存在: {in_path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(in_path, allow_pickle=True)
    out = {}
    changed = []
    for k in data.files:
        v = data[k]
        if k in DOF_KEYS and getattr(v, "ndim", 0) == 2 and v.shape[1] == ndof:
            out[k] = np.asarray(v)[:, perm].astype(v.dtype)
            changed.append(k)
        else:
            out[k] = v
    if not changed:
        print(f"[warn] 未找到可重排的 dof 键({DOF_KEYS}) 且列数={ndof} 的数组，输出与输入一致。",
              file=sys.stderr)

    out_path = pathlib.Path(args.output) if args.output else \
        in_path.with_name(f"{in_path.stem}_{args.dst_order}{in_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.compress:
        np.savez_compressed(out_path, **out)
    else:
        np.savez(out_path, **out)
    print(f"已保存 npz: {out_path}  已重排键={changed}  其余键原样保留")


if __name__ == "__main__":
    main()
