# GMR 脚本与数据格式参考

本文档汇总 `scripts/` 下与 **动捕、Retarget、NPZ/PT/PKL、播放与调试** 相关脚本的用途、**支持格式**与常用参数；并说明 **`ik_configs/`**、`utils/lafan1.py` 与 **`motion_retarget.py`（`GeneralMotionRetargeting`）** 的配置含义与地面对齐逻辑。安装与项目架构见根目录 `CLAUDE.md`；论文与通用用法见根目录 `README.md`。

---

## 1. 共性说明

### 1.1 人体 / 机器人数据流

- **人体**：每帧为 `{body_name: [position(3), rotation_quat(4)]}`，旋转在内部多为 **wxyz**（与 MuJoCo `qpos` 一致）。
- **机器人**：每帧等价于 `qpos`：`root 平移(3) + root 四元数(4) + 关节角(N)`；根四元数 **wxyz**。
- **IK 配置**：`general_motion_retargeting/params.py` 中 `IK_CONFIG_DICT` 将 `src_human`（如 `bvh_lafan1`、`bvh_mocap`、`smplx`、`fbx`、`pt`）映射到 `ik_configs/*.json`。字段级说明见 **§2**。
- **场景 XML**：`ROBOT_XML_DICT[robot]`，播放类脚本（如 `gmr_play.py`、`play_pt_motion.py`）从该字典加载 MuJoCo 场景。
- **BVH 解析**：`load_bvh_file` 见 **§3**；**IK 与 Retarget、地面对齐**见 **§4**。

### 1.2 BVH `--format`（`load_bvh_file`）

| 取值 | 含义（典型用途） |
|------|------------------|
| `lafan1` | LAFAN1 等标准骨架 |
| `nokov` | Nokov 动捕 |
| `sfu` | SFU 等 |
| `noitom` | Noitom |
| `mocap` | 通用 mocap BVH |
| `opt_mocap` | OptiTrack 风格（部分脚本列出） |

未在脚本 `choices` 中出现的格式以对应脚本的 `argparse` 为准。

### 1.3 常见文件后缀与内容

| 后缀 | 常见内容 |
|------|----------|
| `.npz` | NumPy 压缩包：`fps`/`framerate`、`root_pos`、`root_rot`(wxyz)、`dof_pos`、`dof_vel`；或 G1 的 `base_pos_w`、`base_quat_w`、`joint_pos` 等 |
| `.pkl` | Python pickle：字典含上述字段或 `qpos`；部分管线 `root_rot` 存 **xyzw** |
| `.pt` | PyTorch：`dict`，常见键 `base_position`、`joint_position`、`base_pose` 或 `base_quat`、`framerate`/`fps` 等 |

### 1.4 四元数约定（易混点）

- **MuJoCo `qpos` / 本仓库多数 NPZ 导出**：根姿态 **wxyz**。
- **部分 PKL / 旧脚本保存**：可能为 **xyzw**；`play_pt_motion.py` 加载 PKL 时会将 `root_rot` 从 xyzw 转为 wxyz。
- **`gmr_play.py`**：用 `--quat-format wxyz|xyzw` 指明输入根四元数顺序。
- **`play_pt_motion.py` 的 `.pt`**：默认 `base_quat` 为 **xyzw**；若为 wxyz 请加 `--quat-wxyz`。

---

## 2. IK 配置目录 `general_motion_retargeting/ik_configs/`

### 2.1 作用

每个 JSON 描述 **某一种人体数据源**（BVH 骨架名、SMPL-X 关节名等）到 **目标机器人 MuJoCo body / link** 的对应关系，以及 **IK 任务权重、姿态/位置偏置、人形缩放**。`GeneralMotionRetargeting` 在初始化时根据 `src_human` + `tgt_robot` 从 `params.IK_CONFIG_DICT` 打开**唯一**一个配置文件。

### 2.2 与代码的绑定方式

映射表在 `general_motion_retargeting/params.py` 的 **`IK_CONFIG_DICT`**：

- 第一层键 **`src_human`**：数据源类型，须与构造 `GMR(..., src_human=...)` 时一致，例如：
  - `smplx`：SMPL-X / GVHMR
  - `bvh_lafan1`、`bvh_nokov`、`bvh_noitom`、`bvh_mocap`、`bvh_opt_mocap`、`bvh_joint_mocap` 等：对应不同 BVH 解析与骨架约定
  - `fbx` / `fbx_offline`：OptiTrack 等 FBX 管线
  - `pt`：由 G1 `qpos` 经 FK 得到的人体关键点（见 `npz_to_robot_npz.py`）
- 第二层键 **`tgt_robot`**：如 `unitree_g1`、`adam_sp`，指向具体 JSON 路径。

新增机器人或数据源时，通常需要：**新建或复制 JSON** → **在 `IK_CONFIG_DICT` 中注册** → 保证 JSON 里的 **人体 body 名称**与 `lafan1.load_bvh_file` / SMPL 管线输出的 **键名一致**。

### 2.3 JSON 字段含义（摘要）

| 字段 | 含义 |
|------|------|
| `human_root_name` / `robot_root_name` | 人体根关节名（如 `Hips`）、机器人根 body 名（如 `pelvis`），用于缩放时的参考系 |
| `human_height_assumption` | 配置调参时假设的人高；若传入 `actual_human_height`，会按比例缩放 `human_scale_table` |
| `ground_height` | 与任务偏置中的“地面”向量一起参与偏移（见源码中 `self.ground`） |
| `human_scale_table` | 各人体部位相对根的 **位置缩放** 系数（腿长、躯干等可分开调） |
| `use_ik_match_table1` / `use_ik_match_table2` | 是否启用第一 / 二组 IK 任务（两组可分工：例如先躯干四肢、再精细调整） |
| `ik_match_table1` / `ik_match_table2` | 见下表 |

**`ik_match_table*` 中每一项**：`"机器人_body_名": [ "人体关节名", pos_weight, rot_weight, pos_offset(3), rot_offset(4) ]`

- **人体关节名**必须与当前帧 `human_data` 字典的 **key** 一致（例如 BVH 经 `load_bvh_file` 得到的 `LeftFootMod`）。
- **pos_weight / rot_weight**：Mink `FrameTask` 的位置 / 姿态代价权重；为 0 表示该维度不参与优化。
- **pos_offset**：在 **更新后的局部旋转** 下转成全局平移偏置后加到位置上。
- **rot_offset**：四元数 **wxyz**，与人体四元数右乘，用于坐标系或 T-pose 对齐。

仓库内已有大量 `smplx_to_*.json`、`bvh_*_to_*.json`、`fbx_*.json` 等，可按最接近的骨架 **复制后改人体名与机器人 frame 名**。

---

## 3. `lafan1.py`：`load_bvh_file`

### 3.1 职责

从标准 **BVH** 读入骨架，经 `read_bvh` + 前向运动学 `quat_fk` 得到全局位姿，再施加 **固定轴变换**（与历史 LAFAN1 脚本一致），把 **厘米转米**，输出 **每一帧一个字典**：

```text
{ bone_name: [ position(3), orientation_quat(4) ] }  # orientation 为 wxyz
```

### 3.2 `format` 参数（与 `--format` 对应）

不同厂商骨架 **脚趾 / 脚踝命名不同**，脚本用同一套逻辑算全局位姿后，再按 `format` **合成 `LeftFootMod` / `RightFootMod`**（供 IK 里脚底/脚踝任务使用）：

| `format` | 脚部辅助点语义 |
|----------|------------------|
| `lafan1` | 左脚掌位置 + **LeftToe 的朝向**；右脚同理 |
| `nokov` | 使用 `LeftToeBase` 等 |
| `sfu` | 同 nokov 风格（ToeBase） |
| `noitom` | 脚位 + **脚自身**朝向 |
| `mocap` | 脚位 + 脚朝向（类 FBX） |
| `opt_mocap` | 使用 `ankle_l` / `ankle_r` 作为 FootMod |

若传入未实现的 `format`，会 `raise ValueError`。

### 3.3 返回值

- **`frames`**：上述字典的列表，长度 = 动画帧数。
- **`human_height`**：当前实现中为 **常数 `1.75`**（米），供 `GMR(actual_human_height=...)` 使用；若需按真实身高缩放，可在脚本侧传入实测身高覆盖默认逻辑。

### 3.4 与 `bvh_joint_to_robot.py` 的区别

`bvh_joint_to_robot` 使用 **`lafan_vendor.extract.read_bvh` + 另一套后处理**（`bvh_joint_to_robot.py` 内 `load_bvh_frames_direct`），**不经过** `load_bvh_file` 的 FootMod 分支；其 IK 使用 `bvh_joint_mocap` 等配置，骨架关节名需与对应 `ik_configs` 一致。

---

## 4. `motion_retarget.py`：`GeneralMotionRetargeting`

### 4.1 类职责

封装 **MuJoCo 模型 + Mink IK**：把一帧 `human_data` 转成机器人 `qpos`（及 `qvel`，见下）。

### 4.2 初始化参数（常用）

| 参数 | 说明 |
|------|------|
| `src_human` / `tgt_robot` | 选定 `IK_CONFIG_DICT` 与 `ROBOT_XML_DICT` |
| `actual_human_height` | 若给定，按与配置中 `human_height_assumption` 的比例缩放 `human_scale_table` |
| `solver` / `damping` | Mink 求解器（默认 `daqp`）与阻尼 |
| `use_velocity_limit` | 默认 True，为各 actuator 施加速度限幅（见源码） |
| `use_collision_limit` | 预留；当前碰撞相关代码多为注释状态 |
| `base_height_offset` | 与 **`offset_human_data_to_ground_fly`** 一起，在“飞行动画”贴地模式里整体抬高人体（对齐机器人站立高度） |

初始化时会打印机器人 **DoF / Body / Motor** 名称与顺序，便于对齐导出 NPZ 的关节顺序。

### 4.3 `update_targets` / `retarget` 与地面对齐

处理顺序概览：**相邻帧四元数符号连续** → **按根与 `human_scale_table` 缩放** → **IK 配置中的 pos/rot offset** → **`apply_ground_alignment` 分支** → 写入 Mink 任务目标。

#### 4.3.1 源码中的默认参数（注意两处不一致）

| 函数 | `offset_to_ground` | `no_fly` | `apply_ground_alignment` |
|------|--------------------|----------|---------------------------|
| **`retarget(...)`** | 默认 **`True`** | 默认 **`False`** | 默认 **`True`** |
| **`update_targets(...)`** | 默认 **`False`** | 默认 **`False`** | 默认 **`True`** |

直接调用 **`update_targets`** 时若省略参数，会得到 **`offset_to_ground=False`**；而 **`retarget`** 再转给 `update_targets` 时默认传 **`offset_to_ground=True`**。因此**批量脚本应优先通过 `retarget(...)` 调用**，或显式传入与期望一致的三个参数。

#### 4.3.2 `offset_to_ground` 与 `no_fly` 的含义

二者**仅在 `apply_ground_alignment=True` 时**参与分支；逻辑为（见 `motion_retarget.py`）：

```python
if apply_ground_alignment:
    if offset_to_ground and no_fly:
        human_data = offset_human_data_to_ground(human_data)   # 逐帧贴地
    else:
        human_data = offset_human_data_to_ground_fly(human_data)  # “飞行动画”式贴地
```

- **`offset_to_ground`**：字面是“对齐地面”，但实现上**必须与 `no_fly` 同时为 `True`** 才会进入 **`offset_human_data_to_ground`**（逐帧用脚部/全身最低点把 z 压到地面）。单独为 `True` 而 `no_fly=False` 时，仍会走 **`_ground_fly`** 分支。
- **`no_fly`**：与 `offset_to_ground` **成对使用**。名为 “no fly”，表示不要采用“整段轨迹共用一个 z 偏移”的 **`_ground_fly`** 模式，而是采用**逐帧**贴地（跳跃时脚仍跟着地面走，而不是整段被第一次的最低点锁住）。**仅当 `offset_to_ground and no_fly` 同时为 `True`** 时生效。

简要记忆：**逐帧贴地** = `apply_ground_alignment=True` 且 **`offset_to_ground=True` 且 `no_fly=True`**；**首帧定高、全序列平移（fly）** = `apply_ground_alignment=True` 且 **非**（两者同时为 True）。

#### 4.3.3 组合表

| `apply_ground_alignment` | `offset_to_ground` | `no_fly` | 行为 |
|--------------------------|--------------------|-----------|------|
| **`False`** | 任意 | 任意 | **不做**整体 z 向贴地（`offset_human_data_to_ground*` 均不调用）。适合楼梯、跳跃、或已处于绝对高度坐标系的数据。 |
| **True** | **True** | **True** | **`offset_human_data_to_ground`**：每帧按 **含 `Foot`/`foot` 的关节**取最低 z 对齐地面；若无此类命名则回退为全身最低 z。 |
| **True** | 其余（含默认 `retarget` 的 `True, False`） | | **`offset_human_data_to_ground_fly`**：仅在**第一次**调用时用当前帧算 `lowest_pos`，之后整段序列复用该偏移，减轻跳跃被压回地面。 |

脚本 **`npz_to_robot_npz.py`**：`--align-ground` 会同时将 **`offset_to_ground=True`** 与 **`apply_ground_alignment=True`**；`--no-fly` 单独设置 **`no_fly`**。因此 **逐帧贴地**需 **`--align-ground --no-fly`**；仅 `--align-ground` 时仍为 **`_ground_fly`**（与 `retarget` 默认 `no_fly=False` 一致）。

### 4.4 IK 求解与返回值

- 若启用 `ik_match_table1`，先对 `tasks1` 迭代 `solve_ik` 直至误差下降变慢；再同样处理 `tasks2`（若启用）。
- **`retarget` 返回** `(qpos, qvel)`：源码注释说明 **`qvel` 并非沿真实轨迹的运动学速度**；若需要速度，应对 `qpos` 序列做数值差分（如中心差分）。

---

## 5. 脚本一览（按功能）

### 5.1 BVH → 机器人（LAFAN1 管线，`load_bvh_file`）

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `bvh_to_robot.py` | `.bvh` | 默认 `retarget/<robot>/<format>/<name>.pkl` | 单条；`--format`；`--robot`；`--motion_fps` 或从 BVH `Frame Time` 推断；`--start_frame`/`--end_frame`；可视化 + 可选 `--save_path`。`src_human` 为 `bvh_<format>`。 |
| `bvh_to_robot_dataset.py` | 目录下递归 `.bvh` | 镜像目录结构 `.pkl` | 批量；`--target_fps` 默认 30；`--format`；`--override`。 |
| `bvh_to_robot_npz.py` | `.bvh` | `.npz` | 单条数据集风格；`--target_fps`；`--compressed`；`--compute_local_body_pos`；`--height_adjust`/`--perframe_adjust`；新增 `--format jpg`：适配“简化 LAFAN-like 骨架”（缺 Foot/Toe/Hand）。代码会合成 IK 需要的 key：`Spine2=Chest/Spine`、`LeftUpLeg=LeftLeg`、`LeftLeg=LeftShin`、`LeftFootMod=LeftShin`、`LeftHand=LeftForeArm`（右侧同理）；默认保存路径 `retarget/<robot>/<format>/<name>.npz`。 |
| `bvh_to_robot_npz_dataset.py` | `--src_folder` | `--tgt_folder` 下 `.npz` | 批量 NPZ；可选 `--record_video`；`--compressed` 等；机器人列表见脚本内 `choices`（含 `pnd_adam_lite`、`adam_sp` 等）。 |

### 5.2 BVH → 机器人（关节级 / 直接读 BVH，不经 `lafan1` 后处理）

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `bvh_joint_to_robot.py` | `.bvh` | 默认 `retarget/<robot>/<src_dataset>/<name>.npz` | `--src_human`：`bvh_joint_mocap`、`bvh_opt_mocap`、`bvh_opt_mocap_footmod`；`--robot`：`adam_sp` / `adam_sp_pro`；`--compressed`；首帧校验 IK 所需关节；`--print_rot_frame` / `--print_only` 调试。NPZ：`fps`、`root_pos`、`root_rot`(wxyz)、`dof_pos`、`dof_vel`。 |

### 5.3 SMPL-X / GVHMR

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `smplx_to_robot.py` | AMASS 等 **SMPL-X `.npz`** | 可选 `.pkl`（`--save_path`） | `--robot`；`src_human=smplx`；内部重采样 `tgt_fps=30`；`--loop`、`--record_video`、`--rate_limit`。需 `assets/body_models/smplx/`。 |
| `gvhmr_to_robot.py` | GVHMR **`hmr4d_results.pt`** | 可选保存 | `--gvhmr_pred_file`；`--robot`；同样 30fps 对齐；可视化为主。 |
| `gvhmr_to_robot_dataset.py` | 目录递归 **`hmr4d_results.pt`** | 每运动子目录 `*_poses.pkl` + 视频 | `--src_folder`、`--tgt_folder`、`--robot`、`--override`；`--record_video`（默认 True）、`--rate_limit`；`--offset_ground`、`--joint_vel_limit`、`--collision_avoid` 等。**注意**：脚本内 `SMPLX_FOLDER` 指向外部工程路径，跨机使用需改路径或对齐 body_models。 |

### 5.4 NPZ / JSON 转换与其它 Retarget

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `npz_to_robot_npz.py` | **G1 风格 `.npz`**（`base_pos_w`/`root_pos` + `base_quat_w`/`root_rot` + `joint_pos` 等别名） | 目标机器人 `.npz`：`base_pos_w`、`base_quat_w`(wxyz)、`joint_pos`、`joint_names`、`labels`、`framerate` | `--target_robot`（默认 `adam_sp`）；`--quat-xyzw`；`--align-ground`、`--no-fly`；`--no-vis` 纯离线；`--fps`、`--scale`、`--no-human-scale`。 |
| `json_to_robot_npz.py` | 含 `Labels` + `Frames` 的 **JSON** | 与 `bvh_to_robot_npz` 对齐的 **NPZ** | `--robot adam_lite|adam_sp|custom`；`--joint_config`；`--compressed`；缺关节填 0。 |

### 5.5 G1 关节维度互转（`.pt`）

| 脚本 | 作用 |
|------|------|
| `g1_27pt_to_29pt.py` | `joint_position`/`joint_velocity` 在索引 13、14 插入 **waist_roll、waist_pitch=0**，27→29。 |
| `g1_29pt_to_27pt.py` | 去掉索引 13、14，29→27。 |

用法：`python scripts/g1_27pt_to_29pt.py <in.pt> [out.pt]`（另一脚本同理）。

### 5.6 播放与可视化

| 脚本 | 支持格式 | 说明 |
|------|----------|------|
| `gmr_play.py` | `.npz` / `.pkl` / G1 `.npy` `(T,58)` | `python scripts/gmr_play.py <robot> <motion>`；`--quat-format`、`--fps`。多种 NPZ 键组合见脚本内 `load_motion_data`。 |
| `play_pt_motion.py` | `.pt` / `.pkl` | `--robot unitree_g1|adam_sp`；`.pt` 需 `base_position`+`joint_position` + `base_pose` 或 `base_quat`；`--quat-wxyz`；`--fps`、`--no-loop`、`--height-offset`。 |

### 5.7 解析 / 调试 / 实时

| 脚本 | 作用 |
|------|------|
| `parse_npz.py` | 打印 `.npz`/`.npy` 的 keys、shape、推断 fps；可选 `--out_json`。 |
| `parse_g1_pt.py` | 解析 G1 **`.pt`**：keys、shape、推断与 XML 一致的 29 关节 **labels**；`--out_json`。 |
| `print_pt_first_frames.py` | 打印 `.pt` 顶层键及前 `--num_frames` 帧。 |
| `inspect_box_in_pt.py` | 深度检查 `.pt`（含 `link_position`、`box_*`、`chair_*` 等）；`--viz` 交互可视化。 |
| `inspect_bvh_first_frame_rot.py` | 指定 BVH `--format` 与 `--robot`，对 `--frame` 打印/可选 `--visualize` IK 目标姿态。 |
| `optitrack_to_robot.py` | **实时** NatNet：`--server_ip`、`--client_ip`、`--use_multicast`；`src_human=fbx`；需关闭防火墙等。 |

---

## 6. 命令速查（复制前请 `python scripts/<name>.py -h` 核对）

```bash
# BVH 单条 PKL
python scripts/bvh_to_robot.py --bvh_file X.bvh --robot unitree_g1 --format lafan1

# BVH 单条 NPZ
python scripts/bvh_to_robot_npz.py --bvh_file X.bvh --robot adam_sp --format mocap --compressed

# BVH 批量 PKL
python scripts/bvh_to_robot_dataset.py --src_folder ./in --tgt_folder ./out --robot unitree_g1

# BVH 批量 NPZ
python scripts/bvh_to_robot_npz_dataset.py --src_folder ./in --tgt_folder ./out --robot adam_sp --compressed

# 关节级 BVH → NPZ
python scripts/bvh_joint_to_robot.py --bvh_file X.bvh --robot adam_sp --src_human bvh_joint_mocap --compressed

# SMPL-X / GVHMR
python scripts/smplx_to_robot.py --smplx_file X.npz --robot unitree_g1
python scripts/gvhmr_to_robot.py --gvhmr_pred_file hmr4d_results.pt --robot unitree_g1

# G1 NPZ → Adam NPZ
python scripts/npz_to_robot_npz.py g1_motion.npz --target_robot adam_sp

# JSON → NPZ
python scripts/json_to_robot_npz.py --json_file motion.json --robot adam_sp --compressed

# 播放
python scripts/gmr_play.py adam_sp motion.npz --quat-format wxyz
python scripts/play_pt_motion.py data.pt --robot unitree_g1

# 解析
python scripts/parse_npz.py motion.npz
python scripts/parse_g1_pt.py motion.pt --out_json meta.json
```

---

## 7. 机器人模型与 IK 配置注意事项

- **Adam 系列**（`adam_lite`、`adam_sp`、`adam_sp_pro` 等）在机构与 XML 上仍有差异，**不能**混用同一套 IK JSON：需保证 `ik_match_table*` 里的 **机器人 frame 名**与当前 `assets/.../scene.xml` 引用的 **body/link 名**一致，人体侧键名与 `load_bvh_file` 或关节 BVH 输出一致。
- 根目录 **README.md** 保留论文、安装与 **SMPL-X / BVH / GVHMR / FBX** 主流程。
- **本文档** 覆盖本仓库扩展脚本（NPZ 批处理、Adam、关节 BVH、PT 工具链）及 **IK 配置、`lafan1`、`motion_retarget` 核心行为**。若参数与代码不一致，**以源码与 `python scripts/<name>.py -h` 为准**。
