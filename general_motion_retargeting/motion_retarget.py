
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=1e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=True,
        use_collision_limit: bool=True,
        base_height_offset: float = 0.0,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping
        self.use_collision_limit = use_collision_limit

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3 * np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 

        # if self.use_collision_limit:
        #     collision_pairs = [
        #         (["thighLeft_collision", "thighRight_collision", "shinLeft_collision", "shinRight_collision", "wristRollLeft_collision", "wristRollRight_collision"], ["floor"]),
        #         (["wristRollLeft_collision"], ["thighLeft_collision"]),
        #         (["wristRollRight_collision"], ["thighRight_collision"]),
        #         (["wristRollLeft_collision"], ["wristRollRight_collision"]),
        #     ]
        #     self.collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        #         model=self.model,
        #         geom_pairs=collision_pairs,  # type: ignore
        #         minimum_distance_from_collisions=0.01,
        #         # gain=0.3,
        #         collision_detection_distance=0.3,
        #     )
        #     self.ik_limits.append(self.collision_avoidance_limit)
        #     print(f"Collision avoidance limit初始化完成，共 {len(self.collision_avoidance_limit.geom_id_pairs)} 个geom对")


            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0
        self.base_height_offset = base_height_offset

        self.last_human_data = None

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

  
    def update_targets(
        self,
        human_data,
        offset_to_ground=False,
        no_fly=False,
        apply_ground_alignment=True,
    ):
        # === 帧四元数符号对齐 ===
        if self.last_human_data is not None:
            aligned_human_data = {}
            for body_name in human_data:
                pos, quat = human_data[body_name]
                last_quat = self.last_human_data[body_name][1]
                if np.dot(last_quat, quat) < 0:
                    quat = -quat
                aligned_human_data[body_name] = [pos, quat]
            human_data = aligned_human_data
        self.last_human_data = human_data  # 缓存当前帧
        # ==========================
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        # self.ground_offset = self.calculate_foot_bottom_offset()
        if apply_ground_alignment:
            if offset_to_ground and no_fly:
                human_data = self.offset_human_data_to_ground(human_data)
            else:
                human_data = self.offset_human_data_to_ground_fly(human_data)
            
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget(
        self,
        human_data,
        offset_to_ground=True,
        no_fly=False,
        apply_ground_alignment=True,
    ):
        # Update the task targets
        self.update_targets(
            human_data,
            offset_to_ground=offset_to_ground,
            no_fly=no_fly,
            apply_ground_alignment=apply_ground_alignment,
        )

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep

            # if self.use_collision_limit:
            #     self._print_collision_constraints(dt)

            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                # if self.use_collision_limit:
                #     self._print_collision_constraints(dt, num_iter)
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
                
            
        # qpos 由 IK 积分得到；qvel 未按轨迹更新，通常为 0 或最后一轮 IK 速度，不能当作运动速度。
        # 若需要沿轨迹的速度，调用方应用 qpos 序列做中心差分（见 qvel_from_qpos_central）。
        return self.configuration.data.qpos.copy(), self.configuration.data.qvel.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data

    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        lowest_pos = np.inf
        found_foot_like_joint = False

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                found_foot_like_joint = True

        # Fallback for datasets that do not name foot joints with "Foot/foot"
        # (e.g., ankle/toe naming only). This avoids lowest_pos staying +inf.
        if not found_foot_like_joint:
            lowest_pos = min(pos[2] for pos, quat in human_data.values())

        if not np.isfinite(lowest_pos):
            raise ValueError(
                f"Invalid ground reference z={lowest_pos}. "
                "Please check input human_data values."
            )
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos])# - np.array([0, 0, self.ground_offset])
        return offset_human_data
            
    def offset_human_data_to_ground_fly(self, human_data):
        """
        将人体整体沿 z 方向平移，使「最低点」落在 z=0。
        注意：
            - 为避免跳跃动作被强制“粘在地面”，最低点只在**第一次调用**时计算一次，
              之后整个序列都复用这次的 offset。
        """
        # 只在第一次调用时，根据当前帧计算最低高度；之后不再更新，避免跳跃被压回地面
        if not hasattr(self, "_human_ground_lowest_z"):
            lowest_pos = min(pos[2] for pos, quat in human_data.values())
            self._human_ground_lowest_z = lowest_pos
        lowest_pos = self._human_ground_lowest_z

        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # 将全身整体下移 lowest_pos，再上移 base_height_offset（对齐机器人站立高度）
            offset_human_data[body_name][0] = (
                pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, self.base_height_offset])
            )
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, 0.0])
        return human_data
    
    def calculate_foot_bottom_offset(self):
        """计算foot_bottom相对于toe body的固定偏移，取最小值"""
        data = mj.MjData(self.model)
        mj.mj_resetData(self.model, data)
        mj.mj_forward(self.model, data)
        
        offsets = []
        
        # 左脚的偏移
        toe_left_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "toeLeft")
        bottom_left_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "foot_bottom_left")
        
        if toe_left_id != -1 and bottom_left_id != -1:
            offset_z = data.xpos[bottom_left_id][2] - data.xpos[toe_left_id][2]
            if offset_z < 0:  # 确保为负
                offsets.append(offset_z)
                print(f"[INFO] Left foot offset: {offset_z:.6f}m")
        
        # 右脚的偏移  
        toe_right_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "toeRight")
        bottom_right_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "foot_bottom_right")
        
        if toe_right_id != -1 and bottom_right_id != -1:
            offset_z = data.xpos[bottom_right_id][2] - data.xpos[toe_right_id][2]
            if offset_z < 0:  # 确保为负
                offsets.append(offset_z)
                print(f"[INFO] Right foot offset: {offset_z:.6f}m")
        
        if offsets:
            # 取最小的偏移值（最接近地面的）
            min_offset = min(offsets)
            print(f"[INFO] Using minimum offset: {min_offset:.6f}m")
            return min_offset
        
        print("[INFO] Using default offset: -0.065m")
        return -0.065
    
    def _print_collision_constraints(self, dt, iteration=None):
        """打印碰撞约束信息"""
        G, h = self.collision_avoidance_limit.compute_qp_inequalities(self.configuration, dt)
        
        if G is not None:
            active_indices = np.where(h < 1e-3)[0]
            
            if iteration is not None:
                print(f"\n迭代 {iteration}:")
            
            print(f"总碰撞约束数量: {G.shape[0]}, 活跃约束数量: {len(active_indices)}")
            
            if len(active_indices) > 0:
                print(f"{'索引':<5} {'h值':<12} {'距离(m)':<12} {'Geom1':<30} {'Geom2':<30}")
                print("-" * 95)
                
                for idx, (geom1_id, geom2_id) in enumerate(self.collision_avoidance_limit.geom_id_pairs):
                    if idx < len(h) and h[idx] < 1e-3:
                        geom1_name = self.model.geom(geom1_id).name
                        geom2_name = self.model.geom(geom2_id).name
                        
                        fromto = np.empty(6)
                        dist = mj.mj_geomDistance(
                            self.model,
                            self.configuration.data,
                            geom1_id,
                            geom2_id,
                            self.collision_avoidance_limit.collision_detection_distance,
                            fromto,
                        )
                        
                        print(f"{idx:<5} {h[idx]:<12.6f} {dist:<12.6f} {geom1_name:<30} {geom2_name:<30}")
        else:
            print("没有碰撞约束")


def qvel_from_qpos_central(qpos_arr, fps):
    """从 qpos 序列 (T, 7+n_dof) 用中心差分得到 qvel (T, 6+n_dof)，用于替代 IK 返回的无意义 qvel。
    dt = 1/fps；qpos: [pos(3), quat_wxyz(4), joint(ndof)]；qvel: [lin_vel(3), ang_vel(3), joint_vel(ndof)]。
    """
    qpos_arr = np.asarray(qpos_arr, dtype=np.float64)
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    T, nq = qpos_arr.shape
    n_dof = nq - 7
    nv = 6 + n_dof
    qvel = np.zeros((T, nv), dtype=np.float64)

    if T <= 1:
        return qvel.astype(np.float32)

    # 线速度
    pos = qpos_arr[:, :3]
    qvel[0, :3] = (pos[1] - pos[0]) / dt
    qvel[-1, :3] = (pos[-1] - pos[-2]) / dt
    if T > 2:
        qvel[1:-1, :3] = (pos[2:] - pos[:-2]) / (2.0 * dt)

    # 角速度：四元数 wxyz -> 转成 scipy 的 xyzw 后算 rotvec/dt
    quat_wxyz = qpos_arr[:, 3:7]
    def _wxyz_to_xyzw(q):
        return [q[1], q[2], q[3], q[0]]
    for i in range(T):
        if i == 0:
            r_cur = R.from_quat(_wxyz_to_xyzw(quat_wxyz[0]))
            r_next = R.from_quat(_wxyz_to_xyzw(quat_wxyz[1]))
            delta = r_next * r_cur.inv()
            qvel[i, 3:6] = delta.as_rotvec() / dt
        elif i == T - 1:
            r_cur = R.from_quat(_wxyz_to_xyzw(quat_wxyz[i]))
            r_prev = R.from_quat(_wxyz_to_xyzw(quat_wxyz[i - 1]))
            delta = r_cur * r_prev.inv()
            qvel[i, 3:6] = delta.as_rotvec() / dt
        else:
            r_prev = R.from_quat(_wxyz_to_xyzw(quat_wxyz[i - 1]))
            r_next = R.from_quat(_wxyz_to_xyzw(quat_wxyz[i + 1]))
            delta = r_next * r_prev.inv()
            qvel[i, 3:6] = delta.as_rotvec() / (2.0 * dt)

    # 关节速度
    joint = qpos_arr[:, 7:]
    qvel[0, 6:] = (joint[1] - joint[0]) / dt
    qvel[-1, 6:] = (joint[-1] - joint[-2]) / dt
    if T > 2:
        qvel[1:-1, 6:] = (joint[2:] - joint[:-2]) / (2.0 * dt)

    return qvel.astype(np.float32)