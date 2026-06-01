import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def load_bvh_file(bvh_file, format="lafan1"):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    # Keep the same axis conversion as existing BVH pipelines:
    # y-up BVH -> z-up retarget space used by downstream IK.
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    # Most legacy BVH datasets in this repo are in centimeters and need /100.
    # 4D-Human converted BVH uses meter-like offsets, so keep scale as-is.
    position_scale = 1.0 if format == "smpl4d_bvh" else 1.0 / 100.0

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T * position_scale
            result[bone] = [position, orientation]
            
        if format == "lafan1":
            # Add modified foot pose
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]
        elif format == "nokov":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
        elif format == "sfu":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
        elif format == "noitom":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftFoot"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]
        elif format == "mocap":
            # FBX-like foot handling: foot position + foot orientation.
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftFoot"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]
        elif format == "mocap_hands":
            # Same foot synthesis as mocap; use with IK config bvh_mocap_adam_pro_hands.json.
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftFoot"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]
        elif format == "opt_mocap":
            # New opt-mocap skeleton: synthesize FootMod from ankle-roll joints.
            result["LeftFootMod"] = [result["ankle_l"][0], result["ankle_l"][1]]
            result["RightFootMod"] = [result["ankle_r"][0], result["ankle_r"][1]]
        elif format == "smpl4d_bvh":
            # 4D-Human BVH already uses its own label space.
            # Use dedicated IK config bvh_smpl4d_bvh_* for name mapping.
            # Add optional aliases for downstream compatibility/debug.
            if "L_Foot" in result:
                result["LeftFootMod"] = [result["L_Foot"][0], result["L_Foot"][1]]
            if "R_Foot" in result:
                result["RightFootMod"] = [result["R_Foot"][0], result["R_Foot"][1]]
        else:
            raise ValueError(f"Invalid format: {format}")
            
        frames.append(result)
    
    # human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height


