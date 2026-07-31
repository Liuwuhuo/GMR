import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

# Temporal low-pass strength (in frames) for noisy mocap sources. 0 disables it.
OPT_MOCAP_SMOOTH_SIGMA = 2.0


def _smooth_frames(frames, sigma):
    """Low-pass each bone's position and orientation over time to remove
    high-frequency capture noise (which the IK would otherwise reproduce as
    jitter). Quaternions are sign-aligned and renormalized after filtering."""
    if sigma <= 0 or len(frames) < 3:
        return frames
    bones = list(frames[0].keys())
    n = len(frames)
    for bone in bones:
        pos = np.array([frames[i][bone][0] for i in range(n)], dtype=float)
        quat = np.array([frames[i][bone][1] for i in range(n)], dtype=float)
        for i in range(1, n):
            if np.dot(quat[i - 1], quat[i]) < 0:
                quat[i] = -quat[i]
        pos = gaussian_filter1d(pos, sigma, axis=0, mode="nearest")
        quat = gaussian_filter1d(quat, sigma, axis=0, mode="nearest")
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        for i in range(n):
            frames[i][bone] = [pos[i], quat[i]]
    return frames


def load_bvh_file(bvh_file, format="lafan1", already_z_up=False):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }

    Parameters
    ----------
    already_z_up : bool
        If True, skip the Y-up → Z-up world-frame conversion (legacy OptiTrack
        exports that are already z-up). Default False: apply the same conversion
        as other BVH formats so height lands on Z (needed for typical Y-up BVH
        such as PND / standard OptiTrack exports).
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    # Y-up BVH → z-up retarget space used by downstream IK.
    # Both position and orientation are transformed by the same world rotation.
    # Only skip when the capture is already z-up (legacy opt_mocap path).
    if format == "test_mocap":
        rotation_matrix = np.eye(3)
    else:
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
            # Same BVH skeleton and foot handling as mocap (Hips / LeftFoot).
            result["LeftFootMod"]  = [result["LeftFoot"][0],  result["LeftFoot"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]
        elif format == "test_mocap":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
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

    # opt_mocap captures contain high-frequency rotational noise; smooth it so
    # the retargeted robot does not vibrate.
    if format == "opt_mocap":
        frames = _smooth_frames(frames, OPT_MOCAP_SMOOTH_SIGMA)

    # human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height


