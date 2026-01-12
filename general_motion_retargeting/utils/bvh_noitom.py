import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def load_lafan1_file(bvh_file):
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

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = (position, orientation)

        # Add modified foot pose
        # result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftToe"][1])
        # result["RightFootMod"] = (result["RightFoot"][0], result["RightToe"][1])
        
        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftFoot"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightFoot"][1])
        
        frames.append(result)
    
    human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height


def test_lafan():
    frames, human_height = load_lafan1_file("/home/lh/Downloads/lafan1/dance1_subject2.bvh")
    # write to txt file
    with open("lafan_bvh_data.txt", "w") as f:
        f.write(f"Human height: {human_height}\n")
        f.write(str(frames))


def load_noitom_file(bvh_file):
    import bvhio
    root = bvhio.readAsHierarchy(bvh_file)
    bvh = bvhio.readAsBvh(bvh_file)
    print(f"Root: {root.Name}, Frame count: {bvh.FrameCount}, Frame time: {bvh.FrameTime}")
    bones = []
    parents = np.array([], dtype=int)
    
    joint_names_index = {}
    for joint, index, depth in root.layout():
        bones.append(joint.Name)
        joint_names_index[joint.Name] = index
        # print(f"Joint: {joint.Name}, Index: {index}, Depth: {depth} Parent: {joint.Parent.Name if joint.Parent else 'None'}")
        parents = np.append(parents, joint_names_index[joint.Parent.Name] if joint.Parent else -1)

    pos = np.zeros([bvh.FrameCount, len(bones), 3])
    quats = np.zeros([bvh.FrameCount, len(bones), 4])
    print(pos.shape, quats.shape)
    for i in range(bvh.FrameCount):
        frame_pos = np.array([]).reshape((0, 3))
        frame_quats = np.array([]).reshape((0, 4))
        for joint, index, depth in root.layout():
            transform = joint.getKeyframe(i)
            frame_pos = np.append(frame_pos, np.array([transform.Position.to_list()]), axis=0)
            frame_quats = np.append(frame_quats, np.array([transform.Rotation.to_list()]), axis=0)
            
        pos[i] = frame_pos
        quats[i] = frame_quats

    # with open("notiom_data.txt", "w") as f:
    #     f.write(str(bones))
    #     f.write("\n")
    #     f.write(str(parents))
    #     f.write("\n")
    #     f.write(str(pos))
    #     f.write("\n")
    #     f.write(str(quats))
    
    global_data = utils.quat_fk(quats, pos, parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    print(pos.shape[0], len(bones))
    for frame in range(pos.shape[0]):
        result = {}
        for i, bone in enumerate(bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = (position, orientation)
        
        
        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftFoot"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightFoot"][1])
            
        frames.append(result)
    
    # with open("noitom_bvh_data.txt", "w") as f:
    #     f.write(str(frames))
    
    return frames, 1.75  # cm to m

def test_noitom():
    frames, human_height = load_noitom_file("/home/lh/Downloads/take011_chr01.bvh")
    # write to txt file
    with open("noitom_bvh_data.txt", "w") as f:
        f.write(f"Human height: {human_height}\n")
        f.write(str(frames))



if __name__ == "__main__":
    # test_lafan()
    test_noitom()
