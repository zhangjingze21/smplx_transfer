import os
import numpy as np
import torch
import trimesh
import smplx
from tqdm.auto import trange

def save_mesh(vertices, faces, save_path):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(save_path)

def read_smplh(motion_file):
    """ Read smplh motion file and return vertices and joints
    Args:
        motion_file: path to the smplh motion file
    Returns:
        vertices
        joints
    """
    model_folder = "/scratch/stu5/Data/SMPL-MODEL/SMPL_MODEL"
    motion = np.load(motion_file, allow_pickle=True)
    ### Structure of the smplh motion file
    # trans (bs, 3)
    # gender 
    # mocap_framerate ()
    # betas (16)
    # dmpls (batch, 8)
    # poses (batch, 156)
    ##################################
    betas = motion["betas"] if "betas" in motion else np.zeros((10,))
    num_betas = len(betas)
    gender = str(motion["gender"]) if "gender" in motion else "neutral"
    
    model = smplx.create(
        model_path=model_folder,
        model_type="smplh",
        gender=gender,
        use_face_contour=False,
        num_betas=num_betas,
        num_expression_coeffs=10,
        use_pca=False,
        ext="npz",
    )
    betas, expression = torch.tensor(betas).float(), None
    betas = betas.unsqueeze(0)[:, : model.num_betas]
    poses = torch.tensor(motion["poses"]).float() 
    
    ## read pose params
    global_orient = poses[:, :3]
    body_pose = poses[:, 3:66]
    left_hand_pose = poses[:, 66:111]
    right_hand_pose = poses[:, 111:156]

    ret_vertices, ret_joints, ret_faces = [], [], []

    for pose_idx in trange(poses.shape[0]):
        pose_idx = [pose_idx]
        output = model(
            betas=betas,
            global_orient=global_orient[pose_idx],
            body_pose=body_pose[pose_idx],
            left_hand_pose=left_hand_pose[pose_idx],
            right_hand_pose=right_hand_pose[pose_idx],
            return_verts=True,
        )
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()
        ret_vertices.append(vertices)
        ret_joints.append(joints)

    ret_vertices = torch.from_numpy(np.stack(ret_vertices, dtype=np.float32))
    ret_joints = torch.from_numpy(np.stack(ret_joints, dtype=np.float32))
    ret_faces = model.faces.astype(np.float32)
    print(ret_vertices.shape, ret_faces.shape)


    return ret_vertices, ret_faces, ret_joints, poses


