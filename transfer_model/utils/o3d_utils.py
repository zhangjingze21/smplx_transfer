# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
import torch

Vector3d = o3d.utility.Vector3dVector
Vector3i = o3d.utility.Vector3iVector

Mesh = o3d.geometry.TriangleMesh


def np_mesh_to_o3d(vertices, faces):
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()
    mesh = Mesh()
    mesh.vertices = Vector3d(vertices)
    mesh.triangles = Vector3i(faces)
    return mesh

def process_rotation_matrix(rotation_matrix):
    """ Convert rotation matrix to axis-angle """
    if torch.is_tensor(rotation_matrix):
        rotation_matrix = rotation_matrix.detach().cpu().numpy()
    else :
        rotation_matrix = np.array(rotation_matrix)
    frames, param_size, _, _ = rotation_matrix.shape
    res = np.zeros((frames, param_size, 3))
    for i in range(frames):
        for j in range(param_size):
            q = Quaternion(matrix=rotation_matrix[i][j])
            res[i][j][:3] = q.axis * q.angle
    res = res.reshape((frames, -1))
    return res

# ### Process data
def pkl_to_amass(data, npz_save_path):
    dict_data = {}
    dict_data['gender'] = "neutral"
    FRAME = data['betas'].shape[0]
    # dict_data['mocap_frame_length']  = FRAME
    dict_data['mocap_framerate'] = 120.0
    dict_data['betas'] = torch.mean(torch.tensor(data['betas']), dim=0).detach().cpu().numpy()
    dict_data['num_betas'] = dict_data['betas'].shape[0]
    dict_data['trans'] = torch.tensor(data['transl']).detach().cpu().numpy()
    dict_data['root_orient'] = process_rotation_matrix(data['global_orient'].detach().cpu().numpy())
    # dict_data['poses'] = process_rotation_matrix(data['full_pose'].detach().cpu().numpy())
    dict_data['pose_body'] = process_rotation_matrix(data['body_pose'].detach().cpu().numpy())
    left_hand_pose = process_rotation_matrix(data['left_hand_pose'].detach().cpu().numpy()).reshape(FRAME, 45) # [FRAME, 15, 3]
    right_hand_pose = process_rotation_matrix(data['right_hand_pose'].detach().cpu().numpy()).reshape(FRAME, 45) # [FRAME, 15, 3]
    dict_data['pose_hand'] = np.concatenate((left_hand_pose, right_hand_pose), axis=1)
    dict_data['pose_eye'] = np.concatenate((data['leye_pose'].detach().cpu().numpy().reshape(FRAME, -1), data['reye_pose'].detach().cpu().numpy().reshape(FRAME, -1)), axis=1)
    dict_data['jaw_pose'] = process_rotation_matrix(data['jaw_pose'].detach().cpu().numpy())
    dict_data['poses'] = np.concatenate(
        (dict_data['root_orient'], dict_data['pose_body'], dict_data['pose_hand'], dict_data['pose_eye'], dict_data['jaw_pose']), axis=1
    )
    if npz_save_path is not None:
        np.savez(npz_save_path, **dict_data)
    return dict_data