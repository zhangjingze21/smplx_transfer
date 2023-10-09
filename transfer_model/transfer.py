import os
import argparse
import time
import torch
import numpy as np
from loguru import logger
from read_file import read_smplh
from smplx import SMPLXLayer
from transfer_model import run_fitting
from config.defaults import conf as default_conf
from utils import read_deformation_transfer
from omegaconf import OmegaConf
from pyquaternion import Quaternion

class Dataset(torch.utils.data.Dataset):
    def __init__(self, smplh_file):
        self.smplh_file = smplh_file
        self.vertices, self.faces, self.joints, self.poses = read_smplh(smplh_file)
        self.left_hand_pose  = self.poses[:, 66:111]
        self.right_hand_pose = self.poses[:, 111:156]
        self.length = self.poses.shape[0]
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'vertices': self.vertices[idx],
            'faces': self.faces,
            'left_hand_pose': self.left_hand_pose[idx],
            'right_hand_pose': self.right_hand_pose[idx]
        }

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

def save_npz(data, npz_save_path):
    dict_data = {}
    dict_data['gender'] = "neutral"
    FRAME = data['betas'].shape[0]
    # dict_data['mocap_frame_length']  = FRAME
    dict_data['mocap_framerate'] = 120.0
    dict_data['betas'] = torch.mean(torch.tensor(data['betas']), dim=0).detach().cpu().numpy()
    dict_data['num_betas'] = dict_data['betas'].shape[0]
    dict_data['trans'] = torch.tensor(data['transl']).detach().cpu().numpy()
    dict_data['root_orient'] = process_rotation_matrix(data['global_orient'])
    # dict_data['poses'] = process_rotation_matrix(data['full_pose'].detach().cpu().numpy())
    dict_data['pose_body'] = process_rotation_matrix(data['body_pose'])
    left_hand_pose = process_rotation_matrix(data['left_hand_pose']).reshape(FRAME, 45) # [FRAME, 15, 3]
    right_hand_pose = process_rotation_matrix(data['right_hand_pose']).reshape(FRAME, 45) # [FRAME, 15, 3]
    dict_data['pose_hand'] = np.concatenate((left_hand_pose, right_hand_pose), axis=1)
    dict_data['pose_eye'] = np.concatenate((data['leye_pose'].reshape(FRAME, -1), data['reye_pose'].reshape(FRAME, -1)), axis=1)
    dict_data['jaw_pose'] = process_rotation_matrix(data['jaw_pose'])
    dict_data['poses'] = np.concatenate(
        (dict_data['root_orient'], dict_data['pose_body'], dict_data['pose_hand'], dict_data['pose_eye'], dict_data['jaw_pose']), axis=1
    )
    if npz_save_path is not None:
        np.savez(npz_save_path, **dict_data)
    return dict_data


def transfer(cfg, smplh_file, smplx_file, **kwargs):
    """ Transfer smplh motion to smplx motion
    Args:
        smplh_file: path to the input  smplh motion file
        smplx_file: path to the output smplx motion file
    """

    ## Step1: read the smplh file and get the mesh representation 
    smplh_dataset = Dataset(smplh_file)
    smplh_dataloader = torch.utils.data.DataLoader(smplh_dataset, batch_size=len(smplh_dataset), shuffle=False)

    ## Step1.5: build a smplx body model & load config files
    model_folder = kwargs["model_folder"]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    deformation_transfer_path = kwargs["deformation_transfer_path"]
    mask_ids_fname_path = kwargs["mask_ids_fname_path"]

    smplx_model_folder = os.path.join(model_folder, "smplx")
    body_model = SMPLXLayer(
        smplx_model_folder,
        gender="neutral",
        use_face_contour=True,
        use_compressed=False,
    )
    body_model = body_model.to(device=device)
    def_matrix = read_deformation_transfer(deformation_transfer_path, device=device)
    # Read mask for valid vertex ids
    mask_ids = torch.from_numpy(np.load(mask_ids_fname_path)).to(device=device)

    ## Step2: transfer the smplh motion to smplx motion
    result_dict = {}
    for ii, batch in enumerate(smplh_dataloader):
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)

        # TODO: modify the fitting function for correct results
        var_dict = run_fitting(cfg, batch, body_model, def_matrix, mask_ids, device=device)

        # TODO: check whether the for loop is correct
        for keys, values in var_dict.items():
            if torch.is_tensor(values):
                values = values.detach().cpu().numpy()
            var_dict[keys] = values
            if result_dict.get(keys, None) is None:
                result_dict[keys] = var_dict[keys]
            else:
                result_dict[keys] = np.concatenate([result_dict.get(keys), var_dict[keys]], axis=0)

    save_npz(result_dict, smplx_file)

def main(model_folder, smplh_folder, output_folder, cfg):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "CUDA is not available"
    ### write config files and load config files
    deformation_transfer_path = '/scratch/stu5/research_jul_28/Project/Motion/smplx/transfer_data/smplh2smplx_deftrafo_setup.pkl'
    mask_ids_fname_path = '/scratch/stu5/research_jul_28/Project/Motion/smplx/transfer_data/smplx_mask_ids.npy'

    logger.add("outputs/transfer.log" + str(time.time()))
    logger.info("Start transfering smplh motion to smplx motion")
    logger.info(f"Model folder: {model_folder}")
    logger.info(f"SMPLH folder: {smplh_folder}")
    logger.info(f"Output folder: {output_folder}")
    # logger.info(f"Config file: {cfg}")

    for subset in os.listdir(smplh_folder):
        subset_folder = os.path.join(smplh_folder, subset)
        for motion in os.listdir(subset_folder):
            motion_file = os.path.join(subset_folder, motion)
            logger.info(f"Processing {motion_file}")
            output_file = os.path.join(output_folder, subset, motion)
            logger.info(f"Output file: {output_file}")
            os.makedirs(os.path.join(output_folder, subset), exist_ok=True)
            transfer(
                cfg=cfg,
                smplh_file=motion_file,
                smplx_file=output_file,
                model_folder=model_folder,
                deformation_transfer_path=deformation_transfer_path,
                mask_ids_fname_path=mask_ids_fname_path,
            )

if __name__ == "__main__":
    model_folder = "/scratch/stu5/Data/SMPL-MODEL/SMPL_MODEL"
    smplh_folder = "/scratch/stu5/Data/BMLhandball/smplh"
    output_folder = "/scratch/stu5/Data/BMLhandball/smplx"
    cfg = default_conf.copy()

    ## load config from config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_files/smplh2smplx.yaml")
    args = parser.parse_args()
    cfg.merge_with(OmegaConf.load(args.config))

    main(model_folder, smplh_folder, output_folder, cfg)
