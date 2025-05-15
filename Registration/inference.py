import argparse
import os

import ml_collections
import numpy as np
import torch
from tqdm import tqdm

import SimpleITK as sitk

from Registration.dataset import MyRegistrationDataset
from Registration.model import SpatialTransformer
from Registration.train import init_lightning_model, ROOT_OUTPUT, ROOT_CHECKPOINTS, ROOT_DATA
from Utils.utils import match_dvf_to_original_shape, read_image_nii_to_torch, sitk_header, save_image


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--mode', type=str, metavar='', default='test', help='val or test')
    parser.add_argument('--path_to_pretrained_weights', type=str, metavar='', default='None', help='')
    parser.add_argument('-dev', '--device', type=str, metavar='', default='cuda', help='device / gpu used')
    parser.add_argument('--in_channels', type=int, metavar='', default=3, help='number of input channels')
    config = parser.parse_args()

    config.imgsize = (128, 128, 128)  # 128 x 256 x 256
    config.voxel_spacing = [2.0, 2.0, 2.0]

    config.reg_weight = 0
    config.seg_weight = 0
    config.root_output = ROOT_OUTPUT
    config.root_checkpoints = ROOT_CHECKPOINTS
    config.root_data = ROOT_DATA
    config.root_data = "DatasetRegistration"
    config = ml_collections.ConfigDict(dict(**vars(config)))

    print(config)
    return config


if __name__ == '__main__':
    UNITY = False
    config = parse_arguments()

    """ INIT DATA """
    if config.mode == 'val':
        label_gt_paths = os.path.join(config.root_data, 'labelsTr', 'Rect5F_{}_F{}.nii.gz')
    elif config.mode == 'test':
        label_gt_paths = os.path.join(config.root_data, 'labelsTs', 'Rect5F_{}_F{}.nii.gz')

    label_pred_paths = os.path.join(config.root_data,
                                    'prediction_temp_{}'.format(config.mode),
                                    'Rect5F_{}_F{}.nii.gz')

    dataset = MyRegistrationDataset(folder=config.root_data,
                                    train_val_test=config.mode,
                                    target_spacing=tuple(config.voxel_spacing),
                                    device=config.device)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, pin_memory=False)

    """ INIT MODEL """
    config.path_to_pretrained_weights = '/local_scratch/ikolenbr/checkpoints/J06_Rectum/DIR-voxelmorph-0414-153414/epoch=00099.ckpt'
    lightning_model = init_lightning_model(config, path_to_pretrained_weights=config.path_to_pretrained_weights)
    lightning_model.eval()

    """ PREDICT AND EVALUATE """
    config.which = os.path.split(config.path_to_pretrained_weights)[-1].replace('.pth', '').replace('.ckpt', '')

    # Loop over each datapoint and make predictions
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Move batch to the appropriate device
        batch = [b.to(config.device) for b in batch]
        source, target, source_map, target_map = batch
        subject, fraction = dataset.get_subject_fraction(batch_idx)

        label_pred_path = label_pred_paths.format(subject, fraction)
        label_gt_path = label_gt_paths.format(subject, fraction)

        if config.in_channels == 2:
            input_channels = [source, target]
        elif config.in_channels == 3:
            input_channels = [source, target, source_map]

        # Disable gradient calculation for prediction
        with torch.no_grad():
            dvf, source_warped = lightning_model.forward(input_channels)
            label_pred, dvf_resampled = match_dvf_to_original_shape(config, dvf, subject, fraction)
            label_gt = read_image_nii_to_torch(label_gt_path.format(subject, fraction)).to(config.device)
            label_gt_sitk = sitk.ReadImage(label_gt_path.format(subject, fraction))
            _, _, original_voxel_spacing = sitk_header(label_gt_sitk)
            stl = SpatialTransformer(dvf_resampled.shape[2::])
            moving_path, fixed_path = dataset.moving_fixed_paths(batch_idx)
            moving = read_image_nii_to_torch(moving_path).unsqueeze(0).unsqueeze(0)
            moving_warped = stl.forward(moving, dvf_resampled.cpu())

            # Save label prediction
            save_image(label_pred, label_gt_sitk, label_pred_paths.format(subject, fraction))

