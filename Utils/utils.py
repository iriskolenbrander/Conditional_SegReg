import os

import numpy as np
import torch

import SimpleITK as sitk

from Registration.model import SpatialTransformer


def pad_torch(array, shape, pad_value):
    """
    Pads an array with the given padding value to a given shape. Returns the padded_array array and crop slices.
    """
    if array.squeeze().shape == tuple(shape):
        return array, ...
    array = array.squeeze().numpy()
    padded_array = pad_value * np.ones(shape, dtype=array.dtype)
    offsets_low = [((p - v) / 2).__floor__() for p, v in zip(shape, array.shape)]
    offsets_up = [((p - v) / 2).__ceil__() for p, v in zip(shape, array.shape)]
    slices_pad = tuple([slice(offset_low, l + offset_up) if offset_low >= 0 else slice(0, new_shp) for
                        offset_low, offset_up, l, new_shp in
                        zip(offsets_low, offsets_low, array.shape, shape)])
    slices_crop = tuple(
        [slice(-1 * offset_low, l + offset_up) if offset_low < 0 else slice(0, l) for offset_low, offset_up, l in
         zip(offsets_low, offsets_up, array.shape)])
    padded_array[slices_pad] = array[slices_crop]
    padded_array = torch.from_numpy(padded_array).unsqueeze(0)
    return padded_array


def match_dvf_to_original_shape(config, dvf, subject, fraction):
    # Paths to labels
    if config.mode == 'val':
        lbl_fix = 'labelsTr/Rect5F_{}_F{}.nii.gz'
        lbl_mov = 'labelsTr/Rect5F_{}_reftoF{}.nii.gz'
    elif config.mode == 'test':
        lbl_fix = 'labelsTs/Rect5F_{}_F{}.nii.gz'
        lbl_mov = 'labelsTs/Rect5F_{}_reftoF{}.nii.gz'

    # Load ground truth label, source label,
    gt_lbl_sitk = sitk.ReadImage(os.path.join(config.root_data,
                                              lbl_fix.format(subject, fraction)))
    src_lbl_sitk = sitk.ReadImage(os.path.join(config.root_data,
                                               lbl_mov.format(subject, fraction)))
    label_source = torch.from_numpy(sitk.GetArrayFromImage(src_lbl_sitk)).unsqueeze(0).unsqueeze(0).float()
    original_voxel_sp = list(gt_lbl_sitk.GetSpacing())[::-1]
    original_size = list(gt_lbl_sitk.GetSize())[::-1]

    # convert the DVF to numpy and revert preprocessing (crop, resample)
    foo_resampled = torch.nn.functional.interpolate(label_source, scale_factor=tuple(
        [old / new for old, new in zip(original_voxel_sp, config.voxel_spacing)]), mode='trilinear',
                                                    align_corners=False)
    size_intermediate = foo_resampled.shape[2::]

    # convert the DVF to numpy and revert preprocessing (crop, resample)
    dvf_padded = list()
    for i in range(3):
        dvf_padded.append(pad_torch(dvf[0, i].cpu(), size_intermediate, 0.0))  # [125, 200, 200]
    dvf_padded = torch.stack(dvf_padded, dim=1).to(config.device)

    if tuple(original_size) != tuple(size_intermediate):
        dvf_resampled = torch.nn.functional.interpolate(dvf_padded, size=tuple(original_size),
                                                        mode='trilinear',
                                                        align_corners=False)
        factor = [old / new for old, new in zip(config.voxel_spacing, original_voxel_sp)]
        for i in range(3):
            dvf_resampled[0, i, ...] = dvf_resampled[0, i, ...] * factor[i]
    else:
        dvf_resampled = dvf_padded

    stl_bin = SpatialTransformer(dvf_resampled.shape[2::], mode='nearest').to(config.device)
    label_pred = stl_bin.forward(label_source.to(config.device), dvf_resampled).cpu().squeeze().numpy()

    return label_pred, dvf_resampled

def read_image_nii_to_torch(path):
    im = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype('float32')
    return torch.from_numpy(im).to(torch.float32)

def sitk_header(im_sitk):
    dim = np.array(im_sitk.GetSize())
    origin = np.array(im_sitk.GetOrigin())
    voxel_sp = np.array(im_sitk.GetSpacing())
    return dim, origin, voxel_sp

def make_dir(path):
    parent = os.path.split(path)[0]
    parent_parent = os.path.split(parent)[0]
    parent_parent_parent = os.path.split(parent_parent)[0]
    if not os.path.exists(parent_parent_parent):
        os.mkdir(parent_parent_parent)
    if not os.path.exists(parent_parent):
        os.mkdir(parent_parent)
    if not os.path.exists(parent):
        os.mkdir(parent)
    if not os.path.exists(path):
        os.mkdir(path)


def save_image(im, im_src_sitk, image_path):
    # make sitk image
    im_sitk = sitk.GetImageFromArray(im)
    im_sitk.SetOrigin(im_src_sitk.GetOrigin())
    im_sitk.SetDirection(im_src_sitk.GetDirection())
    im_sitk.SetSpacing(im_src_sitk.GetSpacing())

    # write sitk image to file
    make_dir(os.path.split(image_path)[0])
    sitk.WriteImage(im_sitk, image_path)
