import os
from glob import glob

import SimpleITK
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import load_json

from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization

ORGAN_LIST = ['CTVmeso']
DICE_LABELS = [[1]]

class MyRegistrationDataset(torch.utils.data.Dataset):
    def __init__(self, folder, train_val_test,
                 target_spacing=(2.0, 2.0, 2.0),
                 device='cuda', **kwargs):
        self.folder = folder
        self.train_val_test = train_val_test
        self.device = device
        self.overfit = False
        self.target_spacing = target_spacing
        print(self.target_spacing)
        if target_spacing == (2.0, 2.0, 2.0):
            self.shape = (128, 128, 128)

        if self.train_val_test != 'test':
            self.init_paths()
        else:
            self.init_paths_test()

        # Label info
        self.organ_list = ORGAN_LIST
        self.dice_labels = DICE_LABELS

        if self.train_val_test == 'train':
            self.transforms = self.get_transforms()
        else:
            self.transforms = None

        intensityproperties = {
            "max": 4129.10302734375,
            "mean": 995.2545776367188,
            "median": 1075.014892578125,
            "min": -223.6357421875,
            "percentile_00_5": 26.290109634399414,
            "percentile_99_5": 1870.4018194580021,
            "std": 488.85662841796875
        }
        self.normalization_function = ZScoreNormalization(**dict(intensityproperties=intensityproperties))

    def init_paths_test(self):
        subjects = np.unique([int(file.split('_')[1]) for file in os.listdir(os.path.join(self.folder, 'imagesTs'))])
        self.img_paths = glob(os.path.join(self.folder, 'imagesTs', '***.nii.gz'))
        self.img_paths.sort()
        self.seg_paths = glob(os.path.join(self.folder, 'labelsTs', '***.nii.gz'))
        self.seg_paths.sort()

        # Loop over fractions and save as fixed (fraction) and moving (ref) paths
        self.fixed_img_path, self.moving_img_path = [], []
        self.fixed_lbl_path, self.moving_lbl_path = [], []
        file_name = os.path.join(self.folder, 'imagesTs', 'Rect5F_{}_F{}_0000.nii.gz')
        file_name_seg = os.path.join(self.folder, 'labelsTs', 'Rect5F_{}_F{}.nii.gz')

        for subject in subjects:
            case_paths = [path for path in self.img_paths if '{}_F'.format(subject) in path]
            case_paths.sort()
            max_ses = int(os.path.basename(case_paths[-1]).split('_0000')[0][-1])
            # if os.path.isfile(file_name.format(subject, 'reftoF{}'.format(ses)).replace('Fref', 'ref')):
            for ses in range(1, max_ses + 1):
                if os.path.isfile(file_name.format(subject, ses)) and os.path.isfile(
                        file_name_seg.format(subject, ses)) and os.path.isfile(
                    file_name.format(subject, ses).replace('_F', '_reftoF')):
                    self.fixed_img_path.append(file_name.format(subject, ses))
                    self.moving_img_path.append(file_name.format(subject, ses).replace('_F', '_reftoF'))

                    self.fixed_lbl_path.append(file_name_seg.format(subject, ses))
                    self.moving_lbl_path.append(file_name_seg.format(subject, ses).replace('_F', '_reftoF'))
                else:
                    print('Subject {} fraction {} not loaded'.format(subject, ses))
        return

    def init_paths(self):
        split = load_json(os.path.join(self.folder, 'splits_final.json'))[0]
        if self.train_val_test == 'train':
            subjects = np.unique([int(file.split('_')[1]) for file in split['train']])
        elif self.train_val_test == 'val':
            subjects = np.unique([int(file.split('_')[1]) for file in split['val']])
        subjects.sort()

        self.img_paths = glob(os.path.join(self.folder, 'imagesTr', '***.nii.gz'))
        self.img_paths.sort()
        self.seg_paths = glob(os.path.join(self.folder, 'labelsTr', '***.nii.gz'))
        self.seg_paths.sort()

        # Loop over fractions and save as fixed (fraction) and moving (ref) paths
        self.fixed_img_path, self.moving_img_path = [], []
        self.fixed_lbl_path, self.moving_lbl_path = [], []
        file_name = os.path.join(self.folder, 'imagesTr', 'Rect5F_{}_F{}_0000.nii.gz')
        file_name_seg = os.path.join(self.folder, 'labelsTr', 'Rect5F_{}_F{}.nii.gz')

        for subject in subjects:
            case_paths = [path for path in self.img_paths if '{}_F'.format(subject) in path]
            case_paths.sort()
            max_ses = int(os.path.basename(case_paths[-1]).split('_0000')[0][-1])
            for ses in range(1, max_ses + 1):
                if os.path.isfile(file_name.format(subject, ses)) and os.path.isfile(
                        file_name_seg.format(subject, ses)) and os.path.isfile(
                    file_name.format(subject, ses).replace('_F', '_reftoF')):
                    self.fixed_img_path.append(file_name.format(subject, ses))
                    self.moving_img_path.append(file_name.format(subject, ses).replace('_F', '_reftoF'))

                    self.fixed_lbl_path.append(file_name_seg.format(subject, ses))
                    self.moving_lbl_path.append(file_name_seg.format(subject, ses).replace('_F', '_reftoF'))
                else:
                    print('Subject {} fraction {} not loaded'.format(subject, ses))
        return

    @staticmethod
    def get_transforms():
        transforms = []
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        return ComposeTransforms(transforms)

    def moving_fixed_paths(self, i):
        """Get the path to images and labels"""
        # Load in the moving- and fixed image/label
        moving_path, fixed_path = self.moving_img_path[i], self.fixed_img_path[i]
        return moving_path, fixed_path

    def get_subject_fraction(self, i):
        subject = int(os.path.split(self.fixed_img_path[i])[-1].split('_')[1])
        fraction = int(os.path.split(self.fixed_img_path[i])[-1].split('_')[2][1])
        return subject, fraction

    def __len__(self):
        return len(self.fixed_img_path)

    def pad_torch(self, array, shape, pad_value):
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

    @staticmethod
    def read_image_np(path):
        image_np = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path)).astype('float32')
        return image_np

    @staticmethod
    def read_image(path):
        # read image to numpy array and convert to pytorch tensor
        image_np = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path)).astype('float32')
        image_t = torch.from_numpy(image_np).to(torch.float32)
        return image_t

    def read_header_info(self, path):
        # read image to numpy array and convert to pytorch tensor
        image = SimpleITK.ReadImage(path)

        properties = {}
        properties['spacing'] = image.GetSpacing()[::-1]
        properties['origin'] = image.GetOrigin()[::-1]
        properties['direction'] = image.GetDirection()[::-1]
        properties['size'] = image.GetSize()[::-1]
        return properties

    def crop_image(self, im, center0, new_size):
        cropping_idx = torch.tensor(
            [[(round(c * shp) - dim // 2), (round(c * shp) + dim // 2)] for dim, shp, c in
             zip(new_size, im.shape, center0)], dtype=torch.int)

        pad_idx = torch.tensor([[0, new_size[0]], [0, new_size[1]], [0, new_size[2]]], dtype=torch.int)
        new_cropping_idx = cropping_idx.clone()
        for i, (idx, shp) in enumerate(zip(cropping_idx, im.shape)):
            if idx[0] < 0:
                pad_idx[i, 0] = abs(idx[0])
                new_cropping_idx[i, 0] = 0

            if idx[1] > shp:
                pad_idx[i, 1] = shp - idx[1]
                new_cropping_idx[i, 1] = shp

        im_crop = torch.ones(tuple(new_size), dtype=torch.float32) * im.min()

        im_crop[
        pad_idx[0, 0]:pad_idx[0, 1],
        pad_idx[1, 0]:pad_idx[1, 1],
        pad_idx[2, 0]:pad_idx[2, 1]] = im[
                                       int(new_cropping_idx[0, 0]):int(new_cropping_idx[0, 1]),
                                       int(new_cropping_idx[1, 0]):int(new_cropping_idx[1, 1]),
                                       int(new_cropping_idx[2, 0]):int(new_cropping_idx[2, 1])]
        return im_crop

    def moving_fixed_info(self, i):
        """Get the path to images and labels"""
        # Load in the moving- and fixed image/label
        moving_path, fixed_path = self.moving_fixed_paths(i)

        # Load the nii.gz information using simple itk
        moving_dict = self.read_header_info(moving_path)
        fixed_dict = self.read_header_info(fixed_path)
        return moving_dict, fixed_dict

    def subset(self, rand_indices):
        temp = [self.moving_img_path[i] for i in rand_indices]
        self.moving_img_path = temp
        temp = [self.fixed_img_path[i] for i in rand_indices]
        self.fixed_img_path = temp
        temp = [self.moving_lbl_path[i] for i in rand_indices]
        self.moving_lbl_path = temp
        temp = [self.fixed_lbl_path[i] for i in rand_indices]
        self.fixed_lbl_path = temp

    def __getitem__(self, idx):
        # Get image paths and load images
        moving_path, fixed_path = self.moving_fixed_paths(idx)

        moving_img = self.read_image(moving_path).unsqueeze(0).unsqueeze(0)
        fixed_img = self.read_image(fixed_path).unsqueeze(0).unsqueeze(0)

        # Load in the moving- and fixed label
        moving_lbl = self.read_image(self.moving_lbl_path[idx]).unsqueeze(0).unsqueeze(0)
        fixed_lbl = self.read_image(self.fixed_lbl_path[idx]).unsqueeze(0).unsqueeze(0)

        # compute the scale factor
        moving_dict, fixed_dict = self.moving_fixed_info(idx)
        mov_spacing = moving_dict['spacing']
        fix_spacing = fixed_dict['spacing']
        scale_factor_mov = tuple([(old / new).__float__() for old, new in zip(mov_spacing, list(self.target_spacing))])
        scale_factor_fix = tuple([(old / new).__float__() for old, new in zip(fix_spacing, list(self.target_spacing))])

        # resample the images and labels
        moving_img = F.interpolate(moving_img, scale_factor=scale_factor_mov, mode='trilinear', align_corners=False)
        moving_lbl = F.interpolate(moving_lbl, scale_factor=scale_factor_mov, mode='nearest')
        fixed_img = F.interpolate(fixed_img, scale_factor=scale_factor_fix, mode='trilinear', align_corners=False)
        fixed_lbl = F.interpolate(fixed_lbl, scale_factor=scale_factor_fix, mode='nearest')

        # pad the images to new size 192 x 192 x 192
        if self.shape == (128, 128, 128):
            moving_img = self.pad_torch(moving_img, self.shape, moving_img.min().item())
            fixed_img = self.pad_torch(fixed_img, self.shape, fixed_img.min().item())
            moving_lbl = self.pad_torch(moving_lbl, self.shape, 0.0)
            fixed_lbl = self.pad_torch(fixed_lbl, self.shape, 0.0)

        # Normalization
        moving_img = torch.from_numpy(self.normalization_function.run(moving_img.numpy()))
        fixed_img = torch.from_numpy(self.normalization_function.run(fixed_img.numpy()))
        moving_lbl = torch.clamp(moving_lbl, min=0.0, max=1.0)
        fixed_lbl = torch.clamp(fixed_lbl, min=0.0, max=1.0)

        # Apply transforms
        if self.transforms is not None:
            with torch.no_grad():
                tmp = self.transforms(**{'image': moving_img})
                moving_img = tmp['image']

                tmp = self.transforms(**{'image': fixed_img})
                fixed_img = tmp['image']

        return moving_img, fixed_img, \
               moving_lbl, fixed_lbl
