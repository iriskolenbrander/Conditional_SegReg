import os
import shutil
from glob import glob

import numpy as np
import torch
import SimpleITK as sitk
import monai
from batchgenerators.augmentations.spatial_transformations import augment_anatomy_informed
from lightning import seed_everything
from tqdm import tqdm

from Utils.utils import sitk_header, save_image


class DatasetnnUNetPerturbedSegmentation(torch.utils.data.Dataset):
    """
    Dataset class for nnUNet with perturbed images and labels --> Conditional nnUNet
    """

    def __init__(self, folder):
        self.root_data = folder
        self.init_paths()

    def init_paths(self):
        self.input_img_paths = glob(os.path.join(self.root_data, 'imagesTs', '***_0000.nii.gz'))
        self.input_img_paths.sort()
        self.input_lbl_paths = glob(os.path.join(self.root_data, 'imagesTs', '***_0001.nii.gz'))
        self.input_lbl_paths.sort()
        self.seg_paths = glob(os.path.join(self.root_data, 'labelsTs', '***.nii.gz'))
        self.seg_paths.sort()

    def __len__(self):
        return len(self.input_img_paths)

    @staticmethod
    def read_image(path):
        # read image to numpy array and convert to pytorch tensor
        image_np = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype('float32')
        image_t = torch.from_numpy(image_np).to(torch.float32)
        return image_t

    def __getitem__(self, idx):
        input_image = self.read_image(self.input_img_paths[idx])
        input_label = self.read_image(self.input_lbl_paths[idx])
        seg = self.read_image(self.seg_paths[idx])
        return input_image, input_label, seg

    def apply_appearance_deviations(self, image, deviation_type, deviation_magnitude):
        if deviation_type == 'noise':
            transform = monai.transforms.RandRicianNoise(std=deviation_magnitude, prob=1.0, channel_wise=True,  # 1.0
                                                         relative=True, sample_std=False)
        elif deviation_type == 'contrast':
            transform = monai.transforms.AdjustContrast(gamma=deviation_magnitude)  # 0.5
        img_perturbed = transform(image.clone())
        return img_perturbed

    def apply_rigid_deviations(self, label, deviation_type, deviation_magnitude):
        if deviation_type == 'translation_z':
            magn = (deviation_magnitude, 0, 0)
            transform_bin = monai.transforms.Affine(
                rotate_params=None,
                shear_params=None,
                translate_params=magn,
                scale_params=None, affine=None, mode='nearest', padding_mode='border')
        lbl_perturbed, _ = transform_bin(label.clone().unsqueeze(0))
        return lbl_perturbed.squeeze()

    def apply_deformable_transform(self, image, seg, spacing_ratio, deviation_magnitude):
        seg_ = seg.clone()
        seg_[seg_ == 1] = 2
        img_perturbed, seg_perturbed = augment_anatomy_informed(
            data=image.clone().unsqueeze(0).numpy(),
            seg=seg_.numpy(),
            active_organs=[1, 0],
            dilation_ranges=((deviation_magnitude - 500, deviation_magnitude + 500), (-999, 999)),
            directions_of_trans=((0, 1, 1), (1, 1, 1)),
            modalities=(0,),
            spacing_ratio=spacing_ratio,  # 0.3125/3 = ratio of transversal plane spacing and slice thickness
            blur=32,
            anisotropy_safety=True,
            max_annotation_value=3,
            replace_value=0)
        seg_perturbed_ = torch.from_numpy(seg_perturbed)
        seg_perturbed_[seg_perturbed_ == 2] = 1
        img_perturbed = torch.from_numpy(img_perturbed).squeeze()
        return img_perturbed, seg_perturbed_

class DatasetnnUNetPerturbedRegistration(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.init_paths_test()

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

    def __len__(self):
        return len(self.fixed_img_path)

    @staticmethod
    def read_image(path):
        # read image to numpy array and convert to pytorch tensor
        image_np = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype('float32')
        image_t = torch.from_numpy(image_np).to(torch.float32)
        return image_t

    def __getitem__(self, idx):
        # Get image paths and load images
        moving_img = self.read_image(self.moving_img_path[idx])
        fixed_img = self.read_image(self.fixed_img_path[idx])

        # Load in the moving- and fixed label
        moving_lbl = self.read_image(self.moving_lbl_path[idx])
        fixed_lbl = self.read_image(self.fixed_lbl_path[idx])

        return moving_img, fixed_img, moving_lbl, fixed_lbl

    def apply_appearance_deviations(self, image, deviation_type, deviation_magnitude):
        if deviation_type == 'noise':
            transform = monai.transforms.RandRicianNoise(std=deviation_magnitude, prob=1.0, channel_wise=True, #1.0
                                                     relative=True, sample_std=False)
        elif deviation_type == 'contrast':
            transform = monai.transforms.AdjustContrast(gamma=deviation_magnitude) # 0.5
        img_perturbed = transform(image.clone())
        return img_perturbed

    def apply_rigid_deviations(self, image, label, deviation_type, deviation_magnitude):
        if deviation_type == 'translation_z':
            magn = (deviation_magnitude, 0, 0)
            transform = monai.transforms.Affine(
                rotate_params=None,
                shear_params=None,
                translate_params=magn,
                scale_params=None, affine=None, mode='bilinear', padding_mode='zeros')

            transform_bin = monai.transforms.Affine(
                rotate_params=None,
                shear_params=None,
                translate_params=magn,
                scale_params=None, affine=None, mode='nearest', padding_mode='zeros')

        img_perturbed, _ = transform(image.clone().unsqueeze(0))
        lbl_perturbed, _ = transform_bin(label.clone().unsqueeze(0))
        return img_perturbed.squeeze(), lbl_perturbed.squeeze()

    def apply_deformable_transform(self, image, seg, spacing_ratio, deviation_magnitude):
        seg_ = seg.clone()
        seg_[seg_ == 1] = 2
        img_perturbed, seg_perturbed = augment_anatomy_informed(
            data=image.clone().unsqueeze(0).numpy(),
            seg=seg_.numpy(),
            active_organs=[1, 0],
            dilation_ranges=((deviation_magnitude-500, deviation_magnitude+500), (-999, 999)), #((-1200, 1200), (-1200, 1200)),
            directions_of_trans=((0, 1, 1), (1, 1, 1)),
            modalities=(0,),
            spacing_ratio=spacing_ratio,  # 0.3125/3 = ratio of transversal plane spacing and slice thickness
            blur=32,
            anisotropy_safety=True,
            max_annotation_value=3,
            replace_value=0)
        seg_perturbed_ = torch.from_numpy(seg_perturbed)
        seg_perturbed_[seg_perturbed_ == 2] = 1
        img_perturbed = torch.from_numpy(img_perturbed).squeeze()
        return img_perturbed, seg_perturbed_


DEFAULT_TRANSFORM_SETTINGS = {
    'noise': np.linspace(0., 1, 6),
    'contrast': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    'translation_z': [0, 1, 2],
    'deformable': np.linspace(-2000, 2000, 9),
}

def simulated_domain_shifts_Segmentation():
    dataset_folder = ''
    dataset = DatasetnnUNetPerturbedSegmentation(dataset_folder)
    for deviation_type in ['noise', 'contrast']:
        devation_magnitudes = DEFAULT_TRANSFORM_SETTINGS[deviation_type]
        for devation_magnitude in devation_magnitudes:
            print(deviation_type, devation_magnitude)
            seed_everything(1000)
            for idx, (input_image, input_label, seg) in enumerate(tqdm(dataset)):
                # Perturb image
                input_image_perturbed = dataset.apply_appearance_deviations(input_image, deviation_type,
                                                                            devation_magnitude)
                _, _, voxel_spacing = sitk_header(sitk.ReadImage(dataset.input_img_paths[idx]))

                # save image to disk
                dataset_folder_destination = os.path.join(dataset_folder,
                                                          "{}-{}".format(deviation_type, devation_magnitude))
                os.makedirs(os.path.join(dataset_folder_destination, 'imagesTs'), exist_ok=True)
                os.makedirs(os.path.join(dataset_folder_destination, 'labelsTs'), exist_ok=True)

                # Copy unmodified input
                # - input planning segmentation label = second input to the model
                shutil.copy(dataset.input_lbl_paths[idx],
                            dataset.input_lbl_paths[idx].replace(dataset_folder, dataset_folder_destination))
                # - ground truth fraction segmentation label = what the model should predict
                shutil.copy(dataset.seg_paths[idx],
                            dataset.seg_paths[idx].replace(dataset_folder, dataset_folder_destination))

                # Save the perturbed fraction MRI
                image_path_destination = dataset.input_img_paths[idx].replace(dataset_folder,
                                                                              dataset_folder_destination)
                sitk_im = sitk.ReadImage(dataset.input_img_paths[idx])
                save_image(input_image_perturbed.squeeze().numpy(), sitk_im, image_path_destination)

    for deviation_type in ['translation_z']:
        devation_magnitudes = DEFAULT_TRANSFORM_SETTINGS[deviation_type]
        for devation_magnitude in devation_magnitudes:
            print(deviation_type, devation_magnitude)
            seed_everything(1000)
            for idx, (input_image, input_label, seg) in enumerate(tqdm(dataset)):
                # Do perturbation
                input_label_perturbed = dataset.apply_rigid_deviations(input_label,
                                                                       deviation_type, devation_magnitude)

                # save image to disk
                dataset_folder_destination = os.path.join(dataset_folder,
                                                          "{}-{}".format(deviation_type, devation_magnitude))
                os.makedirs(os.path.join(dataset_folder_destination, 'imagesTs'), exist_ok=True)
                os.makedirs(os.path.join(dataset_folder_destination, 'labelsTs'), exist_ok=True)

                # Copy unmodified input
                # - input fraction MRI = first input to the model
                shutil.copy(dataset.input_img_paths[idx],
                            dataset.input_img_paths[idx].replace(dataset_folder, dataset_folder_destination))
                # - ground truth fraction segmentation = what the model should predict
                shutil.copy(dataset.seg_paths[idx],
                            dataset.seg_paths[idx].replace(dataset_folder, dataset_folder_destination))

                # Save perturbed label (as if the planning MRI is not correctly aligned)
                input_label_path_destination = dataset.input_lbl_paths[idx].replace(dataset_folder,
                                                                                    dataset_folder_destination)
                sitk_im = sitk.ReadImage(dataset.input_lbl_paths[idx])
                save_image(input_label_perturbed.squeeze().numpy(), sitk_im, input_label_path_destination)

    for deviation_type in ['deformable']:
        devation_magnitudes = DEFAULT_TRANSFORM_SETTINGS[deviation_type]
        for devation_magnitude in devation_magnitudes:
            print(deviation_type, devation_magnitude)
            seed_everything(1000)
            for idx, (input_image, input_label, seg) in enumerate(tqdm(dataset)):
                # Do perturbation
                _, _, voxel_spacing = sitk_header(sitk.ReadImage(dataset.input_img_paths[idx]))
                spacing_ratio = voxel_spacing[1] / voxel_spacing[2]

                input_image_perturbed, \
                input_label_perturbed = dataset.apply_deformable_transform(input_image, seg,
                                                                           spacing_ratio, devation_magnitude)

                # save image to disk
                dataset_folder_destination = os.path.join(dataset_folder,
                                                          "{}-{}".format(deviation_type, devation_magnitude))
                os.makedirs(os.path.join(dataset_folder_destination, 'imagesTs'), exist_ok=True)
                os.makedirs(os.path.join(dataset_folder_destination, 'labelsTs'), exist_ok=True)

                # Copy unmodified input
                # - input segmentation label = second input to the model
                shutil.copy(dataset.input_lbl_paths[idx],
                            dataset.input_lbl_paths[idx].replace(dataset_folder, dataset_folder_destination))

                # Copy the perturbed fraction MRI...
                input_img_path_destination = dataset.input_img_paths[idx].replace(dataset_folder,
                                                                                  dataset_folder_destination)
                save_image(input_image_perturbed.squeeze().numpy(), sitk.ReadImage(dataset.input_img_paths[idx]),
                           input_img_path_destination)
                #  and fraction segmentation to the destination
                input_img_path_destination = dataset.seg_paths[idx].replace(dataset_folder, dataset_folder_destination)
                save_image(input_label_perturbed.squeeze().numpy(), sitk.ReadImage(dataset.seg_paths[idx]),
                           input_img_path_destination)

def simulated_domain_shifts_Registration():
    dataset_folder = ''
    dataset = DatasetnnUNetPerturbedRegistration(dataset_folder)
    for deviation_type in ['noise', 'contrast']:
        devation_magnitudes = DEFAULT_TRANSFORM_SETTINGS[deviation_type]
        for devation_magnitude in devation_magnitudes:
            print(deviation_type, devation_magnitude)
            seed_everything(1000)
            for idx, (moving_img, fixed_img, moving_lbl, fixed_lbl) in enumerate(tqdm(dataset)):
                # Apply perturbation
                fixed_img_perturbed = dataset.apply_appearance_deviations(fixed_img, deviation_type, devation_magnitude)

                # save image to disk
                dataset_folder_destination = os.path.join(dataset_folder,
                                                          "{}-{}".format(deviation_type, devation_magnitude))
                os.makedirs(os.path.join(dataset_folder_destination, 'imagesTs'), exist_ok=True)
                os.makedirs(os.path.join(dataset_folder_destination, 'labelsTs'), exist_ok=True)

                # save image to disk
                dataset_folder_destination = os.path.join(dataset_folder,
                                                          "{}-{}".format(deviation_type, devation_magnitude))
                os.makedirs(os.path.join(dataset_folder_destination, 'imagesTs'), exist_ok=True)
                os.makedirs(os.path.join(dataset_folder_destination, 'labelsTs'), exist_ok=True)

                # Copy unmodified input
                # - input planning MRI
                shutil.copy(dataset.moving_img_path[idx],
                            dataset.moving_img_path[idx].replace(dataset_folder, dataset_folder_destination))
                # - input planning segmentation label
                shutil.copy(dataset.moving_lbl_path[idx],
                            dataset.moving_lbl_path[idx].replace(dataset_folder, dataset_folder_destination))
                # - ground truth fraction segmentation label
                shutil.copy(dataset.fixed_lbl_path[idx],
                            dataset.fixed_lbl_path[idx].replace(dataset_folder, dataset_folder_destination))

                # Save the perturbed fraction MRI
                image_path_destination = dataset.fixed_img_path[idx].replace(dataset_folder,
                                                                             dataset_folder_destination)
                sitk_im = sitk.ReadImage(dataset.fixed_img_path[idx])
                save_image(fixed_img_perturbed.squeeze().numpy(), sitk_im, image_path_destination)

    for deviation_type in ['translation_z']:
        devation_magnitudes = DEFAULT_TRANSFORM_SETTINGS[deviation_type]
        for devation_magnitude in devation_magnitudes:
            print(deviation_type, devation_magnitude)
            seed_everything(1000)
            for idx, (moving_img, fixed_img, moving_lbl, fixed_lbl) in enumerate(tqdm(dataset)):
                moving_img_perturbed, moving_lbl_perturbed = dataset.apply_rigid_deviations(moving_img, moving_lbl,
                                                                                            deviation_type,
                                                                                            devation_magnitude)
                # save image to disk
                dataset_folder_destination = os.path.join(dataset_folder,
                                                          "{}-{}".format(deviation_type, devation_magnitude))
                os.makedirs(os.path.join(dataset_folder_destination, 'imagesTs'), exist_ok=True)
                os.makedirs(os.path.join(dataset_folder_destination, 'labelsTs'), exist_ok=True)

                # Copy unmodified input
                # - input fraction MRI
                shutil.copy(dataset.fixed_img_path[idx],
                            dataset.fixed_img_path[idx].replace(dataset_folder, dataset_folder_destination))
                # - input fraction segmentation label
                shutil.copy(dataset.fixed_lbl_path[idx],
                            dataset.fixed_lbl_path[idx].replace(dataset_folder, dataset_folder_destination))

                # Copy perturbed input
                # - input planning MRI
                image_path_destination = dataset.moving_img_path[idx].replace(dataset_folder,
                                                                              dataset_folder_destination)
                sitk_im = sitk.ReadImage(dataset.moving_img_path[idx])
                save_image(moving_img_perturbed.squeeze().numpy(), sitk_im, image_path_destination)

                # - input planning segmentation label
                image_path_destination = dataset.moving_lbl_path[idx].replace(dataset_folder,
                                                                              dataset_folder_destination)
                sitk_im = sitk.ReadImage(dataset.moving_lbl_path[idx])
                save_image(moving_lbl_perturbed.squeeze().numpy(), sitk_im, image_path_destination)

    for deviation_type in ['deformable']:
        devation_magnitudes = DEFAULT_TRANSFORM_SETTINGS[deviation_type]
        for devation_magnitude in devation_magnitudes:
            print(deviation_type, devation_magnitude)
            seed_everything(1000)
            for idx in tqdm(range(len(dataset))):
                moving_img, fixed_img, moving_lbl, fixed_lbl = dataset[idx]
                _, _, voxel_spacing = sitk_header(sitk.ReadImage(dataset.fixed_img_path[idx]))
                spacing_ratio = voxel_spacing[1] / voxel_spacing[2]

                # Apply perturbation
                fixed_img_perturbed, fixed_lbl_perturbed = dataset.apply_deformable_transform(fixed_img, fixed_lbl,
                                                                                              spacing_ratio,
                                                                                              devation_magnitude)

                # save image to disk
                dataset_folder_destination = os.path.join(dataset_folder,
                                                          "{}-{}".format(deviation_type, devation_magnitude))
                os.makedirs(os.path.join(dataset_folder_destination, 'imagesTs'), exist_ok=True)
                os.makedirs(os.path.join(dataset_folder_destination, 'labelsTs'), exist_ok=True)

                # Copy unmodified input
                # - input planning MRI
                shutil.copy(dataset.moving_img_path[idx],
                            dataset.moving_img_path[idx].replace(dataset_folder, dataset_folder_destination))
                # - input planning segmentation label
                shutil.copy(dataset.moving_lbl_path[idx],
                            dataset.moving_lbl_path[idx].replace(dataset_folder, dataset_folder_destination))

                # Copy perturbed input
                # - input fraction MRI
                image_path_destination = dataset.fixed_img_path[idx].replace(dataset_folder, dataset_folder_destination)
                save_image(fixed_img_perturbed.squeeze().numpy(), sitk.ReadImage(dataset.fixed_img_path[idx]),
                           image_path_destination)
                # - input fraction segemntation
                image_path_destination = dataset.fixed_lbl_path[idx].replace(dataset_folder, dataset_folder_destination)
                save_image(fixed_lbl_perturbed.squeeze().numpy(), sitk.ReadImage(dataset.fixed_lbl_path[idx]),
                           image_path_destination)

if __name__ == "__main__":
    simulated_domain_shifts_Segmentation()
    simulated_domain_shifts_Registration()




