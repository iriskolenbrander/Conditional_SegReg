from glob import glob

import os

from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd

from Utils.utils import read_image_nii_to_torch, sitk_header
from metrics import *

project_root = '/nfs/rtfs03/storage/home/ikolenbr/Documents/projects/J06_Rectum/code_public_repos'
ROOT_OUTPUT = os.path.join(project_root, 'Output')
ROOT_DATA =os.path.join(project_root, 'Example_data/DatasetRegistration')


def evaluate_contours(label_pred_paths, label_gt_paths, subjects, metrics):
    """ INIT DATA """
    for subject in tqdm(subjects):
        for ses in range(1, 6):
            label_pred_path = label_pred_paths.format(subject, ses)
            label_gt_path = label_gt_paths.format(subject, ses)

            if os.path.exists(label_pred_path) and os.path.exists(label_gt_path):
                label_pred = read_image_nii_to_torch(label_pred_path)
                label_gt = read_image_nii_to_torch(label_gt_path)
                _, _, voxel_spacing = sitk_header(sitk.ReadImage(label_pred_path))

                # Compute metrics
                metrics['subject'].append(subject)
                metrics['fraction'].append(ses)
                metrics = compute_metrics(label_pred.squeeze().cpu(),
                                          label_gt.squeeze().cpu(),
                                          voxel_sp=voxel_spacing,
                                          metrics_dict=metrics,
                                          )
            else:
                print(f'Error processing case {subject}: fraction {ses}')
    return pd.DataFrame(metrics)


if __name__ == '__main__':
    predictions_folder = 'predictionsTs_Reg'
    label_pred_paths = os.path.join(ROOT_DATA, predictions_folder, 'Rect5F_{}_F{}.nii.gz')
    label_gt_paths =  os.path.join(ROOT_DATA, 'labelsTs/Rect5F_{}_F{}.nii.gz')
    pred_paths = glob(label_pred_paths.replace('Rect5F_{}_F{}.nii.gz', '***.nii.gz'))
    pred_paths.sort()
    subjects = np.unique([int(os.path.basename(file).split('_')[1]) for file in pred_paths])

    metrics = evaluate_contours(label_pred_paths, label_gt_paths, subjects, metrics)
    metrics.to_csv(os.path.join(ROOT_OUTPUT,  'evaluation_metrics.csv'))