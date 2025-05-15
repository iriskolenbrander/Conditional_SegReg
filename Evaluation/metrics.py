import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure

from Pytorch_lightning.lightning_metrics import MyHausdorffMultiLabel, MyFolding, MyHausdorffMultiLabelPerRegion
from Pytorch_lightning.lightning_module import MyDiceMultiLabel

metrics = dict(dice_CTVmeso=list(),

               hausdorff_CTVmeso=list(),
               hausdorff_CTVmeso_cranial=list(),
               hausdorff_CTVmeso_caudal=list(),
               hausdorff_CTVmeso_middle=list(),

               ssim=list(),
               folding=list(),

               subject=list(),
               fraction=list())

metrics_baseline = dict(dice_CTVmeso=list(),

                        hausdorff_CTVmeso=list(),
                        hausdorff_CTVmeso_cranial=list(),
                        hausdorff_CTVmeso_caudal=list(),
                        hausdorff_CTVmeso_middle=list(),

                        ssim=list(),
                        folding=list(),

                        subject=list(),
                        fraction=list())


def get_slices(label_gt):
    # Get the upper slice containing the ground truth segmentation
    upper = np.where(label_gt.cpu().numpy().sum(axis=(1, 2)) > 0)[0][-1]
    # Get the lower slice containing the ground truth segmentation
    lower = np.where(label_gt.cpu().numpy().sum(axis=(1, 2)) > 0)[0][0]
    diff = upper - lower
    slices_list = [(0, lower + int(0.3 * diff)),
                   (lower + int(0.3 * diff), lower + int(0.7 * diff)),
                   (lower + int(0.7 * diff), label_gt.shape[0])]
    return slices_list


def compute_metrics(label_1, label_2, voxel_sp, metrics_dict,
                    im_1=None, im_2=None,
                    dvf=None):
    if im_1 is not None:
        ssim = StructuralSimilarityIndexMeasure()(im_1, im_2)
        metrics_dict['ssim'].append(ssim.item())
    else:
        metrics_dict['ssim'].append(np.nan)

    if dvf is not None:
        folding = MyFolding()(dvf)
        metrics_dict['folding'].append(folding.item())
    else:
        metrics_dict['folding'].append(np.nan)

    # Dice and Hausdorff distances
    # Dice
    dice = MyDiceMultiLabel(max_label=1)(label_1, label_2)
    metrics_dict['dice_CTVmeso'].append(dice.item())

    hd = MyHausdorffMultiLabel(max_label=1,
                               voxel_spacing=voxel_sp, percentile=100)(label_1, label_2)
    metrics_dict['hausdorff_CTVmeso'].append(hd.item())

    # 100th percentile (max) Hausdorff distance in different zones
    slices_list = get_slices(label_2)
    post_fix_list = ['_caudal', '_middle', '_cranial']
    for post_fix, slices in zip(post_fix_list, slices_list):
        hd = MyHausdorffMultiLabelPerRegion(max_label=1,
                                            voxel_spacing=voxel_sp,
                                            percentile=100)(label_1, label_2, slices=slices)
        metrics_dict['hausdorff_CTVmeso' + post_fix].append(hd.item())
    return metrics_dict
