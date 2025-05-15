# Incorporating planning contours into deep learning models (conditional Segmentation and Registration)

This repository contains code and resources for training, evaluating,
and applying deep learning models for medical image **segmentation** and **registration**.
The tools (or links to other repositories) for performance evaluation and robustness testing are also provided.

## Citation
If you use this repository or its contents in your research, please cite the following paper:
> [Insert full citation here]
---

## Pretrained Models
Pretrained model weights are available:
- **[Segmentation](https://drive.google.com/file/d/1yp9gxjmbFuxI97AUHCsXCJoHYDnxLQf4/view?usp=sharing)**  
- **[Registration](https://drive.google.com/file/d/1VIDz7vbUY9KGwG1XSQoWKuB4Adi18Hfh/view?usp=sharing)** 

## Data preparation
- **Segmentation**: The only way to bring your data into nnU-Net is by storing it in a specific format, see [documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). 
- **Registration**: We use a similar format for the input data of the registration model. the following structure is expected:


        DatasetRegistration/
        ├── splits_final.json
        ├── imagesTr
        │   ├── Rect5F_001_F1_0000.nii.gz
        │   ├── Rect5F_001_F2_0000.nii.gz
        │   ├── Rect5F_001_F3_0000.nii.gz
        │   ├── Rect5F_001_F4_0000.nii.gz
        │   ├── Rect5F_001_F5_0000.nii.gz
        │   ├── Rect5F_001_reftoF1_0000.nii.gz
        │   ├── Rect5F_001_reftoF2_0000.nii.gz
        │   ├── Rect5F_001_reftoF3_0000.nii.gz
        │   ├── Rect5F_001_reftoF4_0000.nii.gz
        │   ├── Rect5F_001_reftoF5_0000.nii.gz
        │   ├── Rect5F_002_F1_0000.nii.gz
        │   ├── ...
        ├── labelsTr
        │   ├── Rect5F_001_F1.nii.gz
        │   ├── Rect5F_001_F2.nii.gz
        │   ├── Rect5F_001_F3.nii.gz
        │   ├── Rect5F_001_F4.nii.gz
        │   ├── Rect5F_001_F5.nii.gz
        │   ├── Rect5F_001_reftoF1.nii.gz
        │   ├── Rect5F_001_reftoF2.nii.gz
        │   ├── ...
        ├── imagesTs
        │   ├── Rect5F_001_F1_0000.nii.gz
        │   ├── Rect5F_001_F2_0000.nii.gz
        │   ├── Rect5F_001_F3_0000.nii.gz
        │   ├── Rect5F_001_F4_0000.nii.gz
        │   ├── Rect5F_001_F5_0000.nii.gz
        │   ├── Rect5F_001_reftoF1_0000.nii.gz
        │   ├── Rect5F_001_reftoF2_0000.nii.gz
        │   ├── ...
        └── labelsTs
            ├── Rect5F_001_F1.nii.gz
            ├── Rect5F_001_F2.nii.gz
            ├── Rect5F_001_F3.nii.gz
            ├── Rect5F_001_F4.nii.gz
            ├── Rect5F_001_F5.nii.gz
            ├── Rect5F_001_reftoF1.nii.gz
            ├── ...
      
## Model training and inference
### 1. Segmentation
Segmentation models were developed using the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework . 
For training and inference, please use the official nnU-Net repository and documentation.

### 2. Registration
The code for training and inference of the registration model is located in the `Registration/` directory.

Requirements:
- `nnUNetv2` [repository](https://github.com/MIC-DKFZ/nnUNet/tree/master)
- `dynamic-network-architectures` [repository](https://github.com/MIC-DKFZ/dynamic-network-architectures)
- `pytorch-lightning` (only required for **Training**)
- `ml-collections`
- `monai`
- `numpy`
- `SimpleITK`
- `tqdm`

---

## Evaluation
The code to evaluate model performance is located in the `Evaluation/` directory.
- **Dice Similarity Coefficient (DSC)**
- **Hausdorff Distance (HD)** &rarr; Total and in different zones, i.e., caudal, middle, cranial regions
- **Simulated Domain Shifts** &rarr; To evaluate model robustness under varying conditions, we applied several types of perturbations to the input data:
  - **Intensity-based perturbations** using transformations from [MONAI](https://github.com/Project-MONAI/MONAI/tree/dev)
  - **Spatial translations** to simulate misalignments
  - **Anatomy-informed augmentations** from this [repository](https://github.com/MIC-DKFZ/anatomy_informed_DA)
    > [B. Kovacs, N. Netzer, M. Baumgartner, C. Eith, D. Bounias, C. Meinzer, P. F. Jäger, K. S. Zhang,
R. Floca, A. Schrader, et al., Anatomy-informed data augmentation for enhanced prostate cancer
detection, in: International Conference on Medical Image Computing and Computer-Assisted Inter-
vention, Springer, 2023, pp. 531–540.]
  - To promote reproducibility, we provide the full code to generate these perturbations in `Evaluation/simulated_domain_shifts.py`.
---

