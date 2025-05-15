# Incorporating planning contours into deep learning models (Segmentation and Registration)

This repository contains code and resources for training, evaluating,
and applying deep learning models for medical image **segmentation** and **registration**.
The tools (or links to other repositories) for performance evaluation and robustness testing are also provided.

## Citation
If you use this repository or its contents in your research, please cite the following paper:
> [Insert full citation here]
---

## Pretrained Models
Pretrained model weights are available:
- **Segmentation**: `Segmentation/Model_weights_file/`
- **Registration**: `Registration_model/Model_weights_file/`
- **Baseline Segmentation** (SegBase): `Segmentation/Model_weights_file/`
- **Baseline Registration** (RegBase): `Registration_model/Model_weights_file/`

## Model training and inference
### 1. Segmentation
Segmentation models were developed using the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework . 
For training and inference, please use the official nnU-Net repository and documentation.

### 2. Registration
The code for training and inference of the registration model is located in the `Registration/` directory.

Requirements:
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

