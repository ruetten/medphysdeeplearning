###################################################################################################
### Med Phys: Medical Imaging Deep Learning for Predicting Patient Response to Treatment
### @Author: Laik Ruetten
### @git: https://github.com/ruetten/medphys_deep_learning
###################################################################################################
### Inputs: PET imaging (nifti file format)
### Outputs: progression free survival measured in months.
###          alternatively, calssification problem of 'survivor' or 'regressor' past 25(?) months
### Victor is using what he calls hand-crafted features and predicts progression free survival.
### Brayden would like to attempt the same thing but use the raw Image data as input.
### We predict that hand-crafted will do better for smaller dataset
### while deep-learning will work better with more data
### I'm worried about the small size of the dataset being only 34 out of a total of 100
###################################################################################################
import logging
import sys
import os

import numpy as np
import torch

import monai
from monai.data import ImageDataset, DataLoader
import nibabel


# nibable - nifti to numpy
# monai

### Preprocess data
# - equal voxel spacing in all three cardinal directions
# - resample to an isotropic space
# - normalization
# - Brayden is doing this step externally

### Data Generator
# - loads training data with ground truth
# - determine splits between training and test
def main():
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_path = os.sep.join(
        ["W:", "_Data", "NET_DL_Predict", "data", "dynamic", "processed_data", "baseline", "processed_PET",
         "bodyCrop_vox4p5xy8p0z_Standardized"])
    images = [
        "NET01_2018_02_26_crop_rawVox.nii.gz",
        "NET02_2018_04_05_crop_rawVox.nii.gz",
        "NET03_2018_02_15_crop_rawVox.nii.gz",
        "NET04_2017_10_03_crop_rawVox.nii.gz",
        "NET05_2018_04_03_crop_rawVox.nii.gz",
        "NET07_2018_07_16_crop_rawVox.nii.gz",
        "NET08_2018_10_08_crop_rawVox.nii.gz",
        "NET09_2017_08_01_crop_rawVox.nii.gz",
        "NET11_2017_12_11_crop_rawVox.nii.gz",
        "NET14_2018_02_08_crop_rawVox.nii.gz",
        "NET15_2019_05_22_crop_rawVox.nii.gz",
        "NET18_2018_05_16_crop_rawVox.nii.gz",
        "NET19_2020_11_12_crop_rawVox.nii.gz",
        "NET20_2018_09_20_crop_rawVox.nii.gz",
        "NET23_2019_04_05_crop_rawVox.nii.gz",
        "NET29_2017_10_10_crop_rawVox.nii.gz",
        "NET30_2019_09_12_crop_rawVox.nii.gz",
        "NET31_2019_11_01_crop_rawVox.nii.gz",
        "NET41_2020_03_06_crop_rawVox.nii.gz",
        "NET45_2019_04_17_crop_rawVox.nii.gz",
        "NET47_2018_12_14_crop_rawVox.nii.gz",
        "NET53_2018_10_02_crop_rawVox.nii.gz",
        "NET54_2018_08_20_crop_rawVox.nii.gz",
        "NET55_2019_10_18_crop_rawVox.nii.gz",
        "NET70_2020_12_15_crop_rawVox.nii.gz",
        "NET73_2021_01_26_crop_rawVox.nii.gz",
        "NET74_2019_11_26_crop_rawVox.nii.gz",
        "NET77_2021_02_23_crop_rawVox.nii.gz",
        "NET78_2021_04_12_crop_rawVox.nii.gz",
        "NET85_2021_12_02_crop_rawVox.nii.gz",
        "NET86_2020_11_13_crop_rawVox.nii.gz",
        "RP01_tp01_crop_rawVox.nii.gz",
        "RP03_tp01_crop_rawVox.nii.gz",
        "RP08_tp01_crop_rawVox.nii.gz",
        "RP10_tp02_crop_rawVox.nii.gz",
        "RP11_tp01_crop_rawVox.nii.gz"
    ]
    images = [os.sep.join([data_path, f]) for f in images]

    # TODO labels are not correct
    labels = np.array([0, 0, 0,
                       0, 0, 0,
                       0, 0, 0,
                       0, 0, 0,
                       0, 0, 0,
                       0, 0, 0,
                       1, 1, 1,
                       1, 1, 1,
                       1, 1, 1,
                       1, 1, 1,
                       1, 1, 1,
                       1, 1, 1], dtype=np.int64)
    # TODO hyperparameters

    check_ds = ImageDataset(image_files=images, labels=labels)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())
    im, label = monai.utils.misc.first(check_loader)
    print(type(im), im.shape, label)


### Separate data generator for validation

### Set training parameters
# - Epochs and training iterations

### Create/Define/Load model
# monai has some popular models
# - ResNet
# - DenseNet
# - Inception - might not be in Monai
# - Some transformers
# - have some sort of master script, difference between is different architectures

### Train
# Good to save an ongoing plot for each epoch

### Validate

### Test


if __name__ == "__main__":
    main()
