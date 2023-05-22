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

### Imports
# nibable - nifti to numpy
# sklearn?
# monai?
# matplotlib? other?
#
# do I need to setup anaconda? jupyter notebook?

### Preprocess data
# - equal voxel spacing in all three cardinal directions
# - resample to an isotropic space
# - normalization
# - Brayden is doing this step externally

### Data Generator
# - loads training data with ground truth
# - determine splits between training and test

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
