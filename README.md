# medphys_deep_learning
Deep Learning projects with Department of Medical Physics - University of Wisconsin School of Medicine and Public Health.

# BMI/CS 771 Learning Based Methods for Computer Vision

## Dataset used
Not included in submission, we used a subset of the ADNI dataset

## CNN+RNN-2class-1cnn-CLEAN
The CNN+RNN model(s) that we got from the lab.

Main original file: 
mci\_train.py - this file is the original file that trains a CNN and then the RNN and produces a bunch of results in the figures directory

Our important files in this directory:
cnn\_train.py - RNN is removed, only trains the CNN and produces an output predictions.csv file to input into the ViT
vit\_temporal.py - MAIN CODE OF OUR HOMEWORK SUBMISSION, MODIFIED AS NEEDED FOR EXPERIEMENTS; a HEAVILY modified version of vit\_1d.py from the vit-pytorch git repo. Handles input of prediction.csv files and does the all of the training and output of results. Imports loader.py from this directory
loader.py - modified version of loader.py (one directory up) in order to integrate it with the dataloader in vit\_temporal.py

Output files:
predictions.csv - latest run
predictions\_5.csv - renamed saved embed5 vectors (original last layer of CNN)
predictions\_32.csv - renamed saved embed32 vectors (attempted making last layer of CNN bigger and retrained)
predictions\_40.csv - renamed saved embed40 vectors (second to last layer of original CNN)

Other files worth mentioning:
SavedCNNWeights is the saved weights of our CNN model for tensorflow 1.x
figures directory has a lot of good results from the CNN
grouper.py was a prototyping file to figure out how to read in and group images with the same PTIDs together

## Environments
 - environment.yml: Conda environment that was used to train the CNN (note: Code was written a few years ago, so uses Tensorflow 1.X and ran on WIMR lab GPUs)

For completions sake, a couple other environments that were attempted but struggled to get the original CNN code running
 - libs.md: another conda env that was attempted in order to get CNN code up to date
 - requirements.txt: pip3 freeze version of attempting to set up an environment

## models
An earlier attempt to create our own CNN embedder

## vit-tensorflow and vit-pytorch
Subset of their respective repos
 - Pytorch: https://github.com/lucidrains/vit-pytorch
 - Tensorflow: https://github.com/taki0112/vit-tensorflow

## data\_input\_example.py
An example of reading in nifti data

## LP\_ADNIMERGE.csv
Meta data about the images themselves. The data used from this csv file:

 - Class: AD - Alzheimer's Disease; CN - Cognitively Normal; sMCI or pMCI - stable or progressing Mild Cognitive Impairment (not used in this project)
 - PTID: Patient ID
 - Image Data ID: ID number corresponding to the image file name
 - EXAMDATE: Date exam occured
 - Sex
 - Age (current)
 - MMSE: Mini Mental State Examination - 11 questions that tests different areas of cognitive function: orientation, registration, attention, calculation, recall, language

## loader.py
Prototyping loading in PET scans using the csv file to group scans by patient and get exam date embedding and such
