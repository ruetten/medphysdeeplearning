import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize   
from keras import regularizers 
import pickle as pickle
from utils.preprocess import DataLoader
from utils.models import Parameters, CNN_Net, RNN_Net
from utils.heatmapPlotting import heatmapPlotter
from matplotlib import pyplot as plt
import pandas as pd
from scipy import interp
from keras.models import Model, load_model#, load_weights
from keras.layers import Input
from keras.optimizers import Adam
import tensorflow as tf
from IPython.display import Image
import matplotlib.cm as cm
import SimpleITK as sitk
import csv
from copy import deepcopy
import matplotlib.colors as mcolors
import nibabel as nib
import math

import sys
sys.path.append('//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class-1cnn-CLEAN/utils')
from sepconv3D import SeparableConv3D
print(sys.version)

##for 2 class CNN + RNN ##
#Dummy feature vectors are added to feature vectors from CNN (which are fed only the images)


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use id from $ nvidia-smi

target_rows = 91
target_cols = 109
depth = 91
axis = 1
num_clinical = 2
CNN_drop_rate = 0.3
RNN_drop_rate = 0.1
CNN_w_regularizer = regularizers.l2(2e-2)
RNN_w_regularizer = regularizers.l2(1e-6)
CNN_batch_size = 10
RNN_batch_size = 5
val_split = 0.2
optimizer = Adam(lr=1e-5)
final_layer_size = 5

model_filepath = '//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class-1cnn-CLEAN'
mri_datapath = '//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class-1cnn-CLEAN/ADNI_volumes_customtemplate_float32'


params_dict = { 'CNN_w_regularizer': CNN_w_regularizer, 'RNN_w_regularizer': RNN_w_regularizer,
               'CNN_batch_size': CNN_batch_size, 'RNN_batch_size': RNN_batch_size,
               'CNN_drop_rate': CNN_drop_rate, 'epochs': 2,
          'gpu': "/gpu:0", 'model_filepath': model_filepath, 
          'image_shape': (target_rows, target_cols, depth, axis),
          'num_clinical': num_clinical,
          'final_layer_size': final_layer_size,
          'optimizer': optimizer, 'RNN_drop_rate': RNN_drop_rate,}

params = Parameters(params_dict)

seeds = [np.random.randint(1, 5000) for _ in range(1)]

def evaluate_net (seed):
    n_classes = 2
    data_loader = DataLoader((target_rows, target_cols, depth, axis), seed = seed)
    train_data, val_data, test_data,rnn_HdataT1,rnn_HdataT2,rnn_HdataT3,rnn_AdataT1,rnn_AdataT2,rnn_AdataT3, test_mri_nonorm = data_loader.get_train_val_test(val_split, mri_datapath)

    print('Length Val Data[0]: ',len(val_data[0]))

#RUN THE CNN:      
    netCNN = CNN_Net(params)    
    historyCNN, featuresModel_CNN = netCNN.train((train_data, val_data))
    test_lossCNN, test_accCNN  = netCNN.evaluate(test_data)
    test_predsCNN = netCNN.predict(test_data)
    """
    #TO LOAD A PREVIOUS MODEL FOR HEATMAPS: (uncomment this chunk and comment above chunk) #Then add the pickle file and savedWeights to the modelfilepath folder
    #can't seem to figure out how to load the whole model (but am saving it anyway). I'm only able to save and load the weights, so note that the model needs to be recompiled, so it has to be the correct architecture
    picklename = '1820'
    netCNN = CNN_Net(params)
    netCNN.load_the_weights("SavedCNNWeights")
    pickle_in = open(model_filepath+'/'+picklename+'.pickle', 'rb') 
    pickle0=pickle.load(pickle_in)
    pickle_in.close()
    test_data = pickle0[5][0]
    pickle0 = 0  #to save memory
    test_lossCNN, test_accCNN  = netCNN.evaluate(test_data)
    test_predsCNN = netCNN.predict(test_data)
    print('check_lossCNN, check_accCNN: '+ str(test_lossCNN)+', '+ str(test_accCNN))
    """
    
##PREP DATA FOR THE RNN
#Get the feature vectors from the final layer for each training image at each timepoint:
    rnn_HpredsT1 = featuresModel_CNN.predict([rnn_HdataT1[0],rnn_HdataT1[1],rnn_HdataT1[2]])
    rnn_HpredsT2 = featuresModel_CNN.predict([rnn_HdataT2[0],rnn_HdataT2[1],rnn_HdataT2[2]])
    rnn_HpredsT3 = featuresModel_CNN.predict([rnn_HdataT3[0],rnn_HdataT3[1],rnn_HdataT3[2]]) 
    rnn_ApredsT1 = featuresModel_CNN.predict([rnn_AdataT1[0],rnn_AdataT1[1],rnn_AdataT1[2]])
    rnn_ApredsT2 = featuresModel_CNN.predict([rnn_AdataT2[0],rnn_AdataT2[1],rnn_AdataT2[2]])
    rnn_ApredsT3 = featuresModel_CNN.predict([rnn_AdataT3[0],rnn_AdataT3[1],rnn_AdataT3[2]])    
    print(rnn_HpredsT1, rnn_HpredsT1.shape)
    print(rnn_HpredsT2, rnn_HpredsT2.shape)
    print(rnn_HpredsT3, rnn_HpredsT3.shape)
    print(rnn_ApredsT1, rnn_ApredsT1.shape)
    print(rnn_ApredsT2, rnn_ApredsT2.shape)
    print(rnn_ApredsT3, rnn_ApredsT3.shape)
    

#grab the PTIDs for each dataset
    rnn_HptidT1 = rnn_HdataT1[4]
    rnn_HptidT2 = rnn_HdataT2[4]
    rnn_HptidT3 = rnn_HdataT3[4]
    rnn_AptidT1 = rnn_AdataT1[4]
    rnn_AptidT2 = rnn_AdataT2[4]
    rnn_AptidT3 = rnn_AdataT3[4]
#grab the imageIDs for each dataset
    rnn_HimageIDT1 = rnn_HdataT1[5]
    rnn_HimageIDT2 = rnn_HdataT2[5]
    rnn_HimageIDT3 = rnn_HdataT3[5]
    rnn_AimageIDT1 = rnn_AdataT1[5]
    rnn_AimageIDT2 = rnn_AdataT2[5]
    rnn_AimageIDT3 = rnn_AdataT3[5]

#add dummy feature vectors to all missing timepoints
    dummyVector = np.full((final_layer_size),-1)

    #Healthy patients
    rnn_HpredsT1_padded = []
    rnn_HpredsT2_padded = []
    rnn_HpredsT3_padded = []
    rnn_HptidT1_padded = []
    rnn_HptidT2_padded = []
    rnn_HptidT3_padded = []
    rnn_HimageIDT1_padded = []
    rnn_HimageIDT2_padded = []
    rnn_HimageIDT3_padded = []
    j=0
    HrnnT1T2T3 = 0
    HrnnT1T2 = 0
    HrnnT1T3 = 0
    HrnnT1 = 0
    HrnnT2 = 0
    HrnnT2T3 = 0
    HrnnT3 = 0
    HrnnT1Removed = 0
    for ptidT1 in rnn_HptidT1:
        rnn_HpredsT1_padded.append(rnn_HpredsT1[j])
        rnn_HptidT1_padded.append(ptidT1)
        rnn_HimageIDT1_padded.append(rnn_HimageIDT1[j])
        j+=1
        c=0
        k=0
        t2 = False
        t3 = False
        for ptidT2 in rnn_HptidT2:
            c+=1
            if ptidT1 == ptidT2:
                rnn_HpredsT2_padded.append(rnn_HpredsT2[c-1])
                rnn_HptidT2_padded.append(ptidT2)
                rnn_HimageIDT2_padded.append(rnn_HimageIDT2[c-1])
                t2 = True
                for ptidT3 in rnn_HptidT3:
                    k+=1
                    if ptidT1 == ptidT3:
                        rnn_HpredsT3_padded.append(rnn_HpredsT3[k-1])
                        rnn_HptidT3_padded.append(ptidT3)
                        rnn_HimageIDT3_padded.append(rnn_HimageIDT3[k-1])
                        HrnnT1T2T3+=1
                        t3 = True
                        break
                if t3 == False:
                    rnn_HpredsT3_padded.append(dummyVector)
                    rnn_HptidT3_padded.append(ptidT1)
                    rnn_HimageIDT3_padded.append('dummy')
                    HrnnT1T2+=1
                    break
        if t2 == False:
            rnn_HpredsT2_padded.append(dummyVector)
            rnn_HptidT2_padded.append(ptidT1)
            rnn_HimageIDT2_padded.append('dummy')
            for ptidT3 in rnn_HptidT3:
                k+=1
                if ptidT1 == ptidT3:
                    rnn_HpredsT3_padded.append(rnn_HpredsT3[k-1])
                    rnn_HptidT3_padded.append(ptidT3)
                    rnn_HimageIDT3_padded.append(rnn_HimageIDT3[k-1])
                    HrnnT1T3+=1
                    t3 = True
                    break    
            if t3 == False:
                #rnn_HpredsT3_padded.append(dummyVector)
                HrnnT1+=1
                rnn_HpredsT1_padded.pop(-1)   #remove any scans that have only T1
                rnn_HpredsT2_padded.pop(-1)
                rnn_HptidT1_padded.pop(-1)
                rnn_HptidT2_padded.pop(-1)
                rnn_HimageIDT1_padded.pop(-1)
                rnn_HimageIDT2_padded.pop(-1)
                HrnnT1Removed+=1
    c=0
    for ptidT2 in rnn_HptidT2:
        c+=1
        j=0
        k=0
        match = False
        t3=False
        for ptidT1 in rnn_HptidT1:
            j+=1
            if ptidT2 == ptidT1:
                match = True
        if match == False:
            rnn_HpredsT2_padded.append(rnn_HpredsT2[c-1])
            rnn_HpredsT1_padded.append(dummyVector)
            rnn_HptidT2_padded.append(ptidT2)
            rnn_HimageIDT2_padded.append(rnn_HimageIDT2[c-1])
            rnn_HptidT1_padded.append(ptidT1)
            rnn_HimageIDT1_padded.append('dummy')
            for ptidT3 in rnn_HptidT3:
                k+=1
                if ptidT2 == ptidT3:
                    rnn_HpredsT3_padded.append(rnn_HpredsT3[k-1])
                    rnn_HptidT3_padded.append(ptidT2)
                    rnn_HimageIDT3_padded.append(rnn_HimageIDT3[k-1])
                    t3 = True
                    HrnnT2T3+=1
                    break
            if t3 == False:
                rnn_HpredsT3_padded.append(dummyVector)
                rnn_HptidT3_padded.append(ptidT1)
                rnn_HimageIDT3_padded.append('dummy')
                HrnnT2+=1
    k=0
    for ptidT3 in rnn_HptidT3:
        k+=1
        j=0
        c=0
        match1 = False
        for ptidT1 in rnn_HptidT1:
            j+=1
            if ptidT3 == ptidT1:
                match1 = True
#        if match1 == True:
#            break
        if match1 == False:
            match2 = False
            for ptidT2 in rnn_HptidT2:
                c+=1
                if ptidT3 == ptidT2:
                    match2 = True
#            if match2 == True:
#                break
            if match2 == False:
                rnn_HpredsT3_padded.append(rnn_HpredsT3[k-1])
                rnn_HptidT3_padded.append(ptidT3)
                rnn_HimageIDT3_padded.append(rnn_HimageIDT3[k-1])
                rnn_HpredsT1_padded.append(dummyVector)
                rnn_HptidT1_padded.append(ptidT1)
                rnn_HimageIDT1_padded.append('dummy')
                rnn_HpredsT2_padded.append(dummyVector)
                rnn_HptidT2_padded.append(ptidT1)
                rnn_HimageIDT2_padded.append('dummy')
                HrnnT3+=1
            
    #move the data from a list to an array
    j=0
    c=0
    k=0
    LenPadded = len(rnn_HpredsT1_padded)
    rnn_HpredsT1_padArray = np.zeros((LenPadded,final_layer_size), dtype=object)
    rnn_HpredsT2_padArray = np.zeros((LenPadded,final_layer_size), dtype=object)
    rnn_HpredsT3_padArray = np.zeros((LenPadded,final_layer_size), dtype=object)
    for vector in rnn_HpredsT1_padded:
        rnn_HpredsT1_padArray[j] = vector
        j+=1
    for vector in rnn_HpredsT2_padded:
        rnn_HpredsT2_padArray[c] = vector
        c+=1
    for vector in rnn_HpredsT3_padded:
        rnn_HpredsT3_padArray[k] = vector
        k+=1    
        
    with open(model_filepath+'/figures/paddedPreds.txt','w') as paddedPreds:
            paddedPreds.write('Train Preds Sizes: '+'\n') 
            paddedPreds.write('Type of rnn_HpredsT1: '+str(type(rnn_HpredsT1))+'\n')            
            paddedPreds.write('Type of rnn_HpredsT1_padded: '+str(type(rnn_HpredsT1_padded))+'\n')
            paddedPreds.write('Type of rnn_HpredsT1_padArray: '+str(type(rnn_HpredsT1_padArray))+'\n')  
            paddedPreds.write('Type of rnn_HpredsT1 elements: '+str(type(rnn_HpredsT1[0]))+'\n')            
            paddedPreds.write('Type of rnn_HpredsT1_padded elements: '+str(type(rnn_HpredsT1_padded[0]))+'\n')
            paddedPreds.write('Type of rnn_HpredsT1_padArray elements: '+str(type(rnn_HpredsT1_padArray[0]))+'\n')
            paddedPreds.write('Length of rnn_HpredsT1: '+str(len(rnn_HpredsT1))+'\n')
            paddedPreds.write('Length of rnn_HpredsT1_padded: '+str(len(rnn_HpredsT1_padded))+'\n')
            paddedPreds.write('Length of rnn_HpredsT2_padded: '+str(len(rnn_HpredsT2_padded))+'\n')
            paddedPreds.write('Length of rnn_HpredsT3_padded: '+str(len(rnn_HpredsT3_padded))+'\n')
            paddedPreds.write('Length of rnn_HpredsT1_padArray: '+str(len(rnn_HpredsT1_padArray))+'\n')
            paddedPreds.write('Length of rnn_HpredsT2_padArray: '+str(len(rnn_HpredsT2_padArray))+'\n')
            paddedPreds.write('Length of rnn_HpredsT3_padArray: '+str(len(rnn_HpredsT3_padArray))+'\n')
            paddedPreds.write('Length of rnn_HptidT1_padded: '+str(len(rnn_HptidT1_padded))+'\n')
            paddedPreds.write('Length of rnn_HptidT2_padded: '+str(len(rnn_HptidT2_padded))+'\n')
            paddedPreds.write('Length of rnn_HptidT3_padded: '+str(len(rnn_HptidT3_padded))+'\n')
            paddedPreds.write('Length of rnn_HimageIDT1_padded: '+str(len(rnn_HimageIDT1_padded))+'\n')
            paddedPreds.write('Length of rnn_HimageIDT2_padded: '+str(len(rnn_HimageIDT2_padded))+'\n')
            paddedPreds.write('Length of rnn_HimageIDT3_padded: '+str(len(rnn_HimageIDT3_padded))+'\n')
            paddedPreds.write('RNN_HpredsT1_padded: '+str(rnn_HpredsT1_padded)+'\n')
            paddedPreds.write('Compare to RNN_HpredsT1: '+str(rnn_HpredsT1)+'\n')
            paddedPreds.write('RNN_HpredsT1_padArray: '+str(rnn_HpredsT1_padArray)+'\n')
            paddedPreds.write('RNN_HpredsT2_padArray: '+str(rnn_HpredsT2_padArray)+'\n')
            paddedPreds.write('RNN_HpredsT3_padArray: '+str(rnn_HpredsT3_padArray)+'\n')
            paddedPreds.write('Shape of RNN_HpredsT1_padArray: '+str(rnn_HpredsT1_padArray.shape)+'\n')
            paddedPreds.write('Shape of RNN_HpredsT1: '+str(rnn_HpredsT1.shape)+'\n')
            paddedPreds.write('RNN_HpredsT1[0]: '+str(rnn_HpredsT1[0])+'\n')
            paddedPreds.write('rnn_HpredsT1[0][0]: '+str(rnn_HpredsT1[0][0])+'\n')
            paddedPreds.write('rnn_HpredsT1_padArray[0]: '+str(rnn_HpredsT1_padArray[0])+'\n')
            paddedPreds.write('rnn_HpredsT1_padArray[0][0]: '+str(rnn_HpredsT1_padArray[0][0])+'\n')
            paddedPreds.write('# of Hrnn T1 only: '+str(HrnnT1)+'\n')
            paddedPreds.write('# of Hrnn T1 only Removed: '+str(HrnnT1Removed)+'\n')
            paddedPreds.write('# of Hrnn T1+T2: '+str(HrnnT1T2)+'\n')
            paddedPreds.write('# of Hrnn T1+T2+T3: '+str(HrnnT1T2T3)+'\n')
            paddedPreds.write('# of Hrnn T1+T3: '+str(HrnnT1T3)+'\n')
            paddedPreds.write('# of Hrnn T2 only: '+str(HrnnT2)+'\n')
            paddedPreds.write('# of Hrnn T2+T3: '+str(HrnnT2T3)+'\n')
            paddedPreds.write('# of Hrnn T3 only: '+str(HrnnT3)+'\n')

    #AD patients
    rnn_ApredsT1_padded = []
    rnn_ApredsT2_padded = []
    rnn_ApredsT3_padded = []
    rnn_AptidT1_padded = []
    rnn_AptidT2_padded = []
    rnn_AptidT3_padded = []
    rnn_AimageIDT1_padded = []
    rnn_AimageIDT2_padded = []
    rnn_AimageIDT3_padded = []
    j=0
    ArnnT1T2T3 = 0
    ArnnT1T2 = 0
    ArnnT1T3 = 0
    ArnnT1 = 0
    ArnnT2 = 0
    ArnnT2T3 = 0
    ArnnT3 = 0
    ArnnT1Removed = 0
    for ptidT1 in rnn_AptidT1:
        rnn_ApredsT1_padded.append(rnn_ApredsT1[j])
        rnn_AptidT1_padded.append(ptidT1)
        rnn_AimageIDT1_padded.append(rnn_AimageIDT1[j])
        j+=1
        c=0
        k=0
        t2 = False
        t3 = False
        for ptidT2 in rnn_AptidT2:
            c+=1
            if ptidT1 == ptidT2:
                rnn_ApredsT2_padded.append(rnn_ApredsT2[c-1])
                rnn_AptidT2_padded.append(ptidT2)
                rnn_AimageIDT2_padded.append(rnn_AimageIDT2[c-1])
                t2 = True
                for ptidT3 in rnn_AptidT3:
                    k+=1
                    if ptidT1 == ptidT3:
                        rnn_ApredsT3_padded.append(rnn_ApredsT3[k-1])
                        rnn_AptidT3_padded.append(ptidT3)
                        rnn_AimageIDT3_padded.append(rnn_AimageIDT3[k-1])
                        ArnnT1T2T3+=1
                        t3 = True
                        break
                if t3 == False:
                    rnn_ApredsT3_padded.append(dummyVector)
                    rnn_AptidT3_padded.append(ptidT1)
                    rnn_AimageIDT3_padded.append('dummy')
                    ArnnT1T2+=1
                    break
        if t2 == False:
            rnn_ApredsT2_padded.append(dummyVector)
            rnn_AptidT2_padded.append(ptidT1)
            rnn_AimageIDT2_padded.append('dummy')
            for ptidT3 in rnn_AptidT3:
                k+=1
                if ptidT1 == ptidT3:
                    rnn_ApredsT3_padded.append(rnn_ApredsT3[k-1])
                    rnn_AptidT3_padded.append(ptidT3)
                    rnn_AimageIDT3_padded.append(rnn_AimageIDT3[k-1])
                    ArnnT1T3+=1
                    t3 = True
                    break    
            if t3 == False:
                #rnn_ApredsT3_padded.append(dummyVector)
                ArnnT1+=1
                rnn_ApredsT1_padded.pop(-1)   #remove any scans that have only T1
                rnn_ApredsT2_padded.pop(-1)
                rnn_AptidT1_padded.pop(-1)
                rnn_AimageIDT1_padded.pop(-1)
                rnn_AptidT2_padded.pop(-1)
                rnn_AimageIDT2_padded.pop(-1)
                ArnnT1Removed+=1
    c=0
    for ptidT2 in rnn_AptidT2:
        c+=1
        j=0
        k=0
        match = False
        t3=False
        for ptidT1 in rnn_AptidT1:
            j+=1
            if ptidT2 == ptidT1:
                match = True
        if match == False:
            rnn_ApredsT2_padded.append(rnn_ApredsT2[c-1])
            rnn_AptidT2_padded.append(ptidT2)
            rnn_AimageIDT2_padded.append(rnn_AimageIDT2[c-1])
            rnn_ApredsT1_padded.append(dummyVector)
            rnn_AptidT1_padded.append(ptidT1)
            rnn_AimageIDT1_padded.append('dummy')
            for ptidT3 in rnn_AptidT3:
                k+=1
                if ptidT2 == ptidT3:
                    rnn_ApredsT3_padded.append(rnn_ApredsT3[k-1])
                    rnn_AptidT3_padded.append(ptidT3)
                    rnn_AimageIDT3_padded.append(rnn_AimageIDT3[k-1])
                    t3 = True
                    ArnnT2T3+=1
                    break
            if t3 == False:
                rnn_ApredsT3_padded.append(dummyVector)
                rnn_AptidT3_padded.append(ptidT1)
                rnn_AimageIDT3_padded.append('dummy')
                ArnnT2+=1
    k=0
    for ptidT3 in rnn_AptidT3:
        k+=1
        j=0
        c=0
        match1 = False
        for ptidT1 in rnn_AptidT1:
            j+=1
            if ptidT3 == ptidT1:
                match1 = True
#        if match1 == True:
#            break
        if match1 == False:
            match2 = False
            for ptidT2 in rnn_AptidT2:
                c+=1
                if ptidT3 == ptidT2:
                    match2 = True
#            if match2 == True:
#                break
            if match2 == False:
                rnn_ApredsT3_padded.append(rnn_ApredsT3[k-1])
                rnn_AptidT3_padded.append(ptidT3)
                rnn_AimageIDT3_padded.append(rnn_AimageIDT3[k-1])
                rnn_ApredsT1_padded.append(dummyVector)
                rnn_AptidT1_padded.append(ptidT1)
                rnn_AimageIDT1_padded.append('dummy')
                rnn_ApredsT2_padded.append(dummyVector)
                rnn_AptidT2_padded.append(ptidT1)
                rnn_AimageIDT2_padded.append('dummy')
                ArnnT3+=1
            
    #move the data from a list to an array
    j=0
    c=0
    k=0
    LenPadded = len(rnn_ApredsT1_padded)
    rnn_ApredsT1_padArray = np.zeros((LenPadded,final_layer_size), dtype=object)
    rnn_ApredsT2_padArray = np.zeros((LenPadded,final_layer_size), dtype=object)
    rnn_ApredsT3_padArray = np.zeros((LenPadded,final_layer_size), dtype=object)
    for vector in rnn_ApredsT1_padded:
        rnn_ApredsT1_padArray[j] = vector
        j+=1
    for vector in rnn_ApredsT2_padded:
        rnn_ApredsT2_padArray[c] = vector
        c+=1
    for vector in rnn_ApredsT3_padded:
        rnn_ApredsT3_padArray[k] = vector
        k+=1    
        
    with open(model_filepath+'/figures/paddedPreds.txt','a') as paddedPreds:
            paddedPreds.write('Length of rnn_ApredsT1_padArray: '+str(len(rnn_ApredsT1_padArray))+'\n')
            paddedPreds.write('Length of rnn_ApredsT2_padArray: '+str(len(rnn_ApredsT2_padArray))+'\n')
            paddedPreds.write('Length of rnn_ApredsT3_padArray: '+str(len(rnn_ApredsT3_padArray))+'\n')
            paddedPreds.write('Length of rnn_AptidT1_padded: '+str(len(rnn_AptidT1_padded))+'\n')
            paddedPreds.write('Length of rnn_AptidT2_padded: '+str(len(rnn_AptidT2_padded))+'\n')
            paddedPreds.write('Length of rnn_AptidT3_padded: '+str(len(rnn_AptidT3_padded))+'\n')
            paddedPreds.write('Length of rnn_AimageIDT1_padded: '+str(len(rnn_AimageIDT1_padded))+'\n')
            paddedPreds.write('Length of rnn_AimageIDT2_padded: '+str(len(rnn_AimageIDT2_padded))+'\n')
            paddedPreds.write('Length of rnn_AimageIDT3_padded: '+str(len(rnn_AimageIDT3_padded))+'\n')
            paddedPreds.write('# of Arnn T1 only: '+str(ArnnT1)+'\n')
            paddedPreds.write('# of Arnn T1 only Removed: '+str(ArnnT1Removed)+'\n')
            paddedPreds.write('# of Arnn T1+T2: '+str(ArnnT1T2)+'\n')
            paddedPreds.write('# of Arnn T1+T2+T3: '+str(ArnnT1T2T3)+'\n')
            paddedPreds.write('# of Arnn T1+T3: '+str(ArnnT1T3)+'\n')
            paddedPreds.write('# of Arnn T2 only: '+str(ArnnT2)+'\n')
            paddedPreds.write('# of Arnn T2+T3: '+str(ArnnT2T3)+'\n')
            paddedPreds.write('# of Arnn T3 only: '+str(ArnnT3)+'\n')            

#Balance the datasets: (drop the last scans from the H datasets to make the A and H datasets equal. Should be different patients each time because I shuffled in get_filenames
    diff = len(rnn_HpredsT1_padArray)-len(rnn_ApredsT1_padArray)
    for i in range(diff):
        rnn_HpredsT1_padArray = np.delete(rnn_HpredsT1_padArray,-1,0)
        rnn_HpredsT2_padArray = np.delete(rnn_HpredsT2_padArray,-1,0)
        rnn_HpredsT3_padArray = np.delete(rnn_HpredsT3_padArray,-1,0)
    dummyCountHT1 = 0
    dummyCountHT2 = 0
    dummyCountHT3 = 0
    dummyCountAT1 = 0
    dummyCountAT2 = 0
    dummyCountAT3 = 0
    for i in range(len(rnn_HpredsT1_padArray)):
        if rnn_HpredsT1_padArray[i][0] == -1:
            dummyCountHT1 += 1
        if rnn_HpredsT2_padArray[i][0] == -1:
            dummyCountHT2 += 1
        if rnn_HpredsT3_padArray[i][0] == -1:
            dummyCountHT3 += 1
    for i in range(len(rnn_ApredsT1_padArray)):
        if rnn_ApredsT1_padArray[i][0] == -1:
            dummyCountAT1 += 1
        if rnn_ApredsT2_padArray[i][0] == -1:
            dummyCountAT2 += 1
        if rnn_ApredsT3_padArray[i][0] == -1:
            dummyCountAT3 += 1
    with open(model_filepath+'/figures/paddedPreds.txt','a') as paddedPreds:
        paddedPreds.write('Length of rnn_HpredsT1_padArray popped: '+str(len(rnn_HpredsT1_padArray))+'\n')
        paddedPreds.write('Length of rnn_HpredsT2_padArray popped: '+str(len(rnn_HpredsT2_padArray))+'\n')
        paddedPreds.write('Length of rnn_HpredsT3_padArray popped: '+str(len(rnn_HpredsT3_padArray))+'\n')
    with open(model_filepath+'/figures/DataList.txt','a') as datalist:
        datalist.write('Number of scans in HT1 (excluding dummies): '+str(len(rnn_HpredsT1_padArray)-dummyCountHT1)+'\n')
        datalist.write('Number of scans in HT2 (excluding dummies): '+str(len(rnn_HpredsT2_padArray)-dummyCountHT2)+'\n')
        datalist.write('Number of scans in HT3 (excluding dummies): '+str(len(rnn_HpredsT3_padArray)-dummyCountHT3)+'\n') 
        datalist.write('Number of scans in AT1 (excluding dummies): '+str(len(rnn_ApredsT1_padArray)-dummyCountAT1)+'\n')
        datalist.write('Number of scans in AT2 (excluding dummies): '+str(len(rnn_ApredsT2_padArray)-dummyCountAT2)+'\n')
        datalist.write('Number of scans in AT3 (excluding dummies): '+str(len(rnn_ApredsT3_padArray)-dummyCountAT3)+'\n') 
            
#Split RNN data into train/val/test        
    train_predsT1_padArray,train_predsT2_padArray,train_predsT3_padArray,val_predsT1_padArray,val_predsT2_padArray,val_predsT3_padArray,test_predsT1_padArray,test_predsT2_padArray,test_predsT3_padArray, train_labels_padArray,val_labels_padArray,test_labels_padArray, test_ptidT1,test_ptidT2,test_ptidT3,test_imageIDT1,test_imageIDT2,test_imageIDT3 = data_loader.split_data_RNN(rnn_HpredsT1_padArray,rnn_HpredsT2_padArray,rnn_HpredsT3_padArray,rnn_ApredsT1_padArray,rnn_ApredsT2_padArray,rnn_ApredsT3_padArray,rnn_HptidT1_padded,rnn_HptidT2_padded,rnn_HptidT3_padded,rnn_HimageIDT1_padded,rnn_HimageIDT2_padded,rnn_HimageIDT3_padded,rnn_AptidT1_padded,rnn_AptidT2_padded,rnn_AptidT3_padded,rnn_AimageIDT1_padded,rnn_AimageIDT2_padded,rnn_AimageIDT3_padded,val_split)

#RUN THE RNN:
    netRNN = RNN_Net(params)
    historyRNN = netRNN.train(([train_predsT1_padArray,train_predsT2_padArray,train_predsT3_padArray],train_labels_padArray,[val_predsT1_padArray,val_predsT2_padArray,val_predsT3_padArray],val_labels_padArray))

#EVALUATE RNN:
    test_lossRNN, test_accRNN  = netRNN.evaluate (([test_predsT1_padArray,test_predsT2_padArray,test_predsT3_padArray],test_labels_padArray))
    test_predsRNN = netRNN.predict(([test_predsT1_padArray,test_predsT2_padArray,test_predsT3_padArray],test_labels_padArray))
    """
    #TO LOAD A PREVIOUS MODEL INSTEAD: (uncomment this chunk and comment above chunk - all the way up to through the data prep for RNN)
    #note: this is not needed for CNN heatmaps
    #can't seem to figure out how to load the whole model (but am saving it anyway). I'm only able to save and load the weights, so note that the model needs to be recompiled, so it has to be the correct architecture
    #Also, I should check that it works by running the same test set first and making sure I get the same results
    netRNN = RNN_Net(params)
    netRNN.load_the_weights("SavedRNNWeights")
    pickle_in = open(model_filepath+'/'+picklename+'.pickle', 'rb')  #change this to be the pickle filename
    pickle0=pickle.load(pickle_in)
    pickle_in.close()
    test_predsT1_padArray = pickle0[5][1]
    test_predsT2_padArray = pickle0[5][2]
    test_predsT3_padArray = pickle0[5][3]
    test_labels_padArray = pickle0[5][4]
    test_labels_padArray = np.delete(test_labels_padArray,0)
    pickle0 = 0
    print('test_labels_padArray: ',test_labels_padArray)
    test_lossRNN, test_accRNN  = netRNN.evaluate(([test_predsT1_padArray,test_predsT2_padArray,test_predsT3_padArray],test_labels_padArray))
    test_predsRNN = netRNN.predict(([test_predsT1_padArray,test_predsT2_padArray,test_predsT3_padArray],test_labels_padArray))
    print('check_lossRNN, check_accRNN: '+ str(test_lossRNN)+', '+ str(test_accRNN))
    """
#PLOTS FOR THE CNN ALONE    
    #plot accuracy learning curves
    plt.figure()
    plt.plot(historyCNN['acc'],color='red')
    plt.plot(historyCNN['val_acc'],color='blue') 
    plt.title('CNN model accuracy learning curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlabel('1 - Specificity',fontsize=20)
    plt.ylabel('Sensitivity',fontsize=20)
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(model_filepath+'/figures/CNN_LCacc'+str(seed)+'.png', bbox_inches='tight')

    #plot loss learning curves
    plt.figure()
    plt.plot(historyCNN['loss'],color='orange')
    plt.plot(historyCNN['val_loss'],color='purple')
    plt.title('CNN model loss learning curve')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(model_filepath+'/figures/CNN_LCloss'+str(seed)+'.png', bbox_inches='tight')
     
    #plot test ROC curve
    fpr_testCNN = dict()
    tpr_testCNN = dict()
    thresholds_testCNN = dict()
    areaundercurveCNN = dict()
    plt.figure()
    test_predsCNN_class = np.argmax(test_predsCNN,axis=-1)
    test_predsCNN_count = np.bincount(test_predsCNN_class, minlength=n_classes)
    print('test_labelsCNN: ', test_data[3])
    print('test_predCNNclass: ', test_predsCNN_class)
    print('test_predCNNcount: ', test_predsCNN_count)
    
    tROC = True
    for i in range(n_classes):
        if test_predsCNN_count[i]==0:   #skips ROC curve for situation where one class is never predicted
            print('Class ', i, 'is predicted 0 times in CNN testing.')
            print('Cannot plot Test ROC curve for CNN.')
            tROC = False
            break
    if tROC == True:
        if n_classes ==2:
            #fpr_testCNN, tpr_testCNN, thresholds_testCNN = roc_curve(np.array(pd.get_dummies(test_data[3]))[:,1], np.array(test_predsCNN)[:,1])
            fpr_testCNN, tpr_testCNN, thresholds_testCNN = roc_curve(test_data[3], test_predsCNN[:,1])            
            areaundercurveCNN = auc(fpr_testCNN,tpr_testCNN)
            lw = 3
            class_name = ['AD','Healthy']
            plt.plot(fpr_testCNN, tpr_testCNN, lw=lw)
            plt.title('CNN ROC')
            plt.xlabel('1 - Specificity',fontsize=13)
            plt.ylabel('Sensitivity',fontsize=13)
        else:
            for i in range(n_classes):
                fpr_testCNN[i], tpr_testCNN[i], thresholds_testCNN[i] = roc_curve(np.array(pd.get_dummies(test_data[3]))[:,i], np.array(test_predsCNN)[:,i]) 
                areaundercurveCNN[i] = auc(fpr_testCNN[i],tpr_testCNN[i])
                lw = 3
                class_name = ['AD','Healthy']
                plt.plot(fpr_testCNN[i], tpr_testCNN[i],
                    lw=lw, label=str(class_name[i]))
                plt.title('CNN ROC')
                plt.xlabel('1 - Specificity',fontsize=13)
                plt.ylabel('Sensitivity',fontsize=13)

    if tROC==True:   #skips ROC curve and TPRs for situation where one class is never predicted
        #plot testROC
        plt.legend(loc="lower right")
        plt.savefig(model_filepath+'/figures/CNN_ROC'+str(seed)+'.png', bbox_inches='tight')
        #print TPRs for each class
        #print('TPR_AD_CNN = '+str(tpr_testCNN[0]))
        #print('TPR_Healthy_CNN = '+str(tpr_testCNN[1]))
        
    #Confusion matrix
    mci_conf_matrix_testCNN = confusion_matrix(y_true = test_data[3], y_pred = np.round(test_predsCNN_class))
    plt.figure()
    ax = plt.subplot()
    cax = ax.matshow(mci_conf_matrix_testCNN)
    plt.title('Full CNN T1 Confusion Matrix')
    plt.colorbar(cax)
    ax.set_xticklabels(['','AD','Healthy'],fontsize=11)
    ax.set_yticklabels(['','AD','Healthy'],fontsize=11)
    plt.xlabel('Predicted',fontsize=13)
    plt.ylabel('True',fontsize=13)
    plt.savefig(model_filepath+'/figures/CNN_ConfMatrix'+str(seed)+'.png', bbox_inches='tight')

    #Normalized confusion matrix
    mci_conf_matrix_test_normedCNN = mci_conf_matrix_testCNN/(mci_conf_matrix_testCNN.sum(axis=1)[:,np.newaxis])
    plt.figure()
    ax = plt.subplot()
    cax = ax.matshow(mci_conf_matrix_test_normedCNN)
    plt.title('Full CNN T1 Normalized Confusion Matrix')
    plt.colorbar(cax)
    ax.set_xticklabels(['','AD','Healthy'],fontsize=11)
    ax.set_yticklabels(['','AD','Healthy'],fontsize=11)
    plt.xlabel('Predicted',fontsize=13)
    plt.ylabel('True',fontsize=13)
    plt.savefig(model_filepath+'/figures/CNN_ConfMatrixNormed'+str(seed)+'.png', bbox_inches='tight')

    #validation ROC
    val_lossCNN, val_accCNN = netCNN.evaluate ((val_data))
    val_predsCNN = netCNN.predict((val_data)) 
    val_predsCNN_class = np.argmax(val_predsCNN,axis=-1)
    fpr_valCNN = dict()
    tpr_valCNN = dict()
    thresholds_valCNN = dict()
    val_predsCNN_count = np.bincount(val_predsCNN_class, minlength=n_classes)
    print('val_predsCNN_count: ', val_predsCNN_count)

    vROC = True    
    for i in range(n_classes):
        if val_predsCNN_count[i]==0:   #skips ROC curve for situation where one class is never predicted
            print('Class ', i, 'is predicted 0 times in CNN validation.')
            print('Cannot plot vROC curve for CNN.')
            vROC = False
            break
    if vROC == True:
        if n_classes ==2:
            fpr_valCNN, tpr_valCNN, thresholds_valCNN = roc_curve(np.array(pd.get_dummies(val_data[3]))[:,1], np.array(val_predsCNN)[:,1])
        else:
            fpr_valCNN[i], tpr_valCNN[i], thresholds_valCNN[i] = roc_curve(np.array(pd.get_dummies(val_data[3]))[:,i], np.array(val_predsCNN)[:,i])

    mci_conf_matrix_valCNN = confusion_matrix(y_true = val_data[3], y_pred = np.round(val_predsCNN_class)) 
    mci_conf_matrix_val_normedCNN = mci_conf_matrix_valCNN/(mci_conf_matrix_valCNN.sum(axis=1)[:,np.newaxis])

    print("Test CNN accuracy: "+str(test_accCNN))
    print("CNN AUC: " +str(areaundercurveCNN))
    
#PLOTS FOR THE RNN    
    #plot accuracy learning curves
    plt.figure()
    plt.plot(historyRNN['acc'],color='red')
    plt.plot(historyRNN['val_acc'],color='blue') 
    plt.title('RNN model accuracy learning curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(model_filepath+'/figures/RNN_LCacc'+str(seed)+'.png', bbox_inches='tight')

    #plot loss learning curves
    plt.figure()
    plt.plot(historyRNN['loss'],color='orange')
    plt.plot(historyRNN['val_loss'],color='purple')
    plt.title('RNN model loss learning curve')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(model_filepath+'/figures/RNN_LCloss'+str(seed)+'.png', bbox_inches='tight')
     
    #plot 2-class test ROC curve
    fpr_testRNN = dict()
    tpr_testRNN = dict()
    thresholds_testRNN = dict()
    areaundercurveRNN = dict()
    plt.figure()
    test_predsRNN_class = np.argmax(test_predsRNN,axis=-1)
    test_predsRNN_count = np.bincount(test_predsRNN_class, minlength=n_classes)
    print('test_labelsRNN: ', test_labels_padArray)
    print('test_predsRNN_class: ', test_predsRNN_class)
    print('test_predsRNN_count: ', test_predsRNN_count)
    
    tROC = True        
    for i in range(n_classes):
        if test_predsRNN_count[i]==0:   #skips ROC curve for situation where one class is never predicted
            print('Class ', i, 'is predicted 0 times in RNN testing.')
            print('Cannot plot Test ROC curve for RNN.')
            tROC = False
            break
    if tROC == True:
        if n_classes ==2:
            fpr_testRNN, tpr_testRNN, thresholds_testRNN = roc_curve(np.array(pd.get_dummies(test_labels_padArray))[:,i], np.array(test_predsRNN)[:,1]) #changed first 1 from i 
            areaundercurveRNN = auc(fpr_testRNN,tpr_testRNN)
            lw = 3
            class_name = ['AD','Healthy']
            plt.plot(fpr_testRNN, tpr_testRNN, lw=lw)
            plt.title('RNN ROC')
            plt.xlabel('1 - Specificity',fontsize=13)
            plt.ylabel('Sensitivity',fontsize=13)
        else:
            for i in range(n_classes):
                fpr_testRNN[i], tpr_testRNN[i], thresholds_testRNN[i] = roc_curve(np.array(pd.get_dummies(test_labels_padArray))[:,i], np.array(test_predsRNN)[:,i]) 
                areaundercurveRNN[i] = auc(fpr_testRNN[i],tpr_testRNN[i])
                lw = 3
                class_name = ['AD','Healthy']
                plt.plot(fpr_testRNN[i], tpr_testRNN[i],
                    lw=lw, label=str(class_name[i]))
                plt.title('RNN ROC')
                plt.xlabel('1 - Specificity',fontsize=13)
                plt.ylabel('Sensitivity',fontsize=13)

    if tROC==True:   #skips ROC curve and TPRs for situation where one class is never predicted
        #plot testROC
        plt.legend(loc="lower right")
        plt.savefig(model_filepath+'/figures/RNN_ROC'+str(seed)+'.png', bbox_inches='tight')
        #print TPRs for each class
        #print('TPR_AD_RNN = '+str(tpr_testRNN[0]))
        #print('TPR_Healthy_RNN = '+str(tpr_testRNN[1]))
        
    #Confusion matrix
    mci_conf_matrix_testRNN = confusion_matrix(y_true = test_labels_padArray, y_pred = np.round(test_predsRNN_class))
    plt.figure()
    ax = plt.subplot()
    cax = ax.matshow(mci_conf_matrix_testRNN)
    plt.title('RNN Confusion Matrix')
    plt.colorbar(cax)
    ax.set_xticklabels(['','AD','Healthy'],fontsize=11)
    ax.set_yticklabels(['','AD','Healthy'],fontsize=11)
    plt.xlabel('Predicted',fontsize=13)
    plt.ylabel('True',fontsize=13)
    plt.savefig(model_filepath+'/figures/RNN_ConfMatrix'+str(seed)+'.png', bbox_inches='tight')

    #Normalized confusion matrix
    mci_conf_matrix_test_normedRNN = mci_conf_matrix_testRNN/(mci_conf_matrix_testRNN.sum(axis=1)[:,np.newaxis])
    plt.figure()
    ax = plt.subplot()
    cax = ax.matshow(mci_conf_matrix_test_normedRNN)
    plt.title('RNN Normalized Confusion Matrix')
    plt.colorbar(cax)
    ax.set_xticklabels(['','AD','Healthy'],fontsize=11)
    ax.set_yticklabels(['','AD','Healthy'],fontsize=11)
    plt.xlabel('Predicted',fontsize=13)
    plt.ylabel('True',fontsize=13)
    plt.savefig(model_filepath+'/figures/RNN_ConfMatrixNormed'+str(seed)+'.png', bbox_inches='tight')
   
    #validation ROC  
    val_lossRNN, val_accRNN = netRNN.evaluate (([val_predsT1_padArray,val_predsT2_padArray,val_predsT3_padArray],val_labels_padArray))
    val_predsRNN = netRNN.predict(([val_predsT1_padArray,val_predsT2_padArray,val_predsT3_padArray],val_labels_padArray))
    val_predsRNN_class = np.argmax(val_predsRNN,axis=-1)
    fpr_valRNN = dict()
    tpr_valRNN = dict()
    thresholds_valRNN = dict()
    val_predsRNN_count = np.bincount(val_predsRNN_class, minlength=n_classes)
    print('val_predsRNN_count: ', val_predsRNN_count)

    vROC = True    
    for i in range(n_classes):
        if val_predsRNN_count[i]==0:   #skips ROC curve for situation where one class is never predicted
            print('Class ', i, 'is predicted 0 times in RNN validation.')
            print('Cannot plot vROC curve for RNN.')
            vROC = False
            break
    if vROC==True:
        if n_classes == 2:
            fpr_valRNN, tpr_valRNN, thresholds_valRNN = roc_curve(np.array(pd.get_dummies(val_labels_padArray))[:,1], np.array(val_predsRNN)[:,1])
        else:
            for i in range(n_classes):
                fpr_valRNN[i], tpr_valRNN[i], thresholds_valRNN[i] = roc_curve(np.array(pd.get_dummies(val_labels_padArray))[:,i], np.array(val_predsRNN)[:,i])

    mci_conf_matrix_valRNN = confusion_matrix(y_true = val_labels_padArray, y_pred = np.round(val_predsRNN_class)) 
    mci_conf_matrix_val_normedRNN = mci_conf_matrix_valRNN/(mci_conf_matrix_valRNN.sum(axis=1)[:,np.newaxis])

    print("Test RNN accuracy: "+str(test_accRNN))
    print("RNN AUC: " +str(areaundercurveRNN))
    
#TEST SET TABLES
    test_table_CNN = (test_data[4],test_data[5],test_data[3],test_data[6],test_data[7],test_predsCNN_class,test_predsCNN[0],test_predsCNN[1])
    test_table_RNN = (test_ptidT1,test_imageIDT1,test_imageIDT2,test_imageIDT3,test_labels_padArray,test_predsRNN_class,test_predsRNN[0],test_predsRNN[1])
    
#WRITE THE OUTPUT FILE    
    with open(model_filepath+'/figures/Outputs'+str(seed)+'.txt','w') as outputs:
        #RNN
        outputs.write('RNN Confusion Matrix Values:'+'\n')
        outputs.write(str(mci_conf_matrix_testRNN)+'\n')
        outputs.write('RNN Normalized Confusion Matrix Values:'+'\n')
        outputs.write(str(mci_conf_matrix_test_normedRNN)+'\n')
        outputs.write('RNN Test accuracy:'+'\n')
        outputs.write(str(test_accRNN)+'\n')
        outputs.write('RNN AUC:'+'\n')
        outputs.write(str(areaundercurveRNN) +'\n')
        outputs.write('RNN Test Predictions Probabilities'+'\n')
        outputs.write(str(test_predsRNN) +'\n')
        outputs.write('RNN Test Predictions MaxProb Class'+'\n')
        outputs.write(str(test_predsRNN_class) +'\n')
        #CNN
        outputs.write('Full CNN Confusion Matrix Values:'+'\n')
        outputs.write(str(mci_conf_matrix_testCNN)+'\n')
        outputs.write('Full CNN Normalized Confusion Matrix Values:'+'\n')
        outputs.write(str(mci_conf_matrix_test_normedCNN)+'\n')
        outputs.write('Full CNN Test accuracy:'+'\n')
        outputs.write(str(test_accCNN)+'\n')
        outputs.write('Full CNN AUC:'+'\n')
        outputs.write(str(areaundercurveCNN) +'\n')
        outputs.write('Full CNN Test Predictions Probabilities'+'\n')
        outputs.write(str(test_predsCNN) +'\n')
        outputs.write('Full CNN Test Predictions MaxProb Class'+'\n')
        outputs.write(str(test_predsCNN_class) +'\n')
        #outputs.write('Index of best CNN Gmean'+'\n')
        #outputs.write(str(ixC) +'\n')
        #outputs.write('Optimal CNN Threshold'+'\n')
        #outputs.write(str(bestThreshCNN) +'\n')
        #outputs.write('Value of highest Gmean'+'\n')
        #outputs.write(str(highGmeanCNN) +'\n')
        #outputs.write('CNN Accuracy at Optimized Threshold'+'\n')
        #outputs.write(str(OptAccCNN) +'\n'+'\n')    
        #Testset output tables
        outputs.write('test_table_CNN'+'\n')
        outputs.write(str(test_table_CNN)+'\n'+'\n')
        outputs.write('test_table_RNN'+'\n')
        outputs.write(str(test_table_RNN)+'\n'+'\n')

#TEST SET TABLES
    Cptid = test_data[4]
    Cptid = np.insert(Cptid,0,'PTID')
    CimageID = test_data[5]
    CimageID = np.insert(CimageID,0,'imgID')
    Cconfid = test_data[6]
    Cconfid = np.insert(Cconfid,0,'DxConfidence')
    Ccsf = test_data[7]
    Ccsf = np.insert(Ccsf,0,'CSF_Path')
    Clabels = test_data[3]
    Clabels = np.insert(Clabels.astype(str),0,'label')
    test_predsCNN_class = np.insert(test_predsCNN_class.astype(str),0,'prediction')
    probsCAD = [item[0] for item in test_predsCNN]
    probsCNC = [item[1] for item in test_predsCNN]
    probsCAD.insert(0,'prediction probabilities AD')
    probsCNC.insert(0,'prediction probabilities NC')
    test_ptidT1 = np.insert(test_ptidT1,0,'PTID')
    test_imageIDT1 = np.insert(test_imageIDT1,0,'imIDT1')
    test_imageIDT2 = np.insert(test_imageIDT2,0,'imIDT2')
    test_imageIDT3 = np.insert(test_imageIDT3,0,'imIDT3')
    test_labels_padArray = np.insert(test_labels_padArray.astype(str),0,'label')
    test_predsRNN_class = np.insert(test_predsRNN_class.astype(str),0,'prediction')
    probsRAD = [item[0] for item in test_predsRNN]
    probsRNC = [item[1] for item in test_predsRNN]
    probsRAD.insert(0,'prediction probabilities AD')
    probsRNC.insert(0,'prediction probabilities NC')

    with open(model_filepath+'/figures/test_table_'+str(seed)+'.csv','w') as Testcsv:
        Testcsv_writer = csv.writer(Testcsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        Testcsv_writer.writerow(['CNN'])
        Testcsv_writer.writerow(Cptid)
        Testcsv_writer.writerow(CimageID)
        Testcsv_writer.writerow(Clabels)
        Testcsv_writer.writerow(test_predsCNN_class)
        Testcsv_writer.writerow(Cconfid)
        Testcsv_writer.writerow(Ccsf)
        Testcsv_writer.writerow(probsCAD)
        Testcsv_writer.writerow(probsCNC)
        Testcsv_writer.writerow(' ')
        Testcsv_writer.writerow(' ')
        Testcsv_writer.writerow(['RNN'])
        Testcsv_writer.writerow(test_ptidT1)
        Testcsv_writer.writerow(test_imageIDT1)
        Testcsv_writer.writerow(test_imageIDT2)
        Testcsv_writer.writerow(test_imageIDT3)
        Testcsv_writer.writerow(test_labels_padArray)
        Testcsv_writer.writerow(test_predsRNN_class)
        Testcsv_writer.writerow(probsRAD)
        Testcsv_writer.writerow(probsRNC)
    """
    #HEATMAPS:
    HeatmapPlotter = heatmapPlotter(seed)
    
    #if plotting NC data (need to load AD avg):
    picklename = 'loadedGC4912'   #CHANGE to LRP or GGC and change seed
    pickle_in = open(model_filepath+'/'+picklename+'.pickle', 'rb') 
    pickle0=pickle.load(pickle_in)
    pickle_in.close()
    mean_map_AD = pickle0[0]["AD"]
    pickle0=0
    
    #if plotting AD data:
    mean_map_AD = np.zeros((91,109,91))
    
    #RUN LRP
    case_maps_LRP, counts = HeatmapPlotter.LRP(test_data, test_mri_nonorm, model_filepath, netCNN, test_predsCNN)  #Removed CNN_LRP to save memory
    mean_maps_LRP = HeatmapPlotter.plot_avg_maps(case_maps_LRP, counts, 'LRP', test_mri_nonorm, model_filepath, mean_map_AD)
    #WRITE A PICKLE FILE
    with open(model_filepath+'/figures/loadedLRP' + str(seed)+'.pickle', 'wb') as f:
        pickle.dump([mean_maps_LRP, case_maps_LRP, counts], f)  #Removed CNN_LRP to save memory
    
    #RUN GGC
    case_maps_GGC, counts = HeatmapPlotter.GuidedGradCAM(test_data, test_mri_nonorm, model_filepath, netCNN, test_predsCNN) #Removed CNN_gradcam, CNN_gb, CNN_guided_gradcam to save memory
    mean_maps_GGC = HeatmapPlotter.plot_avg_maps(case_maps_GGC, counts, 'Guided GradCAM', test_mri_nonorm, model_filepath, mean_map_AD)
    #WRITE A PICKLE FILE
    with open(model_filepath+'/figures/loadedGC' + str(seed)+'.pickle', 'wb') as f:
        pickle.dump([mean_maps_GGC, case_maps_GGC, counts], f)  #Removed CNN_gradcam, CNN_gb, CNN_guided_gradcam to save memory
    """
#WRITE THE PICKLE FILE
    with open(model_filepath+'/figures/' + str(seed)+'.pickle', 'wb') as f:
        pickle.dump([[fpr_testRNN, tpr_testRNN, thresholds_testRNN, test_lossRNN, test_accRNN, mci_conf_matrix_testRNN, mci_conf_matrix_test_normedRNN, test_predsRNN, test_predsRNN_class, test_labels_padArray, val_predsRNN, val_labels_padArray ], 
                      [fpr_valRNN, tpr_valRNN, thresholds_valRNN, val_lossRNN, val_accRNN, mci_conf_matrix_valRNN, mci_conf_matrix_val_normedRNN],
                      [fpr_testCNN, tpr_testCNN, thresholds_testCNN, test_lossCNN, test_accCNN, mci_conf_matrix_testCNN, mci_conf_matrix_test_normedCNN, test_predsCNN, test_predsCNN_class, test_labels_padArray, val_predsCNN, val_labels_padArray ], 
                      [fpr_valCNN, tpr_valCNN, thresholds_valCNN, val_lossCNN, val_accCNN, mci_conf_matrix_valCNN, mci_conf_matrix_val_normedCNN],
                      [test_table_CNN,test_table_RNN], 
                      [test_data, test_predsT1_padArray,test_predsT2_padArray,test_predsT3_padArray,test_labels_padArray]],f)
                      #[CNN_LRP, mean_maps_LRP],
                      #[CNN_gradcam, CNN_gb, CNN_guided_gradcam, mean_maps_GGC]], f)  

    
#For Recovery of old runs
#    pickle_in = open('/data_wnx1/_Data/AlzheimersDL/MCI-spasov-3class/pickles/' + str(seed)+'.pickle', 'rb')
#    seedpickle=pickle.load(pickle_in)
#    print(seedpickle)
 
#Plots to potentially add in the future 
    #add sMCI+pMCI ROC curve
#    mci_fpr_test = fpr_test[1]+fpr_test[2]
#    mci_tpr_test = tpr_test[1]+tpr_test[2]
#    areaundercurve['all_mci'] = auc(mci_fpr_test,mci_tpr_test)
#    plt.plot(mci_fpr_test, mci_tpr_test,
#        lw=lw, label='All MCI')
    
    #Calculate macros
#    all_fpr_test = np.unique(np.concatenate([fpr_test[i] for i in range(n_classes)]))
#    mean_tpr_test = np.zeros_like(all_fpr_test)
#    for i in range(n_classes):
#        mean_tpr_test += interp(all_fpr_test, fpr_test[i], tpr_test[i])
#    mean_tpr_test /= n_classes
#    fpr_test["macro"] = all_fpr_test
#    tpr_test["macro"] = mean_tpr_test
#    areaundercurve["macro"] = auc(fpr_test["macro"], tpr_test["macro"])
    
#    plt.plot(fpr_test["macro"], tpr_test["macro"],label='ROC curve of macro-average')
#    plt.legend(loc="lower right")
#    plt.savefig(model_filepath+'/figures/ROC'+str(seed)+'.png', bbox_inches='tight')
    
    #Calculate micros
#    test_data_bin = label_binarize(test_data[-1], classes=[0, 1, 2])
#    test_preds_class_bin = label_binarize(test_preds_class, classes=[0, 1, 2])
#    fpr_test["micro"], tpr_test["micro"], _ = roc_curve(test_data_bin.ravel(), test_preds_class_bin.ravel())
#    areaundercurve["micro"] = auc(fpr_test["micro"], tpr_test["micro"])
    
#    plt.plot(fpr_test["macro"], tpr_test["macro"],label='ROC curve of micro-average')
#    plt.legend(loc="lower right")
#    plt.savefig(model_filepath+'/figures/ROC'+str(seed)+'.png', bbox_inches='tight')
    
#RUN IT!
for seed in seeds:
    #Load data
    print('Processing seed number ', seed)
#    data_loader = DataLoader((target_rows, target_cols, depth, axis), seed = seed)
#    train_data, val_data, test_data, healthy_dict = data_loader.get_train_val_test()
#    print('length of train data '+str((train_data)),'; length of val data '+str((val_data)),'; test data '+str((test_data)))
#    print('length of healthy_dict[mri] '+str(len(healthy_dict)))
    evaluate_net(seed)

    

    




