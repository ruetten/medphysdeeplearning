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


params_dict = { 'CNN_w_regularizer': CNN_w_regularizer,
               'CNN_batch_size': CNN_batch_size,
               'CNN_drop_rate': CNN_drop_rate, 'epochs': 200,
          'gpu': "/gpu:0", 'model_filepath': model_filepath, 
          'image_shape': (target_rows, target_cols, depth, axis),
          'num_clinical': num_clinical,
          'final_layer_size': final_layer_size,
          'optimizer': optimizer,}

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
    
#TEST SET TABLES
    test_table_CNN = (test_data[4],test_data[5],test_data[3],test_data[6],test_data[7],test_predsCNN_class,test_predsCNN[0],test_predsCNN[1])
    test_table_RNN = (test_ptidT1,test_imageIDT1,test_imageIDT2,test_imageIDT3,test_labels_padArray,test_predsRNN_class,test_predsRNN[0],test_predsRNN[1])
    
#WRITE THE OUTPUT FILE    
    with open(model_filepath+'/figures/Outputs'+str(seed)+'.txt','w') as outputs:
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
        pickle.dump([[fpr_testCNN, tpr_testCNN, thresholds_testCNN, test_lossCNN, test_accCNN, mci_conf_matrix_testCNN, mci_conf_matrix_test_normedCNN, test_predsCNN, test_predsCNN_class, test_labels_padArray, val_predsCNN, val_labels_padArray ], 
                      [fpr_valCNN, tpr_valCNN, thresholds_valCNN, val_lossCNN, val_accCNN, mci_conf_matrix_valCNN, mci_conf_matrix_val_normedCNN],
                      [test_table_CNN], 
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

    

    




