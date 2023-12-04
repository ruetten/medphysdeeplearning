

import numpy as np
from numpy.random import RandomState
from os import listdir
import nibabel as nib
import math
import csv
import random
from keras.utils import to_categorical
from utils.patientsort import PatientSorter

##for 2 class model CNN + RNN ##

class DataLoader:
    """The DataLoader class is intended to be used on images placed in folder ../ADNI_volumes_customtemplate_float32
        
        naming convention is: class_subjectID_imageType.nii.gz
        masked_brain denotes structural MRI, JD_masked_brain denotes Jacobian Determinant 
        
        stableNL: healthy controls
        stableMCItoAD: progressive MCI
        stableAD: Alzheimer's subjects
    
    Additionally, we use clinical features from csv file ../LP_ADNIMERGE.csv
    """
    
    
    def __init__(self, target_shape, seed = None):
        self.mri_datapath = '//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class-1cnn-CLEAN/ADNI_volumes_customtemplate_float32'
        self.xls_datapath = '//data/data_wnx3/data_wnx1/_Data/AlzheimersDL/CNN+RNN-2class-1cnn-CLEAN'
        self.target_shape = target_shape
        self.seed = seed
  

    def shuffle_dict_lists (self, dictionary): 
        p = RandomState(self.seed).permutation(len(list(dictionary.values())[0])) 
        for key in list(dictionary.keys()):
           dictionary[key] = [dictionary[key][i] for i in p]  

    
    def get_filenames (self,mri_datapath):
        '''Puts filenames in ../ADNI_volumes_customtemplate_float32 in
        dictionaries according to class (stableMCI, MCItoAD, stableNL and stableAD)
        with keys corresponding to image modality (mri and JD)
        '''
        file_names = sorted(listdir(mri_datapath))
        keys = ['mri','PTID','viscode','imageID']  #is it an issue that I added viscodes here?
        healthy_dict, ad_dict = [{key: [] for key in keys} for i in range(2)]  #!!
        healthyBL_dict,healthyM6_dict,healthyM12_dict,healthyM24_dict,healthyM36_dict,healthyM48_dict,healthyM60_dict,healthyM72_dict,healthyM84_dict,healthyM96_dict = [{key: [] for key in keys} for i in range(10)]
        adBL_dict,adM6_dict,adM12_dict,adM24_dict,adM36_dict,adM48_dict,adM60_dict,adM72_dict,adM84_dict,adM96_dict = [{key: [] for key in keys} for i in range(10)] 
        healthyT1_dict,healthyT2_dict,healthyT3_dict,adT1_dict,adT2_dict,adT3_dict = [{key: [] for key in keys} for i in range(6)] 
        healthyT1_Rdict,healthyT2_Rdict,healthyT3_Rdict,adT1_Rdict,adT2_Rdict,adT3_Rdict = [{key: [] for key in keys} for i in range(6)]

        #Get xls info
        with open(self.xls_datapath + '/' + 'LP_ADNIMERGE.csv', 'r') as f:
            reader = csv.reader(f)
            xls = [row for row in reader]  #Extract all data from csv file in a list.
        test_xls=[]
        for _file in file_names:
            for row in xls:
                #imageID = 'I'+row[3]  #prevents shorter imageIDs from matching to longer IDs which contain them
                imageID = row[3]  #use this for loading validation set
                if imageID in _file:
                    test_xls.append(row)
                    break

        #sort the filenames into dicts
        for _file in file_names:
          if _file[-3:] == 'nii':
            if 'stableNL' in _file:
                for row in test_xls:
                    #imageID = 'I'+row[3]  #prevents shorter imageIDs from matching to longer IDs which contain them
                    imageID = row[3]  #use this for loading validation set
                    if imageID in _file:
                        if row[5] == 'bl':
                            healthyBL_dict['mri'].append(_file)
                            healthyBL_dict['PTID'].append(row[2])
                            healthyBL_dict['viscode'].append(row[2])
                            healthyBL_dict['imageID'].append(row[3])
                            break
                        if row[5] == 'm06':
                            healthyM6_dict['mri'].append(_file)
                            healthyM6_dict['PTID'].append(row[2])
                            healthyM6_dict['viscode'].append(row[2])
                            healthyM6_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm12':
                            healthyM12_dict['mri'].append(_file)
                            healthyM12_dict['PTID'].append(row[2])
                            healthyM12_dict['viscode'].append(row[2])
                            healthyM12_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm24':
                            healthyM24_dict['mri'].append(_file)
                            healthyM24_dict['PTID'].append(row[2])
                            healthyM24_dict['viscode'].append(row[2])
                            healthyM24_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm36':
                            healthyM36_dict['mri'].append(_file)
                            healthyM36_dict['PTID'].append(row[2])
                            healthyM36_dict['viscode'].append(row[2])
                            healthyM36_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm48':
                            healthyM48_dict['mri'].append(_file)
                            healthyM48_dict['PTID'].append(row[2])
                            healthyM48_dict['viscode'].append(row[2])
                            healthyM48_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm60':
                            healthyM60_dict['mri'].append(_file)
                            healthyM60_dict['PTID'].append(row[2])
                            healthyM60_dict['viscode'].append(row[2])
                            healthyM60_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm72':
                            healthyM72_dict['mri'].append(_file)
                            healthyM72_dict['PTID'].append(row[2])
                            healthyM72_dict['viscode'].append(row[2])
                            healthyM72_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm84':
                            healthyM84_dict['mri'].append(_file)
                            healthyM84_dict['PTID'].append(row[2])
                            healthyM84_dict['viscode'].append(row[2])
                            healthyM84_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm96':
                            healthyM96_dict['mri'].append(_file)
                            healthyM96_dict['PTID'].append(row[2])
                            healthyM96_dict['viscode'].append(row[2])
                            healthyM96_dict['imageID'].append(row[3])

            elif 'stableAD' in _file:
                for row in test_xls:
                    #imageID = 'I'+row[3]  #prevents shorter imageIDs from matching to longer IDs which contain them
                    imageID = row[3]  #use this for loading validation set
                    if imageID in _file:
                        if row[5] == 'bl':
                            adBL_dict['mri'].append(_file)
                            adBL_dict['PTID'].append(row[2])
                            adBL_dict['viscode'].append(row[2])
                            adBL_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm06':
                            adM6_dict['mri'].append(_file)
                            adM6_dict['PTID'].append(row[2])
                            adM6_dict['viscode'].append(row[2])
                            adM6_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm12':
                            adM12_dict['mri'].append(_file)
                            adM12_dict['PTID'].append(row[2])
                            adM12_dict['viscode'].append(row[2])
                            adM12_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm24':
                            adM24_dict['mri'].append(_file)
                            adM24_dict['PTID'].append(row[2])
                            adM24_dict['viscode'].append(row[2])
                            adM24_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm36':
                            adM36_dict['mri'].append(_file)
                            adM36_dict['PTID'].append(row[2])
                            adM36_dict['viscode'].append(row[2])
                            adM36_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm48':
                            adM48_dict['mri'].append(_file)
                            adM48_dict['PTID'].append(row[2])
                            adM48_dict['viscode'].append(row[2])
                            adM48_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm60':
                            adM60_dict['mri'].append(_file)
                            adM60_dict['PTID'].append(row[2])
                            adM60_dict['viscode'].append(row[2])
                            adM60_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm72':
                            adM72_dict['mri'].append(_file)
                            adM72_dict['PTID'].append(row[2])
                            adM72_dict['viscode'].append(row[2])
                            adM72_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm84':
                            adM84_dict['mri'].append(_file)
                            adM84_dict['PTID'].append(row[2])
                            adM84_dict['viscode'].append(row[2])
                            adM84_dict['imageID'].append(row[3])
                            break
                        elif row[5] == 'm96':
                            adM96_dict['mri'].append(_file)
                            adM96_dict['PTID'].append(row[2])
                            adM96_dict['viscode'].append(row[2])
                            adM96_dict['imageID'].append(row[3])

        #Choose which tps to call T1,T2,T3 ONLY NEEDED IF NOT USING ALL DATA IN CNN (then define healthy_dict_CNN as whichever of these scans you want
        #healthyT1_dict['mri'] = healthyM24_dict['mri']
        #healthyT2_dict['mri'] = healthyM36_dict['mri']
        #healthyT3_dict['mri'] = healthyM48_dict['mri']
        #adT1_dict['mri'] = adBL_dict['mri']
        #adT2_dict['mri'] = adM12_dict['mri']
        #adT3_dict['mri'] = adM24_dict['mri']
        #healthyT1_dict['PTID'] = healthyM24_dict['PTID']
        #healthyT2_dict['PTID'] = healthyM36_dict['PTID']
        #healthyT3_dict['PTID'] = healthyM48_dict['PTID']
        #adT1_dict['PTID'] = adBL_dict['PTID']
        #adT2_dict['PTID'] = adM12_dict['PTID']
        #adT3_dict['PTID'] = adM24_dict['PTID']

        #Use the above dicts for the CNN, now create the dicts for the RNN
        #sort into all healthy and all AD dicts, with PTIDs 
        for _file in file_names:
          if _file[-3:] == 'nii':
            if 'stableNL' in _file:
                for row in test_xls:
                    #imageID = 'I'+row[3]  #prevents shorter imageIDs from matching to longer IDs which contain them
                    imageID = row[3]  #use this for loading validation set
                    if imageID in _file:
                        if row[5] != 'm06':   #throw out all m06 scans
                            healthy_dict['mri'].append(_file)
                            healthy_dict['PTID'].append(row[2])
                            healthy_dict['viscode'].append(row[5])
                            healthy_dict['imageID'].append(row[3])
                            break
            if 'stableAD' in _file:  
                for row in test_xls:
                    #imageID = 'I'+row[3]  #prevents shorter imageIDs from matching to longer IDs which contain them
                    imageID = row[3]  #use this for loading validation set
                    if imageID in _file:
                        if row[5] != 'm06':   #throw out all m06 scans
                            ad_dict['mri'].append(_file)
                            ad_dict['PTID'].append(row[2])
                            ad_dict['viscode'].append(row[5])
                            ad_dict['imageID'].append(row[3])
                            break
        
        #sort RNN data into TP dicts
        patientSorter = PatientSorter(self.seed)
        healthyT1_Rdict,healthyT2_Rdict,healthyT3_Rdict,healthyT4_Rdict,healthyT5_Rdict,healthyT6_Rdict,healthyT7_Rdict,healthyT8_Rdict = patientSorter.sort_patients(healthy_dict,'healthy',self.xls_datapath,first=True)
        adT1_Rdict,adT2_Rdict,adT3_Rdict,adT4_Rdict,adT5_Rdict,adT6_Rdict,adT7_Rdict,adT8_Rdict = patientSorter.sort_patients(ad_dict,'ad',self.xls_datapath)                
        
        with open(self.xls_datapath+'/figures/InitialDicts.txt','w') as InitialDicts:
            InitialDicts.write('healthyBL: '+str(len(healthyBL_dict['mri']))+'\n')
            InitialDicts.write('healthyM6: '+str(len(healthyM6_dict['mri']))+'\n')
            InitialDicts.write('healthyM12: '+str(len(healthyM12_dict['mri']))+'\n')
            InitialDicts.write('healthyM24: '+str(len(healthyM24_dict['mri']))+'\n')
            InitialDicts.write('healthyM36: '+str(len(healthyM36_dict['mri']))+'\n')
            InitialDicts.write('healthyM48: '+str(len(healthyM48_dict['mri']))+'\n')
            InitialDicts.write('healthyM60: '+str(len(healthyM60_dict['mri']))+'\n')
            InitialDicts.write('healthyM72: '+str(len(healthyM72_dict['mri']))+'\n')
            InitialDicts.write('healthyM84: '+str(len(healthyM84_dict['mri']))+'\n')
            InitialDicts.write('healthyM96: '+str(len(healthyM96_dict['mri']))+'\n')
            InitialDicts.write('adBL: '+str(len(adBL_dict['mri']))+'\n')
            InitialDicts.write('adM6: '+str(len(adM6_dict['mri']))+'\n')
            InitialDicts.write('adM12: '+str(len(adM12_dict['mri']))+'\n')
            InitialDicts.write('adM24: '+str(len(adM24_dict['mri']))+'\n')
            InitialDicts.write('adM36: '+str(len(adM36_dict['mri']))+'\n')
            InitialDicts.write('adM48: '+str(len(adM48_dict['mri']))+'\n')
            InitialDicts.write('adM60: '+str(len(adM60_dict['mri']))+'\n')
            InitialDicts.write('adM72: '+str(len(adM72_dict['mri']))+'\n')
            InitialDicts.write('adM84: '+str(len(adM84_dict['mri']))+'\n')
            InitialDicts.write('adM96: '+str(len(adM96_dict['mri']))+'\n')
            InitialDicts.write('healthyBL: '+'\n')
            InitialDicts.write(str(healthyBL_dict['mri'])+'\n')
            InitialDicts.write('healthyM6: '+'\n')
            InitialDicts.write(str(healthyM6_dict['mri'])+'\n')
            InitialDicts.write('healthyM12: '+'\n')
            InitialDicts.write(str(healthyM12_dict['mri'])+'\n')
            InitialDicts.write('healthyM24: '+'\n')
            InitialDicts.write(str(healthyM24_dict['mri'])+'\n')
            InitialDicts.write('healthyM36: '+'\n')
            InitialDicts.write(str(healthyM36_dict['mri'])+'\n')
            InitialDicts.write('healthyM48: '+'\n')
            InitialDicts.write(str(healthyM48_dict['mri'])+'\n')
            InitialDicts.write('adBL: '+'\n')
            InitialDicts.write(str(adBL_dict['mri'])+'\n')
            InitialDicts.write('adM6: '+'\n')
            InitialDicts.write(str(adM6_dict['mri'])+'\n')
            InitialDicts.write('adM12: '+'\n')
            InitialDicts.write(str(adM12_dict['mri'])+'\n')
            InitialDicts.write('adM24: '+'\n')
            InitialDicts.write(str(adM24_dict['mri'])+'\n')
            InitialDicts.write('adM36: '+'\n')
            InitialDicts.write(str(adM36_dict['mri'])+'\n')
            InitialDicts.write('adM48: '+'\n')
            InitialDicts.write(str(adM48_dict['mri'])+'\n')
        
        self.shuffle_dict_lists (healthyBL_dict)
        self.shuffle_dict_lists (healthyM6_dict)
        self.shuffle_dict_lists (healthyM12_dict)        #Randomly shuffle lists healthy_dict ['JD'] and healthy_dict['mri'] in unison       
        self.shuffle_dict_lists (healthyM24_dict)
        self.shuffle_dict_lists (healthyM36_dict)
        self.shuffle_dict_lists (healthyM48_dict)
        self.shuffle_dict_lists (adBL_dict)
        self.shuffle_dict_lists (adM6_dict)
        self.shuffle_dict_lists (adM12_dict)              
        self.shuffle_dict_lists (adM24_dict)
        self.shuffle_dict_lists (adM36_dict)
        self.shuffle_dict_lists (adM48_dict)
        self.shuffle_dict_lists (healthyT1_Rdict)    #This shuffling is actually getting the patients out of order!
        self.shuffle_dict_lists (healthyT2_Rdict)    #But doesn't matter because I use the PTIDs to sort them again later.
        self.shuffle_dict_lists (healthyT3_Rdict)
        self.shuffle_dict_lists (healthyT4_Rdict)
        self.shuffle_dict_lists (healthyT5_Rdict)
        self.shuffle_dict_lists (healthyT6_Rdict)
        self.shuffle_dict_lists (healthyT7_Rdict)
        self.shuffle_dict_lists (healthyT8_Rdict)
        self.shuffle_dict_lists (adT1_Rdict)
        self.shuffle_dict_lists (adT2_Rdict)
        self.shuffle_dict_lists (adT3_Rdict)
        self.shuffle_dict_lists (adT4_Rdict)
        self.shuffle_dict_lists (adT5_Rdict)
        self.shuffle_dict_lists (adT6_Rdict)
        self.shuffle_dict_lists (adT7_Rdict)
        self.shuffle_dict_lists (adT8_Rdict)
        self.shuffle_dict_lists (ad_dict)
        self.shuffle_dict_lists (healthy_dict)

        #return healthyT1_dict,healthyT2_dict,healthyT3_dict,adT1_dict,adT2_dict,adT3_dict,healthyT1_Rdict,healthyT2_Rdict,healthyT3_Rdict,adT1_Rdict,adT2_Rdict,adT3_Rdict  #,healthyExtra_dict,adExtra_dict #, smci_dict, pmci_dict
        #return healthyBL_dict,healthyM6_dict,healthyM12_dict,healthyM24_dict,healthyM36_dict,healthyM48_dict,adBL_dict,adM6_dict,adM12_dict,adM24_dict,adM36_dict,adM48_dict,healthyT1_Rdict,healthyT2_Rdict,healthyT3_Rdict,adT1_Rdict,adT2_Rdict,adT3_Rdict  
        return healthy_dict, ad_dict, healthyT1_Rdict,healthyT2_Rdict,healthyT3_Rdict,healthyT4_Rdict,healthyT5_Rdict,healthyT6_Rdict,healthyT7_Rdict,healthyT8_Rdict,adT1_Rdict,adT2_Rdict,adT3_Rdict,adT4_Rdict,adT5_Rdict,adT6_Rdict,adT7_Rdict,adT8_Rdict
                    
    def split_filenames (self, healthy_dict, ad_dict, val_split =  0.20):

        '''Split filename dictionaries in training/validation and test sets.        
        '''
        keys = ['mri']
        train_dict, val_dict, test_dict = [{key: [] for key in keys} for _ in range(3)]

#        num_test_samples =  int(((len(healthy_dict['mri']) + len(ad_dict['mri']) \
#                            +len(pmci_dict['mri']) + len(smci_dict['mri']))*val_split)/4)
                            
#        num_val_samples =  int(int(math.ceil ((val_split*(len(ad_dict['mri']) + len(healthy_dict['mri']) \
#                            +len(pmci_dict['mri']) + len(smci_dict['mri'])- num_test_samples*4)))/4))

        num_test_ad = int(len(ad_dict['mri'])*val_split)
        num_test_healthy = int(len(healthy_dict['mri'])*val_split)

        num_val_ad = int((len(ad_dict['mri'])-num_test_ad)*val_split)
        num_val_healthy = int((len(healthy_dict['mri'])-num_test_healthy)*val_split)

        with open(self.xls_datapath+'/figures/DataList.txt','w') as dataList:
            dataList.write('FOR CNN'+'\n')
            dataList.write('Dict Sizes:'+'\n')
            dataList.write('#AD_dict '+str(len(ad_dict['mri']))+'#NC_dict '+str(len(healthy_dict['mri']))+'\n'+'\n')
            #dataList.write('#ADT1_dict '+str(len(adT1_dict['mri']))+'#ADT2_dict '+str(len(adT2_dict['mri']))+'#ADT3_dict '+str(len(adT3_dict['mri']))+
            #'#NCT1_dict '+str(len(healthyT1_dict['mri']))+'#NCT2_dict '+str(len(healthyT2_dict['mri']))+'#NCT3_dict '+str(len(healthyT3_dict['mri']))+'\n'+'\n')
            #dataList.write('Test Dict ADT2:'+'\n')
            #dataList.write(str(adT2_dict['mri'])+'\n')            
            dataList.write('Test Data Split by class:'+'\n')
            dataList.write('#ADtestsamples '+str(num_test_ad)+'#NCtestsamples '+str(num_test_healthy)+'\n'+'\n')
            #dataList.write('#ADtestsamplesT1 '+str(num_test_adT1)+'#ADtestsamplesT2 '+str(num_test_adT2)+'#ADtestsamplesT3 '+str(num_test_adT3)+
            #'#NCtestsamplesT1 '+str(num_test_healthyT1)+'#NCtestsamplesT2 '+str(num_test_healthyT2)+'#NCtestsamplesT3 '+str(num_test_healthyT3)+'\n'+'\n')
            dataList.write('Val Data Split by class:'+'\n')
            dataList.write('#ADvalsamples '+str(num_val_ad)+'#NCvalsamples '+str(num_val_healthy)+'\n'+'\n')
            #dataList.write('#ADvalsamplesT1 '+str(num_val_adT1)+'#ADvalsamplesT2 '+str(num_val_adT2)+'#ADvalsamplesT3 '+str(num_val_adT3)+
            #'#NCvalsamplesT1 '+str(num_val_healthyT1)+'#NCvalsamplesT2 '+str(num_val_healthyT2)+'#NCvalsamplesT3 '+str(num_val_healthyT3)+'\n'+'\n')
        
        test_ad = ad_dict['mri'][:num_test_ad]
        test_healthy = healthy_dict['mri'][:num_test_healthy]

        test_dict['mri'] = test_ad + test_healthy
        test_dict['health_state'] = np.concatenate((np.zeros(len(test_ad)),np.ones(len(test_healthy))))
        
        val_ad = ad_dict['mri'][num_test_ad : num_test_ad + num_val_ad]
        val_healthy = healthy_dict['mri'][num_test_healthy : num_test_healthy + num_val_healthy]

        val_dict['mri'] = val_ad + val_healthy 
        val_dict['health_state'] = np.concatenate((np.zeros(len(val_ad)),np.ones(len(val_healthy))))          

        train_ad = ad_dict['mri'][num_test_ad + num_val_ad:]
        train_healthy = healthy_dict['mri'][num_test_healthy + num_val_healthy:]

        train_dict['mri'] = train_ad + train_healthy 
        train_dict['health_state'] = np.concatenate((np.zeros(len(train_ad)),np.ones(len(train_healthy))))
        
        with open(self.xls_datapath+'/figures/DataList.txt','a') as dataList:
            dataList.write('Train Data Split by class:'+'\n')
            dataList.write('#ADtrainsamples '+str(len(train_ad))+'#NCtrainsamples '+str(len(train_healthy))+'\n')
            #dataList.write('#ADtrainsamplesT1 '+str(len(train_adT1))+'#ADtrainsamplesT2 '+str(len(train_adT2))+'#ADtrainsamplesT3 '+str(len(train_adT3))+
            #'#NCtrainsamplesT1 '+str(len(train_healthyT1))+'#NCtrainsamplesT2 '+str(len(train_healthyT2))+'#NCtrainsamplesT3 '+str(len(train_healthyT3))+'\n'+'\n')
            #dataList.write('Number of non-dummy images in train data dictionaries:'+'\n')
            #dataList.wrtie('#ADtrainsamplesT1 '+str(len(train_adT1))+'#ADtrainsamplesT2 '+str(len(train_adT2))+'#ADtrainsamplesT3 '+str(len(train_adT3))+
            #'#NCtrainsamplesT1 '+str(len(train_healthyT1))+'#NCtrainsamplesT2 '+str(len(train_healthyT2))+'#NCtrainsamplesT3 '+str(len(train_healthyT3))+'\n'+'\n')

        return train_dict,val_dict,test_dict

    #SHOULD FOLLOW SAME ORDER OF SUBJECTS AS mri_file_names
    
    def get_data_xls (self, mri_file_names, RNN=False):
        '''Method extracts clinical variables data for all files in mri_file_names list
        Both mri_file_names and LP_ADNIMERGE.csv are in imageID order
        '''
        with open(self.xls_datapath + '/' + 'LP_ADNIMERGE.csv', 'r') as f:
            reader = csv.reader(f)
            xls = [row for row in reader]  #Extract all data from csv file in a list.

        #xls extracts baseline features for patients sorted as in mri_file_names
        test_xls=[]
        for file_name in mri_file_names:
            for row in xls[1:]:
                #imageID = 'I'+row[3]  #prevents shorter imageIDs from matching to longer IDs which contain them
                imageID = row[3]  #use this for loading validation set
                if imageID in file_name:
                    test_xls.append(row)
                    break                 
        #check datalists
        if RNN == False:
            with open(self.xls_datapath+'/figures/DataList.txt','a') as dataList:
                dataList.write('Total CNN Train/Val/Test for each timepoint:'+'\n')
                dataList.write("length of _dict(mri) "+str(len(mri_file_names))+'\n')
                dataList.write("length of test_xls "+str(len(test_xls))+'\n'+'\n')
            """
            with open(self.xls_datapath + '/xlschecks/' + 'dictmri'+str(mri_file_names[1])+'.txt', 'w') as names:
                for line in mri_file_names:
                    names.write(" ".join(line)+"\n")
            with open(self.xls_datapath + '/xlschecks/' + 'testxls'+str(mri_file_names[1])+'.txt', 'w') as testxls:
                for line in test_xls:
                    testxls.write(" ".join(line)+"\n")
            """
        else:
            with open(self.xls_datapath+'/figures/DataList.txt','a') as dataList:
                dataList.write('Total RNN scans in each class for each timepoint (H/A):'+'\n')
                dataList.write("length of _dict(mri) "+str(len(mri_file_names))+'\n')
                dataList.write("length of test_xls "+str(len(test_xls))+'\n'+'\n')
#            with open(self.xls_datapath + '/xlschecks/' + 'dictmri'+str(mri_file_names[1])+'.txt', 'w') as names:
#                for line in mri_file_names:
#                    names.write(" ".join(line)+"\n")
#            with open(self.xls_datapath + '/xlschecks/' + 'testxls'+str(mri_file_names[1])+'.txt', 'w') as testxls:
#                for line in test_xls:
#                    testxls.write(" ".join(line)+"\n")
        #convert gender features to binary variables #removed ethnicity/race
        for row in test_xls:
#            row[8] = float(row[8])
            if row[6] == 'M':
                row[6] = 1.
            else:
                row[6] = 0.
#            row[10] = float(row[10])    
#            if row[11] == 'Hisp/Latino': 
#                row[11] = 1.
#            else:
#                row[11] = 0.                 
#            if row[12] == 'White': #White or non-white only;
#                row[12] = 1. #Cluster Am. Indian, unknown, black, asian
#            else:
#                row[12] = 0.
#            row[13:] = [float(x) for x in row[13:]]

        clinical_features = np.asarray([row[6:8] for row in test_xls]) #only capturing gender and age
        PTIDs = np.asarray([row[2] for row in test_xls])
        imageIDs = np.asarray([row[3] for row in test_xls])
        confids = np.asarray([row[15] for row in test_xls])
        csfs = np.asarray([row[16] for row in test_xls])
        
        return clinical_features, PTIDs, imageIDs, confids, csfs
        
    def get_data_mri (self, filename_dict, mri_datapath, RNN=False):
         '''Loads subject volumes from filename dictionary
         Returns MRI volume and label     
         '''
         mris = np.zeros( (len(filename_dict['mri']),) +  self.target_shape)
         jacs = np.zeros( (len(filename_dict['mri']),) +  self.target_shape)
         if RNN == False:
            labels = filename_dict['health_state']
         else:
            labels = np.zeros(len(filename_dict['mri'])) #just a placeholder bc I never actually use this value
         #keys = ['JD', 'mri']
         keys = ['mri']
         for key in keys:
             for j, filename in enumerate (filename_dict[key]):
                if filename == 'NaN':  #for dummy images, can likely delete
                    mris[j] = np.full((91,109,91,1),-1)
                else:
                    proxy_image = nib.load(mri_datapath + '/' + filename)
                    image = np.asarray(proxy_image.dataobj)
#                   if key == 'JD':
#                       jacs[j] = np.asarray(np.expand_dims(image, axis = -1))
#                   else:
                    mris[j] = np.asarray(np.expand_dims(image, axis = -1))
         with open(self.xls_datapath+'/figures/getdatamri.txt','w') as getdatamri:
            getdatamri.write('Images:'+'\n')  
            getdatamri.write(str(mris)+'\n')            
         return mris.astype('float32'), jacs.astype('float32'), labels

            
    def normalize_data (self, train_im, val_im, test_im, mode):  
        #We use different normalization procedures depending on data type
        if mode != 'mri' and mode != 'jac' and mode != 'xls':
            print ('Mode has to be mri, jac or xls depending on image data type')
        elif mode == 'mri':
            print('length of train_im: ', len(train_im))
            std = np.std(train_im, axis = 0)
            #print('std: ', std)
            mean = np.mean (train_im, axis = 0)
            #print('mean: ', mean)
            train_im = (train_im - mean)/(std + 1e-20)
            print('length of norm train_im: ', len(train_im))
            val_im = (val_im - mean)/(std + 1e-20)
            test_im = (test_im - mean)/(std + 1e-20)
        elif mode == 'jac':
            high = np.max(train_im)
            low = np.min(train_im)
            train_im = (train_im - low)/(high - low)
            val_im = (val_im - low)/(high - low)
            test_im = (test_im - low)/(high - low)
        else:
            high = np.max(train_im, axis = 0)
            low = np.min(train_im, axis = 0)
            train_im = (train_im - low)/(high - low)
            val_im = (val_im - low)/(high - low)
            test_im = (test_im - low)/(high - low) 
        return train_im, val_im, test_im
        
    def normalize_data_RNN (self, dataT1, dataT2, dataT3, mode):  
        #We use different normalization procedures depending on data type
        if mode != 'mri' and mode != 'jac' and mode != 'xls':
            print ('Mode has to be mri, jac or xls depending on image data type')
        elif mode == 'mri':
            stdT1 = np.std(dataT1, axis = 0) 
            meanT1 = np.mean (dataT1, axis = 0)
            dataT1 = (dataT1 - meanT1)/(stdT1 + 1e-20)
            stdT2 = np.std(dataT2, axis = 0) 
            meanT2 = np.mean (dataT2, axis = 0)
            dataT2 = (dataT2 - meanT2)/(stdT2 + 1e-20)
            stdT3 = np.std(dataT3, axis = 0) 
            meanT3 = np.mean (dataT3, axis = 0)
            dataT3 = (dataT3 - meanT3)/(stdT3 + 1e-20)
        return dataT1, dataT2, dataT3


    def split_data_RNN (self, healthy_arrayT1,healthy_arrayT2,healthy_arrayT3, ad_arrayT1,ad_arrayT2,ad_arrayT3,rnn_HptidT1_padded,rnn_HptidT2_padded,rnn_HptidT3_padded,rnn_HimageIDT1_padded,rnn_HimageIDT2_padded,rnn_HimageIDT3_padded,rnn_AptidT1_padded,rnn_AptidT2_padded,rnn_AptidT3_padded,rnn_AimageIDT1_padded,rnn_AimageIDT2_padded,rnn_AimageIDT3_padded, val_split =  0.20):

        '''Split the feature vectors for the RNN into training/validation and test sets.
        Should be the same process as split filenames, but now I have arrays instead of dictionaries  
        Also, I want to split data by patient, not by scan. 
        All timepoint arrays should be organized by patient with dummy vectors as placeholders.
        So I only need to split T1, then the same spots in T2 and T3 can follow.
        '''
        train_arrayT1 = []
        train_arrayT2 = []
        train_arrayT3 = []
        val_arrayT1 = []
        val_arrayT2 = []
        val_arrayT3 = []
        test_arrayT1 = []
        test_arrayT2 = []
        test_arrayT3 = []

        num_test_ad= int(len(ad_arrayT1)*val_split)
        num_test_healthy = int(len(healthy_arrayT1)*val_split)

        num_val_ad = int((len(ad_arrayT1)-num_test_ad)*val_split)
        num_val_healthy = int((len(healthy_arrayT1)-num_test_healthy)*val_split)

        test_adT1 = ad_arrayT1[:num_test_ad]
        test_adT2 = ad_arrayT2[:num_test_ad]
        test_adT3 = ad_arrayT3[:num_test_ad]
        test_healthyT1 = healthy_arrayT1[:num_test_healthy]
        test_healthyT2 = healthy_arrayT2[:num_test_healthy]
        test_healthyT3 = healthy_arrayT3[:num_test_healthy]

        test_arrayT1 = np.concatenate((test_adT1, test_healthyT1),axis=0)
        test_arrayT2 = np.concatenate((test_adT2, test_healthyT2),axis=0) 
        test_arrayT3 = np.concatenate((test_adT3, test_healthyT3),axis=0)
        test_labels = np.concatenate((np.zeros(len(test_adT1)),np.ones(len(test_healthyT1))))
        
        test_AptidT1 = rnn_AptidT1_padded[:num_test_ad]
        test_AptidT2 = rnn_AptidT2_padded[:num_test_ad]
        test_AptidT3 = rnn_AptidT3_padded[:num_test_ad]
        test_AimageIDT1 = rnn_AimageIDT1_padded[:num_test_ad]
        test_AimageIDT2 = rnn_AimageIDT2_padded[:num_test_ad]
        test_AimageIDT3 = rnn_AimageIDT3_padded[:num_test_ad]
        test_HptidT1 = rnn_HptidT1_padded[:num_test_healthy]
        test_HptidT2 = rnn_HptidT2_padded[:num_test_healthy]
        test_HptidT3 = rnn_HptidT3_padded[:num_test_healthy]
        test_HimageIDT1 = rnn_HimageIDT1_padded[:num_test_healthy]
        test_HimageIDT2 = rnn_HimageIDT2_padded[:num_test_healthy]
        test_HimageIDT3 = rnn_HimageIDT3_padded[:num_test_healthy]
        
        test_ptidT1 = np.concatenate((test_AptidT1, test_HptidT1),axis=0)
        test_ptidT2 = np.concatenate((test_AptidT2, test_HptidT2),axis=0)
        test_ptidT3 = np.concatenate((test_AptidT3, test_HptidT3),axis=0)
        test_imageIDT1 = np.concatenate((test_AimageIDT1, test_HimageIDT1),axis=0)
        test_imageIDT2 = np.concatenate((test_AimageIDT2, test_HimageIDT2),axis=0)
        test_imageIDT3 = np.concatenate((test_AimageIDT3, test_HimageIDT3),axis=0)
        
        val_adT1 = ad_arrayT1[num_test_ad : num_test_ad + num_val_ad]
        val_adT2 = ad_arrayT2[num_test_ad : num_test_ad + num_val_ad]
        val_adT3 = ad_arrayT3[num_test_ad : num_test_ad + num_val_ad]
        val_healthyT1 = healthy_arrayT1[num_test_healthy : num_test_healthy + num_val_healthy]
        val_healthyT2 = healthy_arrayT2[num_test_healthy : num_test_healthy + num_val_healthy]
        val_healthyT3 = healthy_arrayT3[num_test_healthy : num_test_healthy + num_val_healthy]

        val_arrayT1 = np.concatenate((val_adT1, val_healthyT1),axis=0)
        val_arrayT2 = np.concatenate((val_adT2, val_healthyT2),axis=0) 
        val_arrayT3 = np.concatenate((val_adT3, val_healthyT3),axis=0)
        val_labels = np.concatenate((np.zeros(len(val_adT1)),np.ones(len(val_healthyT1))))        

        train_adT1 = ad_arrayT1[num_test_ad + num_val_ad:]
        train_adT2 = ad_arrayT2[num_test_ad + num_val_ad:]
        train_adT3 = ad_arrayT3[num_test_ad + num_val_ad:]
        train_healthyT1 = healthy_arrayT1[num_test_healthy + num_val_healthy:]
        train_healthyT2 = healthy_arrayT2[num_test_healthy + num_val_healthy:]
        train_healthyT3 = healthy_arrayT3[num_test_healthy + num_val_healthy:]

        train_arrayT1 = np.concatenate((train_adT1, train_healthyT1),axis=0) 
        train_arrayT2 = np.concatenate((train_adT2, train_healthyT2),axis=0)
        train_arrayT3 = np.concatenate((train_adT3, train_healthyT3),axis=0)
        train_labels = np.concatenate((np.zeros(len(train_adT1)),np.ones(len(train_healthyT1))))

        with open(self.xls_datapath+'/figures/DataList.txt','a') as dataList:
            dataList.write('AFTER CLASS BALANCING'+'\n')
            dataList.write('RNN Train Data Split by class and timepoint:'+'\n')
            dataList.write('#ADtrainsamplesT1 '+str(len(train_adT1))+'#ADtrainsamplesT2 '+str(len(train_adT2))+'#ADtrainsamplesT3 '+str(len(train_adT3))+
            '#NCtrainsamplesT1 '+str(len(train_healthyT1))+'#NCtrainsamplesT2 '+str(len(train_healthyT2))+'#NCtrainsamplesT3 '+str(len(train_healthyT3))+'\n'+'\n')
            dataList.write('RNN Val Data Split by class and timepoint:'+'\n')
            dataList.write('#ADvalsamplesT1 '+str(len(val_adT1))+'#ADvalsamplesT2 '+str(len(val_adT2))+'#ADvalsamplesT3 '+str(len(val_adT3))+
            '#NCvalsamplesT1 '+str(len(val_healthyT1))+'#NCvalsamplesT2 '+str(len(val_healthyT2))+'#NCvalsamplesT3 '+str(len(val_healthyT3))+'\n'+'\n')
            dataList.write('RNN Test Data Split by class and timepoint:'+'\n')
            dataList.write('#ADtestsamplesT1 '+str(len(test_adT1))+'#ADtestsamplesT2 '+str(len(test_adT2))+'#ADtestsamplesT3 '+str(len(test_adT3))+
            '#NCtestsamplesT1 '+str(len(test_healthyT1))+'#NCtestsamplesT2 '+str(len(test_healthyT2))+'#NCtestsamplesT3 '+str(len(test_healthyT3))+'\n'+'\n')

        return train_arrayT1,train_arrayT2,train_arrayT3,val_arrayT1,val_arrayT2,val_arrayT3,test_arrayT1,test_arrayT2,test_arrayT3, train_labels,val_labels,test_labels, test_ptidT1,test_ptidT2,test_ptidT3,test_imageIDT1,test_imageIDT2,test_imageIDT3
       
        
    def get_train_val_test (self, val_split, mri_datapath):
        healthy_dict,ad_dict,healthyT1_Rdict,healthyT2_Rdict,healthyT3_Rdict,healthyT4_Rdict,healthyT5_Rdict,healthyT6_Rdict,healthyT7_Rdict,healthyT8_Rdict,adT1_Rdict,adT2_Rdict,adT3_Rdict,adT4_Rdict,adT5_Rdict,adT6_Rdict,adT7_Rdict,adT8_Rdict = self.get_filenames(mri_datapath)
        #make classes balanced
        diff = len(healthy_dict['mri'])-len(ad_dict['mri'])
        for i in range(diff):
            healthy_dict['mri'].pop(-1)
            healthy_dict['PTID'].pop(-1)
            healthy_dict['viscode'].pop(-1)
            healthy_dict['imageID'].pop(-1)

        train_dict, val_dict, test_dict = self.split_filenames (healthy_dict, ad_dict, val_split = val_split)
        #train_dictT1,train_dictT2,train_dictT3, val_dictT1,val_dictT2,val_dictT3, test_dictT1,test_dictT2,test_dictT3 = self.split_filenames (healthyM24_dict,healthyM36_dict,healthyM48_dict, adBL_dict,adM12_dict,adM24_dict, val_split = val_split)
        #train_dictT4,train_dictT5,train_dictT6, val_dictT4,val_dictT5,val_dictT6, test_dictT4,test_dictT5,test_dictT6 = self.split_filenames (healthyBL_dict,healthyM6_dict,healthyM12_dict, adM6_dict,adM36_dict,adM48_dict, val_split = val_split, first=False)

#        print("length of train_dict[mri]"+str(len(train_dict['mri'])))

        train_mri, train_jac, train_labels  = self.get_data_mri(train_dict,mri_datapath)
        train_xls, train_ptid, train_imageID, train_confid, train_csf = self.get_data_xls (train_dict['mri'])
        val_mri, val_jac, val_labels = self.get_data_mri(val_dict,mri_datapath)
        val_xls, val_ptid, val_imageID, val_confid, val_csf = self.get_data_xls (val_dict['mri'])
        test_mri, test_jac, test_labels = self.get_data_mri(test_dict,mri_datapath)
        test_xls, test_ptid, test_imageID, test_confid, test_csf = self.get_data_xls (test_dict['mri'])

#previously removed normalization because it seemed to be making all my images the exact same...?
#somehow it's ok now though! See normalizedTestData
#now seems to be making the images weird and dark. I normalize in preprocess so I don't think I need to here
        #carry the non-normalized through for grad-cam purposes
        test_mri_nonorm = test_mri 
        #train_mri, val_mri, test_mri = self.normalize_data (train_mri, val_mri, test_mri, mode = 'mri')

        #with open(self.xls_datapath+'/figures/normalizedTestData.txt','w') as normed:
        #    normed.write('Normalized CNN Train Images:'+'\n')  
        #    normed.write(str(train_mri)+'\n')
                       
        test_data = (test_mri, test_mri, test_xls, test_labels, test_ptid, test_imageID, test_confid, test_csf)
        val_data = (val_mri, val_mri, val_xls, val_labels, val_ptid, val_imageID, val_confid, val_csf)
        train_data = (train_mri, train_mri, train_xls, train_labels, train_ptid, train_imageID, train_confid, train_csf)

    #get data lists for RNN
        rnn_HmriT1, rnn_HjacT1, rnn_HlabelsT1 = self.get_data_mri(healthyT1_Rdict,mri_datapath, RNN=True)
        rnn_HxlsT1, rnn_HptidT1, rnn_HimageIDT1, rnn_HconfidT1, rnn_HcsfT1 = self.get_data_xls (healthyT1_Rdict['mri'], RNN=True)
        rnn_HmriT2, rnn_HjacT2, rnn_HlabelsT2 = self.get_data_mri(healthyT2_Rdict,mri_datapath, RNN=True)
        rnn_HxlsT2, rnn_HptidT2, rnn_HimageIDT2, rnn_HconfidT2, rnn_HcsfT2 = self.get_data_xls (healthyT2_Rdict['mri'], RNN=True)
        rnn_HmriT3, rnn_HjacT3, rnn_HlabelsT3 = self.get_data_mri(healthyT3_Rdict,mri_datapath, RNN=True)
        rnn_HxlsT3, rnn_HptidT3, rnn_HimageIDT3, rnn_HconfidT3, rnn_HcsfT3 = self.get_data_xls (healthyT3_Rdict['mri'], RNN=True)
        #normalize:
        #rnn_HmriT1,rnn_HmriT2,rnn_HmriT3  = self.normalize_data_RNN (rnn_HmriT1,rnn_HmriT2,rnn_HmriT3, mode = 'mri')   #Don't have any dummies yet, so this should only affect the actual images
        #rnn_HjacT1,rnn_HjacT2,rnn_HjacT3  = self.normalize_data_RNN (rnn_HjacT1,rnn_HjacT2,rnn_HjacT3, mode = 'jac')   

        rnn_HdataT1 = (rnn_HmriT1, rnn_HjacT1, rnn_HxlsT1, rnn_HlabelsT1, rnn_HptidT1, rnn_HimageIDT1, rnn_HconfidT1, rnn_HcsfT1)
        rnn_HdataT2 = (rnn_HmriT2, rnn_HjacT2, rnn_HxlsT2, rnn_HlabelsT2, rnn_HptidT2, rnn_HimageIDT2, rnn_HconfidT2, rnn_HcsfT2)
        rnn_HdataT3 = (rnn_HmriT3, rnn_HjacT3, rnn_HxlsT3, rnn_HlabelsT3, rnn_HptidT3, rnn_HimageIDT3, rnn_HconfidT3, rnn_HcsfT3)
        
        rnn_AmriT1, rnn_AjacT1, rnn_AlabelsT1 = self.get_data_mri(adT1_Rdict,mri_datapath, RNN=True)
        rnn_AxlsT1, rnn_AptidT1, rnn_AimageIDT1, rnn_AconfidT1, rnn_AcsfT1 = self.get_data_xls (adT1_Rdict['mri'], RNN=True)
        rnn_AmriT2, rnn_AjacT2, rnn_AlabelsT2 = self.get_data_mri(adT2_Rdict,mri_datapath, RNN=True)
        rnn_AxlsT2, rnn_AptidT2, rnn_AimageIDT2, rnn_AconfidT2, rnn_AcsfT2 = self.get_data_xls (adT2_Rdict['mri'], RNN=True)
        rnn_AmriT3, rnn_AjacT3, rnn_AlabelsT3 = self.get_data_mri(adT3_Rdict,mri_datapath, RNN=True)
        rnn_AxlsT3, rnn_AptidT3, rnn_AimageIDT3, rnn_AconfidT3, rnn_AcsfT3 = self.get_data_xls (adT3_Rdict['mri'], RNN=True)
        #normalize:
        #rnn_AmriT1,rnn_AmriT2,rnn_AmriT3  = self.normalize_data_RNN (rnn_AmriT1,rnn_AmriT2,rnn_AmriT3, mode = 'mri')  
        #rnn_AjacT1,rnn_AjacT2,rnn_AjacT3  = self.normalize_data_RNN (rnn_AjacT1,rnn_AjacT2,rnn_AjacT3, mode = 'jac')
        #with open(self.xls_datapath+'/figures/normalizedTestData.txt','a') as normed:
        #    normed.write('Normalized RNN T1 NC Images:'+'\n')  
        #    normed.write(str(rnn_HmriT1)+'\n')
        #    normed.write('Normalized RNN T1 AD Images:'+'\n')  
        #    normed.write(str(rnn_AmriT1)+'\n')
        
        rnn_AdataT1 = (rnn_AmriT1, rnn_AjacT1, rnn_AxlsT1, rnn_AlabelsT1, rnn_AptidT1, rnn_AimageIDT1, rnn_AconfidT1, rnn_AcsfT1)
        rnn_AdataT2 = (rnn_AmriT2, rnn_AjacT2, rnn_AxlsT2, rnn_AlabelsT2, rnn_AptidT2, rnn_AimageIDT2, rnn_AconfidT2, rnn_AcsfT2)
        rnn_AdataT3 = (rnn_AmriT3, rnn_AjacT3, rnn_AxlsT3, rnn_AlabelsT3, rnn_AptidT3, rnn_AimageIDT3, rnn_AconfidT3, rnn_AcsfT3)

        return train_data, val_data, test_data,rnn_HdataT1,rnn_HdataT2,rnn_HdataT3,rnn_AdataT1,rnn_AdataT2,rnn_AdataT3, test_mri_nonorm
