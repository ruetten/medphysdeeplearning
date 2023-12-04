
import numpy as np
from numpy.random import RandomState
from os import listdir
import nibabel as nib
import math
import csv
import random
from itertools import permutations 


class PatientSorter():
    def __init__(self, seed=None):
        self.seed = seed
        
    def sort_patients(self, scan_dict, label, xls_datapath,first=False):
        #from LP_ADNIMERGE, AD patients have at most 4 scans and healthy patients have at most 7 scans; but there are 8 possible timepoints (excluding m06)
        #initialize dicts
        keys = ['mri','PTID','viscode','imageID']
        T1_dict,T2_dict,T3_dict,T4_dict,T5_dict,T6_dict,T7_dict,T8_dict = [{key: [] for key in keys} for i in range(8)]
        dictList = [T1_dict,T2_dict,T3_dict,T4_dict,T5_dict,T6_dict,T7_dict,T8_dict]
        
        #convert viscodes into numbers 
        z=0
        for viscode in scan_dict['viscode']:
            z+=1
            if viscode == 'bl':
                scan_dict['viscode'][z-1] = 0.               
            elif viscode == 'm12':
                scan_dict['viscode'][z-1] = 12.
            elif viscode == 'm24':
                scan_dict['viscode'][z-1] = 24.
            elif viscode == 'm36':
                scan_dict['viscode'][z-1] = 36.
            elif viscode == 'm48':
                scan_dict['viscode'][z-1] = 48.
            elif viscode == 'm60':
                scan_dict['viscode'][z-1] = 60.
            elif viscode == 'm72':
                scan_dict['viscode'][z-1] = 72.
            elif viscode == 'm84':
                scan_dict['viscode'][z-1] = 84. 
            elif viscode == 'm96':
                scan_dict['viscode'][z-1] = 96.
            else:
                scan_dict['viscode'][z-1] = 2000.
                
        #Initialize Counting Variables
        usedPTIDs = []
        scanCount = []
        sortedViscodes = []
        a=0
        #combos2 = combinations([T1,T2,T3,T4,T5,T6,T7],2)  #all combinations of TPs of length 2
        #combos3 = combinations([T1,T2,T3,T4,T5,T6,T7],3)
        #combos4 = combinations([T1,T2,T3,T4,T5,T6,T7],4)
        #combos5 = combinations([T1,T2,T3,T4,T5,T6,T7],5)
        #combos6 = combinations([T1,T2,T3,T4,T5,T6,T7],6)
        #for i in list(combos2):
        #    print i   #at the end, check if all values are true in each i, if so, add to count
         
        #sort scans into TP dicts
        #print("scan_dict['mri']",scan_dict['mri'])
       
        T1only=0
        T1T2=0
        T1T3=0
        T1T2T3=0
        
        for ptid1 in scan_dict['PTID']:
            used = False
            T2=False
            T3=False
            a+=1
            tempList = []
            tempViscodes = []
            tempImageID = []
            sortedTempList = []
            sortedTempViscodes = []
            sortedTempImageID = []
            TempLength = 0
            for usedptid in usedPTIDs:  #check if this ptid has been accounted for already
                if ptid1 == usedptid:
                    used = True
                    break
            if used == False:  #if this PTID has not been accounted for yet... (otherise go to next ptid)
                tempList.append(scan_dict['mri'][a-1])   #add first scan to templist
                tempViscodes.append(scan_dict['viscode'][a-1])   #add viscode of this scan to templist
                tempImageID.append(scan_dict['imageID'][a-1])
                usedPTIDs.append(ptid1)  #add ptid to used list
                for b in range(a,len(scan_dict['PTID'])):  #check for other scans with same ptid
                    if scan_dict['PTID'][b] == ptid1:
                        tempList.append(scan_dict['mri'][b])  #if match, then add that scan to the templist
                        tempViscodes.append(scan_dict['viscode'][b])   #add viscode of this scan to templist
                        tempImageID.append(scan_dict['imageID'][b])
                #record number of scans for that patient
                tempLength = len(tempList)
                scanCount.append(tempLength)
                if tempLength > 1:  #Throw out all scans with only 1 timepoint
                    sortedTempList = [x for _,x in sorted(zip(tempViscodes,tempList))]   #sort the scans in order by viscode
                    sortedTempImageID = [x for _,x in sorted(zip(tempViscodes,tempImageID))]
                    #sortedTempViscodes = [y for _,y in sorted(zip(tempList,tempViscodes))]   #sort the viscodes
                    tempViscodes.sort()  #sort the viscodes in order
                    #print('sortViscodes ',tempViscodes)
                    sortedViscodes.append(tempViscodes)
                    #print('usedPTIDs ',usedPTIDs)
                    #print('length of tempList ',len(tempList))
                    #print('scanCount ',scanCount)
                    T1_dict['mri'].append(sortedTempList[0])  #add first scan to T1_dict
                    T1_dict['PTID'].append(ptid1)
                    T1_dict['viscode'].append(tempViscodes[0])
                    T1_dict['imageID'].append(sortedTempImageID[0])
                    diff = []
                    for i in range(tempLength-1):
                        diff.append(tempViscodes[i+1] - tempViscodes[0])  #so that diff[0] applies to the difference between temp1 and temp0 and diff[1] is diff bw temp2 and temp0
                    for i in range(len(diff)):   #sort scans into the appropriate list based on time diff from T1
                        if diff[i] == 12:
                            T2_dict['mri'].append(sortedTempList[i+1]) 
                            T2_dict['PTID'].append(ptid1)
                            T2_dict['viscode'].append(tempViscodes[i+1])
                            T2_dict['imageID'].append(sortedTempImageID[i+1])
                            T2=True
                        elif diff[i] == 24:
                            T3_dict['mri'].append(sortedTempList[i+1])
                            T3_dict['PTID'].append(ptid1)
                            T3_dict['viscode'].append(tempViscodes[i+1])
                            T3_dict['imageID'].append(sortedTempImageID[i+1])
                            T3=True
                        elif diff[i] == 36:
                            T4_dict['mri'].append(sortedTempList[i+1])
                            T4_dict['PTID'].append(ptid1)
                            T4_dict['viscode'].append(tempViscodes[i+1])
                            T4_dict['imageID'].append(sortedTempImageID[i+1])
                        elif diff[i] == 48:
                            T5_dict['mri'].append(sortedTempList[i+1])
                            T5_dict['PTID'].append(ptid1)
                            T5_dict['viscode'].append(tempViscodes[i+1])
                            T5_dict['imageID'].append(sortedTempImageID[i+1])
                        elif diff[i] == 60:
                            T6_dict['mri'].append(sortedTempList[i+1])
                            T6_dict['PTID'].append(ptid1)
                            T6_dict['viscode'].append(tempViscodes[i+1])
                            T6_dict['imageID'].append(sortedTempImageID[i+1])
                        elif diff[i] == 72:
                            T7_dict['mri'].append(sortedTempList[i+1])
                            T7_dict['PTID'].append(ptid1)
                            T7_dict['viscode'].append(tempViscodes[i+1])
                            T7_dict['imageID'].append(sortedTempImageID[i+1])
                        elif diff[i] == 84:
                            T8_dict['mri'].append(sortedTempList[i+1])
                            T8_dict['PTID'].append(ptid1)
                            T8_dict['viscode'].append(tempViscodes[i+1])
                            T8_dict['imageID'].append(sortedTempImageID[i+1])
                    if T2==True and T3==False:
                        T1T2+=1
                    if T2==True and T3==True:
                        T1T2T3+=1
                    if T2==False and T3==True:
                        T1T3+=1
                    if T2==False and T3==False:
                        T1only+=1
            
        #all scans have been sorted
        #get counts of scanCounts
        scans1tp = sum(1 for i in scanCount if i == 1)
        scans2tp = sum(1 for i in scanCount if i == 2)
        scans3tp = sum(1 for i in scanCount if i == 3)
        scans4tp = sum(1 for i in scanCount if i == 4)
        scans5tp = sum(1 for i in scanCount if i == 5)
        scans6tp = sum(1 for i in scanCount if i == 6)
        scans7tp = sum(1 for i in scanCount if i == 7)
        scans8tp = sum(1 for i in scanCount if i == 8)        

        if first == True:
            with open(xls_datapath+'/figures/RNNDicts.txt','w') as InitialDicts:
                InitialDicts.write('Label: '+label+'\n')
                InitialDicts.write('Length of T1_dict: '+str(len(T1_dict['mri']))+'\n')
                InitialDicts.write('Length of T2_dict: '+str(len(T2_dict['mri']))+'\n')
                InitialDicts.write('Length of T3_dict: '+str(len(T3_dict['mri']))+'\n')
                InitialDicts.write('Length of T4_dict: '+str(len(T4_dict['mri']))+'\n')
                InitialDicts.write('Length of T5_dict: '+str(len(T5_dict['mri']))+'\n')
                InitialDicts.write('Length of T6_dict: '+str(len(T6_dict['mri']))+'\n')
                InitialDicts.write('Length of T7_dict: '+str(len(T7_dict['mri']))+'\n')
                InitialDicts.write('Length of T8_dict: '+str(len(T8_dict['mri']))+'\n')
                InitialDicts.write('ScanCount: '+str(scanCount)+'\n')
                InitialDicts.write('Number of patients with 1 scan: '+str(scans1tp)+'\n')
                InitialDicts.write('Number of patients with 2 scans: '+str(scans2tp)+'\n')
                InitialDicts.write('Number of patients with 3 scans: '+str(scans3tp)+'\n')
                InitialDicts.write('Number of patients with 4 scans: '+str(scans4tp)+'\n')
                InitialDicts.write('Number of patients with 5 scans: '+str(scans5tp)+'\n')
                InitialDicts.write('Number of patients with 6 scans: '+str(scans6tp)+'\n')
                InitialDicts.write('Number of patients with 7 scans: '+str(scans7tp)+'\n')
                InitialDicts.write('Number of patients with 8 scans: '+str(scans8tp)+'\n')
                InitialDicts.write('Used PTIDs: '+str(usedPTIDs)+'\n')
                InitialDicts.write('Sorted Viscodes: '+str(sortedViscodes)+'\n')
                InitialDicts.write('For first 3 timepoints...'+'\n')
                InitialDicts.write('T1only: '+str(T1only)+'\n')
                InitialDicts.write('T1T2: '+str(T1T2)+'\n')
                InitialDicts.write('T1T3: '+str(T1T3)+'\n')
                InitialDicts.write('T1T2T3: '+str(T1T2T3)+'\n')
        else:
            with open(xls_datapath+'/figures/RNNDicts.txt','a') as InitialDicts:
                InitialDicts.write('Label: '+label+'\n')
                InitialDicts.write('Length of T1_dict: '+str(len(T1_dict['mri']))+'\n')
                InitialDicts.write('Length of T2_dict: '+str(len(T2_dict['mri']))+'\n')
                InitialDicts.write('Length of T3_dict: '+str(len(T3_dict['mri']))+'\n')
                InitialDicts.write('Length of T4_dict: '+str(len(T4_dict['mri']))+'\n')
                InitialDicts.write('Length of T5_dict: '+str(len(T5_dict['mri']))+'\n')
                InitialDicts.write('Length of T6_dict: '+str(len(T6_dict['mri']))+'\n')
                InitialDicts.write('Length of T7_dict: '+str(len(T7_dict['mri']))+'\n')
                InitialDicts.write('Length of T8_dict: '+str(len(T8_dict['mri']))+'\n')
                InitialDicts.write('ScanCount: '+str(scanCount)+'\n')
                InitialDicts.write('Number of patients with 1 scan: '+str(scans1tp)+'\n')
                InitialDicts.write('Number of patients with 2 scans: '+str(scans2tp)+'\n')
                InitialDicts.write('Number of patients with 3 scans: '+str(scans3tp)+'\n')
                InitialDicts.write('Number of patients with 4 scans: '+str(scans4tp)+'\n')
                InitialDicts.write('Number of patients with 5 scans: '+str(scans5tp)+'\n')
                InitialDicts.write('Number of patients with 6 scans: '+str(scans6tp)+'\n')
                InitialDicts.write('Number of patients with 7 scans: '+str(scans7tp)+'\n')
                InitialDicts.write('Number of patients with 8 scans: '+str(scans8tp)+'\n')
                InitialDicts.write('Used PTIDs: '+str(usedPTIDs)+'\n')
                InitialDicts.write('Sorted Viscodes: '+str(sortedViscodes)+'\n')
                InitialDicts.write('For first 3 timepoints...'+'\n')
                InitialDicts.write('T1only: '+str(T1only)+'\n')
                InitialDicts.write('T1T2: '+str(T1T2)+'\n')
                InitialDicts.write('T1T3: '+str(T1T3)+'\n')
                InitialDicts.write('T1T2T3: '+str(T1T2T3)+'\n')
            
        return  T1_dict,T2_dict,T3_dict,T4_dict,T5_dict,T6_dict,T7_dict,T8_dict

            
#    def run_sort(self, healthy_dict, ad_dict):
#        healthy_sorted = sort_patients(healthy_dict)
#        ad_sorted = sort_patients(ad_dict)
#    
#    return healthy_sorted, ad_sorted
        
        
        
        
        