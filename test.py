import csv
from datetime import datetime as dt
import numpy as np
import nibabel as nib
import os.path

def init():

    rows = []
    with open("LP_ADNIMERGE.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            # By default, reader gives us a list with one element (string)
            row = row[0].split(",")

            if row[0] == "AD" or row[0] == "CN":
                rows.append(row[0:4])
                
    rows.sort(key=lambda e: e[2])

    counts = {'0': {"ids": [], "count": 0}}
    cur_id = '0'
    count = 0
    ones_ad = 0
    ones_cn = 0
    for row in rows:
        if row[2] != cur_id:
            
            # Add the count for the Patient ID we just finished
            if count == 1:
                if row[0] == "AD":
                    ones_ad += 1
                else:
                    ones_cn += 1
            counts[cur_id]["count"] = count

            # Initialize variables for next Patient ID
            count = 1
            cur_id = row[2]
            counts[cur_id] = {"ids": [row[3]], "count": 0}

        else:
            count += 1
            counts[cur_id]["ids"].append(row[3]) 

    # print("AD: %d \nCN: %d" % (ones_ad, ones_cn))
    return counts

def parseData(ptid):
    ptids = init()

    data = ptids[ptid]
    print(data)

    # Initialize tensor with appropriate dimensions, and channels for each scan
    tensor = np.zeros([91, 109, 91, data["count"]])

    for i in range(data["count"]):
        img_id = data["ids"][i]
        
        try:
            img_path = "AD_FDGPET_preprocessed/" + "Inf_NaN_stableAD__I" +img_id +"_masked_brain.nii.nii"
            img = nib.load(img_path)

            tensor[:, :, :, i] = img.get_fdata()

        except:
            print("ERR file does not exist %d" % i)

    print(tensor)

    # XXX There are many Image IDs in the CSV file that do not have corresponding filenames in the ZIP directory.


parseData(input("PTID from CSV: "))