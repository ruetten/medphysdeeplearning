import csv
from datetime import datetime as dt
import numpy as np
import nibabel as nib
import re
import os.path

# NOTICE: This only works if the patient is either AD or CN. Anything else breaks.

# Convert a formatted date (MM/DD/YYYY or DD-MM-YYYY) to milliseconds since Jan 1, 1970
def convertDate(formattedDate):
    p1 = "%m/%d/%Y"
    p2 = "%d-%m-%Y"
    epoch = dt(1970, 1, 1)

    if re.compile("\d+/\d+/\d{2,4}").fullmatch(formattedDate):
        return (dt.strptime(formattedDate, p1) - epoch).total_seconds()
    elif re.compile("\d+-\d+-\d{2,4}").fullmatch(formattedDate):
        return (dt.strptime(formattedDate, p2) - epoch).total_seconds()

# Return a dictionary with image ids for each patient id
def init():

    rows = []
    with open("LP_ADNIMERGE.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            # By default, reader gives us a list with one element (string)
            row = row[0].split(",")

            if row[0] == "AD" or row[0] == "CN":
                rows.append(row[0:5])

    # Sort by patient ID       
    rows.sort(key=lambda e: e[2])

    counts = {'0': {"ids": [], "count": 0, "examdate": []}}
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
            counts[cur_id] = {"ids": [row[3]], "count": 0, "examdate": [convertDate(row[4])]}

        else:
            count += 1
            counts[cur_id]["ids"].append(row[3])
            counts[cur_id]["examdate"].append(convertDate(row[4]))

    # print("AD: %d \nCN: %d" % (ones_ad, ones_cn))
    return counts

# Load a tensor with all the image data for the provided patient id
def parseData(ptid):
    ptids = init()

    data = ptids[ptid]
    print(data)

    # Find time delta using initial exam dates. Time in milliseconds
    delta = max(data["examdate"]) - min(data["examdate"])
    print(delta)

    # Initialize tensor with appropriate dimensions, and channels for each scan
    # NOTE: Samarth said that the first dimension should be the channel count
    tensor = np.zeros([data["count"], 91, 109, 91])

    # For each image id, load the data into the tensor
    for i in range(data["count"]):
        img_id = data["ids"][i]
        
        # Variables help keep the lines somewhat condensed
        ad_dir = "AD_FDGPET_preprocessed"
        ad_name = "Inf_NaN_stableAD__I" +img_id +"_masked_brain.nii.nii"

        nc_dir = "NC_FDGPET_preprocessed"
        nc_name = "Inf_NaN_stableNL__I" +img_id +"_masked_brain.nii.nii"

        # Check for Alzheimer's scans (with and without extra _s)
        if os.path.exists(ad_dir+"/" +ad_name):
            img = nib.load(ad_dir +"/" +ad_name)
        elif os.path.exists(ad_dir +"/" +"Inf_NaN_stableAD_I" +img_id +"_masked_brain.nii.nii"):
            img = nib.load(ad_dir +"/" +"Inf_NaN_stableAD_I" +img_id +"_masked_brain.nii.nii")
        
        # Check for CN scans (with and without extra _s)
        elif os.path.exists(nc_dir+"/" +nc_name):
            img = nib.load(nc_dir +"/" +nc_name)
        elif os.path.exists(nc_dir +"/" +"Inf_NaN_stableNL_I" +img_id +"_masked_brain.nii.nii"):
            img = nib.load(nc_dir +"/" +"Inf_NaN_stableNL_I" +img_id +"_masked_brain.nii.nii")

        # If a file is not found, show an error and skip
        else:
            print("NIFTI image with %d ID (%d) not found" % (i, img_id))
            continue

        # Load up the tensor with numerical data now that we have an image
        tensor[i, :, :, :] = img.get_fdata()

        # Possibly return?
        # return tensor

    # For debugging
    print(tensor.shape)
    
# Doesn't have to be a text input, here just for example
parseData(input("PTID from CSV: "))