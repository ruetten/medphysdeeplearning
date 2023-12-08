import csv
from datetime import datetime as dt
import numpy as np
import nibabel as nib
import re
import os.path
import sys

# NOTICE: This only works if the patient is either AD or CN. Anything else breaks.
# You'll probably only need to call parseData() and parseTimestamps()

# Number of consecutive images to group together in a list
NGRAMS = 3

# Convert a formatted date (MM/DD/YYYY or DD-MM-YYYY) to seconds since Jan 1, 1970
def convertDate(formattedDate):
    p1 = "%m/%d/%Y"
    p2 = "%d-%m-%Y"
    epoch = dt(1970, 1, 1)

    if re.compile("\d+/\d+/\d{2,4}").fullmatch(formattedDate):
        return (dt.strptime(formattedDate, p1) - epoch).total_seconds()
    elif re.compile("\d+-\d+-\d{2,4}").fullmatch(formattedDate):
        return (dt.strptime(formattedDate, p2) - epoch).total_seconds()

# Return a dictionary with image ids for each patient id
def getData():

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

    fields = {'0': {"ids": [], "count": 0, "examdate": []}}
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
            fields[cur_id]["count"] = count

            # Initialize variables for next Patient ID
            count = 1
            cur_id = row[2]
            fields[cur_id] = {"ids": [row[3]], "count": 0, "examdate": [convertDate(row[4])]}

        else:
            count += 1
            fields[cur_id]["ids"].append(row[3])
            fields[cur_id]["examdate"].append(convertDate(row[4]))

    # Debugging
    # print("AD: %d \nCN: %d" % (ones_ad, ones_cn))
    return fields

###
# Load a list with all the image data for the provided patient id.
# ptid: string. The patient ID
# fields: List of dictionaries. You would get this by running getData()
# RETURNS: A List of Lists, where each sublist is of length 3 and contains the NIFTI data as a tensor of floats
###
def parseData(ptid, fields):

    data = {"ids": [], "count": 0, "examdate": []}
    try:
        data = fields[ptid]
    except:
        print("parseData(): Invalid PTID", file=sys.stderr)
        return None
    
    # Debugging
    # print(data)

    # Assert that there are 3 images in the image ID list
    if data["count"] < 3:
        print("parsedata(): Less than 3 images in source", file=sys.stderr)
        return None

    # Find time delta using initial exam dates. Time in milliseconds
    # delta = max(data["examdate"]) - min(data["examdate"])
    # print(delta)
    
    listOfTriplets = []

    # For each set of NGRAMS (default is 3)
    for i in range(int(data["count"] - NGRAMS + 1)):
        indices = [i, i+1, i+2]

        # Load up the tensor with numerical data now that we have an image
        listOfTriplets.append(parseDataOne(data, indices))

    # For debugging
    # print(len(listOfTriplets))
    return listOfTriplets

# Analogous to parseData(), but sublists contain timestamps (in days) instead of tensors
def parseTimestamps(ptid, fields):

    data = {"ids": [], "count": 0, "examdate": []}
    try:
        data = fields[ptid]
    except:
        print("parseData(): Invalid PTID", file=sys.stderr)
        return None
    
    # Debugging
    # print(data)

    # Assert that there are 3 images in the image ID list
    if data["count"] < 3:
        print("parsedata(): Less than 3 images in source", file=sys.stderr)
        return None

    listOfTriplets = []

    # For each set of NGRAMS (default is 3)
    for i in range(int(data["count"] - NGRAMS + 1)):
        indices = [i, i+1, i+2]

        # Load up the list with the timestamps
        listOfTriplets.append(parseTimestampOne(data, indices))

    # For debugging
    # print(len(listOfTriplets))
    return listOfTriplets

###
# Get the list of tensors given a dictionary with all relevant images and which images to use
# data: The image ids, dates, and things like that for a given PTID
# indices: Which set of 3 images to include in the list
# RETURNS: a List of 3 tensors, each with NIFTI image data as floats
###
def parseDataOne(data, indices):

    if data == None or data["ids"] == None or data["examdate"] == None or data["count"] < 3:
        print("parseDataOne(): Poorly formatted data", file=sys.stderr)

    if len(indices) != 3:
        print("parseDataOne(): Indices is of incorrect length", file=sys.stderr)
        return None

    # Find time delta using initial exam dates. Time in milliseconds
    delta = max(data["examdate"]) - min(data["examdate"])

    listOfTensors = []

    # For each image id, load the data into the tensor
    for i in range(3):
        cur_id = indices[i]

        img_id = data["ids"][cur_id]
        # print(data["examdate"][cur_id])
        
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
            print("parseDataOne(): NIFTI image with ID (", img_id, ") not found", file=sys.stderr)
            continue

        # Append the image data tensor to the list
        listOfTensors.append(img.get_fdata())

    return listOfTensors

    # For debugging
    # print(tensor.shape)

# Analogous to parseDataOne(), loads timestamp (in days) instead of image data    
def parseTimestampOne(data, indices):

    if data == None or data["ids"] == None or data["examdate"] == None or data["count"] < 3:
        print("parseDataOne(): Poorly formatted data", file=sys.stderr)

    if len(indices) != 3:
        print("parseDataOne(): Indices is of incorrect length", file=sys.stderr)
        return None

    # Find time delta using initial exam dates. Time in milliseconds
    delta = max(data["examdate"]) - min(data["examdate"])

    listOfDates = []
    firstDate = min(data["examdate"])

    SEC_TO_HR = 3600
    HR_TO_DAY = 24

    # For each image id, load the data into the tensor
    for i in range(3):
        cur_id = indices[i]

        delta = data["examdate"][cur_id] - firstDate

        # Error if the time delta is negative, inlude in list anyways
        if delta < 0:
            print("Negative time difference encountered for image %d" % data["ids"][cur_id], file=sys.stderr)

        deltaDays = delta / SEC_TO_HR / HR_TO_DAY

        # Append the image data tensor to the list
        listOfDates.append(int(deltaDays))

    # For debugging
    # print(listOfDates)
 
    return listOfDates

    
# Entry point code here just for debugging
# the_ptid = input("(loader.py) PTID from CSV: ")
# the_fields = getData()

# the_result = parseTimestamps(the_ptid, the_fields)
# print(len(the_result[0]))
