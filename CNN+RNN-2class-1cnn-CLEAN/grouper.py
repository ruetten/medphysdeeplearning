import numpy as np
import csv
import torch

num_features = 5
# Load your CSV file into a NumPy array
#data = np.genfromtxt('predictions_5.csv', delimiter=',', dtype=str, skip_header=1)
data = []
labels = []
PTIDs = []

with open('predictions_5.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    i = 0
    for row in csv_reader:
        label = row['Class']
        features = torch.tensor([float(item) for item in row['Predictions'].strip('[]').split()]).reshape(1, num_features)
        ptid = row['PTIDs']

        data.append(features)
        labels.append(torch.tensor(int(label)).view(1))
        PTIDs.append(ptid)

# Create a dictionary to store unique Features for each PTID
ptid_features_dict = {}
ptid_labels_dict = {}

# Iterate through the data and populate the dictionary
for ptid, feature, label in zip(PTIDs, data, labels):
    if ptid in ptid_features_dict:
        ptid_features_dict[ptid].append(feature)
    else:
        ptid_features_dict[ptid] = [feature]
        ptid_labels_dict[ptid] = label

#print(ptid_features_dict)
for key, value in ptid_features_dict.items():
    result_tensor = np.concatenate(value, axis=1)

    # Check if the result tensor is less than 15 columns
    if result_tensor.shape[1] < 15:
        # Calculate the number of columns to pad
        padding_size = 15 - result_tensor.shape[1]

        # Create a padding array with -1 and concatenate it to the result_tensor
        padding_array = np.full((1, padding_size), -1)
        result_tensor = np.concatenate([result_tensor, padding_array], axis=1)
    print(f"Key: {key}, Class: {ptid_labels_dict[key]}, Value: {result_tensor}")
