# Project: Detecting Cyber-Attacks on SCADA Systems using a Transformer Neural Network 
# and Naive Bayes on the Electra Railway Dataset
# Authors: Aditi Shah, Christian Martin, Colby Tyree, Jessica Elkins
# Leojaris Brujan, Mary Scholl, Tatiana Kontoulakos
# Class: UAH IS 692 Spring 2023
# Description: This program uses the Scikit-Learn's Random Forest Classifier
# to create a Random Forest model based on the Electra Modbus dataset.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


dtype_dict = {"Time": 'Int64', "smac": str, "dmac": str, "sip": str, "dip": str, "request": bool, "fc": int,
              "error": int, "address": int, "data": int, "label": str}

# Import the data into a pandas dataframe
electra_modbus = pd.read_csv(r'C:\Users\jessi\Downloads\electra_modbus.csv', header=0, dtype=dtype_dict)

# Sort the data based on time
electra_modbus.sort_values(by='Time', ascending=True, inplace=True)

# Replace the MITM label with the normal label
electra_modbus = electra_modbus.replace('MITM_UNALTERED', 'NORMAL')

# Initialize the One-Hot Encoder and Label Encoder functions
ohe = OneHotEncoder()
ohe_ip = OneHotEncoder()
le = LabelEncoder()

# Label encode the request and label columns
electra_modbus['request_n'] = le.fit_transform(electra_modbus['request'])
electra_modbus['label_n'] = le.fit_transform(electra_modbus['label'])

# One-Hot Encoding the MAC and IP addresses
mac_array = ohe.fit_transform(electra_modbus[['smac', 'dmac']]).toarray()
ip_array = ohe_ip.fit_transform(electra_modbus[['sip', 'dip']]).toarray()

mac_labels = np.array(ohe.categories_).ravel()
ip_labels = np.array(ohe_ip.categories_).ravel()

# Renaming the One-Hot Encoding (OHE) categories to match source and destination
for i in range(4):
    mac_labels[i] = 'smac_{}'.format(mac_labels[i])
    
for i in range(4):
    mac_labels[i] = 'dmac_{}'.format(mac_labels[i])    
    
for i in range(4):
    ip_labels[i] = 'sip_{}'.format(ip_labels[i])
    
for i in range(4):
    ip_labels[i] = 'dip_{}'.format(ip_labels[i])

# Turning the OHE arrays into pandas dataframes
mac_df = pd.DataFrame(mac_array, columns = mac_labels)
ip_df = pd.DataFrame(ip_array, columns = ip_labels)

# Adding the MAC and IP dataframes onto the original dataframe
electra_modbus = pd.concat([electra_modbus, mac_df, ip_df], axis=1)

# Split the dataset into 60% training, 10% validation, 15% test, and 15% challenge.
train_size = 0.60
validation_size = 0.10
test_size = 0.15
challenge_size = 0.15

# Calculating the indexes to split the df at
train_index_modbus = int(len(electra_modbus) * train_size)
valid_index_modbus = int(len(electra_modbus) * validation_size)
test_index_modbus = int(len(electra_modbus) * test_size)
challenge_index_modbus = int(len(electra_modbus) * challenge_size)

# Slicing the dataset based on the indexes
train_df = electra_modbus[0:train_index_modbus]
validation_df = electra_modbus[train_index_modbus:train_index_modbus+valid_index_modbus]
test_df = electra_modbus[train_index_modbus+valid_index_modbus:train_index_modbus+valid_index_modbus+test_index_modbus]
challenge_df = electra_modbus[train_index_modbus+valid_index_modbus+test_index_modbus:]

feature_columns = [mac_labels[0], mac_labels[1], mac_labels[2], mac_labels[3], mac_labels[4], 
                   mac_labels[5], mac_labels[6], mac_labels[7], ip_labels[0], ip_labels[1], 
                   ip_labels[2], ip_labels[3], ip_labels[4], ip_labels[5], ip_labels[6], 
                   ip_labels[7], 'request_n', 'fc', 'error', 'address', 'data']

# Assigning features to X and labels to y
train_X = train_df[feature_columns]
train_y = train_df['label_n']

validation_X = validation_df[feature_columns]
validation_y = validation_df['label_n']

test_X = test_df[feature_columns]
test_y = test_df['label_n']

challenge_X = challenge_df[feature_columns]
challenge_y = challenge_df['label_n']

# Initializing the Random Forest Classifier
randomforest_model = RandomForestClassifier()

# Training the model
randomforest_model.fit(train_X, train_y)

# Getting predictions for the validaton set
y_pred = randomforest_model.predict(validation_X)

# Calculating validation set accuracy
validation_accuracy = accuracy_score(validation_y, y_pred) * 100
print(f'Random Forest Modbus Validation Accuracy: {validation_accuracy:.4f}')

# Getting predictions for the test set
test_pred = randomforest_model.predict(test_X)

# Calculating test set accuracy
test_accuracy = accuracy_score(test_y, test_pred) * 100
print(f'Random Forest Modbus Test Accuracy: {test_accuracy:.4f}')

# Getting predictions for the challenge set
challenge_pred = randomforest_model.predict(challenge_X)

# Calculating the challenge set accuracy
challenge_accuracy = accuracy_score(challenge_y, challenge_pred) * 100
print(f'Random Forest Modbus Challenge Accuracy: {challenge_accuracy:.4f}')