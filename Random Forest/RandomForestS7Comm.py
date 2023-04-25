# Project: Detecting Cyber-Attacks on SCADA Systems using a Transformer Neural Network 
# and Naive Bayes on the Electra Railway Dataset
# Authors: Aditi Shah, Christian Martin, Colby Tyree, Jessica Elkins
# Leojaris Brujan, Mary Scholl, Tatiana Kontoulakos
# Class: UAH IS 692 Spring 2023
# Description: This program uses the Scikit-Learn's Random Forest Classifier
# to create a Random Forest model based on the Electra S7 Comm dataset.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


dtype_dict = {"Time": 'Int64', "smac": str, "dmac": str, "sip": str, "dip": str, "request": bool, "fc": int,
              "error": int, "address": int, "data": int, "label": str}

# Initialize the Random Forest Classifier model
randomforest_model = RandomForestClassifier()

def parse_data(electra_s7):
    # Replace the MITM label with the normal label
    electra_s7 = electra_s7.replace('MITM_UNALTERED', 'NORMAL')
    
    # Initialize the Label Encoder functions
    le = LabelEncoder()

    # Label encoding the request and label columns
    electra_s7['request_n'] = le.fit_transform(electra_s7['request'])
    electra_s7['label_n'] = le.fit_transform(electra_s7['label'])
    
    # Initializing the One-Hot Encoder (OHE) functions
    ohe = OneHotEncoder()
    ohe_ip = OneHotEncoder()
    
    # One-Hot Encoding the MAC and IP addresses
    mac_array = ohe.fit_transform(electra_s7[['smac', 'dmac']]).toarray()
    ip_array = ohe_ip.fit_transform(electra_s7[['sip', 'dip']]).toarray()
    
    mac_labels = np.array(ohe.categories_).ravel()
    ip_labels = np.array(ohe_ip.categories_).ravel()
     
    # Turning the OHE arrays into pandas dataframes
    mac_df = pd.DataFrame(mac_array, columns = mac_labels)
    ip_df = pd.DataFrame(ip_array, columns = ip_labels)
    
    # Adding the MAC and IP dataframes onto the original dataframe
    electra_s7 = pd.concat([electra_s7, mac_df, ip_df], axis=1)
    
    feature_columns = [mac_labels[0], mac_labels[1], mac_labels[2], mac_labels[3], mac_labels[4], 
                       mac_labels[5], mac_labels[6], mac_labels[7], ip_labels[0], ip_labels[1], 
                       ip_labels[2], ip_labels[3], ip_labels[4], ip_labels[5], ip_labels[6], 
                       ip_labels[7], 'request_n', 'fc', 'error', 'address', 'data']
    
    # Assigning features to X and labels to y
    X = electra_s7[feature_columns]
    y = electra_s7['label_n']

    return X, y

# ----- Training Set Loop -----
# Loops over 60% of the S7 Comm dataset that was split into 61 chunks of 6,400,000 lines each
for i in range(36):
    # Reads in the file chunk as pandas dataframe
    filename = f"electra_s7comm-{i}.csv"
    electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)
    
    train_X, train_y = parse_data(electra_s7)
    
    # Training the Random Forest model
    randomforest_model.fit(train_X, train_y)
    
    # Deleting the chunk to save RAM
    del electra_s7

# ------- Validation Set Loop -------
# Loops over 10% of the S7 Comm dataset that was split into 61 chunks of 6,400,000 lines each
validation_chunk_len = 6 
validation_total_accuracy = 0

for i in range(36,42):
    # Reads in the file chunk as pandas dataframe
    filename = f"electra_s7comm-{i}.csv"
    electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)
    
    # Assigning features to X and labels to y
    validation_X, validation_y = parse_data(electra_s7)
    
    # Getting predictions for the validation chunk
    validation_predictions = randomforest_model.predict(validation_X)

    # Calculating the accuracy of the chunk and adding it to the total accuracy
    chunk_accuracy = accuracy_score(validation_y, validation_predictions)
    validation_total_accuracy += chunk_accuracy
    
    # Deleting the chunk to save RAM
    del electra_s7

# Printing the total validation set accuracy
validation_accuracy = (validation_total_accuracy / validation_chunk_len) * 100
print(f"Random Forest S7 Comm Validation Accuracy: {validation_accuracy:.4f}")


# ----- Test Set Loop -----
# Loops over 15% of the S7 Comm dataset that was split into 61 chunks of 6,400,000 lines each
test_chunk_len = 9
test_total_accuracy = 0

for i in range(42,51):
    # Reads in the file chunk as pandas dataframe
    filename = f"electra_s7comm-{i}.csv"
    electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)
    
    test_X, test_y = parse_data(electra_s7)
    
    # Getting predictions for the test chunk
    test_predictions = randomforest_model.predict(test_X)

    # Calculating the accuracy of the chunk and adding it to the total accuracy
    chunk_accuracy = accuracy_score(test_y, test_predictions)
    test_total_accuracy += chunk_accuracy
    
    # Deleting the chunk to save RAM
    del electra_s7
    
# Printing the total test set accuracy
test_accuracy = (test_total_accuracy / test_chunk_len) * 100
print(f"Random Forest S7 Comm Test Accuracy: {test_accuracy:.4f}")


# ----- Challenge Set Loop -----
# Loops over 15% of the S7 Comm dataset that was split into 61 chunks of 6,400,000 lines each
challenge_chunk_len = 10
challenge_total_accuracy = 0

for i in range(51,61):
    # Reads in the file chunk as pandas dataframe
    filename = f"electra_s7comm-{i}.csv"
    electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)
    
    challenge_X, challenge_y = parse_data(electra_s7)
    
    # Getting predictions for the challenge chunk
    challenge_predictions = randomforest_model.predict(challenge_X)

    # Calculating the accuracy of the chunk and adding it to the total accuracy
    chunk_accuracy = accuracy_score(challenge_y, challenge_predictions)
    challenge_total_accuracy += chunk_accuracy

    # Deleting the chunk to save RAM
    del electra_s7
    
# Printing the total challenge set accuracy
challenge_accuracy = (challenge_total_accuracy / challenge_chunk_len) * 100
print(f"Random Forest S7 Comm Challenge Accuracy: {challenge_accuracy:.4f}")