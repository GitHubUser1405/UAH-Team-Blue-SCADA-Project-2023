# Project: Detecting Cyber-Attacks on SCADA Systems using a Transformer Neural Network 
# and Naive Bayes on the Electra Railway Dataset
# Authors: Aditi Shah, Christian Martin, Colby Tyree, Jessica Elkins
# Leojaris Brujan, Mary Scholl, Tatiana Kontoulakos
# Class: UAH IS 692 Spring 2023
# Description: This program uses the Scikit-Learn's Gaussian Naive Bayes
# to create a GNB model based on the Electra Modbus dataset.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dtype_dict = {"Time": 'Int64', "smac": str, "dmac": str, "sip": str, "dip": str, "request": bool, "fc": int,
              "error": int, "address": int, "data": int, "label": str}

# Import the dataset into a pandas dataframe
electra_modbus = pd.read_csv(r'C:\Users\jessi\Downloads\electra_modbus.csv', header=0, dtype=dtype_dict)

# Replacing the Man In The Middle label with the Normal label
electra_modbus = electra_modbus.replace('MITM_UNALTERED', 'NORMAL')

# Intialize the label encoder & encoding the label column
le = LabelEncoder()
electra_modbus['label_n'] = le.fit_transform(electra_modbus['label'])

feature_columns = ['Time', 'smac', 'dmac', 'sip', 'dip', 'request', 'fc', 'error', 'address', 'data']
electra_modbus_features = electra_modbus[feature_columns]
electra_modbus_y = electra_modbus['label_n']

# Turning the electra modbus dataframe to a list
electra_modbus_list = electra_modbus_features.values.tolist() 

# creating a list to store the new rows on
data_list = []

# Going through each row in the dataset and joining the elements together
for row in electra_modbus_list:
    new_row = " ".join(str(v) for v in row)
    data_list.append(new_row)

# Tokenizing each row and encoding it into a word vector
token_docs = [doc.split() for doc in data_list]
all_tokens = set([word for sentence in token_docs for word in sentence])
word_to_idx = {token:idx+1 for idx, token in enumerate(all_tokens)}

encoded_array = np.array([[word_to_idx[token] for token in token_doc] for token_doc in token_docs], dtype=object)

# Turning the encoded array to a pandas dataframe
encoded_df = pd.DataFrame(encoded_array, columns = feature_columns)

# Split the dataset into 60% training, 10% validation, 15% test, and 15% challenge.
train_size = 0.60
validation_size = 0.10
test_size = 0.15
challenge_size = 0.15

# Calculating the indexes to split the df at
train_index_modbus = int(len(encoded_df) * train_size)
valid_index_modbus = int(len(encoded_df) * validation_size)
test_index_modbus = int(len(encoded_df) * test_size)
challenge_index_modbus = int(len(encoded_df) * challenge_size)

# Slicing the dataset based on the indexes
train_X = encoded_df[0:train_index_modbus]
validation_X = encoded_df[train_index_modbus:train_index_modbus+valid_index_modbus]
test_X = encoded_df[train_index_modbus+valid_index_modbus:train_index_modbus+valid_index_modbus+test_index_modbus]
challenge_X = encoded_df[train_index_modbus+valid_index_modbus+test_index_modbus:]

train_y = electra_modbus_y[0:train_index_modbus]
validation_y = electra_modbus_y[train_index_modbus:train_index_modbus+valid_index_modbus]
test_y = electra_modbus_y[train_index_modbus+valid_index_modbus:train_index_modbus+valid_index_modbus+test_index_modbus]
challenge_y = electra_modbus_y[train_index_modbus+valid_index_modbus+test_index_modbus:]

# Setting the seed with a value of 13
np.random.seed(13)

# Initializing the Gaussian Naive Bayes model
model = GaussianNB()

# Training the model
model.fit(train_X, train_y)

# Getting predictions for the validaton set
y_pred = model.predict(validation_X)

# Calculating validation set accuracy
validation_accuracy = accuracy_score(validation_y, y_pred) * 100
print(f'GNB Modbus Validation Accuracy: {validation_accuracy:.4f}')

# Getting predictions for the test set
test_pred = model.predict(test_X)

# Calculating test set accuracy
test_accuracy = accuracy_score(test_y, test_pred) * 100
print(f'GNB Modbus Test Accuracy: {test_accuracy:.4f}')

# Getting predictions for the challenge set
challenge_pred = model.predict(challenge_X)

# Calculating the challenge set accuracy
challenge_accuracy = accuracy_score(challenge_y, challenge_pred) * 100
print(f'GNB Modbus Challenge Accuracy: {challenge_accuracy:.4f}')