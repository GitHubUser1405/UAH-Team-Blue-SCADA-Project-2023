# Project: Detecting Cyber-Attacks on SCADA Systems using a Transformer Neural Network 
# and Naive Bayes on the Electra Railway Dataset
# Authors: Aditi Shah, Christian Martin, Colby Tyree, Jessica Elkins
# Leojaris Brujan, Mary Scholl, Tatiana Kontoulakos
# Class: UAH IS 692 Spring 2023
# Description: This program uses the Scikit-Learn's Gaussian Naive Bayes
# to create a GNB model based on the Electra S7 Comm dataset.

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Initialize the model
model = GaussianNB()

dtype_dict = {"Time": 'Int64', "smac": str, "dmac": str, "sip": str, "dip": str, "request": bool, "fc": int,
              "error": int, "address": int, "data": int, "label": str}

def parse_data(electra_s7):
    # Replace the MITM label with the normal label
    electra_s7 = electra_s7.replace('MITM_UNALTERED', 'NORMAL')
    
    # Initialize the Label Encoder functions
    le = LabelEncoder()

    # Label encoding the label column
    electra_s7['label_n'] = le.fit_transform(electra_s7['label'])
    
    feature_columns = ['Time', 'smac', 'dmac', 'sip', 'dip', 'request', 'fc', 'error', 'address', 'data']
    
    # Assigning features to X and labels to y
    X = electra_s7[feature_columns]
    y = electra_s7['label_n']
    
    # Turning the training dataframe to a list
    X_list = X.values.tolist()

    # Creating an empty list to store the new rows in
    new_X_list = []
    
    # Looping through each row in the dataset and joining the elements together
    for row in X_list:
        new_row = " ".join(str(v) for v in row)
        new_X_list.append(new_row)
       
    # Tokenizing each row and encoding it into a word vector
    token_docs = [doc.split() for doc in new_X_list]
    all_tokens = set([word for sentence in token_docs for word in sentence])
    word_to_idx = {token:idx+1 for idx, token in enumerate(all_tokens)}
    
    encoded_array = np.array([[word_to_idx[token] for token in token_doc] for token_doc in token_docs], dtype=object)

    # Turning the encoded array to a pandas dataframe
    X_df = pd.DataFrame(encoded_array)

    return X_df, y


# ----- Training Set Loop -----
# Loops over 60% of the S7 Comm dataset that was split into 61 chunks of 6,400,000 lines each
for i in range(36):
    # Reads in the file chunk as pandas dataframe
    filename = f"electra_s7comm-{i}.csv"
    electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)

    # Parse the dataset into X and y
    train_X, train_y = parse_data(electra_s7)
    
    # Training the model
    model.fit(train_X, train_y)
    
    # Deleting the chunk to save RAM
    del electra_s7


# ----- Validation Set Loop -----
# Loops over 10% of the S7 Comm dataset that was split into 61 chunks of 6,400,000 lines each
validation_chunk_length = 6
validation_total_accuracy = 0

for i in range(36, 42):
    # Reads in the file chunk as pandas dataframe
    filename = f"electra_s7comm-{i}.csv"
    electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)

    # Parse the dataset into X and y
    validation_X, validation_y = parse_data(electra_s7)
    
    # Getting predictions on the validation chunk
    validation_predictions = model.predict(validation_X)

    # Calculating the accuracy of the chunk and adding it to the total accuracy
    chunk_accuracy = accuracy_score(validation_y, validation_predictions)
    validation_total_accuracy += chunk_accuracy
    
    # Deleting the chunk to save RAM
    del electra_s7
    
# Printing the total validation set accuracy
validation_accuracy = (validation_total_accuracy / validation_chunk_length) * 100
print(f"GNB S7 Comm Validation Accuracy: {validation_accuracy:.4f}")


# ----- Test Set Loop -----
# Loops over 15% of the S7 Comm dataset that was split into 61 chunks of 6,400,000 lines each
test_chunk_length = 9
test_total_accuracy = 0

for i in range(42, 51):
    # Reads in the file chunk as pandas dataframe
    filename = f"electra_s7comm-{i}.csv"
    electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)

    # Parse the dataset into X and y
    test_X, test_y = parse_data(electra_s7)
    
    # Getting predictions on the test chunk
    test_predictions = model.predict(test_X)

    # Calculating the accuracy of the chunk and adding it to the total accuracy
    chunk_accuracy = accuracy_score(test_y, test_predictions)
    test_total_accuracy += chunk_accuracy
    
    # Deleting the chunk to save RAM
    del electra_s7
    
# Printing the total test set accuracy
test_accuracy = (test_total_accuracy / test_chunk_length) * 100
print(f"GNB S7 Comm Test Accuracy: {test_accuracy:.4f}")


# ----- Challenge Set Loop -----
# Loops over 15% of the S7 Comm dataset that was split into 61 chunks of 6,400,000 lines each
challenge_chunk_length = 10
challenge_total_accuracy = 0

for i in range(51, 61):
    # Reads in the file chunk as pandas dataframe
    filename = f"electra_s7comm-{i}.csv"
    electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)

    # Parse the dataset into X and y
    challenge_X, challenge_y = parse_data(electra_s7)
    
    # Getting predictions on the challenge chunk
    challenge_predictions = model.predict(challenge_X)

    # Calculating the accuracy of the chunk and adding it to the total accuracy
    chunk_accuracy = accuracy_score(challenge_y, challenge_predictions)
    challenge_total_accuracy += chunk_accuracy
    
    # Deleting the chunk to save RAM
    del electra_s7
    
# Printing the total test set accuracy
challenge_accuracy = (challenge_total_accuracy / challenge_chunk_length) * 100
print(f"GNB S7 Comm Challenge Accuracy: {challenge_accuracy:.4f}")
