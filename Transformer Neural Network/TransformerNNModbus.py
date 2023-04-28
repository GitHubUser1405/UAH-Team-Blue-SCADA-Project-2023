# Project: Detecting Cyber-Attacks on SCADA Systems using a Transformer Neural Network 
# and Naive Bayes on the Electra Railway Dataset
# Authors: Aditi Shah, Christian Martin, Colby Tyree, Jessica Elkins
# Leojaris Brujan, Mary Scholl, Tatiana Kontoulakos
# Class: UAH IS 692 Spring 2023
# Description: This program uses PyTorch to create a Transformer Neural Network that does
# multi-class classification based on the Electra Modbus dataset.

import torch  # torch 1.9.0+cu111 - Python 3.9.10
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer  # torchtext 0.10.0
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

# Asking the user if they want to train or test the model
while(True):
    user_option1 = input('Do you want to train or test the model? \n').strip().lower()

    if (user_option1 == 'test'):
        break
    elif (user_option1 == 'train'):
        break
    else:
        print("Error. Please type 'test' or 'train'.")

print("Importing the dataset...")
dtype_dict = {"Time": 'Int64', "smac": str, "dmac": str, "sip": str, "dip": str, "request": bool, "fc": int,
              "error": int, "address": int, "data": int, "label": str}

# Import the dataset as a pandas dataframe
electra_modbus = pd.read_csv('electra_modbus.csv', header=0, dtype=dtype_dict)

# # Sort the data based on time
electra_modbus.sort_values(by='Time', ascending=True, inplace=True)

# Replacing the Man In The Middle label with the Normal label
electra_modbus = electra_modbus.replace('MITM_UNALTERED', 'NORMAL')

# Initializing the label encoder
le = LabelEncoder()

# Label Encoding the label column
electra_modbus['label_n'] = le.fit_transform(electra_modbus['label'])

# Formatting the MAC and IP columns
electra_modbus['smac'] = electra_modbus['smac'].str.replace(':', '')
electra_modbus['dmac'] = electra_modbus['dmac'].str.replace(':', '')
electra_modbus['sip'] = electra_modbus['sip'].str.replace('.', '')
electra_modbus['dip'] = electra_modbus['dip'].str.replace('.', '')

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

feature_columns = ['smac', 'dmac', 'sip', 'dip', 'request', 'fc', 'error', 'address', 'data']

# Putting the features in X and the labels in y
train_X = train_df[feature_columns]
train_y = train_df['label_n']

validation_X = validation_df[feature_columns]
validation_y = validation_df['label_n']

test_X = test_df[feature_columns]
test_y = test_df['label_n']

challenge_X = challenge_df[feature_columns]
challenge_y = challenge_df['label_n']

# Function that takes in a dataframes, turns it to a list, tokenizes & encodes
# the data into a vocab, and returns an ecoded Torch tensor
def parse_data(df):
    # Turn Dataframe into List
    df_list = df.values.tolist()

    # Creating a list to store the new rows on
    data_list = []

    # Going through each row in the dataset and joining the elements together
    for row in df_list:
        new_row = " ".join(str(v) for v in row)
        data_list.append(new_row)

    # Define tokenizer
    tokenizer = get_tokenizer('basic_english')

    # Define datas
    data_iter = iter(data_list)

    # Build vocabulary from tokenized data
    vocab = build_vocab_from_iterator(map(tokenizer, data_iter))

    # Tokenize and encode data
    data_iter = iter(data_list)
    data = [torch.tensor([vocab[token] for token in tokenizer(header)], dtype=torch.long) for header in data_iter]

    # Pad sequences to the same length
    data = pad_sequence(data, batch_first=True, padding_value=0)
    
    return data

# Function that takes in the model's output and the
# actual labels and returns the accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, dim=1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Seeing if the GPU is available for computation
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Device: " + device)

# Define hyperparameters
input_size = 17906 # size of the input vocabulary
hidden_size = 128  # dimension of the hidden state
num_classes = 7    # number of classes
num_layers = 2     # number of transformer layers
num_heads = 8      # number of attention heads
dropout = 0.1      # dropout probability

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout), num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # aggregate the sequence representations to a fixed-length vector
        x = self.fc(x)
        return x

class MultiClassDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Initialize the TNN
model = TransformerModel(input_size, hidden_size, num_classes, num_layers, num_heads, dropout)

# Send the model to the GPU
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    # Parsing the training data into a torch tensor
    data = parse_data(train_X)

    # Turning the training labels into a Torch tensor
    labels_list = train_y.values.tolist()
    labels = torch.tensor(labels_list)

    # Create the dataset and data loader
    dataset = MultiClassDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Train the model
    for epoch in range(10):
        for batch_idx, (data, labels) in enumerate(data_loader):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            outputs = outputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if batch_idx % 100 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch+1, batch_idx, loss.item()))

    # Save the trained model's state dictionary
    model_path = 'transformer_neural_network_modbus.pth'
    torch.save(model.state_dict(), model_path)

def validation():
    # Loading the trained model's state dictionary
    model.load_state_dict(torch.load('transformer_neural_network_modbus.pth'))

    # Puts the model in evaluation mode
    model.eval()

    # Turning the X and y validation sets into tensors
    X_valid = parse_data(validation_X)
    validation_y_tensor = torch.tensor(validation_y.values)

    # Putting X and y on the GPU
    X_valid = X_valid.to(device)
    validation_y_tensor = validation_y_tensor.to(device)

    # Creating the MultiClassDataset object and passing it to the dataloader
    validation_set = MultiClassDataset(X_valid, validation_y_tensor)
    validation_loader = DataLoader(validation_set, batch_size=64, shuffle=False)

    total_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
        
            # Forward pass
            outputs = model(inputs)
        
            # Calculate batch accuracy
            batch_accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += batch_accuracy
        
        # Calculate Validation Set accuracy
        validation_accuracy = (total_accuracy / len(validation_loader)) * 100
        print(f"TNN Modbus Validation Accuracy: {validation_accuracy:.4f}")

def test():
    # Loading the trained model's state dictionary
    model.load_state_dict(torch.load('transformer_neural_network_modbus.pth'))

    # Puts the model in evaluation mode
    model.eval()

    # Turning the X and y test sets into tensors
    X_test = parse_data(test_X)
    test_y_tensor = torch.tensor(test_y.values)

    # Putting X and y on the GPU
    X_test = X_test.to(device)
    test_y_tensor = test_y_tensor.to(device)

    # Creating the MultiClassDataset object and passing it to the dataloader
    test_set = MultiClassDataset(X_test, test_y_tensor)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)

    total_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
        
            # Forward pass
            outputs = model(inputs)
        
            # Calculate accuracy
            batch_accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += batch_accuracy
        
        # Calculate Test Set accuracy
        test_accuracy = (total_accuracy / len(test_dataloader)) * 100
        print(f"TNN Modbus Test Accuracy: {test_accuracy:.4f}")


def challenge():
    # Loading the trained model's state dictionary
    model.load_state_dict(torch.load('transformer_neural_network_modbus.pth'))

    # Puts the model in evaluation mode
    model.eval()

    # Turning the X and y challenge sets into tensors
    X_challenge = parse_data(challenge_X)
    challenge_y_tensor = torch.tensor(challenge_y.values)

    # Putting X and y on the GPU
    X_challenge = X_challenge.to(device)
    challenge_y_tensor = challenge_y_tensor.to(device)

    # Creating the MultiClassDataset object and passing it to the dataloader
    challenge_set = MultiClassDataset(X_challenge, challenge_y_tensor)
    challenge_dataloader = DataLoader(challenge_set, batch_size=64, shuffle=False)

    total_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in challenge_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
        
            # Forward pass
            outputs = model(inputs)
        
            # Calculate batch accuracy
            batch_accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += batch_accuracy
        
        # calculate overall accuracy
        challenge_accuracy = (total_accuracy / len(challenge_dataloader)) * 100
        print(f"TNN Modbus Challenge Accuracy: {challenge_accuracy:.4f}")

if user_option1 == 'train':
    train()
elif user_option1 == 'test':
    validation()
    test()
    challenge()