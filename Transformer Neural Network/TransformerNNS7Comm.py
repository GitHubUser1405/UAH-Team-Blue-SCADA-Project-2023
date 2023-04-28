# Project: Detecting Cyber-Attacks on SCADA Systems using a Transformer Neural Network 
# and Naive Bayes on the Electra Railway Dataset
# Authors: Aditi Shah, Christian Martin, Colby Tyree, Jessica Elkins
# Leojaris Brujan, Mary Scholl, Tatiana Kontoulakos
# Class: UAH IS 692 Spring 2023
# Description: This program uses PyTorch to create a Transformer Neural Network that does
# multi-class classification based on the Electra S7 Comm dataset.

import torch
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

dtype_dict = {"Time": 'Int64', "smac": str, "dmac": str, "sip": str, "dip": str, "request": bool, "fc": int,
              "error": int, "address": int, "data": int, "label": str}

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Device: " + device)

# Define hyperparameters
input_size = 40688 # size of the input vocabulary
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
        x = x.mean(dim=1) 
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

# Initialize the model
model = TransformerModel(input_size, hidden_size, num_classes, num_layers, num_heads, dropout)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def parse_data(electra_s7):
    electra_s7 = electra_s7.replace('MITM_UNALTERED', 'NORMAL')
    le = LabelEncoder()
    electra_s7['label_n'] = le.fit_transform(electra_s7['label'])

    electra_s7['smac'] = electra_s7['smac'].str.replace(':', '')
    electra_s7['dmac'] = electra_s7['dmac'].str.replace(':', '')
    electra_s7['sip'] = electra_s7['sip'].str.replace('.', '')
    electra_s7['dip'] = electra_s7['dip'].str.replace('.', '')
    
    feature_columns = ['smac', 'dmac', 'sip', 'dip', 'request', 'fc', 'error', 'address', 'data']
    electra_X = electra_s7[feature_columns]
    electra_y = electra_s7['label_n']

    labels_list = electra_y.values.tolist()
    labels = torch.tensor(labels_list)

    # Turn Dataframe into List
    df_list = electra_X.values.tolist()

    data_list = []

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
    
    dataset = MultiClassDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    return data_loader

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, dim=1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def train():
    # Loop that imports one chunk of the s7 comm, encodes the data, trains on it, and then frees it from memory
    for i in range(36):
        filename = f"electra_s7comm-{i}.csv"
        electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)
    
        # Parse the data
        data_loader = parse_data(electra_s7)

        # Train the model
        for epoch in range(10):
            for batch_idx, (data, labels) in enumerate(data_loader):
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                if batch_idx % 100 == 0:
                    print('Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch+1, batch_idx, loss.item()))
        
            # Save the trained model's state dictionary
            model_path = 'transformer_neural_network_s7.pth'
            torch.save(model.state_dict(), model_path) 
    
        # Delete the chunk to save RAM
        del electra_s7

def validation():
    total_accuracy = 0
    total_len = 0

    # Loading the trained model's state dictionary
    model.load_state_dict(torch.load('transformer_neural_network_s7.pth'))

    # Puts the model in evaluation mode
    model.eval()

    for i in range(36, 42):
        filename = f"electra_s7comm-{i}.csv"
        electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)
    
        validation_data_loader = parse_data(electra_s7)
        total_len += len(validation_data_loader)
    
        with torch.no_grad():
            for inputs, labels in validation_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
        
                # forward pass
                outputs = model(inputs)
        
                #calculate accuracy
                batch_accuracy = calculate_accuracy(outputs, labels)
                total_accuracy += batch_accuracy
        
        del electra_s7
    
    # Calculate Validation Set accuracy
    validation_accuracy = (total_accuracy / total_len) * 100
    print(f"TNN S7 Comm Validation Accuracy: {validation_accuracy:.4f}")  

def test():
    total_accuracy = 0
    total_len = 0

    # Loading the trained model's state dictionary
    model.load_state_dict(torch.load('transformer_neural_network_s7.pth'))

    # Puts the model in evaluation mode
    model.eval()

    for i in range(42, 51):
        filename = f"electra_s7comm-{i}.csv"
        electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)
    
        test_data_loader = parse_data(electra_s7)
        total_len += len(test_data_loader)
    
        with torch.no_grad():
            for inputs, labels in test_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
        
                # forward pass
                outputs = model(inputs)
        
                #calculate accuracy
                batch_accuracy = calculate_accuracy(outputs, labels)
                total_accuracy += batch_accuracy
        
        del electra_s7
    
    # Calculate Test Set accuracy
    test_accuracy = (total_accuracy / total_len) * 100
    print(f"TNN S7 Comm Test Accuracy: {test_accuracy:.4f}")

def challenge():
    total_accuracy = 0
    total_len = 0

    # Loading the trained model's state dictionary
    model.load_state_dict(torch.load('transformer_neural_network_s7.pth'))

    # Puts the model in evaluation mode
    model.eval()

    for i in range(51, 61):
        filename = f"C:/Users/jessi/Downloads/electra_s7comm-{i}.csv"
        electra_s7 = pd.read_csv(filename, header=0, dtype=dtype_dict)
    
        challenge_data_loader = parse_data(electra_s7)
        total_len += len(challenge_data_loader)
    
        with torch.no_grad():
            for inputs, labels in challenge_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
        
                # forward pass
                outputs = model(inputs)
        
                #calculate accuracy
                batch_accuracy = calculate_accuracy(outputs, labels)
                total_accuracy += batch_accuracy
        
        del electra_s7
    
    # Calculate Challenge Set accuracy
    challenge_accuracy = (total_accuracy / total_len) * 100
    print(f"TNN S7 Comm Challenge Accuracy: {challenge_accuracy:.4f}")

if user_option1 == 'train':
    train()
elif user_option1 == 'test':
    validation()
    test()
    challenge()