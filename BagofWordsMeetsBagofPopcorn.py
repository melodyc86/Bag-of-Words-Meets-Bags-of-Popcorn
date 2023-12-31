# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

"""# Load the train and test data"""

train_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/testData.tsv', header=0, delimiter="\t", quoting=3)

"""# Split the training and validation sets"""

train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

"""#Use CountVectorizer & Convert labels to tensors"""

# Convert the text data into numerical vectors using CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(train_data['review'])
train_vectors = vectorizer.transform(train_data['review'])
val_vectors = vectorizer.transform(val_data['review'])
test_vectors = vectorizer.transform(test_data['review'])

# Convert the labels to PyTorch tensors
train_labels = torch.tensor(train_data['sentiment'].values)
val_labels = torch.tensor(val_data['sentiment'].values)

"""#Initalize TextDataset & convert into objects"""

# Convert the data into PyTorch Dataset and DataLoader objects
class TextDataset(Dataset):
    def __init__(self, vectors, labels=None):
        self.vectors = vectors
        self.labels = labels

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.vectors[index].toarray()[0], dtype=torch.float32)
        if self.labels is not None:
            y = torch.as_tensor(self.labels[index], dtype=torch.long)
            return x, y
        else:
            return x

train_dataset = TextDataset(train_vectors, train_labels)
val_dataset = TextDataset(val_vectors, val_labels)
test_dataset = TextDataset(test_vectors)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

"""#Define neural network model <3"""

# Define the neural network model
class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

"""#Train the model"""

hiddenSize = 500 #set back to 256

model = TextClassifier(train_vectors.shape[1], hiddenSize, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1): #changed this back to 10 later
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validate the model
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('Epoch [{}/{}], Validation Accuracy: {:.2f}%'.format(epoch+1, 10, accuracy))

"""Test data predictions"""

import torch.nn.functional as F

# Make predictions on the test data
with torch.no_grad():
    predicted_probs = []
    for inputs in test_loader:
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        predicted_probs.extend(probs[:, 1].cpu().numpy()) # Use the probability of positive sentiment as the prediction

# Convert the predicted probabilities to binary labels
predicted_labels = [1 if p >= 0.5 else 0 for p in predicted_probs]

# Remove double quotes from the 'id' column
test_data['id'] = test_data['id'].str.replace('"', '')
# Save the predictions to a CSV file
submission_df = pd.DataFrame({'id': test_data['id'], 'sentiment': predicted_probs})
submission_df.to_csv('submission10.csv', index=False)
