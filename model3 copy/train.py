import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import dataset
import model
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from dataset import MaqamDataset
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def pad_to_max_length(self, max_length):
    for i in range(len(self)):
        padded_data = F.pad(self.data[i][0], (0, max_length - len(self.data[i][0])), 'constant', 0)
        padded_data = padded_data.unsqueeze(0) if len(padded_data.shape) == 1 else padded_data
        padded_data = padded_data.unsqueeze(1)
        padded_data = padded_data.repeat(1, 32, 1, 1)
        self.data[i] = (padded_data, self.data[i][1])

def MFCC_plot(mfcc):
        plt.figure(figsize=(10, 4))
        mfcc = mfcc.detach().numpy()
        mfcc = mfcc.mean(axis=2).T
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()

def custom_collate(batch):
    inputs, labels, mfcc = zip(*batch)
    inputs = [F.pad(input, (0, max_length - input.size(0)), 'constant', 0) for input in inputs]
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels)
    return inputs, labels, mfcc

#clean GPU torch cache
torch.cuda.empty_cache()
# Define hyperparameters
batch_size = 4 # should be 64 according to page 7
learning_rate = 0.0001 #page 7 0.0001
num_epochs = 10 #should be 35 according to page 7

# Load the dataset
train_dataset = dataset.MaqamDataset(mode='train')


# Find the maximum length of the input tensors
# print("Finding the maximum length of the input tensors!")
# max_length = max(inputs.shape[0] for inputs, _, _ in train_dataset)
max_length = 1440000
# print("Found the maximum length of the input tensors! = ", max_length)

# Pad all input tensors to the maximum length
# MaqamDataset.pad_to_max_length(1440000)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Define the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.MaqamCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(torch.cuda.is_available())
# Train the model
print("Starting training!")
yesy = 0
counter = 0
while(yesy == 0 or input("Do You Want To Continue? [y,n]\n") == "y"):
    yesy = 1
    for epoch in range(num_epochs):
        counter +=1
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels, mfcc = data
            # MFCC_plot(mfcc)
            labels = labels.to(device)
            inputs = inputs.unsqueeze(1).unsqueeze(3).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_size1 = outputs.size(0)
            padding_size = max_length - outputs.size(1)
            padding = torch.zeros(batch_size1, padding_size).to(device)
            padded_outputs = torch.cat((outputs, padding), dim=1)
            loss = criterion(padded_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d, loss: %.3f' % (counter, running_loss / len(train_loader)))

# Save the model
torch.save(model.state_dict(), 'maqam_cnn5.pth')