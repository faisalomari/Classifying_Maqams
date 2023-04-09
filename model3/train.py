import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import dataset
import model
import torch.nn.functional as F

def pad_to_max_length(self, max_length):
    for i in range(len(self)):
        padded_data = F.pad(self.data[i][0], (0, max_length - len(self.data[i][0])), 'constant', 0)
        padded_data = padded_data.unsqueeze(0) if len(padded_data.shape) == 1 else padded_data
        padded_data = padded_data.unsqueeze(1)
        padded_data = padded_data.repeat(1, 32, 1, 1)
        self.data[i] = (padded_data, self.data[i][1])


# Define hyperparameters
batch_size = 2
learning_rate = 0.001
num_epochs = 10

# Load the dataset
train_dataset = dataset.MaqamDataset(mode='train')

# Find the maximum length of the input tensors
max_length = 0
for i in range(len(train_dataset)):
    inputs, labels = train_dataset[i]
    if inputs.shape[0] > max_length:
        max_length = inputs.shape[0]

# Pad all input tensors to the maximum length
train_dataset.pad_to_max_length(1440000)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.MaqamCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        print("in process number ", i)
        inputs, labels = data
        print(inputs.shape)
        inputs = inputs.unsqueeze(1).unsqueeze(3)
        optimizer.zero_grad()
        outputs = model(inputs)
        print("Outputs shape = ", outputs.shape)
        batch_size1 = outputs.size(0)
        padding_size = max_length - outputs.size(1)
        padding = torch.zeros(batch_size1, padding_size).to(device)
        padded_outputs = torch.cat((outputs, padding), dim=1)
        loss = criterion(padded_outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

# Save the model
torch.save(model.state_dict(), 'maqam_cnn.pth')