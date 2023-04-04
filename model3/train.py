import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import dataset
import model

# Define hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Load the dataset
train_dataset = dataset.MaqamDataset(mode='train')
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = model.MaqamCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.unsqueeze(1).unsqueeze(3)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

# Save the model
torch.save(model.state_dict(), 'maqam_cnn.pth')
