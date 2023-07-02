import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MaqamDataset
from model import MaqamClassifier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set data directories and hyperparameters
data_dir = r"C:\Users\omari\Documents\GitHub\smalldataset"
maqam = "Kurd"
num_classes = 1
batch_size = 32
learning_rate = 1e-3
num_epochs = 10

# Initialize dataset and data loader
dataset = MaqamDataset(data_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = MaqamClassifier(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in tqdm(enumerate(loader), total=len(loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

# Save the trained model
model_path = os.path.join("models", f"maqam_{maqam}.pt")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
