import torch
from tqdm import tqdm  # Import tqdm for the progress bar

# Assuming you have initialized num_epochs, model, criterion, optimizer, and train_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model for a specified number of epochs
num_epochs = 45
print("Starting training")

for epoch in range(num_epochs):
    # Set the model to training mode for the current epoch
    model.train()

    # Create a progress bar for the train_loader loop
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

    # Initialize variables to track the loss and number of correct predictions
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, data in enumerate(train_loader_tqdm):
        inputs, targets = data  # MFCCs and labels
        targets = targets.to(device)
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update the loss and accuracy metrics
        running_loss += loss.item()
        _, predicted_labels = torch.max(outputs, 1)
        correct_predictions += (predicted_labels == targets).sum().item()
        total_samples += len(targets)

        # Update the progress bar description
        train_loader_tqdm.set_postfix({'Loss': running_loss / (i + 1), 'Accuracy': 100 * correct_predictions / total_samples})

    # Calculate and print average loss and accuracy for the current epoch
    avg_loss = running_loss / len(train_loader)
    avg_accuracy = 100 * correct_predictions / total_samples
    print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss={avg_loss:.5f}, Train Accuracy={avg_accuracy:.5f}%')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
