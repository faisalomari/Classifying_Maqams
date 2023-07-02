import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MaqamDataset
from model import MaqamClassifier

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)

    print('Train Epoch: {} Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        epoch, train_loss, correct, len(train_loader.dataset), train_acc))

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 32

    # Load the dataset
    train_dataset = MaqamDataset(root_dir=r"C:\Users\omari\Documents\GitHub\smalldataset")
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    # Initialize the model
    model = MaqamClassifier(num_classes=len(train_dataset.classes)).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(1, num_epochs+1):
        train(model, device, train_loader, optimizer, epoch)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

if __name__ == '__main__':
    main()
