import torch
import torch.nn as nn
import torch.utils.data as data
import dataset
import model
import torch.nn.functional as F

def collate_fn(batch):
    waveforms = [sample[0] for sample in batch]
    labels = [sample[1] for sample in batch]
    waveforms_padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    return waveforms_padded, torch.tensor(labels)

# Define hyperparameters
batch_size = 8

# Load the dataset
test_dataset = dataset.MaqamDataset(mode='test')

# Define the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.MaqamCNN().to(device)
model.load_state_dict(torch.load('maqam_cnn2.pth'))

# Test the model
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.unsqueeze(1).unsqueeze(3).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Accuracy of the network on the test dataset: %d %%' % accuracy)
