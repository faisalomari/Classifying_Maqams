import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(3,3), padding="same")
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,1))
        self.dropout1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), padding="valid")
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,1))
        self.dropout2 = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.dropout6 = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(32, 8)
        self.dropout5 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Reshape the CNN output to match LSTM input
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, sequence_length, input_size)

        x, _ = self.lstm(x)

        x = self.fc1(x[:, -1, :])
        
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        
        x = F.relu(self.fc3(x))
        # x = self.dropout4(x)

        # x = self.fc3(x)
        # x = self.dropout5(x)
        x = F.softmax(x, dim=1)
        return x
