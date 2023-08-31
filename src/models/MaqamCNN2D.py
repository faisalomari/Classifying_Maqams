import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm

class MaqamCNN2D(nn.Module):
    def __init__(self):
        super(MaqamCNN2D, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=2048, kernel_size=1, stride=7, padding=2)
        self.bn1 = nn.BatchNorm2d(2048)
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=7)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=1, stride=7, padding=2)
        self.bn2 = nn.BatchNorm2d(4096)
        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=7)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(8192, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(4096, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout6 = nn.Dropout(p=0.2)

        self.fc5 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.dropout7 = nn.Dropout(p=0.2)

        self.output_layer = nn.Linear(64, 8)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        x = self.fc3(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        
        x = self.fc4(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.fc5(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.dropout7(x)

        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return x
