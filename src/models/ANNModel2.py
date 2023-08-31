import torch.nn as nn
import torch.nn.functional as F

class ANNModel2(nn.Module):
    def __init__(self):
        super(ANNModel2, self).__init__()

        self.fc1 = nn.Linear(51680, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=0.25)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(p=0.25)
        
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(p=0.1)
        
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(p=0.1)
        
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(p=0)
        
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.dropout6 = nn.Dropout(p=0)

        self.output_layer = nn.Linear(64, 8)

    def forward(self, x):
        # Flatten the input data to [batch_size, 30 * 469]
        x = x.view(x.size(0), -1)
        x = x.float()
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return x