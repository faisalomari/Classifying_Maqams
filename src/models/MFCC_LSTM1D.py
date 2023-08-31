import torch.nn as nn
import torch.nn.functional as F

class MFCC_LSTM1D(nn.Module):
    def __init__(self):
        super(MFCC_LSTM1D, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=12, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(64, 256)
        self.dropout4 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(256, 1024)
        self.dropout5 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(1024, 256)
        self.dropout6 = nn.Dropout(p=0)

        self.fc4 = nn.Linear(256, 8)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x = x[:,-1,:]
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout6(x)

        
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return x
