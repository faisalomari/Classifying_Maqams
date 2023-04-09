import torch.nn as nn

class MaqamCNN(nn.Module):
    def __init__(self):
        super(MaqamCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.fc1 = nn.Linear(32*90000, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        print(x.shape)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        print("x shape = ", x.shape)
        return x
