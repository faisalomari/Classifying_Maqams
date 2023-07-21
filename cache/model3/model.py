import torch.nn as nn
import torch
import numpy as np
class MaqamCNN(nn.Module):
    def __init__(self):
        super(MaqamCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.dropout4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool1d(kernel_size=2)
        self.dropout5 = nn.Dropout(p=0.2)

        self.conv6 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool1d(kernel_size=2)
        self.dropout6 = nn.Dropout(p=0.2)

        self.conv7 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool1d(kernel_size=2)
        self.dropout7 = nn.Dropout(p=0.2)
        
        self.conv8 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU()
        self.pool8 = nn.MaxPool1d(kernel_size=2)
        self.dropout8 = nn.Dropout(p=0.2)

        self.conv9 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU()
        self.pool9 = nn.MaxPool1d(kernel_size=2)
        self.dropout9 = nn.Dropout(p=0.2)

        self.conv10 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU()
        self.pool10 = nn.MaxPool1d(kernel_size=2)
        self.dropout10 = nn.Dropout(p=0.2)
        
        self.conv11 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.pool11 = nn.MaxPool1d(kernel_size=2)
        self.dropout11 = nn.Dropout(p=0.2)

        # self.conv12 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        # self.relu12 = nn.ReLU()
        # self.pool12 = nn.MaxPool1d(kernel_size=2)
        # self.dropout12 = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(312, 265)
        self.dropout13 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(265, 128)
        self.dropout14 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(128, 64)
        self.dropout15 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.squeeze(x, 3)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        x = self.dropout6(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.pool7(x)
        x = self.dropout7(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.pool8(x)
        x = self.dropout8(x)

        x = self.conv9(x)
        x = self.relu9(x)
        x = self.pool9(x)
        x = self.dropout9(x)

        x = self.conv10(x)
        x = self.relu10(x)
        x = self.pool10(x)
        x = self.dropout10(x)

        x = self.conv11(x)
        x = self.relu11(x)
        x = self.pool11(x)
        x = self.dropout11(x)

        # x = self.conv12(x)
        # x = self.relu12(x)
        # x = self.pool12(x)
        # x = self.dropout12(x)

        x = x.view(x.size(0), -1)   

        x = self.fc1(x)
        x = self.dropout13(x)

        x = self.fc2(x)
        x = self.dropout14(x)

        x = self.fc3(x)
        x = self.dropout15(x)
        return x
