import torch
import torch.nn as nn
import torch.nn.functional as F

class MaqamClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(MaqamClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        # Compute the shape of the output of the conv layers, based on the input shape
        self._to_linear = None
        self._compute_conv_output_shape()
        
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def _compute_conv_output_shape(self):
        # Create a dummy tensor with the input shape
        x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        
        # Pass it through the conv layers and compute the output shape
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
