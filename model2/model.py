import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class MaqamClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MaqamClassifier, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return F.softmax(x, dim=1)
