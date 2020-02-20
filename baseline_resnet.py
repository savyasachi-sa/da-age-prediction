import torch.nn as nn
import torchvision.models as models


class BaselineResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 1, bias=True),
        )

    def forward(self, x):
        return self.resnet.forward(x)
