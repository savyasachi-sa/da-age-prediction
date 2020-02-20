import torch.nn as nn
import torchvision.models as models

class BaselineResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False #TODO: shouldn't involve fc layer??
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 1, bias=True),
        )

    def forward(self, x):
        print('in forward now')
        return self.resnet.forward(x)
