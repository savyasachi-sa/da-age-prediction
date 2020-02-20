import torch.nn as nn
import torchvision.models as models

class BaselineResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False #TODO: shouldn't involve fc layer??
        self.resnet.fc = nn.Linear(512, 1, bias=True)

    def forward(self, x):
        print('in forward now')
        return self.resnet.forward(x)
