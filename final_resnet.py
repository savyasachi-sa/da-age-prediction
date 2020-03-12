import torch.nn as nn
import torch
import torchvision.models as models
from nntools import NeuralNetwork
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


class FinalResnet(NeuralNetwork):
    def __init__(self, rank, pairwise, fine_tuning = True, loss = 'L1'):
        super().__init__()
        
        print('Parameterised ResNet with options : Pairwise {} \t Rank loss : {}'.format(pairwise, rank))
        
        self.rank = rank
        self.pairwise = pairwise
        self.reg_loss_type = loss
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*[i for i in self.resnet.children()][:-1])
        
        if self.pairwise:
            #modify the first layer of the model to take 6 channel input
            self.resnet_list = []
            self.resnet_list.append(nn.Conv2d(6, 64, kernel_size=(1, 1), stride=(1, 1), padding=(3, 3), bias=True))
            for j in [i for i in self.resnet.children()][1:]: #remove the top classifier layer
                self.resnet_list.append(j)

            self.resnet = nn.Sequential(*[i for i in self.resnet_list])
        
        
        self.linear_out = nn.Linear(128, 1, bias = True)
        if self.rank :
            #modify the output to give 2 outputs instead of the single regression output
            self.linear_out = nn.Linear(128, 2, bias=True)
            self.rank_loss = nn.BCELoss()
            self.sig = nn.Sigmoid()
        
        if self.reg_loss_type == 'L1':
            self.reg_loss = nn.SmoothL1Loss()
        elif self.reg_loss_type == 'L2':
            self.reg_loss = nn.MSELoss()
        

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            self.linear_out
        )
        
        for param in self.resnet.parameters():
            param.requires_grad = fine_tuning
        for param in self.fc.parameters():
            param.requires_grad = fine_tuning
        
        self.fc.apply(init_weights)

 
    def forward(self, x):
        x = self.resnet.forward(x)
        x = torch.squeeze(x)
        y = self.fc(x)
       
        if self.rank:
            y[:,1] = self.sig(y[:,1])
        
        return y
    
    def criterion(self, y, d):
        if self.pairwise:
            d = d.view([len(d), 2])
        else:
            d = d.view([len(d), 1])
            
        if self.rank:   
            reg_l = self.reg_loss(y[:,0],d[:,0])
            rank_l = self.rank_loss(y[:,1], d[:,1])
            loss = 0.7 * reg_l + 0.3 * rank_l
        else:
            reg_l = self.reg_loss(y[:,0], d[:,0])
            loss = reg_l
            
        return loss