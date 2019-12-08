import os
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as td 
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt 

import cGAN_model.nntools as nt

class NNRegressor(nt.NeuralNetwork):
    
    def __init__(self):
        super(NNRegressor, self).__init__()
        self.mse = nn.MSELoss()
    
    def criterion(self, y, d):
        return self.mse(y, d)


class DnCNN(NNRegressor):

    def __init__(self, D, C=64): 
        super(DnCNN, self).__init__() 
        self.D = D
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, C, 3, padding=1))
        # COMPLETE
        
        self.bn = nn.ModuleList()
        for k in range(D):
            self.bn.append(nn.BatchNorm2d(C))
            self.conv.append(nn.Conv2d(C, C, 3, padding=1))
        self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
        
    def forward(self, x): 
        D = self.D
        h = F.relu(self.conv[0](x))
        # COMPLETE
        for i in range(D):
            h = F.relu(self.bn[i](self.conv[i + 1](h)))
        
        y = self.conv[D+1](h) + x
        
        return y

class DenoisingStatsManager(nt.StatsManager):
    def __init__(self):
        super(DenoisingStatsManager, self).__init__()
        self.accPSNR = 0
        
    def init(self):
        super(DenoisingStatsManager, self).init()
        self.accPSNR = 0
        
        
    def accumulate(self, loss, x, y, d): 
        super(DenoisingStatsManager, self).accumulate(loss, x, y, d)
        mse = nn.MSELoss()
        psnr = 10 * np.log10(4 / mse(y, d).item())
        self.accPSNR += psnr
        
        
    def summarize(self):
        loss = super(DenoisingStatsManager, self).summarize() 
        avgPSNR = self.accPSNR / self.number_update
        return {"loss": loss, "psnr":avgPSNR}