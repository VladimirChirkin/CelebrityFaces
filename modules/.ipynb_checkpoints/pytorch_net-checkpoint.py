import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class MyVGG(nn.Module):
    
    def __init__(self):
        super(MyVGG, self).__init__()
        self.features = torchvision.models.squeezenet1_0(pretrained=False).features
        self.embeddings = nn.Sequential(nn.Linear(66560, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                        nn.Linear(1024, 128))
        self.classif = nn.Sequential(self.embeddings, nn.ReLU(), nn.Linear(128, 40))
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return F.sigmoid(self.classif(x))
    
    def get_embeddings(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.embeddings(x)
 
        return F.normalize(x, dim=1)