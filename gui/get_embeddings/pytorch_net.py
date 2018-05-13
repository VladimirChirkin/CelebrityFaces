import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import transform


def test_transform(X):
    X = transform.resize(X, (218, 178))[None]
    height = X.shape[1]
    width = X.shape[2]
    X_out = np.zeros((X.shape[0], height, width, 3), dtype=np.float32)
    for idx, x in enumerate(X):
        x = transform.resize(x, (height, width))
        X_out[idx] = x
    X_out = X_out.transpose([0, 3, 1, 2])
    return X_out


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
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