import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, stochastic):
        super().__init__()

        self.stochastic = stochastic

        self.fc = nn.Linear(784, 500)
        self.out = nn.Linear(500, 10)

    def forward(self, image):
        x = image.view(-1, 784)

        x = F.sigmoid(self.fc(x))
        if not self.training and self.stochastic:
            x = torch.bernoulli(x)

        x = self.out(x)
        if not self.training:
            x = F.softmax(x)

        return x
