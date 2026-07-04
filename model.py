import torch
import torch.nn as nn


class PhysicsMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()                  

        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, 4)
        )


    def forward(self, x):
        output = self.net(x)
        return output

