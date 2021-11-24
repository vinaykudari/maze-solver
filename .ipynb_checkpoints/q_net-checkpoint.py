import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseNetwork


class QNetwork(BaseNetwork):
    def __init__(
        self,
        state_dim,
        action_dim,
        op_dim=1,
        hidden_layers=[32, 32],
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.out = nn.Linear(hidden_layers[0], action_dim)

    def forward(self, state):
        state = state.float()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        # return q-value
        return x