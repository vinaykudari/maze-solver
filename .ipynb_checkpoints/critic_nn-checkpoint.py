import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, state):
        state.to(self.device)
        output = self.fc1(state)
        output = F.relu(output)
        state_val = self.fc2(output)

        return state_val