import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Critic_CNN(nn.Module):
    def __init__(self,img_dim, w, h):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.conv1 = nn.Conv2d(img_dim, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state):
        state.to(self.device)
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = nn.Flatten()(x)
        output = self.fc1(x)
        output = F.relu(output)
        state_val = self.fc2(output)
        
        return state_val