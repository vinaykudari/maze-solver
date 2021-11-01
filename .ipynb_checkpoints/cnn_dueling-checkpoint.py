import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingCNN(nn.Module):
    def __init__(self, img_dim, w, h, input_dim, output_dim, dueling_type='mean'):
        super().__init__()
        self.dueling_type = dueling_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(img_dim, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        self.fc_value = nn.Linear(linear_input_size, 1)
        self.fc_action_adv = nn.Linear(linear_input_size, output_dim)

    def forward(self, x):
        x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        
        v = self.fc_value(x)
        a = self.fc_action_adv(x)
        
        if self.dueling_type == 'max':
            q = v + a - a.max()
        else:
            q = v + a - a.mean()
        
        return q