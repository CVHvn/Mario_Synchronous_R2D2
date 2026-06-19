import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# change from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, (nn.LSTM, nn.LSTMCell)):
        for name, param in layer.named_parameters():
            if "weight" in name:
                torch.nn.init.orthogonal_(param, std)
            elif "bias" in name:
                torch.nn.init.constant_(param, bias_const)
    else:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, use_layer_init):
        super(Model, self).__init__()
        self.conv1 = layer_init(nn.Conv2d(1, 32, 8, stride=4)) if use_layer_init else nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = layer_init(nn.Conv2d(32, 64, 4, stride=2)) if use_layer_init else nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = layer_init(nn.Conv2d(64, 64, 3, stride=1)) if use_layer_init else nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = layer_init(nn.Linear(3136, 512)) if use_layer_init else nn.Linear(3136, 512)
        self.lstm = layer_init(nn.LSTMCell(512+output_dim+1, 512)) if use_layer_init else nn.LSTMCell(512+output_dim+1, 512)
        self.linearV = layer_init(nn.Linear(512, 512)) if use_layer_init else nn.Linear(512, 512)
        self.value = layer_init(nn.Linear(512, 1)) if use_layer_init else nn.Linear(512, 1)
        self.linearA = layer_init(nn.Linear(512, 512)) if use_layer_init else nn.Linear(512, 512)
        self.advantage = layer_init(nn.Linear(512, output_dim)) if use_layer_init else nn.Linear(512, output_dim)

    def forward(self, x, h, c, action, reward):
        x = F.relu(self.conv1(x/255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        x = torch.cat([x, action, reward], -1)
        h, c = self.lstm(x, (h, c))

        a = F.relu(self.linearA(h))
        a = self.advantage(a)

        v = F.relu(self.linearV(h))
        v = self.value(v)

        q = v + a - a.mean(1, keepdim=True)
        return q, h, c