#Installing packages and importing
#pip install gymnasium
#pip install 'gymansium[atari, accept-rom-license]
#brew install swig
#pip install 'gymnasium[box2d]

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

#creating the architecture of neural network
class Network(nn.Module):
    def __init__(self, action_size, seed = 42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.full_connection1 = nn.Linear(10 * 10 * 128, 512)
        self.full_connection2 = nn.Linear(512, 256)
        self.full_connection3 = nn.Linear(256, action_size)
    
    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.full_connection1(x))
        x = F.relu(self.full_connection2(x))
        return self.full_connection3(x)
    
#setting up the enviroment
import ale_py
import gymnasium as gym

env = gym.make('MsPacmanDeterministic-v0', full_action_space = False)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('state shapes: ', state_shape)
print('state size: ', state_size)
print('number of actions: ', number_actions)

#initializing the hyperparameters
learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99

#preprocessing the frames
from PIL import Image
from torchvision import transforms

def preprocessing_frame(frame):
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    return preprocess(frame).unsqueeze(0)