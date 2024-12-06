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