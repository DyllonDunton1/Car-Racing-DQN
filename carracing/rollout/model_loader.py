import gymnasium as gym
import pygame
import PIL
from torchvision.transforms.functional import to_pil_image
from input_testing import prepare_state_into_tensor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
import torch
from torchvision import io, transforms, models
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import gc
from torch.cuda.amp import autocast, GradScaler
from collections import deque
import random

class Policy(nn.Module):
    def __init__(self, state_vector_len=11, output_vector_len=25):
        super(Policy, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_vector_len, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,output_vector_len)
        )

    def forward(self, input):
        return self.model(input)

    def get_weights(self):
        return self.state_dict()
    
    def set_weights(self, weights):
        self.load_state_dict(weights)

def get_policy(device, weights_path=''):
    
    model = Policy().to(device)

    if weights_path != '':
        model.set_weights(torch.load(weights_path))
    return model