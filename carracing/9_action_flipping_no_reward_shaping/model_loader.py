import gymnasium as gym
import pygame
import PIL
from torchvision.transforms.functional import to_pil_image
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

class MLP_Policy(nn.Module):
    def __init__(self, state_vector_len=60, output_vector_len=9):
        super(MLP_Policy, self).__init__()

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_vector_len, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,output_vector_len)
        )

    def forward(self, input):
        return self.mlp(input)

    def get_weights(self):
        return self.state_dict()
    
    def set_weights(self, weights):
        self.load_state_dict(weights)

class CNN_Policy(nn.Module):
    def __init__(self, input_shape, output_vector_len=9):
        super(CNN_Policy, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.features = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_vector_len)
        )

    def forward(self, input):
        #Dueling DQN
        x = self.cnn(input)
        x = self.features(x)
        value = self.value_head(x)
        advantages = self.adv_head(x)
        q_values = value + (advantages - advantages.mean(dim=1,keepdim=True)) #Value of state plus the relative advantages of each action
        return q_values

    def get_weights(self):
        return self.state_dict()
    
    def set_weights(self, weights):
        self.load_state_dict(weights)

def get_policy(device, policy_type, weights_path='', state_vector_len=60, input_shape=(4,80,80),output_vector_len=9):
    
    if policy_type == "CNN":
        model = CNN_Policy(input_shape=input_shape,output_vector_len=output_vector_len).to(device)
    else:
        model = MLP_Policy(state_vector_len=state_vector_len,output_vector_len=output_vector_len).to(device)

    if weights_path != '':
        model.set_weights(torch.load(weights_path))
    return model