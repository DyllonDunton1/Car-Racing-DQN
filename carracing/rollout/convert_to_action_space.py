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


def make_action(action_num):
    # steering [-1, -0.5, 0, 0.5, 1]
    # gas [0, 0.5, 1]
    # brake [0, 0.5, 1]

    action_space = [
        # Steering = -1.0
        [-1.0, 0.0, 0.0], [-1.0, 0.5, 0.0], [-1.0, 1.0, 0.0],  # Gas
        [-1.0, 0.0, 0.5], [-1.0, 0.0, 1.0],  # Brake

        # Steering = -0.5
        [-0.5, 0.0, 0.0], [-0.5, 0.5, 0.0], [-0.5, 1.0, 0.0],  # Gas
        [-0.5, 0.0, 0.5], [-0.5, 0.0, 1.0],  # Brake

        # Steering = 0.0
        [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0],  # Gas
        [0.0, 0.0, 0.5], [0.0, 0.0, 1.0],  # Brake

        # Steering = 0.5
        [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 1.0, 0.0],  # Gas
        [0.5, 0.0, 0.5], [0.5, 0.0, 1.0],  # Brake

        # Steering = 1.0
        [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0],  # Gas
        [1.0, 0.0, 0.5], [1.0, 0.0, 1.0],  # Brake
    ]

    return np.array(action_space[action_num])
