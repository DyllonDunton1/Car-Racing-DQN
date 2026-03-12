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



def calc_ray_locs(N=8, points_per_ray=35, height=60, H=80):
    #N is num rays
    # o1,o2 are the origin
    # alpha is the angle range +-alpha of the rays
    # H is the height in pixels of the square img

    # (0,0) is top left
    
    length_line_plot = 2*height + H
    lenght_per_ray = int(length_line_plot / (N-2))


    ray_heads = []
    full_rays = []

    for i in range(N-1):
        ray_location_on_line = i * lenght_per_ray

        if ray_location_on_line <= height:
            ray_heads.append((height - ray_location_on_line, 0))
        elif ray_location_on_line <= height + H:
            ray_heads.append((0, ray_location_on_line - height))
        else:
            ray_heads.append((ray_location_on_line - H - height, H-1))
    #Add in a reverse ray
    ray_heads.append((H-1, H/2))
    
    #print(ray_heads)
    for ray_num, (y,x) in enumerate(ray_heads):
        forward_travel = y - height
        right_travel = x - H/2

        forward_delta = forward_travel / (points_per_ray-1)
        right_delta = right_travel / (points_per_ray-1)

        ray = []
        for i in range(points_per_ray):
            ray_y = min(H-1, int(height + forward_delta * i))
            ray_x = min(H-1, int(H/2 + right_delta * i)) - 4*int(ray_num == N-1)
            ray.append((ray_y, ray_x))

        full_rays.append(ray)
    
    return full_rays