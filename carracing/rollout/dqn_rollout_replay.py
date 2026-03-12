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

import time

from model_loader import get_policy
from convert_to_action_space import make_action
from ray_generation import calc_ray_locs

full_rays = calc_ray_locs()

def car_speed(env):
    return  np.linalg.norm(env.unwrapped.car.hull.linearVelocity)

def car_heading():
    return ((env.unwrapped.car.hull.angle + np.pi) / (2*np.pi) ) % 1

def car_steering():
    return env.unwrapped.car.wheels[0].joint.angle + 0.5

def car_angle_vel():
    return env.unwrapped.car.hull.angularVelocity

def is_off_road(state_image):
        
    img = state_image.squeeze(0)
    front_tuple = full_rays[0][0]
    front_val = img[front_tuple[0], front_tuple[1]]

    #print(avg_value)
    return (front_val > 110)

def gen_ray_vals(img):
    ray_vals = [0 for ray in full_rays]
    for i, ray in enumerate(full_rays):
        for j, (y,x) in enumerate(ray):
            if img[y][x] > 110:
                ray_vals[i] = j
                break
            ray_vals[i] = j
    return ray_vals

def gen_turn_vals(img):
        left_side = torch.tensor([line[0] for line in img])
        right_side = torch.tensor([line[-1] for line in img])
        top_side = torch.tensor(img[0])

        sides_have_road = []
        if left_side.min() < 110:
            sides_have_road.append(1)
        else:
            sides_have_road.append(-1)
        if top_side.min() < 110:
            sides_have_road.append(1)
        else:
            sides_have_road.append(-1)
        if right_side.min() < 110:
            sides_have_road.append(1)
        else:
            sides_have_road.append(-1)
        
        return sides_have_road


seed = 3
env = gym.make('CarRacing-v3', render_mode="human",lap_complete_percent=0.95, domain_randomize=False, continuous=True)
state, _ = env.reset(seed=seed)

actions_to_start =  np.loadtxt('rollout.txt')
actions_to_start = [int(action) for action in actions_to_start]


time_between_action = 0
slowmo_frames = 0

for i, action in enumerate(actions_to_start):
    # Render the game window
    
    env.render()
    
    
    state, reward, done, truncated, info = env.step(make_action(action))

    state_tensor = prepare_state_into_tensor(state).squeeze(0)
    ray_vals = gen_ray_vals(state_tensor)
    turn_vals = gen_turn_vals(state_tensor)
    off_road = is_off_road(state_tensor)
    speed = car_speed(env)

    print(ray_vals)
    print(turn_vals)
    print(off_road)
    print(speed)

    centerness = (ray_vals[0] - ray_vals[-1])
    print(centerness)

    time.sleep(time_between_action)
    if len(actions_to_start) - i < slowmo_frames:
        time.sleep(1)

    if done:
        break

# Close the environment
env.close()
