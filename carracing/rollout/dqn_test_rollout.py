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





def car_speed(env):
    return  np.linalg.norm(env.unwrapped.car.hull.linearVelocity)

def car_heading(env):
    return ((env.unwrapped.car.hull.angle + np.pi) / (2*np.pi) ) % 1

def car_steering(env):
    return env.unwrapped.car.wheels[0].joint.angle + 0.5

def car_angle_vel(env):
    return env.unwrapped.car.hull.angularVelocity

def is_off_road(state_image, full_rays):
    front_tuple = full_rays[0][0]
    front_val = state_image[front_tuple[0], front_tuple[1]]

    #print(avg_value)
    return (front_val > 110)

def gen_ray_vals(img, full_rays):
    ray_vals = [0 for ray in full_rays]
    for i, ray in enumerate(full_rays):
        for j, (y,x) in enumerate(ray):
            if img[y][x] > 110:
                ray_vals[i] = j
                break
            ray_vals[i] = j
    return ray_vals

def pick_action(policy, full_state):
    full_state_tensor = torch.tensor(full_state).to('cuda').to(torch.float32)
    
    #print(state_tensor.size())
    q_values = policy(full_state_tensor)
    arg_max = torch.argmax(q_values).item()
    #print(arg_max)
    return arg_max


class Rollout_Policy():
    def __init__(self, base_policy, seed, rollout_render_mode='state_pixels', render_mode='human'):
        super(Rollout_Policy, self).__init__()

        self.seed = seed
        self.base_policy = base_policy
        self.action_list = []
        self.action_size = 25
        self.truncation_limit = 40
        self.frames_to_run = 0
        self.starting_frames = 0
        self.min_tile_percentage = 97
        
        self.main_env = gym.make('CarRacing-v3', render_mode=None,lap_complete_percent=0.95, domain_randomize=False, continuous=True).unwrapped
        self.current_state, _ = self.main_env.reset(seed=self.seed)

        self.full_rays = calc_ray_locs()
        
    def gen_ray_vals(self, img):
        ray_vals = [0 for ray in self.full_rays]
        for i, ray in enumerate(self.full_rays):
            for j, (y,x) in enumerate(ray):
                if img[y][x] > 110:
                    ray_vals[i] = j
                    break
                ray_vals[i] = j
        return ray_vals

    def generate_new_env_and_catch_up(self):
        rollout_env = gym.make('CarRacing-v3', render_mode=None,lap_complete_percent=0.95, domain_randomize=False, continuous=True)
        state, _ = rollout_env.reset(seed=self.seed)

        for action in self.action_list:
            state, reward, done, truncated, info = rollout_env.step(make_action(action))

        return rollout_env, state, rollout_env.unwrapped.tile_visited_count

    def find_rollout_best_action(self, current_frame_num):

        reward_sums = []

        #run base policy for each action and store reward sums
        for action in range(self.action_size):
            print(f"ACTION: {action}")
            off_road_count = 0
            too_slow_count = 0
            test_env, state, tiles_visited = self.generate_new_env_and_catch_up()

            state, reward, done, truncated, info = test_env.step(make_action(action))
            state_tensor = prepare_state_into_tensor(state).squeeze(0)

            state_speed = car_speed(test_env)
            state_angle_vel = car_angle_vel(test_env)
            state_heading = car_heading(test_env)
            state_steering = car_steering(test_env)
            state_off_road = is_off_road(state_tensor, self.full_rays)
            full_state = torch.tensor(gen_ray_vals(state_tensor, self.full_rays) + [state_speed, state_angle_vel, state_heading, state_steering])
            
            
            if state_speed < 0.25:
                too_slow_count += 1
                #print('too slow')
                if too_slow_count > 50:
                    reward -= 100
            else:
                too_slow_count = 0

            if (current_frame_num - self.start_frames) > 100 and state_off_road:
                off_road_count += 1
                reward -= 5
                if off_road_count > 10:
                    reward = -100

            else:
                off_road_count = 0

            reward += 0.5*min(1,state_speed/70)

            #run a base policy to 3500 max
            reward_sum = reward
            for frame in range(self.truncation_limit):
                action_picked = pick_action(self.base_policy, full_state)

                state, reward, done, truncated, info = test_env.step(make_action(action))
                state_tensor = prepare_state_into_tensor(state).squeeze(0)

                state_speed = car_speed(test_env)
                state_angle_vel = car_angle_vel(test_env)
                state_heading = car_heading(test_env)
                state_steering = car_steering(test_env)
                state_off_road = is_off_road(state_tensor, self.full_rays)
                full_state = torch.tensor(gen_ray_vals(state_tensor, self.full_rays) + [state_speed, state_angle_vel, state_heading, state_steering])
                
                
                if state_speed < 0.25:
                    too_slow_count += 1
                    #print('too slow')
                    if too_slow_count > 50:
                        reward -= 100
                else:
                    too_slow_count = 0

                if (current_frame_num - self.start_frames) > 20 and state_off_road:
                    off_road_count += 1
                    reward -= 5
                    if off_road_count > 10:
                        reward = -100
                else:
                    off_road_count = 0

                reward += 0.5*min(1,state_speed/70)

                reward_sum += reward

                if done or truncated:
                    break
            
            reward_sums.append(reward_sum)
        
        #Best action is the one with best reward sum
        print(np.array(reward_sums))
        best_action = np.argmax(np.array(reward_sums))
        return best_action, tiles_visited

    def rollout_run(self):

        for i in range(self.frames_to_run):
            print(f"ROLLOUT: {i}")
            action, tiles_visited = self.find_rollout_best_action(i)
            self.action_list.append(action)
            self.current_state, reward, done, truncated, info = self.main_env.step(make_action(action))
            np.savetxt('rollout.txt', np.array(self.action_list))

            #unwrapped_env = .unwrapped
            total_tiles = len(self.main_env.track)
            completion_percentage = tiles_visited / total_tiles

            print(f"Completed {tiles_visited}/{total_tiles} = {completion_percentage * 100}% | Need {int((self.min_tile_percentage / 100) * total_tiles) + 1} to finish!")

            if (completion_percentage) > (self.min_tile_percentage / 100):
                done = True

            if done:
                break

    def render_actions(self):
        
        render_env = gym.make('CarRacing-v3', render_mode="human",lap_complete_percent=0.95, domain_randomize=False, continuous=True)
        current_state, _ = render_env.reset(seed=self.seed)

        for i, action in enumerate(self.action_list):
            render_env.render()
            current_state, reward, done, truncated, info = render_env.step(make_action(action))
            print(i)
            time.sleep(0.05)

            if done or truncated:
                break



policy = get_policy("cuda", "carracing.pth")

seed = 3

frames_to_run = 1000

rollout_policy = Rollout_Policy(policy, seed)
actions_to_start =  np.loadtxt('rollout.txt')
actions_to_start = [int(action) for action in actions_to_start]
print(f"Adding {len(actions_to_start)} actions from previous rollout!")
rollout_policy.action_list.extend(actions_to_start)
rollout_policy.start_frames = len(actions_to_start)
rollout_policy.frames_to_run = frames_to_run

rollout_policy.rollout_run()
rollout_policy.render_actions()


