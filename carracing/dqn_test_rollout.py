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
import time


from make_env import get_environment
from model_loader import get_policy



def pick_action(policy, full_state): 
    full_state_tensor = full_state 
    
    q_values = policy(full_state_tensor) 
    arg_max = torch.argmax(q_values).item() 
    
    return arg_max

class Rollout_Policy():
    def __init__(self, base_policy, seed, rollout_render_mode='state_pixels', render_mode='human'):
        super(Rollout_Policy, self).__init__()

        self.seed = seed
        self.base_policy = base_policy
        self.action_list = []
        self.action_size = 9
        self.truncation_limit = 40
        self.frames_to_run = 0
        self.starting_frames = 0
        self.min_tile_percentage = 100
        
        self.env_type = "CNN"
        self.stack_size = 4
        self.device = 'cuda'
        self.main_env = get_environment(self.env_type, self.stack_size, render_mode, do_flips=False)
        self.current_state, _ = self.main_env.reset(seed=self.seed)
        self.current_state = torch.tensor(self.current_state).to(torch.float32).to(device).unsqueeze(0) / 255.0

        



    def generate_new_env_and_catch_up(self):
        rollout_env = get_environment(self.env_type, self.stack_size, None, do_flips=False)
        state, _ = rollout_env.reset(seed=self.seed)

        for action in self.action_list:
            state, rewards, done, trunc, info = rollout_env.step(action)
            state = torch.tensor(state).to(torch.float32).to(device).unsqueeze(0) / 255.0

        
        return rollout_env

    def find_rollout_best_action(self, current_frame_num):

        reward_sums = []

        #Catch up all envs  
        for action in range(self.action_size):
            print(f"ACTION: {action}")
            reward_sum = 0          
            rollout_env = self.generate_new_env_and_catch_up()

            #Run branch actions
            state, reward, done, trunc, info = rollout_env.step(action)
            state = torch.tensor(state).to(torch.float32).to(device).unsqueeze(0) / 255.0
            reward_sum += reward


            # Run the envs to truncation limit
            for frame in range(self.truncation_limit):
                action_picked = pick_action(self.base_policy, state)

                state, reward, done, trunc, info = rollout_env.step(action_picked)
                state = torch.tensor(state).to(torch.float32).to(device).unsqueeze(0) / 255.0
                reward_sum += reward

                if done or trunc:
                    reward_sum += 1000
                    break
            
            reward_sums.append(reward_sum)

        #Best action is the one with best reward sum
        print(reward_sums)
        max_val = np.max(reward_sums)
        actions = np.where(reward_sums == max_val)[0]
        if len(actions) == 1:
            print("clear best")
            best_action = actions[0]
        else:
            tie_break = pick_action(self.base_policy, self.current_state)
            if tie_break in actions:
                best_action = tie_break
                print(f"Broke tie!")
            else:
                print("no tie break, take first")
                best_action = actions[0]
        print(f"Took action {best_action}")
        return best_action

    def rollout_run(self):

        for action in self.action_list:
            self.current_state, reward, done, truncated, info = self.main_env.step(action)
            self.current_state = torch.tensor(self.current_state).to(torch.float32).to(device).unsqueeze(0) / 255.0
            #print(np.linalg.norm(self.main_env.unwrapped.car.hull.linearVelocity))

        for i in range(self.starting_frames, self.frames_to_run):
            print(f"ROLLOUT: {i}")
            action = self.find_rollout_best_action(i)
            self.action_list.append(action)
            self.current_state, reward, done, truncated, info = self.main_env.step(action)
            self.current_state = torch.tensor(self.current_state).to(torch.float32).to(device).unsqueeze(0) / 255.0
            np.savetxt('rollout.txt', np.array(self.action_list))

            #unwrapped_env = .unwrapped
            tiles_visited = getattr(self.main_env.unwrapped, "tile_visited_count", 0)
            total_tiles = len(self.main_env.unwrapped.track)
            completion_percentage = tiles_visited / total_tiles

            print(f"Completed {tiles_visited}/{total_tiles} = {completion_percentage * 100}% | Need {int((self.min_tile_percentage / 100) * total_tiles) + 1} to finish!")

            if (completion_percentage) >= (self.min_tile_percentage / 100) or tiles_visited == total_tiles:
                done = True

            if done or truncated:
                break

    def render_actions(self):
        
        render_env = gym.make('CarRacing-v3', render_mode="human",lap_complete_percent=0.95, domain_randomize=False, continuous=True)
        current_state, _ = render_env.reset(seed=self.seed)
        current_state = torch.tensor(current_state).to(torch.float32).to(device).unsqueeze(0) / 255.0

        for i, action in enumerate(self.action_list):
            render_env.render()
            current_state, reward, done, truncated, info = render_env.step(action)
            current_state = torch.tensor(current_state).to(torch.float32).to(device).unsqueeze(0) / 255.0
            print(i)
            time.sleep(0.05)

            if done or truncated:
                break

if __name__ == "__main__":
    env_type = "CNN"
    device = 'cuda'
    policy = get_policy(device, env_type, weights_path="carracing.pth")
    policy.eval()
    seed = 3

    frames_to_run = 1000

    rollout_policy = Rollout_Policy(policy, seed)
    actions_to_start =  list(np.loadtxt('rollout.txt'))
    #print(actions_to_start)
    actions_to_start = [int(action) for action in actions_to_start]
    print(f"Adding {len(actions_to_start)} actions from previous rollout!")
    rollout_policy.action_list.extend(actions_to_start)
    rollout_policy.starting_frames = len(actions_to_start)
    rollout_policy.frames_to_run = frames_to_run

    rollout_policy.rollout_run()
    rollout_policy.render_actions()


