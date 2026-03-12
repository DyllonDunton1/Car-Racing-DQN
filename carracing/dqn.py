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
from collections import deque
import random
import copy


from make_env import get_environment
from model_loader import get_policy



class DQNLearner():
    def __init__(self, env_type, lr, replay_buffer_max_size, train_start_buffer_length, steps_between_fit, target_update_step_interval, batch_size, total_steps, exploration_portion, epsilon_final, gamma, stack_size=4, render_mode=None, do_flips=False, pickup=False, pickup_current_step=0):
        self.env_type = env_type
        self.lr = lr
        self.replay_buffer_max_size = replay_buffer_max_size
        self.train_start_buffer_length = train_start_buffer_length
        self.steps_between_fit = steps_between_fit
        self.target_update_step_interval = target_update_step_interval
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.exploration_portion = exploration_portion
        self.epsilon_final = epsilon_final
        self.gamma = gamma
        self.do_flips = do_flips

        self.current_episode = 0
        self.current_step = 0
        self.epsilon_step = 0
        self.num_updates = 0
        self.start_epsilon = 1
        self.exploration_step_quantity = self.exploration_portion * self.total_steps
        self.epsilon_drop_per_step = (1 - self.epsilon_final) / (self.exploration_step_quantity)

        self.max_episode_length = 1000
        self.s_replay_buffer = deque(maxlen=self.replay_buffer_max_size)
        self.a_replay_buffer = deque(maxlen=self.replay_buffer_max_size)
        self.r_replay_buffer = deque(maxlen=self.replay_buffer_max_size)
        self.ns_replay_buffer = deque(maxlen=self.replay_buffer_max_size)
        self.d_replay_buffer = deque(maxlen=self.replay_buffer_max_size)
        self.reward_sums = []
        self.tile_records = []
        self.competion_records = []
        
        self.save_name = "carracing.pth"
        self.csv_path = "data.csv"
        self.log_columns = ["episode", "reward", "avg reward", "tiles", "avg tiles", "completed", "avg completed", "epsilon"]
        self.logs = pd.DataFrame(columns=self.log_columns)

        assert self.env_type in ["CNN", "MLP"]
        self.device = 'cuda'
        self.env = get_environment(env_type, stack_size, render_mode, do_flips=do_flips)
        if pickup:
            self.online_net = get_policy(self.device, self.env_type, weights_path="self.save_name")
            self.current_episode = current_step
            self.epsilon_step = self.current_episode + self.train_start_buffer_length
            self.logs = pd.read_csv(self.csv_path)
            self.reward_sums = df["reward"].tolist()
            self.tile_records = df["tiles"].tolist()
        else:
            self.online_net = get_policy(self.device, self.env_type)
        self.target_net = get_policy(self.device, self.env_type)
        self.target_net.set_weights(self.online_net.get_weights())
        self.optimizer=optim.Adam(self.online_net.parameters(), lr=self.lr, eps=1e-5)

        
    



    def current_epsilon(self):
        if self.current_step == 0:
            return 1.0
        return max(1.0 - self.epsilon_drop_per_step * self.epsilon_step, self.epsilon_final)

    def convert_to_torch(self, numpy_tensor):
        return torch.from_numpy(numpy_tensor).to(torch.float).to(self.device).unsqueeze(0) / 255.0

    def pick_action(self, state):

        if np.random.random() < self.current_epsilon():
            return np.random.randint(self.env.action_space.n)

        with torch.no_grad():      
            return torch.argmax(self.online_net(self.convert_to_torch(state))).item()


    def episodic_train(self):
        
        
        while self.current_step < self.total_steps:
                        
            state, _ = self.env.reset()
            state = state.astype(np.uint8)
            reward_sum = 0

            tiles_visited = 0
            for i in range(self.max_episode_length):
                
                if len(self.s_replay_buffer) >= self.train_start_buffer_length:
                    self.current_step += 1
                self.epsilon_step += 1

                action = self.pick_action(state)
                next_state, reward, is_terminal, truncated, _ = self.env.step(action)
                reward_sum += reward
                if reward > 0:
                    tiles_visited += 1
                next_state = next_state.astype(np.uint8)

                self.s_replay_buffer.append(state)
                self.a_replay_buffer.append(action)
                self.r_replay_buffer.append(reward)
                self.ns_replay_buffer.append(next_state)
                self.d_replay_buffer.append(float(is_terminal))
                
                if len(self.s_replay_buffer) >= self.train_start_buffer_length and self.current_step%self.steps_between_fit == 0 and self.current_step > 0:
                    self.train_net()
                
                state = next_state

                if self.num_updates > 0 and self.num_updates%self.target_update_step_interval==0:
                    self.target_net.set_weights(self.online_net.get_weights())

                if is_terminal or truncated:
                    break


            total_tiles = len(getattr(self.env.unwrapped, "track", []))
            visited_tiles = int(getattr(self.env.unwrapped, "tile_visited_count", 0))

            self.reward_sums.append(reward_sum)
            self.tile_records.append(tiles_visited)
            print("-------------------------------")
            print(f"Episode Reward: {reward_sum}")
            print(f"Avg 100Ep Reward: {np.mean(np.array(self.reward_sums[-100:]))}")
            print(f"Tiles Visited: {tiles_visited}")
            print(f"Avg 100Ep Tiles Visited: {np.mean(np.array(self.tile_records[-100:]))}")
            print(f"Epsilon: {self.current_epsilon()}")
            print(f"Steps Completed: {self.current_step}")
            if len(self.s_replay_buffer) < self.train_start_buffer_length:
                print(f"Buffer Length: {len(self.s_replay_buffer)}/{self.train_start_buffer_length}")
            
            new_entry = pd.DataFrame({
                "episode": [self.current_episode],
                "reward": [reward_sum],
                "avg reward": [np.mean(np.array(self.reward_sums[-100:]))],
                "tiles": [tiles_visited],
                "avg tiles": [np.mean(np.array(self.tile_records[-100:]))],
                "epsilon": [self.current_epsilon()]
            })
            self.logs = pd.concat([self.logs,new_entry], ignore_index=True)
            self.logs.to_csv(self.csv_path, index=False)

            self.current_episode += 1

            # --- 1) Epsilon over episodes ---
            plt.figure(figsize=(8,4))
            plt.plot(self.logs["episode"], self.logs["epsilon"])
            plt.xlabel("Episode")
            plt.ylabel("Epsilon")
            plt.title("Epsilon Over Episodes")
            plt.grid(True)
            plt.savefig("plots/epsilon_over_episodes.png")
            plt.close()

            # --- 2) Reward and Avg Reward ---
            plt.figure(figsize=(8,4))
            plt.plot(self.logs["episode"], self.logs["reward"], label="Reward")
            plt.plot(self.logs["episode"], self.logs["avg reward"], label="Avg Reward (smoothed)")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Reward Progress")
            plt.legend()
            plt.grid(True)
            plt.savefig("plots/reward_over_episodes.png")
            plt.close()

            # --- 3) Tiles and Avg Tiles ---
            plt.figure(figsize=(8,4))
            plt.plot(self.logs["episode"], self.logs["tiles"], label="Tiles per Episode")
            plt.plot(self.logs["episode"], self.logs["avg tiles"], label="Avg Tiles (smoothed)")
            plt.xlabel("Episode")
            plt.ylabel("Tiles")
            plt.title("Tiles Visited Progress")
            plt.legend()
            plt.grid(True)
            plt.savefig("plots/tiles_over_episodes.png")
            plt.close()

            # --- 3) Avg Reward and Avg Tiles ---
            plt.figure(figsize=(8,4))
            plt.plot(self.logs["episode"], self.logs["avg reward"], label="Avg Reward (smoothed)")
            plt.plot(self.logs["episode"], self.logs["avg tiles"], label="Avg Tiles (smoothed)")
            plt.xlabel("Episode")
            plt.ylabel("Tiles")
            plt.title("Reward / Tile relationship")
            plt.legend()
            plt.grid(True)
            plt.savefig("plots/rewards_and_tiles_over_episodes.png")
            plt.close()



    def train_net(self):

        idxs = random.sample(range(len(self.s_replay_buffer)), self.batch_size)
        
        s  = torch.as_tensor(torch.cat([self.convert_to_torch(self.s_replay_buffer[i]) for i in idxs],dim=0), dtype=torch.float, device=self.device)
        ns = torch.as_tensor(torch.cat([self.convert_to_torch(self.ns_replay_buffer[i]) for i in idxs],dim=0), dtype=torch.float, device=self.device)
        a  = torch.as_tensor([self.a_replay_buffer[i] for i in idxs], dtype=torch.long,   device=self.device)
        r  = torch.as_tensor([self.r_replay_buffer[i] for i in idxs], dtype=torch.float, device=self.device).clamp_(-1.0,1.0)
        d  = torch.as_tensor([self.d_replay_buffer[i] for i in idxs], dtype=torch.float, device=self.device)

        self.online_net.train()
        self.target_net.eval()

        q_all = self.online_net(s)
        q_taken = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if False: #DQN
                q_next_max = self.target_net(ns).max(dim=1).values
                y = r + self.gamma * q_next_max * (1.0 - d)
            else:    #DDQN
                next_actions = self.online_net(ns).argmax(dim=1) 
                q_next_tgt = self.target_net(ns).gather(1, next_actions.unsqueeze(1)).squeeze(1) 
                y = r + self.gamma * q_next_tgt * (1.0 - d)

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_taken, y) # Huber loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.num_updates += 1

        if self.current_step%10000==0:
            torch.save(self.online_net.get_weights(), self.save_name)



