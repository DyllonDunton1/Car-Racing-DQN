import torch
from torch import nn, optim
import pygame
import gymnasium as gym
from collections import deque
import numpy as np



class DQNModel(nn.Module):
    def __init__(self, state_vector_len, output_vector_len):
        super(DQNModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_vector_len, 128),
            nn.ReLU(),
            nn.Linear(128,56),
            nn.ReLU(),
            nn.Linear(56,output_vector_len)
        )

    def forward(self, input):
        return self.model(input)

    def get_weights(self):
        return self.state_dict()
    
    def set_weights(self, weights):
        self.load_state_dict(weights)


env = gym.make('CartPole-v1', render_mode='human')
target_net = DQNModel(4,2)
target_net.set_weights(torch.load('cartpole.pth', map_location='cuda'))

episode = 0
while True:
    rewards_list = []
    print(f"Starting episode {episode}")
    reward_sum = 0

    state,_ = env.reset()

    is_terminal = False
    while not is_terminal:
        action = torch.argmax(target_net(torch.tensor(state))).item()
        next_state, reward, is_terminal, _, _ = env.step(action)
        reward_sum += reward
        print(reward_sum)
        state = next_state

    reward_sum = np.sum(rewards_list)
    print(f"Sum or Rewards for Episode: {reward_sum}")
    episode += 1