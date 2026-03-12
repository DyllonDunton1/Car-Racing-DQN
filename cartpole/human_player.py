import pygame
from pygame.locals import *
import gymnasium as gym
import numpy as np
import time

env = gym.make('CartPole-v1', render_mode='human')
left = 0
right = 0

print(f"Starting Game")

state,_ = env.reset()
is_terminal = False
reward_sum = 0
while not is_terminal:
    time.sleep(1/10)
    #monitor left and right key presses
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                left = 1
                right = 0
            elif event.key == K_RIGHT:
                right = 1
                left = 0
        if event.type == KEYUP:
            if event.key == K_LEFT:
                left = 0
            elif event.key == K_RIGHT:
                right = 0

    if left:
        action = 0
    else:
        action = 1

    next_state, reward, is_terminal, _, _ = env.step(action)
    reward_sum += reward
    state = next_state

    print(f"Sum or Rewards for Episode: {reward_sum}")


