import gymnasium as gym
import torch
import pygame
  



env = gym.make('CartPole-v1', render_mode='human')
state = env.reset()
print(state)
tensor_state = torch.tensor(state[0], dtype=torch.float32)
print(tensor_state)

for _ in range(100):
    # Render the game window
    env.render()

    # Take a random action
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)

    if done or truncated:
        state, info = env.reset()
        break

# Close the environment
env.close()

