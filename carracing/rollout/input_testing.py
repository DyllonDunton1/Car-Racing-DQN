import gymnasium as gym
import torch
import pygame
import numpy as np
import PIL
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Grayscale


def prepare_state_into_tensor(state):
    channel_order = [0,2,1]
    height_range = 80
    cutoff = int((96-height_range) / 2)

    state_tensor = torch.tensor(state, dtype=torch.float32).permute(2,0,1)[channel_order,0:height_range,cutoff:96-cutoff]
    return Grayscale()(state_tensor)

'''
env = gym.make('CarRacing-v3', render_mode='human',lap_complete_percent=0.95, domain_randomize=False, continuous=True)
state = env.reset()
#print(state)
tensor_state = torch.tensor(state[0], dtype=torch.float32)
#print(tensor_state)
print(env.action_space)

for _ in range(100):
    # Render the game window
    state, reward, done, truncated, info = env.step(np.array([0.0, 0.5, 0.0]))
    tensor_state = torch.tensor(state, dtype=torch.float32)
    #print(tensor_state)
    if done:
        state, info = env.reset()
        break



# Close the environment
env.close()

state_tensor = prepare_state_into_tensor(state)

state_img = to_pil_image(state_tensor)

state_img.show()
'''