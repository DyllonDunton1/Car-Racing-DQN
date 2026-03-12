import time
import torch
from make_env import get_environment
from model_loader import get_policy
import numpy as np

DO_ROLLOUT = False

env_type = "CNN"
stack_size = 4
render_mode = "human"
device = 'cuda'
env = get_environment(env_type, stack_size, render_mode, do_flips=False)
policy = get_policy(device, env_type, weights_path="carracing.pth")

state, _ = env.reset(seed=3)
print(state.shape)
state = torch.tensor(state).to(torch.float32).to(device).unsqueeze(0) / 255.0

time_steps = 10000
time_delay = 0.01

if DO_ROLLOUT:
    actions_to_start =  np.loadtxt('rollout.txt')
    actions_to_start = [int(action) for action in actions_to_start]
    time_steps = len(actions_to_start)


for i in range(time_steps):

    if DO_ROLLOUT:
        action = actions_to_start[i]
    else:
        action = torch.argmax(policy(state))
    #if np.random.random() < 0.05:
    #    action = np.random.randint(env.action_space.n)
    state, reward, is_terminal, _, _ = env.step(action)
    state = torch.tensor(state).to(torch.float32).to(device).unsqueeze(0) / 255.0
    #print(state.shape)
    time.sleep(time_delay)



