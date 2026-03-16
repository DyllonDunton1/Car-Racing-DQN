import time
import torch
from make_env import get_environment
from model_loader import get_policy
import numpy as np

DO_ROLLOUT = False

env_type = "CNN"
stack_size = 4
render_mode = "rgb_array"
device = 'cuda'
env = get_environment(env_type, stack_size, render_mode, do_flips=True)
policy = get_policy(device, env_type, weights_path="carracing.pth")

state, _ = env.reset(seed=3)
state = torch.tensor(state).to(torch.float32).to(device).unsqueeze(0) / 255.0

vid_len = 800
time_steps = vid_len
time_delay = 0



if DO_ROLLOUT:
    actions_to_start =  np.loadtxt('rollout.txt')
    actions_to_start = [int(action) for action in actions_to_start]
    time_steps = len(actions_to_start)


vid_frames = []
for i in range(time_steps):

    if DO_ROLLOUT:
        action = actions_to_start[i]
    else:
        action = torch.argmax(policy(state))

    #if np.random.random() < 0.05:
    #    action = np.random.randint(env.action_space.n)
    state, reward, is_terminal, _, raw = env.step(action)
    frame = env.render()
    print(frame.shape)
    vid_frames.append(frame)
    state = torch.tensor(state).to(torch.float32).to(device).unsqueeze(0) / 255.0
    #print(state.shape)
    time.sleep(time_delay)
    angle_vel = env.unwrapped.car.hull.angularVelocity
    #print(np.linalg.norm(env.unwrapped.car.hull.linearVelocity))
    #print(reward)

print(len(vid_frames))
print("TIME STEPS", vid_len)
while len(vid_frames) < vid_len:
    vid_frames.append(vid_frames[-1].copy())
print("NEW", len(vid_frames))
import imageio.v2 as imageio

vid_name = "flipped.gif"
vid_path = f"vids/{vid_name}"
print(len(vid_frames))
imageio.mimsave(vid_path, vid_frames, fps=45)
