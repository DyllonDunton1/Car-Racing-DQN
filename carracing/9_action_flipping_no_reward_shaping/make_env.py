import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import DQN
import numpy as np
import os
from torchvision.transforms import Grayscale
import torch 
import random 

from ray_generation import calc_ray_locs


class DiscreteCarRacing(gym.ActionWrapper):
    def __init__(self, env, env_type="CNN", do_flips=False, verbose_flips=False):
        super().__init__(env)
        self.env_type = env_type
        assert self.env_type in ["CNN", "MLP"]
        self.do_flips = do_flips
        if self.env_type == "CNN":
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(80, 80), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-50.0, high=50.0, shape=(15,), dtype=np.float32
            )

        self.DISCRETE= np.array([
            [-1.0, 1.0, 0.0],  [ 0.0, 1.0, 0.0],  [ 1.0, 1.0, 0.0],  # gas, turn left/straight/right
            [-1.0, 0.0, 0.0],  [ 0.0, 0.0, 0.0],  [ 1.0, 0.0, 0.0],  # coast (steer only)
            [-1.0, 0.0, 1.0],  [ 0.0, 0.0, 1.0],  [ 1.0, 0.0, 1.0],  # hard brake while steering
        ], dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.DISCRETE))

        self.full_rays = calc_ray_locs()

        self.verbose_flips = verbose_flips

    def is_off_road(self,state_image):
        
        img = state_image
        front_tuple = self.full_rays[0][0]
        front_val = img[front_tuple[0], front_tuple[1]]

        #print(avg_value)
        return (front_val > 110)
    
    def car_speed(self):
        return  np.linalg.norm(self.env.unwrapped.car.hull.linearVelocity)

    def car_heading(self):
        return ((self.unwrapped.car.hull.angle + np.pi) / (2*np.pi) ) % 1

    def car_steering(self):
        return self.unwrapped.car.wheels[0].joint.angle + 0.5

    def car_angle_vel(self):
        return self.unwrapped.car.hull.angularVelocity
        
    
    def gen_ray_vals(self, img, off_road):
        ray_vals = [0 for ray in self.full_rays]

        #Check to see if we are offroad to see if we need negatives
        if not off_road:
            for i, ray in enumerate(self.full_rays):
                for j, (y,x) in enumerate(ray):
                    if img[y][x] > 110:
                        ray_vals[i] = j
                        break
                    ray_vals[i] = j
            return ray_vals
        else:
            for i, ray in enumerate(self.full_rays):
                for j, (y,x) in enumerate(ray):
                    if img[y][x] < 110:
                        ray_vals[i] = -1 * j
                        break
                    ray_vals[i] = -1 * j
        
            #switch around the rays make more sense

            #switch front and back
            temp = ray_vals[3]
            ray_vals[3] = ray_vals[7]
            ray_vals[7] = temp

            #switch left and right
            temp = ray_vals[0]
            ray_vals[0] = ray_vals[6]
            ray_vals[6] = temp

            #switch upleft and upright
            temp = ray_vals[1]
            ray_vals[1] = ray_vals[5]
            ray_vals[5] = temp

            #switch upupleft and upupright
            temp = ray_vals[2]
            ray_vals[2] = ray_vals[4]
            ray_vals[4] = temp

            return ray_vals

    def get_turn_vals(self, img):
        left_side = torch.tensor([line[0] for line in img])
        right_side = torch.tensor([line[-1] for line in img])
        top_side = img[0]

        sides_have_road = []
        if left_side.min() < 110:
            sides_have_road.append(1)
        else:
            sides_have_road.append(0)
        if top_side.min() < 110:
            sides_have_road.append(1)
        else:
            sides_have_road.append(0)
        if right_side.min() < 110:
            sides_have_road.append(1)
        else:
            sides_have_road.append(0)
        
        return sides_have_road

    def gather_state(self, state_tensor):
        state_speed = self.car_speed()
        state_angle_vel = self.car_angle_vel()
        state_heading = self.car_heading()
        state_steering = self.car_steering()
        state_off_road = self.is_off_road(state_tensor)
        ray_vals = self.gen_ray_vals(state_tensor, state_off_road)
        turn_vals = self.get_turn_vals(state_tensor)
        R = ray_vals[6]
        L = ray_vals[0]
        offset = (R - L) / (R + L + 1e-6)
        centerness = 1 - abs(offset)
        if state_off_road:
            centerness = 0

        auxilary = [state_speed, state_angle_vel, state_steering, centerness]

        full_state = torch.tensor(ray_vals + turn_vals + auxilary)
        return full_state

    def prepare_state_into_tensor(self, state):
        channel_order = [0,2,1]
        height_range = 80
        cutoff = int((96-height_range) / 2)

        state_tensor = torch.tensor(state, dtype=torch.uint8).permute(2,0,1)[channel_order,0:height_range,cutoff:96-cutoff]
        return Grayscale()(state_tensor).squeeze(0)

    def observation(self, obs):
        state_tensor = self.prepare_state_into_tensor(obs)
        if self.env_type == "CNN":
            return (state_tensor).cpu()
        elif self.env_type == "MLP":
            state_mlp = self.gather_state(state_tensor)
            return state_mlp.cpu()

    def action_conv(self, act):
        return self.DISCRETE[act]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.do_flips and random.choice([True,False]):
            if self.verbose_flips:
                print("FLIP")
            # Flip the car's angle by 180 degrees (pi radians)
            start_angle = self.env.unwrapped.car.hull.angle
            flip_angle = start_angle + np.pi
            if flip_angle > np.pi: # Ensure the angle is within [-pi, pi]
                flip_angle -= 2 * np.pi 
            self.env.unwrapped.car.hull.angle = flip_angle
            # Reset car velocity to prevent sliding after flipping
            self.env.unwrapped.car.hull.linearVelocity = (0, 0)
        elif self.verbose_flips:
            print("NO FLIP")
        return self.observation(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(self.action_conv(action))
        return self.observation(obs), rew, term, trunc, info

def get_environment(env_type, stack_size, render_mode, do_flips=False):
    env = gym.make("CarRacing-v3", domain_randomize=False, continuous=True, render_mode=render_mode)
    env = DiscreteCarRacing(env, env_type=env_type, do_flips=do_flips)
    if stack_size < 1:
        stack_size = 1
    if stack_size > 1: 
        env = FrameStackObservation(env, stack_size=stack_size)
    return env