import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense

from gymnasium.spaces import Box, Discrete
from gymnasium import Env
from collections import deque
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, load_model
import os
import random
import matplotlib.pyplot as plt




#---------------------------------------------------------------------------------------------


class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "car_racing.keras"
        self.state_size         = state_size
        self.action_size        = action_size
        self.action_count       = 5
        self.memory             = deque(maxlen=100000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.95
        self.brain              = self._build_model()
        self.frame_m2 = np.zeros(shape=(1, 96,96,3))
        self.frame_m1 = np.zeros(shape=(1, 96,96,3))
        self.state = np.zeros(shape=(1, 96,96,3))
        self.next_state = None


    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=(96,96,9))
        #6
        x = keras.layers.Flatten()(inputs)
        #7
        x = keras.layers.Dense(1024, activation="relu", kernel_initializer='normal')(x)
        x = keras.layers.Dense(256, activation="relu", kernel_initializer='normal')(x)
        x = keras.layers.Dense(32, activation="relu", kernel_initializer='normal')(x)
        #8
        x = keras.layers.Dense(5, activation="relu", kernel_initializer='normal')(x)
        #x = keras.layers.Softmax()(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def _build_model_experimental(self):
        inputs = tf.keras.layers.Input(shape=(96, 96, 9))
        #1
        x = keras.layers.Conv2D(9, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1", use_bias=True,)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        #2
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

        #3
        x = keras.layers.Conv2D(9, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2", use_bias=True,)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        #4
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

        #5
        x = keras.layers.Conv2D(9, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv3", use_bias=True,)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        #6
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

        #7
        x = keras.layers.Conv2D(9, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv4", use_bias=True,)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        #8
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

        #9
        x = keras.layers.Conv2D(9, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv5", use_bias=True,)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        #10
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

        #11
        x = keras.layers.Flatten()(x)

        #12
        x = keras.layers.Dense(1024, activation="relu", kernel_initializer='normal')(x)

        #13
        x = keras.layers.Dense(5, activation="relu", kernel_initializer='normal')(x)
        x = keras.layers.Softmax()(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self):
            self.state_new = np.column_stack((self.state, self.frame_m1, self.frame_m2)).reshape(1, 96, 96, 9)
            act_values = self.brain.predict(self.state_new)
            action =  np.argmax(act_values[0])
            print("Predicted: " + str(action))
            if np.random.rand() <= self.exploration_rate:
                print("RANDOM")
                action = random.randrange(self.action_count)
            self.frame_m2 = self.frame_m1
            self.frame_m1 = self.state
            print("ACTION: " + str(action))
            return action

           

    def remember(self, action, reward, done):
            self.memory.append((self.frame_m1, self.state, self.state_new, action, reward, self.next_state, done))

    def replay(self, sample_batch_size):
            if len(self.memory) < sample_batch_size:
                return
            sample_batch = random.sample(self.memory, sample_batch_size)
            for frame1, state, state_new, action, reward, next_state, done in sample_batch:
                #print("frame1: " + str(frame1.shape))
                #print("state: " + str(state.shape))
                #print("state_new: " + str(state_new.shape))
                #print("action: " + str(action))
                #print("reward: " + str(reward))
                #print("next_state: " + str(next_state.shape))
                #print("done: " + str(done))
                target = reward
                if not done:
                    new_next = np.column_stack((next_state, state, frame1)).reshape(1, 96, 96, 9)
                    #print("new_next: " + str(new_next.shape))
                    target = reward + self.gamma * np.amax(self.brain.predict(new_next)[0])
                target_f = self.brain.predict(state_new)
                target_f[0][action] = target
                self.brain.fit(state_new, target_f, epochs=1, verbose=True)
            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_decay

#---------------------------------------------------------------------------------------------
class CarRacing():
    def __init__(self):
        self.sample_batch_size = 64
        self.episodes          = 325
        self.max_episode_length = 600
        self.env               = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")
        self.state_size        = self.env.observation_space.shape
        self.action_size       = self.env.action_space.shape
        self.agent             = Agent(self.state_size, self.action_size)
        self.all_episode_rewards = []
        self.running_reward = 0

    def run(self):
        try:
            for index_episode in range(self.episodes):
                
                state, _ = self.env.reset()
                self.agent.state = np.array(state).reshape(1, *self.state_size)
                self.running_reward = 0
                #print(state.shape)
                done = False
                index = 0
                while not done:
                    self.env.render()
                    action = self.agent.act()
                    print(action)
                    next_state, reward, done, _, _ = self.env.step(action)
                    self.agent.next_state = np.array(next_state).reshape(1, 96, 96, 3)
                    self.agent.remember(action, reward, done)
                    self.agent.state = self.agent.next_state
                    self.running_reward += reward
                    print(reward)
                    print("Done: {}".format(done))
                    index += 1
                    print("Index: {}".format(index))
                    if index > self.max_episode_length: done = True
                print("Episode #{} Score: {}".format(index_episode, self.running_reward))
                self.all_episode_rewards.append(self.running_reward)
                self.agent.replay(self.sample_batch_size)
        finally:
            
            print(str(self.all_episode_rewards))
            episodes = range(1,self.episodes+1)
            cumsum = 0
            running_avg = []
            for i in range(0, self.episodes):
                 cumsum += self.all_episode_rewards[i]
                 print(i+1)
                 running_avg.append(cumsum/(i+1))
                 
            plt.plot(episodes, self.all_episode_rewards, color="blue")
            plt.plot(episodes, running_avg, color="red")
            plt.title("Episode Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.legend()
            plt.show()
            self.agent.save_model()

#---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    car_racing = CarRacing()
    car_racing.run()
    car_racing.env.close()
    
