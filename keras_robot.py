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



#---------------------------------------------------------------------------------------------


class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "car_racing.keras"
        self.state_size         = state_size
        self.action_size        = action_size
        self.action_count       = 5
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.95
        self.brain              = self._build_model()

    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=(96,96,3))
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

    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self, state):
            if np.random.rand() <= self.exploration_rate:
                return random.randrange(self.action_count)
            act_values = self.brain.predict(state)
            print("ACT VALUES: " + str(act_values))
            return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
            if len(self.memory) < sample_batch_size:
                return
            sample_batch = random.sample(self.memory, sample_batch_size)
            for state, action, reward, next_state, done in sample_batch:
                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
                target_f = self.brain.predict(state)
                target_f[0][action] = target
                self.brain.fit(state, target_f, epochs=1, verbose=True)
            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_decay

#---------------------------------------------------------------------------------------------
class CarRacing():
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 1000
        self.env               = gym.make("CarRacing-v2", domain_randomize=True, continuous=False, render_mode="human")
        self.state_size        = self.env.observation_space.shape
        self.action_size       = self.env.action_space.shape
        self.agent             = Agent(self.state_size, self.action_size)

    def run(self):
        try:
            for index_episode in range(self.episodes):
                
                state, _ = self.env.reset()
                state = np.array(state).reshape(1, *self.state_size)
                print(state.shape)
                done = False
                index = 0
                while not done:
                    self.env.render()
                    action = self.agent.act(state)
                    print(action)
                    next_state, reward, done, _, _ = self.env.step(action)
                    next_state = np.array(next_state).reshape(1, *self.state_size)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    print(reward)
                    print("Done: {}".format(done))
                    index += 1
                    print("Index: {}".format(index))
                    if index > 500: done = True
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()

#---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    while True:
        car_racing = CarRacing()
        car_racing.run()
    