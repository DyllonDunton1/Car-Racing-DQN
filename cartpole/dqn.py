import torch
from torch import nn, optim
import torch.nn.functional as F
import pygame
import gymnasium as gym
from collections import deque
import numpy as np
import random


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

   


class DQNLearner():
    def __init__(self, env, state_shape=(4,1), out_shape=(2,1), replay_buffer_size=300, sampling_batch_size=100, update_timer_thresh=100, num_episodes=100, epsilon=0.1, gamma=1, lr=0.0001, save_name='cartpole', pick_up=False):
        self.env = env
        self.state_shape=state_shape
        self.out_shape=out_shape
        self.replay_buffer_size=replay_buffer_size
        self.sampling_batch_size=sampling_batch_size
        self.update_timer_thresh=update_timer_thresh
        self.num_episodes=num_episodes
        self.epsilon=epsilon
        self.gamma=gamma
        self.lr=lr
        self.save_name=save_name


        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.update_timer = 0
        self.reward_sums_per_episode = []
        self.actions_list = []
        self.epsilon_per_episode = []
        
        self.online_net = DQNModel(self.state_shape[0],self.out_shape[0]).to('cuda')
        self.target_net = DQNModel(self.state_shape[0],self.out_shape[0]).to('cuda')

        self.optimizer=optim.Adam(self.online_net.parameters(), lr = self.lr)

        if pick_up:
            self.online_net.set_weights(torch.load(f"{self.save_name}.pth", map_location='cuda'))

        self.target_net.set_weights(self.online_net.get_weights())

        self.max_loss = 0

    def loss_function(self,predictions, truth):
        loss = 0

        for i, action in enumerate(self.actions_list):
            error = truth[i][action] - predictions[i][action]
            loss += error*error
        
        loss /= self.sampling_batch_size
        return loss

    def fit(self, state_batch, output_batch, epochs=100):
        for epoch in range(epochs):
            self.online_net.train()
            train_loss = 0

            self.optimizer.zero_grad()
            predicted_outputs = self.online_net(state_batch)
            train_loss = self.loss_function(predicted_outputs, output_batch)
            train_loss.backward()
            self.optimizer.step()
            
            #print(f"Epoch training loss: {train_loss.item()}")


    def episodic_train(self):
        
        for episode in range(self.num_episodes):
            rewards_list = []
            print(f"Starting episode {episode}")

            #if episode > 450:
            #    self.env.render_mode = 'human'

            state,_ = self.env.reset()

            is_terminal = False
            while not is_terminal:
                action = self.pick_action(state,episode)
                next_state, reward, is_terminal, _, _ = self.env.step(action)
                rewards_list.append(reward)
                
                self.replay_buffer.append((state,action,reward,next_state,is_terminal))

                #print(f"Replay_buffer length: {len(self.replay_buffer)}")
                if len(self.replay_buffer) >= self.replay_buffer_size:
                    self.train_net()

                state = next_state




            reward_sum = np.sum(rewards_list)
            print(f"Sum or Rewards for Episode: {reward_sum}")
            self.reward_sums_per_episode.append(reward_sum)
            self.epsilon_per_episode.append(self.epsilon)
            torch.save(self.online_net.get_weights(), f'{self.save_name}.pth')

    def pick_action(self, state, episode):

        if episode == 0:
            return np.random.choice([0,1])
        
        random_num = np.random.random()

        if episode > 200:
            self.epsilon *= 0.999
        
        if random_num < self.epsilon:
            return np.random.choice([0,1])
        
        #print(state)
        q_values = self.online_net(torch.tensor(state).to('cuda'))
        arg_max = torch.argmax(q_values).item()
        #print(arg_max)
        return arg_max

    def train_net(self):
        #print(self.replay_buffer)
        
        random_sampling = random.sample(self.replay_buffer, self.sampling_batch_size)

        device = 'cuda'
        s  = torch.as_tensor([b[0] for b in random_sampling], dtype=torch.float32, device=device)
        a  = torch.as_tensor([b[1] for b in random_sampling], dtype=torch.long,   device=device)
        r  = torch.as_tensor([b[2] for b in random_sampling], dtype=torch.float32, device=device)
        ns = torch.as_tensor([b[3] for b in random_sampling], dtype=torch.float32, device=device)
        d  = torch.as_tensor([b[4] for b in random_sampling], dtype=torch.float32, device=device)  # 1.0 if done else 0.0

        #print(s.shape, a.shape, r.shape, ns.shape, d.shape)
        #print(a)

        self.online_net.train()
        self.target_net.eval()

        q_all = self.online_net(s)
        q_taken = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        #print(q_all.shape, q_taken.shape)

        with torch.no_grad():
            next_actions = self.online_net(ns).argmax(dim=1)
            q_next_tgt = self.target_net(ns).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            y = r + self.gamma * q_next_tgt * (1.0 - d)
            y = y.clamp(-100.0, 100.0)

        self.optimizer.zero_grad()
        loss = F.mse_loss(q_taken, y)
        if loss > self.max_loss:
            self.max_loss = loss
            print(loss)
        loss = loss.clamp(-100.0, 100.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()


        tau = 0.005  # “update speed”
        with torch.no_grad():
            for p, tp in zip(self.online_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
        torch.save(self.online_net.get_weights(), f'{self.save_name}.pth')
        #self.update_timer += 1
        #if self.update_timer >= self.update_timer_thresh:
        #    self.update_timer = 0
        #    #print("Copying weights to target network")
        #    self.target_net.set_weights(self.online_net.get_weights())
        #    torch.save(self.online_net.get_weights(), f'{self.save_name}.pth')



    def train_net_old(self):
        #print(self.replay_buffer)
        random_indeces = np.random.choice(list(range(self.replay_buffer_size)), self.sampling_batch_size, replace=False)
        #print(random_indeces)
        random_sampling = [self.replay_buffer[i] for i in random_indeces]

        state_batch = np.zeros(shape=(self.sampling_batch_size, self.state_shape[0]))
        next_state_batch = np.zeros(shape=(self.sampling_batch_size, self.state_shape[0]))

        for state_index, state_values in enumerate(random_sampling):
            state_batch[state_index] = state_values[0] #current state
            next_state_batch[state_index] = state_values[3] #next state


        q_values_for_next_state_using_target_net = self.target_net(torch.tensor(next_state_batch).to(torch.float32))
        q_values_for_state_using_online_net = self.online_net(torch.tensor(state_batch).to(torch.float32))

        output_batch = torch.zeros(size=(self.sampling_batch_size,self.out_shape[0]))

        self.actions_list = []

        for state_index, (state, action, reward, next_state, is_terminal) in enumerate(random_sampling):

            y = reward
            if not is_terminal:
                #print(q_values_for_next_state_using_target_net[state_index])
                y += self.gamma*(torch.max(q_values_for_next_state_using_target_net[state_index]))
            
            self.actions_list.append(action)

            output_batch[state_index] = q_values_for_state_using_online_net[state_index]
            output_batch[state_index][action] = y

        print("Fitting Online Net")
        self.fit(torch.tensor(state_batch).to(torch.float32), torch.tensor(output_batch).to(torch.float32))

        self.update_timer += 1
        if self.update_timer >= self.update_timer_thresh:
            self.update_timer = 0
            print("Copying weights to target network")
            self.target_net.set_weights(self.online_net.get_weights())





    


episodes_to_train = 1000


env = gym.make('CartPole-v1', render_mode=None)
dqn_learner = DQNLearner(env,num_episodes = episodes_to_train,pick_up=False)
dqn_learner.episodic_train()
print(f"Reward List: {dqn_learner.reward_sums_per_episode}")
print(f"Epsilon List: {dqn_learner.epsilon_per_episode}")

