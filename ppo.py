
import argparse
import pickle
from collections import namedtuple

import os
import numpy as np
import matplotlib.pyplot as plt

import time
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# # Parameters
# parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
# parser.add_argument(
#     '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
# parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
# parser.add_argument('--render', action='store_true', default=True, help='render the environment')
# parser.add_argument(
#     '--log-interval',
#     type=int,
#     default=10,
#     metavar='N',
#     help='interval between training status logs (default: 10)')
# args = parser.parse_args()

# env = gym.make('MarketEnv').unwrapped
# num_state = env.observation_space.shape[0]
# num_action = env.action_space.shape[0]
# torch.manual_seed(args.seed)
# env.seed(args.seed)

# Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state'])
# TrainRecord = namedtuple('TrainRecord',['episode', 'reward'])

class Actor(nn.Module):
    def __init__(self,num_state, num_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64,8)
        self.mu_head = nn.Linear(8, num_action)
        self.sigma_head = nn.Linear(8, num_action)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        mu = self.mu_head(x)
        sigma = self.sigma_head(x)

        return mu, sigma

class Critic(nn.Module):
    def __init__(self,num_state):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 8)
        self.state_value= nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.state_value(x)
        return value

class PPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self, num_state, num_action):
        self.actor_net = Actor(num_state, num_action).float()
        self.critic_net = Critic(num_state).float()
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.actor_net(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = torch.clamp(action, -1.0, 1.0)
        return action.item(), action_log_prob.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self, gamma):
        self.training_step += 1

        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor(
            [t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        # Normalize rewards
        reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        with torch.no_grad():
            target_v = reward + gamma * self.critic_net(next_state)
        advantage = (target_v - self.critic_net(state)).detach()

        for _ in range(self.ppo_epoch):
            sampler = BatchSampler(
                SubsetRandomSampler(range(len(self.buffer))),
                batch_size=self.batch_size,
                drop_last=False)
            for indices in sampler:
                indices = torch.tensor(indices)
                mu, sigma = self.actor_net(state[indices])
                dist = Normal(mu, sigma)
                action_log_prob = dist.log_prob(action[indices])
                ratio = torch.exp(action_log_prob - old_action_log_prob[indices])

                surr1 = ratio * advantage[indices]
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[indices]
                action_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[indices]), target_v[indices])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]
        self.counter = 0

# def main():

#     agent = PPO()

#     training_records = []
#     running_reward = -1000

#     for i_epoch in range(1000):
#         score = 0
#         state = env.reset()
#         if args.render: env.render()
#         for t in range(200):
#             action, action_log_prob = agent.select_action(state)
#             next_state, reward, done, info = env.step(action)
#             trans = Transition(state, action, reward, action_log_prob, next_state)
#             if args.render: env.render()
#             if agent.store_transition(trans):
#                 agent.update()
#             score += reward
#             state = next_state

#         running_reward = running_reward * 0.9 + score * 0.1
#         training_records.append(TrainingRecord(i_epoch, running_reward))
#         if i_epoch % 10 ==0:
#             print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
#         if running_reward > -200:
#             print("Solved! Moving average score is now {}!".format(running_reward))
#             env.close()
#             agent.save_param()
#             break

# if __name__ == '__main__':
#     main()