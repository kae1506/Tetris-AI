import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
import numpy as np
from cpprb import ReplayBuffer

"""
Deuling for Tetris AI
"""

np.random.seed(0)

class Network(nn.Module):
    def __init__(self, alpha, inputShape, numActions):
        super().__init__()
        self.inputShape = inputShape
        self.numActions = numActions
        self.fc1Dims = 1024
        self.fc2Dims = 512

        self.fc1 = nn.Linear(*self.inputShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)

        self.value = nn.Linear(self.fc2Dims, 1)
        self.advantage = nn.Linear(self.fc2Dims, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.value(x)
        a = self.advantage(x)

        return v, a

class Agent():
    def __init__(self, lr, input_shape, n_actions, eps_dec=0.001, eps_min=0.001):
        self.lr = lr
        self.gamma = 0.99
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.surprise = 0.5

        self.learn_cntr = 0
        self.replace = 100

        self.eps = 1
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        self.model = Network(lr, self.input_shape, self.n_actions)
        self.target = Network(lr, self.input_shape, self.n_actions)
        self.memory = ReplayBuffer(1_000_000,{"obs": {"shape": self.input_shape},
                               "act": {"shape": 1},
                               "rew": {},
                               "next_obs": {"shape": self.input_shape},
                               "done": {},
                               "player": {"dtype": {np.short}}})



    def choose_action(self, state):
        if np.random.random() > self.eps:
            state = T.Tensor(state).to(self.model.device)
            states = state.unsqueeze(0)
            _, actions = self.model(states)
            action = T.argmax(actions).item()
        else:
            action = env.action_space.sample()

        return action

    def replace_ntwrk(self):
        self.target.load_state_dict(self.model.state_dict())

    def save_checkpoint(self, filename, memoryfile):
        if filename == None:
            filename = "DDDQN.weights"

        save_dict = {'policy': self.model.state_dict(),
             'optim': self.model.optimizer.state_dict(),
             'target': self.target.state_dict()
        }

        T.save(save_dict, filename)
        self.memory.save(memoryfile)

    def load_checkpoint(self, filename, memoryfile):
        load_dict = T.load(filename)

        self.model.load_state_dict(load_dict['policy'])
        self.model.optimizer.load_state_dict(load_dict['optim'])
        self.target.load_state_dict(load_dict['target'])

        self.model.train()
        self.target.train()
        self.memory.load(memoryfile)

    def learn(self, batchSize):
        if self.memory.memCount < batchSize:
            return

        self.model.optimizer.zero_grad()

        if self.learn_cntr % self.replace == 0:
            self.replace_ntwrk()

        state, action, reward, state_, done = self.memory.sample(batchSize)

        states  = T.Tensor(state).to(T.float32 ).to(self.model.device)
        actions = T.Tensor(action).to(T.int64   ).to(self.model.device)
        rewards = T.Tensor(reward).to(T.float32 ).to(self.model.device)
        states_ = T.Tensor(state_).to(T.float32 ).to(self.model.device)
        dones   = T.Tensor(done).to(T.bool    ).to(self.model.device)

        batch_indices = np.arange(batchSize, dtype=np.int64)
        vState, aState = self.model(states)
        vNext, aNext = self.target(states_)

        qValue = T.add(vState, (aState - aState.mean(dim=1, keepdim=True)))[batch_indices, actions]

        qValue_ = T.add(vNext, (aNext - aNext.mean(dim=1, keepdim=True))).max(dim=1)[0]
        qValue_[dones] = 0.0

        td = rewards + self.gamma * qValue_
        loss = self.model.loss(td, qValue).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()

        self.eps -= self.eps_dec
        if self.eps < self.eps_min:
            self.eps = self.eps_min

        self.learn_cntr += 1

