"""
THis is the DQN file for Tetris AI
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import cpprb


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
        self.fc3 = nn.Linear(self.fc2Dims, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Agent():
    def __init__(self, lr, input_shape, n_actions, eps_dec=0.001, eps_min=0.001, reward_shape=2):
        self.lr = lr
        self.gamma = 0.99
        self.reward_shape = reward_shape
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
        self.memory = cpprb.ReplayBuffer(1_000_000,{"obs": {"shape": self.input_shape},
                               "act": {"shape": 1},
                               "rew": {},
                               "next_obs": {"shape": self.input_shape},
                               "done": {},
                               })


    def choose_action(self, state):
        if np.random.random() > self.eps:
            state = torch.Tensor(state).to(self.model.device)
            states = state.unsqueeze(0)
            actions = self.model(states)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice([i for i in range(self.n_actions)])

        return action

    def replace_ntwrk(self):
        self.target.load_state_dict(self.model.state_dict())

    def learn(self, batchSize):
        if self.memory.memCount < batchSize:
            return

        self.model.optimizer.zero_grad()

        if self.learn_cntr % self.replace == 0:
            self.replace_ntwrk()

        state, action, reward, state_, done, players = self.memory.sample(batchSize)

        states  = torch.Tensor(state).to(torch.float32 ).to(self.model.device)
        actions = torch.Tensor(action).to(torch.int64   ).to(self.model.device)
        rewards = torch.Tensor(reward).to(torch.float32 ).to(self.model.device)
        states_ = torch.Tensor(state_).to(torch.float32 ).to(self.model.device)
        dones   = torch.Tensor(done).to(torch.bool    ).to(self.model.device)

        batch_indices = np.arange(batchSize, dtype=np.int64)
        qValue = self.model(states, players)[batch_indices, actions]

        qValues_ = self.target(states_)
        qValue_ = torch.max(qValues_, dim=1)[0]
        qValue_[dones] = 0.0

        td = rewards + self.gamma * qValue_
        loss = self.model.loss(td, qValue)
        loss.backward()
        self.model.optimizer.step()

        #   PER
        error = td - qValue


        self.eps -= self.eps_dec
        if self.eps < self.eps_min:
            self.eps = self.eps_min

        self.learn_cntr += 1
