"""
DZKK创建DQN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class Q_Network(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc2 = nn.Linear(256, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(64, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)

    def forward(self, x):
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out = self.fc3(out2)
        return out

class DQN(object):

    def __init__(self,
                 state_dim,
                 action_dim,
                 gamma,
                 memory_size,
                 learning_rate,
                 epsilon,
                 batch_size,
                 delay_update):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        # self.device = device
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = 0.0003
        self.batch_size = batch_size
        self.delay_update = delay_update
        self.loss = []
        self.counter = 0
        self.Q_eval_network, self.Q_target_network = Q_Network(self.state_dim, self.action_dim), \
                                                     Q_Network(self.state_dim, self.action_dim)
        self.replay_memory = np.zeros((self.memory_size, self.state_dim * 2 + 1 + 2))
        self.index_memory = 0
        self.optimizer = torch.optim.Adam(self.Q_eval_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def choose_action(self, observation):

        s = torch.unsqueeze(torch.FloatTensor(observation), 0)
        a = self.Q_eval_network(s)

        if self.epsilon > 0.1:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = 0.1
        if np.random.uniform() > self.epsilon:

            action = torch.max(a, 1)[1].detach().numpy()[0]
        else:
            action = np.random.randint(0, self.action_dim)

        return action

    def store_memory(self, s, a, r, s_, d):

        memory = np.hstack((s, a, r, s_, d))
        index1 = self.index_memory % self.memory_size
        self.replay_memory[index1, :] = memory
        self.index_memory += 1

    def sample_memory(self):
        sample_memory_index = np.random.choice(min(self.memory_size, self.index_memory), self.batch_size)
        sample_memory = self.replay_memory[sample_memory_index, :]
        sample_memory_s = torch.FloatTensor(sample_memory[:, : self.state_dim])
        sample_memory_a = torch.LongTensor(sample_memory[:, self.state_dim: 1 + self.state_dim])
        sample_memory_r = torch.FloatTensor(sample_memory[:, - self.state_dim - 2: - self.state_dim - 1])
        sample_memory_s_ = torch.FloatTensor(sample_memory[:, - self.state_dim - 1: -1])
        sample_memory_d = torch.FloatTensor(sample_memory[:, -1:])
        return sample_memory_s, sample_memory_a, sample_memory_r, sample_memory_s_, sample_memory_d


    def learn(self):
        self.counter += 1
        # 每过delay_update步长跟新target_Q
        if self.counter % self.delay_update == 0:
            for parm, target_parm in zip(self.Q_eval_network.parameters(), self.Q_target_network.parameters()):
                target_parm.data.copy_(parm.data)
        s, a, r, s_, d = self.sample_memory()
        # 根据a来选取对应的Q(s,a)
        q_eval = self.Q_eval_network(s).gather(1, a)
        # 计算target_Q
        q_target = self.gamma * torch.max(self.Q_target_network(s_), 1)[0].reshape(self.batch_size, 1)
        y = r + (1 - d) * q_target

        # 网络更新
        loss = self.loss_function(y, q_eval)
        self.loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()