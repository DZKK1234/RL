"""
DZKK创建
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.distributions import Normal


class Actor(nn.Module): # 建立策略网络为一个分布
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # 得出分布的均值的神经网络
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh())

        # 得出分布的方差的神经网络
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, s):

        action_mean = self.actor_mean(s) * 2
        action_logstd = self.actor_logstd.expand_as(action_mean)
        std = torch.exp(action_logstd)
        dist = Normal(action_mean, std)

        return dist

class V_Net(nn.Module): # 建立Q网络

    def __init__(self, state_dim):
        super(V_Net, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1))

    def forward(self, s):

        out = self.critic(s)
        return out

class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 epochs,
                 learning_rate_P,
                 learning_rate_V,
                 eps,
                 gamma,
                 Lambda,
                 device):
        self.state_dim = state_dim  # 状态维度
        self.action_dim = action_dim  # 动作维度
        self.learning_rate_P = learning_rate_P  # 学习率(策略网络)
        self.learning_rate_V = learning_rate_V  # 学习率(V网络)
        self.eps = eps # 截断常数
        self.gamma = gamma  # 折扣因子
        self.epochs = epochs
        self.memory = {'states': [], 'actions': [], 'log_a': [], 'next_states': [], 'rewards': [], 'dones': []}
        self.Lambda = Lambda
        self.device = device
        #  策略网络以及V网络
        self.policy_net = Actor(self.state_dim, self.action_dim).to(self.device)
        self.V_net = V_Net(self.state_dim).to(self.device)

        # 建立网络loss函数，使用均方误差(MSE)
        self.mse_loss_V = nn.MSELoss()

        # 建立优化器(Adam)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate_P)
        self.optimizer_V = torch.optim.Adam(self.V_net.parameters(), lr=self.learning_rate_V)


    # 选取动作函数
    def choose_action(self, s):
        # 选取动作
        s = torch.FloatTensor(s)
        s = torch.unsqueeze(s, dim=0).to(self.device)
        dist = self.policy_net(s)
        action = dist.sample().detach()
        log_a = dist.log_prob(action).detach() # 得出动作的概率
        return action[0].cpu().numpy(), log_a[0].cpu().numpy()

    # 存储样本
    def store_memory(self, s, a, log_a, r, s_, done):
        self.memory['states'].append(s)
        self.memory['actions'].append(a)
        self.memory['next_states'].append(s_)
        self.memory['rewards'].append(r)
        self.memory['dones'].append(done)
        self.memory['log_a'].append(log_a)

    def compute_advantage(self, td_error):

        Adv = []
        adv = 0
        for i in td_error[::-1]:
            adv = self.gamma * self.Lambda * adv + i[0]
            Adv.insert(0, adv)
        return torch.FloatTensor(Adv).reshape(-1, 1).to(self.device)

    # 随机采样
    def sample(self, transitions):
        sample_memory_s = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        sample_memory_a = torch.FloatTensor(np.array(self.memory['actions'])).to(self.device)
        sample_memory_r = torch.FloatTensor(np.array(self.memory['rewards'])).reshape(sample_memory_s.shape[0], 1).to(self.device)
        sample_memory_s_ = torch.FloatTensor(np.array(self.memory['next_states'])).to(self.device)
        sample_memory_d = torch.FloatTensor(np.array(self.memory['dones'])).reshape(sample_memory_s.shape[0], 1).to(self.device)
        sample_memory_log_a = torch.FloatTensor(np.array(self.memory['log_a'])).reshape(sample_memory_s.shape[0], 1).to(self.device)
        # print(sample_memory_log_a.shape)
        self.memory = {'states': [], 'actions': [], 'log_a': [], 'next_states': [], 'rewards': [], 'dones': []}

        return sample_memory_s, sample_memory_a, sample_memory_log_a, sample_memory_r, sample_memory_s_, sample_memory_d

    def learn(self):

        s, a, log_old, r, s_, d = self.sample(self.memory)

        with torch.no_grad():
            # 这里为求优势函数，不需要梯度
            V_value = self.V_net(s)  # 当前V值
            V_value_ = self.V_net(s_)
            td_error = (r + (1 - d) * self.gamma * V_value_ - V_value).detach().cpu().numpy()  # TD误差
            A = self.compute_advantage(td_error)
            td_target = r + (1 - d) * self.gamma * V_value_

        for _ in range(self.epochs): # 这里可以也可以利用batch_size进行更新
            dist_new = self.policy_net(s)
            log_new = dist_new.log_prob(a)  # 新策略的轨迹概率
            dist_entropy = dist_new.entropy().sum(1, keepdim=True) # 求熵，为下面的策略损失做准备
            ratio = torch.exp(log_new - log_old)
            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * A # 以上两行为PPO中的clip函数
            policy_loss = - torch.min(surr1, surr2) - 0.01 * dist_entropy # 策略网络loss  - 0.01 * dist_entropy
            # 策略网络更新
            self.policy_net.zero_grad()
            policy_loss.mean().backward()
            self.optimizer_policy.step()
            # V网络更新
            loss_v = self.mse_loss_V(td_target, self.V_net(s)) # V网络loss
            self.optimizer_V.zero_grad()
            loss_v.backward()
            self.optimizer_V.step()

