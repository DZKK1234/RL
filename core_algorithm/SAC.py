import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from copy import deepcopy
from torch.distributions import Normal
class Q_Net(nn.Module): # 建立Q网络

    def __init__(self, state_dim, action_dim):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc2 = nn.Linear(32, 16)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(16, 1)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)

    def forward(self, s, a):

        x = torch.cat((s, a), 1)
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out = self.fc3(out2)
        return out

#建立策略网络
class policy_NET(nn.Module): # 建立策略网络为一个分布
    def __init__(self, state_dim, action_dim):
        super(policy_NET, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)

        # 得出分布的均值的神经网络
        self.fc_mean = nn.Linear(32, action_dim)
        self.fc_mean.weight.data.normal_(0, 0.1)
        self.fc_mean.bias.data.normal_(0.1)
        # 得出分布的方差的神经网络
        self.fc_std = nn.Linear(32, action_dim)
        self.fc_std.weight.data.normal_(0, 0.1)
        self.fc_std.bias.data.normal_(0.1)

    def forward(self, s):

        out = torch.relu(self.fc1(s))
        out = torch.relu(self.fc2(out))
        #得出分布的均值
        mean = self.fc_mean(out)
        #得出分布的方差
        std = self.fc_std(out)
        std = torch.clamp(std, -20, 2)
        std = torch.exp(std)
        # 构建均值为mean，方差为std的高斯分布
        normal1 = Normal(mean, std)
        # 采样动作
        v_a = normal1.rsample()
        # 将动作映射到[-1, 1]
        action = torch.tanh(v_a)
        # 计算当前动作的概率密度
        logp_pi = (normal1.log_prob(v_a).sum(axis=1, keepdim=True) -
                   (2 * (np.log(2) - v_a - F.softplus(-2 * v_a))).sum(axis=1, keepdim=True))

        return action, logp_pi

class SAC(object):

    def __init__(self,
                 state_dim,
                 action_dim,
                 memory_size,
                 batch_size,
                 learning_rate_P,
                 learning_rate_Q,
                 learning_rate_alpha,
                 gamma,
                 alpha,
                 TAU,
                 target_entropy):

        self.state_dim = state_dim   # 状态维度
        self.action_dim = action_dim  # 动作维度
        self.memory_size = memory_size  # 经验池大小
        self.batch_size = batch_size  # 选取批量的大小
        self.learning_rate_P = learning_rate_P   # 学习率(Q网络)
        self.learning_rate_Q = learning_rate_Q   # 学习率(策略网络)
        self.learning_rate_alpha = learning_rate_alpha   # 学习率(权衡熵的影响的系数)
        self.gamma = gamma   # 折扣因子
        # self.alpha = alpha  # 权衡熵的影响的系数
        self.log_alpha = torch.log(torch.tensor(alpha)) # 权衡熵的影响的系数
        self.TAU = TAU   # 软更新系数
        self.target_entropy = target_entropy # 更新alpha
        self.index_memory = 0   # 来计数经验样本的条数
        # self.memory_ = 20000
        # self.device = device
        self.memory = np.zeros((self.memory_size, self.state_dim * 2 + self.action_dim + 2))

        #建立网络，其中有4个Q网络：两个Q网络（来选取Q值较小的网络），两个目标Q网络
        self.Q1 = Q_Net(self.state_dim, self.action_dim)
        self.Q2 = Q_Net(self.state_dim, self.action_dim)
        self.target_Q1 = deepcopy(self.Q1)
        self.target_Q2 = deepcopy(self.Q2)
        #  一个策略网络
        self.policy_net = policy_NET(self.state_dim, self.action_dim)

        #建立网络loss函数，使用均方误差(MSE)
        self.mse_loss_Q1 = nn.MSELoss()
        self.mse_loss_Q2 = nn.MSELoss()

        # 建立优化器(Adam)
        self.optimizer_Q1 = torch.optim.Adam(self.Q1.parameters(), lr=self.learning_rate_Q)
        self.optimizer_Q2 = torch.optim.Adam(self.Q2.parameters(), lr=self.learning_rate_Q)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate_P)
        self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=self.learning_rate_alpha)

    # 选取动作函数
    def choose_action(self, s):

        s = torch.FloatTensor(s)
        s = torch.unsqueeze(s, dim=0)
        action = self.policy_net(s)[0].detach().cpu().numpy()
        return action[0]

    #存储样本
    def store_memory(self, s, a, r, s_, done):

        memory = np.hstack((s, a, r, s_, done))
        index = self.index_memory % self.memory_size
        self.memory[index, :] = memory
        self.index_memory += 1

    #随机采样
    def sample(self):

        sample_memory_index = np.random.choice(min(self.memory_size, self.index_memory), self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, : self.state_dim]
        sample_memory_s_ = sample_memory[:, - self.state_dim - 1: -1]
        sample_memory_a = sample_memory[:, self.state_dim: self.state_dim + self.action_dim]
        sample_memory_r = sample_memory[:, self.state_dim + self.action_dim: self.state_dim + self.action_dim + 1]
        sample_memory_done = sample_memory[:, -1:]
        return sample_memory_s, sample_memory_a, sample_memory_r, sample_memory_s_, sample_memory_done

    #计算目标值
    def Q_target(self, r, s_, d):

        # 输入下一个状态选取动作并计算当前策略下的动作概率密度
        action_, log_pi_ = self.policy_net(s_)
        entropy = - log_pi_
        # 选取小的Q值减小过估计
        q1_target = self.target_Q1(s_, action_)
        q2_target = self.target_Q2(s_, action_)
        min_target = torch.min(q1_target, q2_target)
        next_value = min_target + self.log_alpha.exp() * entropy
        y_value = r + (1 - d) * self.gamma * next_value

        return y_value

    #学习
    def learning(self):
        # 随机采样
        sample_memory_s, sample_memory_a, sample_memory_r, sample_memory_s_, sample_memory_done = self.sample()

        #更新Q网络
        target_value = self.Q_target(sample_memory_r, sample_memory_s_, sample_memory_done)

        loss_q1 = self.mse_loss_Q1(target_value, self.Q1(sample_memory_s, sample_memory_a))
        loss_q2 = self.mse_loss_Q2(target_value, self.Q2(sample_memory_s, sample_memory_a))
        #Q1网络更新
        self.optimizer_Q1.zero_grad()
        loss_q1.backward(retain_graph=True)
        self.optimizer_Q1.step()

        # Q2网络更新
        self.optimizer_Q2.zero_grad()
        loss_q2.backward()
        self.optimizer_Q2.step()

        #策略网络更新
        action_s, log_pi = self.policy_net(sample_memory_s)
        entropy = - log_pi
        Q1_action = self.Q1(sample_memory_s, action_s)
        Q2_action = self.Q2(sample_memory_s, action_s)
        min_on = torch.min(Q1_action, Q2_action)
        policy_loss = torch.mean(- self.log_alpha.exp() * entropy - min_on)
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.optimizer_alpha.zero_grad()
        alpha_loss.requires_grad_(True)
        alpha_loss.backward()
        self.optimizer_alpha.step()

        for parm, target_parm in zip(self.Q1.parameters(), self.target_Q1.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Q2.parameters(), self.target_Q2.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)