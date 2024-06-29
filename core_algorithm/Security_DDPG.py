"""
DZKK创建
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy



class Actor_Net(nn.Module): # 创建Actor网络

    def __init__(self, state_dim, action_dim):
        super(Actor_Net, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(32, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out = torch.tanh(self.fc3(out2))

        return out

class Critic_Net(nn.Module): # 创建Critic网络

    def __init__(self, state_dim, action_dim):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(32, 1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, s, a):

        x = torch.cat((s, a), 1)
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out = self.fc3(out2)
        return out


class DDPG(object):

    def __init__(self,
                 state_counters,
                 action_counters,
                 batch_size,
                 memory_size,
                 device,
                 LR_A,
                 LR_C,
                 gamma,
                 TAU):

        self.state_counters = state_counters # 状态维度
        self.action_counters = action_counters # 动作维度
        self.memory_size = memory_size # 经验池大小
        self.batch_size = batch_size # 每个更新的batch大小
        self.gamma = gamma # 折扣因子
        self.TAU = TAU # 目标网络更新幅度
        self.device = device # 将数据上传至GPU训练
        self.index_memory = 0 # 来计数样本个数
        self.memory = np.zeros((memory_size, self.state_counters * 2 + self.action_counters + 2)) # 构建经验池
        # 构建在线Actor网络以及目标Actor网络
        self.Actor_Net_eval, self.Actor_Net_target = Actor_Net(self.state_counters, self.action_counters).to(self.device), \
                                                     Actor_Net(self.state_counters, self.action_counters).to(self.device)
        # 构建在线Critic网络以及目标Critic网络
        self.Critic_Net_eval, self.Critic_Net_target = Critic_Net(self.state_counters, self.action_counters).to(self.device), \
                                                       Critic_Net(self.state_counters, self.action_counters).to(self.device)

        # 构建在线代价Critic网络以及目标Critic网络
        self.Critic_Net_eval_cost, self.Critic_Net_target_cost = Critic_Net(self.state_counters, self.action_counters).to(
            self.device), Critic_Net(self.state_counters, self.action_counters).to(self.device)
        # 构建在线Actor网络的优化器
        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=LR_A)
        # 构建在线Critic网络的优化器
        self.optimizer_C = torch.optim.Adam(self.Critic_Net_eval.parameters(), lr=LR_C)
        # 构建在线代价Critic网络的优化器
        self.optimizer_C_cost = torch.optim.Adam(self.Critic_Net_eval_cost.parameters(), lr=LR_C)

        # 构建损失函数即均方误差函数
        self.loss = nn.MSELoss()

    # 构建动作选取函数
    def choose_action(self, observation):

        observation = torch.FloatTensor(observation).to(self.device)
        observation = torch.unsqueeze(observation, 0)
        action = self.Actor_Net_eval(observation)[0].detach()
        return action.cpu().numpy()

    # 将样本存储到经验池中:(s,a,r,s')→D
    def store_memory(self, s, a, r, c, s_):

        memory = np.hstack((s, a, r, c, s_))
        index = self.index_memory % self.memory_size
        self.memory[index, :] = memory
        self.index_memory += 1

    # 学习函数
    def learn_critic_network(self, lamba):
        # 首先从经验池中随机选取batch_size个样本来学习
        sample_memory_index = np.random.choice(min(self.memory_size, self.index_memory), self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory).to(self.device)
        sample_memory_s = sample_memory[:, : self.state_counters]
        sample_memory_s_ = sample_memory[:, - self.state_counters:]
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
        sample_memory_r = sample_memory[:, - self.state_counters - 2 : - self.state_counters - 1]
        sample_memory_c = sample_memory[:, - self.state_counters - 1: - self.state_counters]

        # 根据公式Q*(s,a)=E[R + maxQ*(s',a')],不在选取期望，而是直接走一步看看(即TD(0)思想)，便可以将期望拿掉
        a_ = self.Actor_Net_target(sample_memory_s_)
        a_s = self.Actor_Net_eval(sample_memory_s)

        q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a_)
        q_target_cost = sample_memory_c + self.gamma * self.Critic_Net_target(sample_memory_s_, a_)
        q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
        q_eval_cost = self.Critic_Net_eval_cost(sample_memory_s, sample_memory_a)
        # 最小化Q*(s,a)和r + maxQ*(s',a')之间的差距，我的理解就是目标Critic网络是来拟合maxQ*的
        loss_c = self.loss(q_target, q_eval)
        self.optimizer_C.zero_grad()
        loss_c.backward()
        self.optimizer_C.step()

        loss_c_cost = self.loss(q_target_cost, q_eval_cost)
        self.optimizer_C_cost.zero_grad()
        loss_c_cost.backward()
        self.optimizer_C_cost.step()

        # 根据公式J(π)=E[Q(s,a)],我们最大化J即可求解最优确定性策略,加一个负号即最小化-J
        loss_a = - (torch.mean(self.Critic_Net_eval(sample_memory_s, a_s) -
                               lamba * self.Critic_Net_eval_cost(sample_memory_s, a_s)))
        self.optimizer_A.zero_grad()
        loss_a.backward()
        self.optimizer_A.step()

        d_lamba = lamba

        # 软更新目标网络
        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Critic_Net_eval_cost.parameters(), self.Critic_Net_target_cost.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)