"""
DZKK创建TD3算法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class Actor_Net(nn.Module): # 创建Actor网络

    def __init__(self, state_dim, action_dim):
        super(Actor_Net, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(64, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out = torch.tanh(self.fc3(out2))

        return out

class Critic_Net(nn.Module): # 创建Critic网络

    def __init__(self, state_dim, action_dim):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, s, a):

        x = torch.cat((s, a), 1)
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out = self.fc3(out2)
        return out

class TD3(object):

    def __init__(self,
                 state_counters,
                 action_counters,
                 batch_size,
                 memory_size,
                 device,
                 LR_A,
                 LR_C,
                 gamma,
                 TAU,
                 d,
                 noise):

        self.state_counters = state_counters # 状态维度
        self.action_counters = action_counters # 动作维度
        self.memory_size = memory_size # 经验池大小
        self.batch_size = batch_size # 每个更新的batch大小
        self.gamma = gamma # 折扣因子
        self.TAU = TAU # 目标网络更新幅度
        self.device = device # 将数据上传至GPU训练
        self.index_memory = 0 # 来计数样本个数
        self.step = 0 # 计数更新步数
        self.d = d # 延迟更新步数
        self.noise = noise # 噪声大小
        self.memory = np.zeros((self.memory_size, self.state_counters * 2 + self.action_counters + 2)) # 构建经验池
        # 构建在线Actor网络以及目标Actor网络
        self.Actor_Net_eval, self.Actor_Net_target = Actor_Net(self.state_counters, self.action_counters).to(self.device), \
                                                     Actor_Net(self.state_counters, self.action_counters).to(self.device)
        # 构建在线Critic网络以及目标Critic网络(其中有4个网络，为防止过估计)
        self.Critic_Net_eval1 = Critic_Net(self.state_counters, self.action_counters).to(self.device)
        self.Critic_Net_target1 = deepcopy(self.Critic_Net_eval1)
        self.Critic_Net_eval2 = Critic_Net(self.state_counters, self.action_counters).to(self.device)
        self.Critic_Net_target2 = deepcopy(self.Critic_Net_eval2)

        # 构建在线Actor网络的优化器
        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=LR_A)
        # 构建在线Critic网络的优化器
        self.optimizer_C1 = torch.optim.Adam(self.Critic_Net_eval1.parameters(), lr=LR_C)
        self.optimizer_C2 = torch.optim.Adam(self.Critic_Net_eval2.parameters(), lr=LR_C)
        # 构建损失函数即均方误差函数
        self.loss = nn.MSELoss()

    # 构建动作选取函数
    def choose_action(self, observation):

        observation = torch.FloatTensor(observation).to(self.device)
        observation = torch.unsqueeze(observation, 0)
        action = self.Actor_Net_eval(observation)[0].detach()
        return action.cpu().numpy()

    # 将样本存储到经验池中:(s,a,r,s')→D
    def store_memory(self, s, a, r, s_, done):

        memory = np.hstack((s, a, r, s_, done))
        index = self.index_memory % self.memory_size
        self.memory[index, :] = memory
        self.index_memory += 1

    # 学习函数
    def learn(self):
        self.step += 1
        # 首先从经验池中随机选取batch_size个样本来学习
        sample_memory_index = np.random.choice(min(self.memory_size, self.index_memory), self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory).to(self.device)
        sample_memory_s = sample_memory[:, : self.state_counters]
        sample_memory_s_ = sample_memory[:, - self.state_counters - 1: -1]
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
        sample_memory_r = sample_memory[:, - self.state_counters -2 : - self.state_counters - 1]
        sample_memory_d = sample_memory[:,  -1:]
        with torch.no_grad():
            # 根据公式Q*(s,a)=E[R + maxQ*(s',a')],不在选取期望，而是直接走一步看看(即TD(0)思想)，便可以将期望拿掉
            a_ = self.Actor_Net_target(sample_memory_s_)
            # 在s_的选取动作上加入噪声，能够在该值的附近进行探索，使得学的策略更加健壮
            a_noise = (a_ + torch.normal(0, 1, a_.shape).clamp(- self.noise, self.noise).to(self.device)).clamp(-1, 1)
            # 算出两个网络的y值，并比较大小取小来防止Q值的过估计
            q_target1 = sample_memory_r + self.gamma * self.Critic_Net_target1(sample_memory_s_, a_noise) * (1 - sample_memory_d)
            q_target2 = sample_memory_r + self.gamma * self.Critic_Net_target2(sample_memory_s_, a_noise) * (1 - sample_memory_d)
            q_target = torch.min(q_target1, q_target2)
        # 计算两个Q网络的Q值
        q_eval1 = self.Critic_Net_eval1(sample_memory_s, sample_memory_a)
        q_eval2 = self.Critic_Net_eval2(sample_memory_s, sample_memory_a)
        # 最小化Q*(s,a)和r + maxQ*(s',a')之间的差距，我的理解就是目标Critic网络是来拟合maxQ*的，两个网络分别更新
        # 第一个Q网络更新
        loss_c1 = self.loss(q_target, q_eval1)
        self.optimizer_C1.zero_grad()
        loss_c1.backward()
        self.optimizer_C1.step()
        # 第二个Q网络更新
        loss_c2 = self.loss(q_target, q_eval2)
        self.optimizer_C2.zero_grad()
        loss_c2.backward()
        self.optimizer_C2.step()
        if self.step % self.d == 0:
            # 根据公式J(π)=E[Q(s,a)],我们最大化J即可求解最优确定性策略,加一个负号即最小化-J，同样需要取较小的Q值

            loss_a = - torch.mean(torch.min(self.Critic_Net_eval1(sample_memory_s, self.Actor_Net_eval(sample_memory_s)),
                                            self.Critic_Net_eval2(sample_memory_s, self.Actor_Net_eval(sample_memory_s))))
            self.optimizer_A.zero_grad()
            loss_a.backward()
            self.optimizer_A.step()

            # 软更新目标网络
            for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
            for parm, target_parm in zip(self.Critic_Net_eval1.parameters(), self.Critic_Net_target1.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
            for parm, target_parm in zip(self.Critic_Net_eval2.parameters(), self.Critic_Net_target2.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
