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

    def __init__(self, state_dim_total, action_dim_total):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim_total + action_dim_total, 256)
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

class MADDPG(object):

    def __init__(self,
                 n_agent,
                 state_counters_list,
                 action_counter_list,
                 batch_size,
                 memory_size,
                 LR_A,
                 LR_C,
                 gamma,
                 TAU):

        self.n_agent = n_agent # 输入智能体编号,例如[0, 1, 2]
        self.number_of_agent = len(self.n_agent)
        # 构建存放状态和动作维度的字典
        self.state_counters_dict = dict(zip(self.n_agent, state_counters_list)) # 例如{“0”：10， “1”：11， “2”：12}
        self.action_counter_dict = dict(zip(self.n_agent, action_counter_list)) # 例如{“0”：2， “1”：3， “2”：4}
        #构建存放状态和动作的字典
        self.state_dict = dict(zip(self.n_agent, self.state_counters_list))
        self.action_dict = dict(zip(self.n_agent, self.action_counter_dict))
        self.state_next_dict = dict(zip(self.n_agent, self.state_counters_list))
        self.reward_dict = dict(zip(self.n_agent, self.action_counter_dict))
        # 构建存放随机采样的样本的字典
        self.sample_dict = dict(zip(self.n_agent, self.action_counter_dict))
        # 构建存放loss的集合
        self.loss_dict_C = dict(zip(self.n_agent, self.action_counter_dict))
        self.loss_dict_A = dict(zip(self.n_agent, self.action_counter_dict))

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU
        self.index_memory = 0

        #构建经验池字典
        self.memory_list = [np.zeros((self.memory_size, self.state_counters_dict[i]  * 2 + self.action_counter_dict[i] + 1))
                            for i in action_counter_dict]
        self.memory_dict = dict(zip(self.n_agent, self.memory_list))

        # 构建Actor网络字典
        self.Actor_Net_list = [Actor_Net(self.state_counters_dict[i], self.action_counter_dict[i]) for i in action_counter_dict]
        self.Actor_Net_dict = dict(zip(self.n_agent, self.Actor_Net_list))
        self.Actor_Net_target_dict = deepcopy(self.Actor_Net_dict)

        # 构建Critic网络字典
        self.Critic_Net_list = [Critic_Net(sum(self.state_counters_dict[i] for i in self.state_counters_dict),
                                           sum(self.action_counter_dict[i] for i in self.action_counter_dict)) for j in range(self.number_of_agent)]
        self.Critic_Net_dict = dict(zip(self.n_agent, self.Critic_Net_list))
        self.Critic_Net_target_dict = deepcopy(self.Critic_Net_dict)

        # 构建优化器字典
        self.optimizer_A_dict = {}
        self.optimizer_C_dict = {}
        for i in self.Actor_Net_dict:
            self.optimizer_A_dict.update({i, torch.optim.Adam(self.Actor_Net_dict[i].parameters(), lr=LR_A)})
        for i in self.Critic_Net_dict:
            self.optimizer_C_dict.update({i, torch.optim.Adam(self.Critic_Net_dict[i].parameters(), lr=LR_C)})

        # 构建损失函数
        self.loss = nn.MSELoss()

    def choose_action(self, observation):

        for i in self.state_counters_dict:
            observation_i = self.observation[i]
            observation_i = torch.FloatTensor(observation_i)
            observation_i = torch.unsqueeze(observation_i, 0)

            action_i = self.Actor_Net_dict[i](observation_i)[0].detach().cpu().numpy()
            self.action_dict[i] = action_i
        # action = np.hstack((action1, action2, action3))
        return self.action_dict


    def agg_state_action(self):

        state_total = np.array([])
        state_next_total = np.array([])
        action_total = np.array([])
        reward_total = np.array([])
        for i in self.n_agent:
            state_total = np.hstack((state_total, self.state_dict[i]))
            action_total = np.hstack((action_total, self.action_dict[i]))
            reward_total = np.hstack((reward_total, self.reward_dict[i]))
            state_total_next = np.hstack((state_next_total, self.state_next_dict[i]))

        return state_total, action_total, reward_total, state_next_total


    def store_memory(self, s, a, r, s_):

        for i in s:
            memory_i = np.hstack((s[i], a[i], r[i], s_[i]))
            index = self.index_memory % self.memory_size
            self.memory_dict[i][index, :] = memory_i
        self.index_memory += 1


    def learn(self):
        # 采样学习，将不同智能体的采样样本放入各自的字典中
        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        for i in self.sample_dict:
            sample_memory = self.memory_dict[i][sample_memory_index, :]
            sample_memory = torch.FloatTensor(sample_memory)
            self.sample_dict[i] = sample_memory
        # 在学习中需要所有智能体的状态集合以及动作集合(包括采样集合中的状态集合以及动作集合，还有在s'下的动作集合)
        a_dict = dict(zip(self.n_agent, self.action_counter_list))
        a_next_total = torch.tensor([])
        x_next_total = torch.tensor([])
        a_total = torch.tensor([])
        x_total = torch.tensor([])

        for i in a_dict:
            a_agent_i = self.Actor_Net_dict[i](self.sample_dict[i][:, - self.state_counters_dict[i]:])
            a_dict[i] = self.sample_dict[i][:, self.state_counters_dict[i] : self.state_counters_dict[i] + self.action_counter_dict[i]]
            a_next_total = torch.hstack((a_next, a_agent_i))
            a_total = torch.hstack((a_total, self.sample_dict[i][:, self.state_counters_dict[i] : self.state_counters_dict[i] + self.action_counter_dict[i]]))
            x_total = torch.hstack((x_total, self.sample_dict[i][:, : self.state_counters_dict[i]]))
            x_next_total = torch.hstack((x_next_total, self.sample_dict[i][:, - self.state_counters_dict[i]:]))
        # 对于每一个智能体来进行更新学习
        for i in self.Critic_Net_dict:
            q_target_i = (self.sample_dict[i][:, - self.state_counters_dict[i] - 1 : - self.state_counters_dict[i]]
                         + self.gamma * self.Critic_Net_target_dict[i](x_next_total, a_next_total))
            q_eval_i = self.Critic_Net_dict[i](x_total, a_total)

            self.loss_dict_C[i] = self.loss(q_target1, q_eval_1)
            self.optimizer_C_dict[i].zero_grad()
            self.loss_dict_C[i].backward()
            self.optimizer_C_dict[i].step()

            a_s_agent_i = self.Actor_Net_dict[i](self.sample_dict[i][:, 0:self.state_counters_dict[i]])
            a_dict_copy = deepcopy(a_dict)
            a_dict_copy[i] = a_s_agent_i
            a_new = torch.tensor([])
            for j in a_dict:
                torch.hstack((a_new, a_dict_copy[j]))
            self.loss_dict_A[i] = - torch.mean(self.Critic_Net_dict[i](x_total, a_new))
            self.optimizer_A_dict[i].zero_grad()
            self.loss_dict_A[i].backward()
            self.optimizer_A_dict[i].step()

            for parm, target_parm in zip(self.Actor_Net_dict[i].parameters(), self.Actor_Net_target_dict[i].parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
            for parm, target_parm in zip(self.Critic_Net_dict[i].parameters(), self.Critic_Net_target_dict[i].parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
