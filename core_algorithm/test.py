import numpy as np
import torch
from PPO import PPO
import gym
import random
from TD3 import TD3



def PPO_main():
    env = gym.make('Pendulum-v1')
    EP_MAX = 10000
    HORIZON = 128
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = PPO(state_dim=3,
                action_dim=1,
                epochs=8,
                learning_rate_P=2e-4,
                learning_rate_V=2e-4,
                eps=0.2,
                gamma=0.99,
                Lambda=1,
                device=device)

    for e in range(EP_MAX):
        s, _ = env.reset()
        start = True
        rewards = 0
        while start:
            for i in range(HORIZON):
                env.render()

                a, log_a = agent.choose_action(s)
                s_,  r, done, _, _ = env.step(a)

                rewards += r
                agent.store_memory(s, a, log_a, r / 10, s_, float(done))
                if done or i == HORIZON - 1:
                    start = False
                    break
                s = s_
            agent.learn()
        if e % 10 == 0:
            print("Episode:{}, reward:{}".format(e, rewards))


# DDPG,TD3这种都类似
def TD3_main():
    env = gym.make('Pendulum-v1')
    EP_MAX = 10000
    HORIZON = 128
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = TD3(state_counters=3,
                action_counters=1,
                batch_size=32,
                memory_size=10000,
                device=device,
                LR_A=2e-3,
                LR_C=2e-3,
                gamma=0.99,
                TAU=0.005,
                d=3,
                noise=0.5)

    for e in range(EP_MAX):
        s, _ = env.reset()
        start = True
        rewards = 0
        while start:
            for i in range(HORIZON):
                env.render()
                a = agent.choose_action(s)
                a_ture = a * 2
                s_,  r, done, _, _ = env.step(a_ture)

                rewards += r
                agent.store_memory(s, a, r / 10, s_, float(done))
                if done or i == HORIZON - 1:
                    start = False
                    break
                s = s_
                if agent.index_memory > agent.memory_size:
                    agent.learn()
        if e % 10 == 0:
            print("Episode:{}, reward:{}".format(e, rewards)) 



if __name__ == '__main__':
    # PPO_main()
    TD3_main()