import gym

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import warnings
warnings.filterwarnings("ignore")

EPISODE = 3000  # episode限制
STEP = 300  # 采样上限
BATCH_SIZE = 32
GAMMA = 0.9

# # 就用这个来代替我们的experience replay把
# Transition = namedtuple("Transition", ("states", "actions", "rewards"))
# 这样的做法确实很方便，但是这并非我们传统意义上的结构体，比如我们不能自由地访问其成员

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def Store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def Empty(self):
        self.states = []
        self.actions = []
        self.rewards = []

class PolicyNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(PolicyNet, self).__init__()
        self.hidden = nn.Linear(in_features, 20)
        self.output = nn.Linear(20, out_features)

    def forward(self, x):
        """ 返回概率值 """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float, requires_grad=True)
        x = F.relu(self.hidden(x))
        # 由于对于动作的选择是一个二分类问题，所以我这里使用sigmoid做激活函数
        x = F.softmax(self.output(x))
        return x


class PG:
    def __init__(self):
        # Environment part
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        # Memory replay
        self.memory = Memory()
        # Policy network(原来这里才引入了policy network,之前的DQN里面根本就没有这个概念)
        self.policy_network = PolicyNet(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.2)

    def __SelectAction(self, state):
        policy = self.policy_network(state).detach().numpy()
        # 用random choice只是其中一种写法而已, pytorch已经集成了一种更好的写法
        return np.random.choice(policy.shape[0], p=policy)

    def __Train(self, step):
        # 说实话，这里的长度就是我们的step
        # Step1: Get state's value
        values = np.zeros(step)
        temp = 0
        for i in reversed(range(step)):
            temp = self.memory.rewards[i] + temp * GAMMA
            values[i] = temp
        # Whiten
        values -= np.mean(values)
        values /= np.std(values)

        # Step2: Calculate reinforce
        # 确实，即便是MCPG，也需要接受一个或者多个样本
        # 之前在做竞赛的时候，就没有碰到过要输入一个单独样本的情况
        actions = torch.tensor(self.memory.actions)
        values = torch.tensor(values, dtype=torch.float)
        policies = self.policy_network(self.memory.states)
        self.optimizer.zero_grad()

        for i in range(step):
            m = Categorical(policies[i])
            loss = -m.log_prob(actions[i]) * values[i]
            # Loss倒确实是标量，我们之前搞图像的时候，loss也是标量
            loss.backward(retain_graph=True)

        self.optimizer.step()

    def Episode(self):
        state = env.reset()
        for step in range(STEP):
            action = self.__SelectAction(state)
            next_state, reward, done, _ = env.step(action)
            self.memory.Store(state, action, reward)
            state = next_state
            if done:
                self.__Train(step)
                self.memory.Empty()
                break

    def Test(self, episode, TEST=10):
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = self.__SelectAction(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward / TEST
            print("Episode: ", (episode + 100), "Evaluation Average Reward:", avg_reward)


if __name__ == "__main__":
    # 环境部分
    env = gym.make('CartPole-v0')
    solution = PG()
    for episode in range(EPISODE):
        solution.Episode()
        solution.Test(episode)
    env.close()
