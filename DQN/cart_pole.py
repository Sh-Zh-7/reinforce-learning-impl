import math
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import warnings
warnings.filterwarnings("ignore")

# 超参数部分
EPISODE = 3000  # episode限制
STEP = 300  # 采样上限
BATCH_SIZE = 32
GAMMA = 0.9


class ReplayMemory:
    """
    其实就相当于维护了一个新的数据结构
    然后使用往里面存数据，取数据，就是这么简单
    还有一点比较重要的就是里面存放的是五元组
    """
    def __init__(self, capacity):
        # 一个新的类，专门用来存储五元组
        self.Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "is_done"))
        # 模仿deque创建一个用来存储多个五元组的数据结构
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """ 用类似循环队列的方式来添加新的元素"""
        if len(self.memory) < self.capacity:
            self.memory.append(self.Transition(*args))
        else:
            self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ 就是往这个结构中取出数据喽，只不过每次取的个数不唯一罢了 """
        # 注意返回的是tuple的列表
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(QNet, self).__init__()
        self.hidden = nn.Linear(in_features, 20)
        self.output = nn.Linear(20, out_features)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


class DQN:
    def __init__(self):
        # Environment part
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        # Q net part
        self.Q_net = QNet(self.state_dim, self.action_dim)  # Choose actions
        self.optimizer = optim.SGD(self.Q_net.parameters(), lr=0.2)
        self.loss_fn = nn.MSELoss()
        # Other
        self.count = 0
        self.replay_buffer = ReplayMemory(10000)

    def __SelectAction(self, state):
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        # 这里很奇怪的一点是, 我们能使用类似宏定义的变量，却不能使用普通的全局变量，得通过global声明以后才能用
        # 更新Epsilon
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * self.count / EPS_DECAY)
        self.count += 1
        # 使用epsilon贪心法选择动作
        # 注意我们要返回整型
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            # 看来每次返回一个action的时候，要对我们的action进行编码
            return np.argmax(self.Q_net(state).detach().numpy())

    def __Train(self):
        mini_batch = self.replay_buffer.sample(BATCH_SIZE)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state = [data[3] for data in mini_batch]

        # Forward
        # 本质是把后续状态对现在状态的估计当做真值。把对现在状态当做估计值（什么乱七八糟的）
        # 每次只能更新一个动作
        y_true_batch = []
        Q_value_batch = self.Q_net(next_state)  # 神经网络的输出
        for i in range(BATCH_SIZE):
            done = mini_batch[i][4]
            if done:
                y_true_batch.append(reward_batch[i])
            else:
                y_true_batch.append(reward_batch[i] + GAMMA * torch.max(Q_value_batch[i], dim=0)[0])
        y_true_batch = torch.tensor(y_true_batch, dtype=torch.float, requires_grad=True).unsqueeze(1)
        # 我们估计的状态
        action_batch = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1)
        y_pred_batch = self.Q_net(state_batch).gather(1, action_batch)

        loss = self.loss_fn(y_pred_batch, y_true_batch)  # 注意这里是当前选择ACTION的Q值
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def Episode(self):
        state = env.reset()     # State就是各个状态的编码(在这里是传感器的返回值)，多个不同的状态堆叠形成向量
        for step in range(STEP):
            # 选择动作并与环境交互, 获得奖励
            action = self.__SelectAction(state)
            next_state, reward, done, _ = env.step(action)
            reward = -1 if done else 0.1
            # 更新
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            if len(self.replay_buffer) > BATCH_SIZE:
                self.__Train()
            if done:
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
    solution = DQN()
    for episode in range(EPISODE):
        solution.Episode()
        solution.Test(episode)
    env.close()



