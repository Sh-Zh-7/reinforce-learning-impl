import gym
import math
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Hyper-parameters
EPISODE = 3000      # Episode limit
STEP = 300          # Sample's upper bound
BATCH_SIZE = 32
GAMMA = 0.9
C = 10


class ReplayMemory:
    def __init__(self, capacity=10000):
        # Structure for five element tuple
        self.Transition = namedtuple("Transition", ("cur_state", "action", "reward", "next_state", "is_done"))
        # List to store the data
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def Store(self, *args):
        item = self.Transition(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.index] = item
        self.index = (self.index + 1) % self.capacity

    def Sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(QNet, self).__init__()
        self.in_size = in_feature
        self.out_size = out_feature
        # 这里只使用了最简单的三层神经网络
        # Hidden layer
        self.hidden = nn.Linear(in_feature, 20)
        # Value function
        self.value_out = nn.Linear(20, 1)
        # Advantage function
        self.advantage_out = nn.Linear(20, out_feature)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        mid_out = F.relu(self.hidden(x))
        # 分别计算两个子网络的输出
        values = self.value_out(mid_out).expand(x.shape[0], self.out_size)
        advantages = self.advantage_out(mid_out)
        # 将优势函数和价值函数的结果线性组合
        ret = values + advantages - torch.mean(advantages, dim=1, keepdim=True).expand(x.shape[0], self.out_size)
        # 如果是mini-batch的话，就会输出[[]]这种形式的二维tensor
        # 虽然有点反直觉，但是谁叫你神经网络每一个数据最后都只有一个输出呢
        return ret

class DDQN:
    def __init__(self, memory_size=10000):
        # Environment part
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        # Q net part
        self.policy_net = QNet(self.state_dim, self.action_dim)
        self.target_net = QNet(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.2)
        self.loss_fn = nn.MSELoss()
        # Others
        self.replay_buffer = ReplayMemory(memory_size)
        self.count = 0

    def __SelectAction(self, state):
        # Constant
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        # Epsilon
        self.count += 1
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * self.count / EPS_DECAY)
        if random.random() < epsilon:
            return random.randrange(0, self.action_dim - 1)
        else:
            # 想不到把，这里的格式也要变化
            ret_q_net = self.policy_net(state.reshape(1, -1)).detach().numpy()
            return np.argmax(ret_q_net.squeeze())

    def __Train(self):
        # Get data set
        mini_batch = self.replay_buffer.Sample(BATCH_SIZE)
        cur_states_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_states_batch = [data[3] for data in mini_batch]
        # Get expectation
        max_action = torch.max(self.policy_net(next_states_batch), dim=1)[1]
        y_true = []
        for i in range(BATCH_SIZE):
            done = mini_batch[i][4]
            if done:
                y_true.append(reward_batch[i])
            else:
                next_states = torch.tensor(next_states_batch[i], dtype=torch.float).unsqueeze(0)
                y_true.append(reward_batch[i] +
                              GAMMA * self.target_net(next_states).squeeze()[max_action[i]])
        y_true = torch.tensor(y_true).unsqueeze(1)
        y_pred = self.policy_net(cur_states_batch).gather(1, torch.tensor(action_batch).unsqueeze(1))
        # backward
        loss = self.loss_fn(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def Episode(self):
        state = env.reset()
        for step in range(STEP):
            action = self.__SelectAction(state)
            next_state, reward, done, _ = env.step(action)
            reward = -1 if done else 0.1
            self.replay_buffer.Store(state, action, reward, next_state, done)
            state = next_state
            if len(self.replay_buffer) > BATCH_SIZE:
                self.__Train()
            if done:
                break

    def Update(self, episode, C=10):
        if episode % C == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def Test(self, episode, TEST=10):
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
        print("Episode: ", (episode + 100), "Evaluation", avg_reward)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    solution = DDQN()
    for episode in range(EPISODE):
        solution.Episode()
        solution.Update(episode=episode)

        if episode % 100 == 0:
            solution.Test(episode)
    env.close()





