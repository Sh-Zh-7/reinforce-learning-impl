import gym
import math
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

# Hyper-parameters
EPISODE = 3000      # Episode limit
STEP = 300          # Sample's upper bound
BATCH_SIZE = 32
GAMMA = 0.9
C = 10

class MemoryReplay:
    def __init__(self, capacity):
        # The five tuple class
        self.Transition = namedtuple("Transition", ("cur_state", "action", "reward", "next_state", "is_done"))
        # Initialize members
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def Push(self, *args):
        target = self.Transition(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(target)
        else:
            self.memory[self.position] = target
        self.position = (self.position + 1) % self.capacity

    def Sample(self, batch_size):
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
        self.replay_buffer = MemoryReplay(memory_size)
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
            ret_q_net = self.policy_net(state).detach().numpy()
            return np.argmax(ret_q_net)

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
                y_true.append(reward_batch[i] + GAMMA * self.target_net(next_states_batch[i])[max_action[i]])
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
            self.replay_buffer.Push(state, action, reward, next_state, done)
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




