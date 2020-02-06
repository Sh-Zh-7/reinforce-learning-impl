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

# --------------------------------Helper Structure---------------------------------------------------------
class ReplayMemory:
    def __init__(self, capacity):
        # New class: the five element tuple
        self.Transition = namedtuple("Transition", ("cur_state", "action", "reward", "next_state", "is_done"))
        # Members
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """ Convert your args to Transition object and push it to memory. """
        if len(self.memory) < self.capacity:
            self.memory.append(self.Transition(*args))
        else:
            self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ return batch_size samples saved in memory. """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(QNet, self).__init__()
        self.linear1 = nn.Linear(in_features, 20)
        self.linear2 = nn.Linear(20, out_features)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# ----------------------------------------Main Structure------------------------------------------
class NatureDQN:
    def __init__(self):
        # Environment part
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        # 2 Q nets
        # 仔细体会一下官方文档为其命名的含义
        self.policy_net = QNet(self.state_dim, self.action_dim)  # Choose actions
        self.target_net = QNet(self.state_dim, self.action_dim)  # Calculate Q
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.2)
        self.loss_fn = nn.MSELoss()
        # Other
        self.iter_time = 0
        self.replay_buffer = ReplayMemory(10000)

    def __SelectAction(self, state):
        """ Epsilon-greedy method. """
        # Constant
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        # Choose actions
        self.iter_time += 1
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * self.iter_time / EPS_DECAY)
        if random.random() < epsilon:
            return random.randrange(0, self.action_dim - 1)
        else:
            # Action values represent action's value
            # And the index represent action's code
            actions_values = self.policy_net(state).detach().numpy()
            return np.argmax(actions_values)

    def __Train(self):
        """
        Train the policy net with mini_batch data set.
        The Q net defined here would return all action and its Q values by given state.
        But we only need one best action to optimize at a time.
        """
        # Step 1: Get data set
        mini_batch = self.replay_buffer.sample(BATCH_SIZE)
        cur_states_batch = [data[0] for data in mini_batch]
        actions_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_states_batch = [data[3] for data in mini_batch]

        # Step 2: Forward
        # Calculate prediction
        y_true = []
        Q_values = self.target_net(next_states_batch)
        for i in range(BATCH_SIZE):
            done = mini_batch[i][4]
            if done:
                y_true.append(reward_batch[i])
            else:
                y_true.append(reward_batch[i] + GAMMA * torch.max(Q_values[i], dim=0)[0])
        y_true = torch.tensor(y_true, dtype=torch.float, requires_grad=True).unsqueeze(1)
        # Get true labels
        y_pred = self.policy_net(cur_states_batch).gather(1, torch.tensor(actions_batch).unsqueeze(1))

        # Step 3:Backward
        # 注意两者的形状必须适配！！
        loss = self.loss_fn(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def Episode(self):
        """ Do episode due to the paper. """
        state = env.reset()
        for step in range(STEP):
            # Step 1: Choose action and interact with environment
            action = self.__SelectAction(state)
            next_state, reward, done, __ = env.step(action)
            # Rewrite reward(optional)
            reward = -1 if done else 0.1
            # Step 2: Update
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            # Step 3: Training
            if len(self.replay_buffer) > BATCH_SIZE:
                self.__Train()
            if done:
                break

    def Update(self, episode, C=10):
        """ Update policy net due to the episode """
        if episode % C == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def Test(self, episode, TEST=10):
        """ Test the Nature DQN by using CartPole-v0 """
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
    # Initialize
    env = gym.make('CartPole-v0')
    solution = NatureDQN()
    # Do episode
    for episode in range(EPISODE):
        # Episode
        solution.Episode()
        solution.Update(episode=episode)
        # Test
        if episode % 100 == 0:
            solution.Test(episode)
    # Done
    env.close()
