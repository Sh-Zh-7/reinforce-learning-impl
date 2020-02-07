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
STEP = 3000  # 采样上限
GAMMA = 0.95

class ActorNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(ActorNet, self).__init__()
        self.hidden = nn.Linear(in_features, 20)
        self.output = nn.Linear(20, out_features)

    def forward(self, x):
        """ 返回每个动作的概率值 """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float, requires_grad=True)
        x = F.relu(self.hidden(x))
        # 由于对于动作的选择是一个二分类问题，所以我这里使用sigmoid做激活函数
        # 当然这里用softmax也是可以的
        x = F.softmax(self.output(x))
        return x


class CriticNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(CriticNet, self).__init__()
        self.hidden = nn.Linear(in_features, 20)
        self.output = nn.Linear(20, out_features)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        x = F.relu(self.hidden(x))
        x = self.output(x)
        # 其实这里是输出状态价值的估计值
        return x


class ActorCritic:
    def __init__(self):
        # Environment part
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        # Network part
        self.actor = ActorNet(self.state_dim, self.action_dim)              # 这个是输出策略的
        self.critic = CriticNet(self.state_dim, 1)                          # 这个是输出价值的
        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=0.1)
        self.critic_optimizer = optim.SGD(self.critic.parameters(), lr=0.1)

    def __GetV(self, cur_state, next_state, reward):
        """ 利用Critic网络计算出对应动作的价值，来提示我们Actor网络的更新 """
        # 得到Q值输出
        cur_value = self.critic(cur_state)
        next_value = self.critic(next_state)
        # 计算TD误差
        td_error = reward + GAMMA * next_value - cur_value      # 如果你的计算图中包含这种中间结果，那么就要retain_graph=True
        # 反向传播优化critic网络
        mse_loss = td_error ** 2
        self.critic_optimizer.zero_grad()
        mse_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        return td_error

    def __SelectAction(self, state):
        # 如果是策略网络的话，我们都不会采用贪心法进行更新
        # 这两个方法是有差别的, 一个是基本上选择最大的价值，还有一个则是随机的
        policy = self.actor(state).detach().numpy()             # require_grad=True的tensor必须要使用detach
        # 如果你想把单个数据整成二维tensor的表现形式，一个做法就是在使用np.newaxis
        # 但是在pytorch里面，毫无必要
        return np.random.choice(policy.shape[0], p=policy)

    def __Learn(self, action, state, td_error):
        # 初始化
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        policy = self.actor(state)
        self.actor_optimizer.zero_grad()
        # 反向传播优化actor网络
        m = Categorical(policy)
        loss = -m.log_prob(action) * td_error
        loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def Episode(self):
        state = env.reset()
        for step in range(STEP):
            action = self.__SelectAction(state)
            next_state, reward, done, _ = env.step(action)
            reward = -1 if done else 0.1
            td_error = self.__GetV(state, next_state, reward)
            self.__Learn(action, state, td_error)
            state = next_state
            if done:
                break

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
    solution = ActorCritic()
    for episode in range(EPISODE):
        solution.Episode()
        if episode % 100 == 0:
            solution.Test(episode)
    env.close()
