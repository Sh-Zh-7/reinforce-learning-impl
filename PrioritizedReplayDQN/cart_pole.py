import gym
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Hyper-parameters
EPISODE = 3000  # Episode limit
STEP = 300  # Sample's upper bound
BATCH_SIZE = 128
GAMMA = 0.9
C = 10


class SumTree:
    """
    设计这个类的目的就是能让我们的SumTree和我们五元组的列表
    能够以相同的索引来访问对应的元素
    """

    def __init__(self, capacity):
        self.capacity = capacity
        # 指定大小但是不指定类型(后来改成float)的列表(实际上构建了一颗空的满二叉树
        self.nodes = np.zeros(2 * self.capacity - 1, dtype=float)  # 这里建议不要使用python的list，方便批量更新

    def Update(self, position, data):
        """ 更新某个位置上的节点 """
        # 还有一点，就是这里不能使用最好不要批量处理（即便你批量处理也得出错）
        # 注意不要忘了底层节点更新，其对应父母和祖先节点也得更新
        data = float(data)
        index = position + self.capacity - 1
        delta = data - self.nodes[index]
        self.nodes[index] = data
        while index != 0:
            index = (index - 1) // 2
            self.nodes[index] += delta

    def Search(self, x):
        """ 给定一个值，返回这个值所在区间的位置和大小 """
        parent, left, right = 0, 1, 2
        while left < len(self.nodes):
            if x > self.nodes[left]:
                x -= self.nodes[left]
                parent = right
            else:
                parent = left
            left = 2 * parent + 1
            right = 2 * parent + 2

        ret_index = parent - self.capacity + 1
        # 时刻注意，返回的是在叶节点对应的索引
        return ret_index, self.nodes[parent]

    # 以下都是辅助方法
    @property
    def Range(self):
        """ 返回这个SumTree所表示的范围 """
        return self.nodes[0]

    def Max(self):
        # 一开始我们选择了全零初始化，所以哪怕是取最大值，我们依然要把0当特殊情况处理
        ret_val = np.max(self.nodes[-self.capacity:])
        if ret_val == 0:
            ret_val = 0.001
        return ret_val

    def Min(self):
        # 前几轮（这个轮数还不小）我们的叶子节点都是没有被填满的，所以取最小值的时候我们得把0当特殊情况处理
        ret_val = np.min(self.nodes[-self.capacity:])
        if ret_val == 0:
            ret_val = 0.00001
        return ret_val


class Memory:
    """ 用来更新价值以及对应元素的类 """

    def __init__(self, capacity):
        """ 这里我们既创建了tree，也创建了list，所以我们在存储和采样的时候两者都要考虑 """
        self.capacity = capacity
        # 一个SumTree用来存储优先级(刚进来的赋予已知最大的优先级，后面用误差绝对值进行更新)
        self.sum_tree = SumTree(capacity)
        # 一个data_list用来存放五元组，当然我们这里并不知道这是五元组
        self.data_list = np.zeros(self.capacity, dtype=object)
        self.index = 0

    def Store(self, data):
        """ 存储我们的data，我们还要给他分配优先级 """
        # 更新data_list
        self.data_list[self.index] = data
        # 更新sum_tree
        max_priority = self.sum_tree.Max()
        self.sum_tree.Update(self.index, max_priority)
        # 更新指针
        self.index = (self.index + 1) % self.capacity

    def Sample(self, batch_size):
        """ 从中取样, 注意要返回对应值和优先级"""
        index_batch = np.zeros(batch_size, dtype=int)
        priority_batch = np.zeros(batch_size, dtype=float)

        priority_length = self.sum_tree.Range / batch_size
        for i in range(batch_size):
            left, right = priority_length * i, priority_length * (i + 1)
            sample_num = random.uniform(left, right)
            index_batch[i], priority_batch[i] = self.sum_tree.Search(sample_num)
        data_batch = self.data_list[index_batch]
        return index_batch, priority_batch, data_batch

    # 关于PRDQN的一个特殊的操作——那就是批量更新
    def BatchUpdate(self, index_batch, errors_batch):
        """ 利用error更新你的SumTree(本身就是error越大越具有启发性，所以这个才是重点)"""
        # 修建你的误差的范围
        errors_batch += 0.01
        priorities = np.minimum(errors_batch, 1)
        for i in range(len(index_batch)):
            self.sum_tree.Update(index_batch[i], priorities[i])  # 正是因为同一层节点的大小没什么规律，这里才要传入index


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


class ModifiedMSE(nn.Module):
    # 这里是把误差也给重新调整了
    def __init__(self, weight=None):
        super(ModifiedMSE, self).__init__()
        if weight is not None:
            self.weight = torch.tensor(weight, dtype=torch.float)

    def forward(self, inputs, targets, weight=None):
        # 注意下溢的问题
        # 还有这里我们最好全部使用torch.Tensor类型的变量
        if weight is not None:
            self.weight = torch.tensor(weight, dtype=torch.float)
        if not hasattr(self, "weight"):
            raise AttributeError
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        diff = torch.pow((targets - inputs), 2) * self.weight
        return torch.mean(diff)


class PRQNet:
    beta = 0.4

    def __init__(self):
        # Environment part
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        # QNet part
        self.policy_net = QNet(self.state_dim, self.action_dim)
        self.target_net = QNet(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.2)
        self.loss = ModifiedMSE()
        # Others
        self.count = 0
        self.replay_time = 0
        self.replay_buffer = Memory(10000)

    def __SelectAction(self, state):
        # Constant
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        # Choose actions
        self.count += 1
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * self.count / EPS_DECAY)
        # Epsilon
        if random.random() < epsilon:
            return random.randrange(0, self.action_dim)
        else:
            actions = self.policy_net(state).detach().numpy()
            return np.argmax(actions)

    def __Sample(self, batch_size=BATCH_SIZE):
        index_batch, priority_batch, data_batch = self.replay_buffer.Sample(batch_size)
        # 计算要返回的权重
        min_priority = self.replay_buffer.sum_tree.Min()
        weight_batch = np.power(priority_batch / min_priority, -self.beta)
        return index_batch, data_batch, weight_batch

    def __Train(self, data_batch, weight_batch):
        # Sample
        cur_states_batch = [data[0] for data in data_batch]
        actions_batch = [data[1] for data in data_batch]
        rewards_batch = [data[2] for data in data_batch]
        next_states_batch = [data[3] for data in data_batch]
        # Get prediction and y labels
        y_true = []
        max_action = torch.max(self.policy_net(next_states_batch), dim=1)[1]
        for i in range(BATCH_SIZE):
            done = data_batch[i][4]
            # 其实这个就是根据reward进行函数的调整
            if done:
                y_true.append(rewards_batch[i])
            else:
                y_true.append(rewards_batch[i] + GAMMA * self.target_net(next_states_batch[i])[max_action[i]])
        y_true = torch.tensor(y_true, dtype=torch.float).unsqueeze(1)
        y_pred = self.policy_net(cur_states_batch).gather(1, torch.tensor(actions_batch).unsqueeze(1))
        # backward
        loss = self.loss(y_pred, y_true, weight_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 把y_true和y_pred转化为nd.array后返回
        return y_true.detach().numpy(), y_pred.detach().numpy()

    def __UpdateSumTree(self, index_batch, errors):
        # Update SumTree
        priorities = np.abs(errors)
        self.replay_buffer.BatchUpdate(index_batch, priorities)

    def Episode(self):
        state = env.reset()
        for step in range(STEP):
            action = self.__SelectAction(state)
            next_state, reward, done, _ = env.step(action)
            reward = -1 if done else 0.1
            self.replay_buffer.Store((state, action, reward, next_state, done))
            self.replay_time += 1
            if self.replay_time > BATCH_SIZE:
                index_batch, data_batch, weight_batch = self.__Sample(BATCH_SIZE)
                y_true, y_pred = self.__Train(data_batch, weight_batch)
                self.__UpdateSumTree(index_batch, y_true - y_pred)
            state = next_state
            # 注意有done, 所以每一个episode并不会恒定地添加STEP个五元组
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
    solution = PRQNet()
    for episode in range(EPISODE):
        solution.Episode()
        solution.Update(episode=episode)

        if episode % 100 == 0:
            solution.Test(episode)
    env.close()
