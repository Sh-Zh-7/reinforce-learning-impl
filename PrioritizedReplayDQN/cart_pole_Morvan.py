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
    # 说说叫SumTree，其实还带有一个列表的结构
    # 这棵树存在的意义就是把我们的优先级的大小转化为区间的长度，然后进行采样
    # 注意由于我们SumTree本身的性质，这棵树是一棵满二叉树
    # 每一层的节点之间并没有大小关系，只有节点对于其儿子节点和父节点之间有大小关系（当然这是在我们的节点值都>0的前提下
    def __init__(self, capacity):
        # 在倒入numpy包以后，我们就可以使用np里面的方法来声明数组了
        self.capacity = capacity
        # 这里的树专门用来存储优先级, 后面的列表用来存储五元组
        self.priority_tree = np.zeros(2 * self.capacity - 1)
        self.data_list = np.zeros(self.capacity, dtype=object)  # 很神奇的一个做法
        # 指向transition_data的指针，这个指针指向的transition_data索引和我们的priority_tree有一一对应关系
        self.data_index = 0

    def Push(self, priority, data):
        # Update data list
        self.data_list[self.data_index] = data
        # Update priority tree
        tree_index = self.capacity - 1 + self.data_index
        self.UpdatePriorityOnly(tree_index, priority)
        # Update index
        self.data_index = (self.data_index + 1) % self.capacity

    def UpdatePriorityOnly(self, tree_index, priority):
        # 根据给定价值更新priority_tree
        delta = priority - self.priority_tree[tree_index]
        self.priority_tree[tree_index] = priority
        # 从最底层迭代更新
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.priority_tree[tree_index] += delta

    def GetData(self, x):
        """ 给定取样点，返回对应数据 """
        parent_index = 0
        left_index = 1
        right_index = 2
        # 这里不是capacity
        while left_index < len(self.priority_tree):
            if self.priority_tree[left_index] < x:
                parent_index = right_index
                x -= self.priority_tree[left_index]
            else:
                parent_index = left_index
            left_index = 2 * parent_index + 1
            right_index = 2 * parent_index + 2
        data_index = parent_index + 1 - self.capacity
        # 既要返回优先级，又要返回对应的五元组
        return parent_index, self.data_list[data_index], self.priority_tree[parent_index]

    # 这个就是我刚才所想的，如何在修改一个变量的同时，其他与之相关的变量都跟着修改
    # 做法是，不在__init__中声明这些变量
    # 而是定义一个@property修饰的方法，在方法中写入他们的函数
    @property
    def Range(self):
        return self.priority_tree[0]


class ReplayMemory:
    # 这里做了一个很好的封装
    # 那就是我们不用计算优先级，优先级都是由我们的类自动帮我们计算的
    alpha = 0.6
    beta = 0.4

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def Push(self, transition):
        # 注意放入transition的时候， 优先级得我们自己指定
        priority = np.max(self.tree.priority_tree[-self.tree.capacity:])
        # 这条语句还真的不能漏，你看第一次进来的时候是不是全是0
        if priority == 0:
            priority = 0.001
        self.tree.Push(priority, transition)

    def Sample(self, batch_size):
        # 待返回的坐标，五元组数据，以及由优先级计算而来的权重
        ret_index_batch = np.zeros((batch_size,), dtype=int)
        ret_data_batch = np.zeros((batch_size, 5), dtype=object)
        ret_weight = np.zeros((batch_size,))
        # 利用返回的priority batch计算待返回的权重
        min_priority = np.min(self.tree.priority_tree[-self.tree.capacity:])
        if min_priority == 0:
            min_priority = 0.00001
        range_length = self.tree.Range / batch_size
        for i in range(batch_size):
            # 更加均匀地采样
            left, right = i * range_length, (i + 1) * range_length
            sample_num = random.uniform(left, right)
            # 获得三个返回值
            ret_index_batch[i], ret_data_batch[i, :], priority_i = self.tree.GetData(sample_num)
            ret_weight[i] = np.power(priority_i / min_priority, -self.beta)
        return ret_index_batch, ret_data_batch, ret_weight

    def BatchUpdate(self, tree_index, errors):
        # 这是因为我们的优先级正比于我们的误差
        # 这里绝对不能出现0，这里如果不对0进行特殊处理的话，我们就会一直得到0
        # 不过，这里的更新只跟你已经存入SumTree的节点有关系，对于那些初始值为0的，依然没有办法(主要是从Range中可以看出)
        errors += 0.01  # convert to abs and avoid 0
        priorities = np.minimum(errors, 1)
        for index, priority in zip(tree_index, priorities):
            self.tree.UpdatePriorityOnly(index, priority)

    def __len__(self):
        return self.tree.data_index


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
    def __init__(self, weight):
        super(ModifiedMSE, self).__init__()
        self.weight = torch.tensor(weight, dtype=torch.float)

    def forward(self, inputs, targets):
        # 注意下溢的问题
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        diff = torch.pow((targets - inputs), 2) * self.weight
        return torch.mean(diff)


class PRQNet:
    def __init__(self):
        # Environment part
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        # QNet part
        self.policy_net = QNet(self.state_dim, self.action_dim)
        self.target_net = QNet(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.2)
        # Others
        self.count = 0
        self.replay_time = 0
        self.replay_buffer = ReplayMemory(10000)

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

    def __Train(self):
        # Sample
        index_batch, data_batch, weight_batch = self.replay_buffer.Sample(BATCH_SIZE)
        cur_states = [data[0] for data in data_batch]
        actions = [data[1] for data in data_batch]
        rewards = [data[2] for data in data_batch]
        next_states = [data[3] for data in data_batch]
        # Get prediction and y labels
        y_true = []
        max_action = torch.max(self.policy_net(next_states), dim=1)[1]
        for i in range(BATCH_SIZE):
            done = data_batch[i][4]
            # 其实这个就是根据reward进行函数的调整
            if done:
                y_true.append(rewards[i])
            else:
                y_true.append(rewards[i] + GAMMA * self.target_net(next_states[i])[max_action[i]])
        y_true = torch.tensor(y_true, dtype=torch.float).unsqueeze(1)
        y_pred = self.policy_net(cur_states).gather(1, torch.tensor(actions).unsqueeze(1))
        # backward
        loss = ModifiedMSE(weight_batch)(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update SumTree
        error = torch.abs(y_pred - y_true).detach().numpy()
        self.replay_buffer.BatchUpdate(index_batch, error)

    def Episode(self):
        state = env.reset()
        for step in range(STEP):
            action = self.__SelectAction(state)
            next_state, reward, done, _ = env.step(action)
            reward = -1 if done else 0.1
            self.replay_buffer.Push((state, action, reward, next_state, done))
            self.replay_time += 1
            if self.replay_time > BATCH_SIZE:
                self.__Train()
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
