import numpy as np

# 超参数
EPSILON = 0.1       # 探索率
ALPHA = 0.5         # Sarsa的步长
REWARD = -1.0       # 奖励

# 环境
# 世界的大小以及风速
WORLD_HEIGHT = 7
WORLD_WIDTH = 10
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

class State:
    """
    在这里我们的state仅仅是agent在空间中的坐标,
    由于python没有专门的struct，所以这里只能用class来实现了
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        """ 这个是实现hashable对象所必备的方法 """
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        """ 只要x值和y值相同, 我们就把它放到同一个桶中 """
        return hash((self.x, self.y))

# Agent's actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

class QValue:
    """
    关于q价值函数的一个类，
    主要是实现像Q[S, A]这样的访问方式
    """
    def __init__(self, state_size, actions):
        """ Q值构造函数 """
        # 初始化类的所有成员
        self.state_size = state_size
        self.actions = actions
        self.q_values = {}
        # 初始化所有格子的q值
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                self.q_values[State(i, j)] = [0] * len(self.actions)

    def __getitem__(self, item):
        """ 实现Q[S, A]访问对象 """
        return self.q_values[item[0]][item[1]]
    
    def Update(self, cur_state, next_state, cur_action, next_action, reward=-1, alpha=ALPHA):
        """ 跟新对应状态上某一个动作的q值 """
        self.q_values[cur_state][cur_action] += \
            alpha * (reward + self.q_values[next_state][next_action] - self.q_values[cur_state][cur_action])


class Sarsa:
    def __init__(self):
        # 起点和终点
        self.start_point = State(3, 0)
        self.end_point = State(3, 7)
        # Q values
        self.q_values = QValue((WORLD_HEIGHT, WORLD_WIDTH), ACTIONS)

    def ChooseAction(self, state):
        """ 利用epsilon-贪心法选择动作 """
        if np.random.binomial(1, EPSILON):
            # 探索
            choice = np.random.choice(ACTIONS)
        else:
            # 选择最有价值的动作
            # 这里是获取所有动作价值的列表，从中取出最有价值的
            all_actions_values = self.q_values[state, :]
            # 为了防止有多个动作拥有相同的价值
            # 把所有动作全部都列出来，然后随机抽样
            choice = np.random.choice([action_index
                                       for action_index, action_value in enumerate(all_actions_values)
                                       if action_value == np.max(all_actions_values)])
            # 这里情况比较特殊：
            # 首先我每一个状态都有且仅有4个移动的方向
            # 其次是我把四个移动的方向都设置成了0~3四个数字
        return choice

    @staticmethod
    def Step(state, action):
        """ 利用action对当前状态进行转移 """
        # 注意我们不能对state进行修改，因为我们在更新q的时候要同时利用原来的状态和现在的状态
        # 注意这里上下左右都是有界限的，所以说我们这里每一次ACTION都要判断边界
        # 还要注意边界其实是0, 0, WORLD_HEIGHT-1和WORLD_WIDTH-1
        x = state.x
        y = state.y
        # 注意这里与风有关的只有y
        if action == ACTION_UP:
            x = max(x - 1 - WIND[y], 0)
        elif action == ACTION_DOWN:
            # 往下走是最复杂的，既要防止agent往上跑，还要防止agent往下跑
            x = min(max(x + 1 - WIND[y], 0), WORLD_HEIGHT - 1)
        elif action == ACTION_LEFT:
            x = max(x - WIND[y], 0)
            y = max(y - 1, 0)
        elif action == ACTION_RIGHT:
            x = max(x - WIND[y], 0)
            y = min(y + 1, WORLD_WIDTH - 1)
        else:
            assert False

        return State(x, y)

    def Episode(self):
        """ 进行一次episode """
        # 设定初始状态和动作
        cur_state = self.start_point
        cur_action = self.ChooseAction(cur_state)
        # 进行迭代
        while cur_state != self.end_point:
            # 获得下一个状态和动作
            next_state = self.Step(cur_state, cur_action)
            next_action = self.ChooseAction(next_state)
            # 根据下一个动作和状态进行更新
            self.q_values.Update(cur_state, next_state, cur_action, next_action)
            cur_state = next_state
            cur_action = next_action

    def OptimizedPolicy(self):
        """ 这个是为了获得所有各自格子中对应的最佳走法 """
        all_best_actions = []
        for i in range(WORLD_HEIGHT):
            row_best_actions = []
            for j in range(WORLD_WIDTH):
                # 判断是否到达了终点
                state = State(i, j)
                if state == self.end_point:
                    row_best_actions.append("G")
                    continue
                # 获得最佳行动并添加到list中去
                best_action = np.argmax(self.q_values[state, :])
                if best_action == ACTION_UP:
                    row_best_actions.append("U")
                elif best_action == ACTION_DOWN:
                    row_best_actions.append("D")
                elif best_action == ACTION_LEFT:
                    row_best_actions.append("L")
                elif best_action == ACTION_RIGHT:
                    row_best_actions.append("R")
                else:
                    raise False
            all_best_actions.append(row_best_actions)
        return all_best_actions


if __name__ == "__main__":
    solution = Sarsa()
    iter_time = 10000
    index = 0
    while index < iter_time:
        solution.Episode()
        index += 1
        print("{} round done.".format(index))
    optimized_policy = solution.OptimizedPolicy()
    for i in optimized_policy:
        print(i)
    print([str(wind) for wind in WIND])

