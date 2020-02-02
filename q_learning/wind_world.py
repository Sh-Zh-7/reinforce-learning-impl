import numpy as np

# Hyper-parameters
EPSILON = 0.1  # Exploration rate
ALPHA = 0.5  # Sarsa step size
REWARD = -1.0  # Reward

# Environment
WORLD_HEIGHT = 7
WORLD_WIDTH = 10
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


# State structure
class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return "({}.{})".format(self.x, self.y)


# Actions marco
ACTION_UP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN]


# Q_values class
class QValues:
    def __init__(self, state_size, action_size):
        self.q_values = {}
        for i in range(state_size[0]):
            for j in range(state_size[1]):
                state = State(i, j)
                self.q_values[state] = [0] * action_size

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.q_values[item[0]][item[1]]
        elif isinstance(item, State):
            return self.q_values[item]
        else:
            raise TypeError

    def Update(self, cur_state, next_state, action, alpha=ALPHA, reward=REWARD):
        next_state_best_action = int(np.argmax(self.q_values[next_state]))
        self.q_values[cur_state][action] += \
            alpha * (reward + self.q_values[next_state][next_state_best_action] - self.q_values[cur_state][action])


class QLearning:
    def __init__(self):
        self.start_point = State(3, 0)
        self.end_point = State(3, 7)
        self.q_values = QValues((WORLD_HEIGHT, WORLD_WIDTH), 4)

    def ChooseAction(self, state):
        # Epsilon-greedy method
        if np.random.binomial(1, EPSILON):
            return np.random.choice(ACTIONS)
        else:
            best_values = np.max(self.q_values[state, :])
            actions = np.random.choice([action_index
                       for action_index, action_values in enumerate(self.q_values[state, :])
                       if action_values == best_values])
            return actions

    @staticmethod
    def Step(state, action):
        # You need both step and action to calculate your new state
        x = state.x
        y = state.y
        if action == ACTION_UP:
            x = max(x - 1 - WIND[y], 0)
        elif action == ACTION_LEFT:
            x = max(x - WIND[y], 0)
            y = max(y - 1, 0)
        elif action == ACTION_RIGHT:
            x = max(x - WIND[y], 0)
            y = min(y + 1, WORLD_WIDTH - 1)
        elif action == ACTION_DOWN:
            x = min(max(x + 1 - WIND[y], 0), WORLD_HEIGHT - 1)
        else:
            assert False

        return State(x, y)

    def Episode(self):
        cur_state = self.start_point
        while cur_state != self.end_point:
            action = self.ChooseAction(cur_state)
            next_state = self.Step(cur_state, action)
            self.q_values.Update(cur_state, next_state, action)
            cur_state = next_state

    def OptimizedPolicy(self):
        optimized_policy = []
        for i in range(WORLD_HEIGHT):
            row_policy = []
            for j in range(WORLD_WIDTH):
                state = State(i, j)
                if state == self.end_point:
                    row_policy.append("G")
                    continue
                best_action = np.argmax(self.q_values[state])
                if best_action == ACTION_UP:
                    row_policy.append("U")
                elif best_action == ACTION_LEFT:
                    row_policy.append("L")
                elif best_action == ACTION_RIGHT:
                    row_policy.append("R")
                elif best_action == ACTION_DOWN:
                    row_policy.append("D")
                else:
                    assert False
            optimized_policy.append(row_policy)

        return optimized_policy

if __name__ == "__main__":
    solution = QLearning()
    # Iter enough times to converge the q-values functions
    for i in range(10000):
        solution.Episode()
        print("Now round {}")
    result = solution.OptimizedPolicy()
    # Output result
    for row_policy in result:
        print(row_policy)
    # Output wind condition
    print([str(wind) for wind in WIND])
