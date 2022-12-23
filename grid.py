from copy import deepcopy
import numpy as np
from citylearn.citylearn import CityLearnEnv
from ai_dm.Search.best_first_search import breadth_first_search
from ai_dm.base.problem import Problem
from ai_dm.Search.utils import State, Node


schema_path = 'ai_dm/data/schema.json'
num_features = 28
num_buildings = 1
num_values_per_feature = 5
num_actions = 3


env = CityLearnEnv(schema=schema_path)

# for i in range(10):
#     result = self.env.step([[0.001]])[0][0]
#     print(result)
#     print("  ".join([str(x) for x in zip(result, self.env.observation_names[0])]))

# actions_box = self.env.action_space[0]
# print(len(self.env.action_space))

actions = np.linspace(-1, 1, num_actions)
action_space = np.array(np.meshgrid(*[actions for _ in range(num_buildings)])).T.reshape(len(actions)**num_buildings, -1)
action_space = [[(b, 0, 0, 0) for b in a] for a in action_space]


low, high = env.observation_space[0].low, env.observation_space[0].high
assert low.size == num_features and high.size == num_features
# print("\n".join([str(x) for x in enumerate(zip(low, high, self.env.observation_names[0]))]))
discrete_features_indices = [0, 1, 2]
features_values = []
for i in range(num_features):
    n = num_values_per_feature
    if i in discrete_features_indices:
        n = round(high[i] - low[i] + 1)
    features_values.append(np.linspace(low[i], high[i], n))


index = 1


class GridState:
    def __init__(self, env, state, done, reward, depth):
        # assert len(state) == num_buildings
        # assert len(state[0]) == num_features
        global index
        self._index = index
        index += 1
        self.env = env
        self.reward = reward
        self.done = done
        self.depth = depth

        _state = []
        for s in state:
            discrete_s = [min(features_values[i], key=lambda x: abs(x - s[i])) for i in range(num_features)]
            _state.append(tuple(discrete_s))
        self._state = tuple(_state)
        # soc = [env.buildings[i].electrical_storage.soc[-1] for i in range(num_buildings)]

    def successor(self, action):
        env_copy = deepcopy(self.env)
        state, reward, done, info = env_copy.step(action)
        reward = sum(reward)
        average_reward = (self.reward * self.depth + reward) / (self.depth + 1)
        return GridState(env_copy, state, done, average_reward, self.depth + 1)

    def get_applicable_actions(self):
        # TODO return only the optional actions instead of all of them
        return action_space

    def get_transition_path_string(self):
        return "!"

    def is_done(self):
        return self.done or (self.depth >= 5 and self.reward > -0.5)

    def __str__(self):
        return f" [{self._index}|{self.depth}|{self.reward:.2f}] "

    def __repr__(self):
        return self.__str__()


class CityGym(Problem):
    def __init__(self):
        state = tuple([tuple(env.observation_space[0].sample()) for _ in range(num_buildings)])
        super().__init__(initial_state=GridState(env, state, False, 0, 0), constraints=[])

    def get_applicable_actions(self, node):
        return node.state.get_key().get_applicable_actions()

    def get_successors(self, action, node):
        # The cost is only when charging the battery  TODO check
        cost = self.get_action_cost(action, node.state)
        return [Node(State(node.state.get_key().successor(action)), node, action, node.path_cost + cost)]

    def get_action_cost(self, action, state):
        return 0  # sum([sum([c for c in b if c > 0]) for b in action])

    def is_goal_state(self, state):
        return state.get_key().is_done()

    def apply_action(self, action):
        pass


city_gym = CityGym()
best_value, best_node, best_plan, explored_count, ex_terminated = breadth_first_search(problem=city_gym, log=True)
