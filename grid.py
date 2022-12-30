from copy import deepcopy
import numpy as np
from citylearn.citylearn import CityLearnEnv
from ai_dm.Search.best_first_search import breadth_first_search, a_star, depth_first_search, greedy_best_first_search
from ai_dm.base.problem import Problem
from ai_dm.Search.utils import State, Node
from mc_heuristic import heuristic_montecarlo


schema_path = 'data/schema.json'
num_features = 28
num_buildings = 1
num_values_per_feature = 5
num_actions = 5


actions = np.linspace(-1, 1, num_actions)
action_space = np.array(np.meshgrid(*[actions for _ in range(num_buildings)])).T.reshape(len(actions)**num_buildings, -1)
action_space = [[(b, 0, 0, 0) for b in a] for a in action_space]

env = CityLearnEnv(schema=schema_path)

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


class CityState:
    def __init__(self, env, state, done, rewards):
        # assert len(state) == num_buildings
        # assert len(state[0]) == num_features
        global index
        self._index = index
        index += 1
        self.env = env
        self.rewards = rewards
        self.done = done

        _state = []
        for s in state:
            discrete_s = [min(features_values[i], key=lambda x: abs(x - s[i])) for i in range(num_features)]
            _state.append(tuple(discrete_s))
        self._state = tuple(_state)
        # soc = [env.buildings[i].electrical_storage.soc[-1] for i in range(num_buildings)]

    def successor(self, action):
        env_copy = deepcopy(self.env)
        state, reward, done, info = env_copy.step(action)
        return CityState(env_copy, state, done, self.rewards + [sum(reward)])

    def get_applicable_actions(self):
        # TODO return only the optional actions instead of all of them
        return action_space

    def get_transition_path_string(self):
        return "!"

    def is_done(self):
        return self.done or (len(self.rewards) >= 5 and np.mean(self.rewards) > -0.2)

    def __str__(self):
        return f"[{self._index}|{len(self.rewards)}|{np.mean(self.rewards):.4f}]"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        # The opposite because the rewards are negative
        return np.mean(self.rewards) < np.mean(other.rewards)


class CityProblem(Problem):
    def __init__(self, random_start=False):
        if random_start:
            state = tuple([tuple(env.observation_space[0].sample()) for _ in range(num_buildings)])
        else:
            building = (8.0, 1.0, 1.0, 22.456795, 24.281796, 24.544542, 22.8684, 92.53819, 70.58964, 71.346596, 85.56254, 832.66046, 936.07117, 208.70508, 347.4476, 741.96875, 235.003, 108.57185, 715.0383, 0.20709743, 0.85116667, 0.0, 0.88911945, -7.3135595, 0.31945515, 0.3977238, 0.2844061, 0.5062929)
            state = tuple([building for _ in range(num_buildings)])
        super().__init__(initial_state=CityState(env, state, False, [0]), constraints=[])

    def get_applicable_actions_at_state(self, state):
        return state.get_key().get_applicable_actions()

    def get_applicable_actions_at_node(self, node):
        # print("@@@@@@@@@@@@@@@@", node.get_path_cost(self)[0])
        return self.get_applicable_actions_at_state(node.state)

    def get_successors(self, action, node):
        # The cost is only when charging the battery  TODO check
        cost = self.get_action_cost(action, node.state)
        successor = node.state.get_key().successor(action)
        return [Node(State(successor, successor.is_done()), node, action, node.path_cost + cost)]

    def get_action_cost(self, action, state):
        return abs(state.get_key().rewards[-1])

    def is_goal_state(self, state):
        return state.get_key().is_done()

    def apply_action(self, action):
        pass

    def reset_env(self):
        pass


def reward_heuristic(node):
    state = node.state.get_key()
    return abs(np.mean(state.rewards))


def astar_heuristic(node):
    state = node.state.get_key()
    return len(state.rewards)


city_gym = CityProblem()


# result = breadth_first_search(problem=city_gym, log=True)
# result = depth_first_search(problem=city_gym, log=True)

# result = a_star(problem=city_gym, heuristic_func=astar_heuristic, log=True)
# result = a_star(problem=city_gym, log=True)

# result = greedy_best_first_search(problem=city_gym, heuristic_func=heuristic_montecarlo, log=True)
result = greedy_best_first_search(problem=city_gym, heuristic_func=reward_heuristic, log=True)
