from copy import deepcopy
import numpy as np
from citylearn.citylearn import CityLearnEnv
from ai_dm.Search.best_first_search import breadth_first_search, a_star, depth_first_search, greedy_best_first_search
from ai_dm.base.problem import Problem
from ai_dm.Search.utils import State, Node
from mc_heuristic import heuristic_montecarlo
from mcts import mcts, default_selection_policy, default_expansion_policy, default_rollout_policy


schema_path = 'data/schema.json'
num_features = 28
num_buildings = 1
feature_num_values = 5
soc_num_values = 10
num_actions = 5


actions = np.linspace(-1, 1, num_actions)
action_space = np.array(np.meshgrid(*[actions for _ in range(num_buildings)])).T.reshape(len(actions)**num_buildings, -1)
action_space = [[(b, 0, 0, 0) for b in a] for a in action_space]


env = CityLearnEnv(schema=schema_path)


names, low, high = env.observation_names[0], env.observation_space[0].low, env.observation_space[0].high
assert low.size == num_features and high.size == num_features and len(names) == num_features
discrete_features_indices = [0, 1, 2]
features_values = {}
for i in range(num_features):
    n = feature_num_values
    if i in discrete_features_indices:
        n = round(high[i] - low[i] + 1)
    features_values[names[i]] = np.linspace(low[i], high[i], n)


for building in env.buildings:
    features = ["energy_simulation", "weather", "pricing", "carbon_intensity"]
    for a in features:
        a = building.__dict__[f"_Building__{a}"]
        fields = [x for x in vars(a).keys() if x in building.active_observations]
        for x in fields:
            arr = a.__dict__[x]
            for i in range(arr.size):
                arr[i] = min(features_values[x], key=lambda v: abs(v - arr[i]))


index = 1


class CityState:
    def __init__(self, env, done, rewards):
        # assert len(state) == num_buildings
        # assert len(state[0]) == num_features
        global index
        self._index = index
        index += 1
        self.env = env
        self.rewards = rewards
        self.done = done

    def successor(self, action):
        env = deepcopy(self.env)
        state, reward, done, info = env.step(action)
        for building in env.buildings:
            features = [f for f in ["cooling_storage_soc", "heating_storage_soc", "dhw_storage_soc", "electrical_storage_soc"] if f in building.active_observations]
            for a in features:
                aa = f"_Building__{a[:-4]}"
                values = np.linspace(0, building.__dict__[aa].capacity, soc_num_values)
                building.__dict__[aa].soc[-1] = min(values, key=lambda v: abs(v - building.__dict__[aa].soc[-1]))
                # print(building.__dict__[f"_Building__{a[:-4]}"].soc)
        return CityState(env, done, self.rewards + [sum(reward)])

    def get_applicable_actions(self):
        # TODO return only the optional actions instead of all of them
        return action_space

    def get_transition_path_string(self):
        return "!"

    def is_done(self):
        return self.done or (len(self.rewards) >= 5 and self.result() > -0.2)

    def result(self):
        return np.mean(self.rewards)

    def __str__(self):
        return f"[{self._index}|{len(self.rewards)}|{self.result():.4f}]"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        # The opposite because the rewards are negative
        return self.result() < other.result()


class CityProblem(Problem):
    def __init__(self):
        super().__init__(initial_state=CityState(env, False, [0]), constraints=[])

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
    return abs(state.result())


def astar_heuristic(node):
    state = node.state.get_key()
    return len(state.rewards)


city_gym = CityProblem()


# result = breadth_first_search(problem=city_gym, log=True)
# result = depth_first_search(problem=city_gym, log=True)

# result = a_star(problem=city_gym, heuristic_func=astar_heuristic, log=True)
# result = a_star(problem=city_gym, log=True)

# result = greedy_best_first_search(problem=city_gym, heuristic_func=heuristic_montecarlo, log=True)
# result = greedy_best_first_search(problem=city_gym, heuristic_func=reward_heuristic, log=True)

mcts(city_gym, 100, default_selection_policy, default_expansion_policy, default_rollout_policy)
