from copy import deepcopy
import numpy as np
from citylearn.citylearn import CityLearnEnv
from ai_dm.base.problem import Problem
from ai_dm.Search.utils import State, Node


class Infrastructure:
    schema_path = 'data/schema.json'

    min_steps_to_finish = 5
    min_result_to_finish = 0
    max_states = 10000

    num_actions = 5
    num_features = 28
    feature_num_values = 5
    soc_num_values = 10

    def __init__(self, building_indices):
        # Init the CityLearn environment
        self.env = CityLearnEnv(schema=self.schema_path)
        self.env.buildings = [self.env.buildings[i] for i in building_indices]
        self.num_buildings = len(building_indices)
        self.discovered_states = 0
        self.best_result = []
        self.deepest_depth = 0

        # Init the action space
        actions = np.linspace(-1, 1, self.num_actions)
        action_space = np.array(np.meshgrid(*[actions for _ in range(self.num_buildings)]))\
            .T.reshape(len(actions)**self.num_buildings, -1)
        self.action_space = [[(b, 0, 0, 0) for b in a] for a in action_space]

        # Init the states space values
        names = self.env.observation_names[0]
        low, high = self.env.observation_space[0].low, self.env.observation_space[0].high
        assert low.size == self.num_features and high.size == self.num_features and len(names) == self.num_features
        discrete_features_indices = [0, 1, 2]
        features_values = {}
        for i in range(self.num_features):
            n = self.feature_num_values
            if i in discrete_features_indices:
                n = round(high[i] - low[i] + 1)
            features_values[names[i]] = np.linspace(low[i], high[i], n)

        # Round the states to the states space values
        for building in self.env.buildings:
            features = ["energy_simulation", "weather", "pricing", "carbon_intensity"]
            for a in features:
                a = building.__dict__[f"_Building__{a}"]
                fields = [x for x in vars(a).keys() if x in building.active_observations]
                for x in fields:
                    arr = a.__dict__[x]
                    for i in range(arr.size):
                        arr[i] = min(features_values[x], key=lambda v: abs(v - arr[i]))

    def update_data(self, state):
        self.discovered_states += 1

        if len(state.rewards) > self.deepest_depth:
            self.deepest_depth = len(state.rewards)

        if len(state.rewards) >= self.min_steps_to_finish and \
                (self.best_result == [] or state.result() > self.best_result[-1][2]):
            self.best_result.append((self.discovered_states, self.deepest_depth, state.result()))

        return self.discovered_states


class CityState:
    def __init__(self, problem, env, done, rewards):
        self.problem = problem
        self.env = env
        self.rewards = rewards
        self.done = done
        self.index = problem.infrastructure.update_data(self)

    def successor(self, action):
        env = deepcopy(self.env)
        state, reward, done, info = env.step(action)
        for building in env.buildings:
            features = [f for f in ["cooling_storage_soc", "heating_storage_soc", "dhw_storage_soc", "electrical_storage_soc"]
                        if f in building.active_observations]
            for a in features:
                aa = f"_Building__{a[:-4]}"
                values = np.linspace(0, building.__dict__[aa].capacity, self.problem.infrastructure.soc_num_values)
                building.__dict__[aa].soc[-1] = min(values, key=lambda v: abs(v - building.__dict__[aa].soc[-1]))
        return CityState(self.problem, env, done, self.rewards + [sum(reward)])

    def get_applicable_actions(self):
        return self.problem.infrastructure.action_space

    def get_transition_path_string(self):
        return ""

    def is_done(self):
        return self.done or \
            self.problem.infrastructure.discovered_states >= self.problem.infrastructure.max_states or \
            (len(self.rewards) >= self.problem.infrastructure.min_steps_to_finish and
                self.result() > self.problem.infrastructure.min_result_to_finish)

    def result(self):
        return np.mean(self.rewards)

    def __str__(self):
        return f"[{self.index}|{len(self.rewards)}|{self.result():.4f}]"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        # The opposite because the rewards are negative
        return self.result() < other.result()


class CityProblem(Problem):
    def __init__(self, building_indices):
        self.infrastructure = Infrastructure(building_indices)
        initial_state = CityState(self, self.infrastructure.env, False, [0])
        super().__init__(initial_state=initial_state, constraints=[])

    def get_applicable_actions_at_state(self, state):
        return state.get_key().get_applicable_actions()[:]

    def get_applicable_actions_at_node(self, node):
        return self.get_applicable_actions_at_state(node.state)[:]

    def get_successors(self, action, node):
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
