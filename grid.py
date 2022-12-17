from typing import Tuple, Optional, Union, List

import numpy as np
from citylearn.citylearn import CityLearnEnv
from gym.core import ActType, ObsType

import ai_dm.Search.defs as defs
from ai_dm.Environments.gym_envs.gym_problem import GymProblem
from ai_dm.Search.best_first_search import breadth_first_search
import gym


class Constants:
    episodes = 3
    schema_path = 'ai_dm/data/schema.json'


class CityGym(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = CityLearnEnv(schema=Constants.schema_path)

        for i in range(10):
            result = self.env.step([[0.001]])[0][0]
            print(result)
            print("  ".join([str(x) for x in zip(result, self.env.observation_names[0])]))
        raise Exception("========")

        # num_states = 1000000

        # actions_box = self.env.action_space[0]

        # print(len(self.env.action_space))

        self.action_space = np.arange(-1.0, 1.1, 0.1)
        self.num_buildings = 1

        low, high = self.env.observation_space[0].low, self.env.observation_space[0].high
        print("\n".join([str(x) for x in enumerate(zip(low, high, self.env.observation_names[0]))]))
        discrete_features_indices = [0, 1, 2]
        features_values = []
        for i in range(low.size):
            num_values = 5
            if i in discrete_features_indices:
                num_values = round(high[i] - low[i] + 1)
            features_values.append(np.linspace(low[i], high[i], num_values))

        optional_actions = {action: None for action in self.action_space}

        class P:
            def __init__(self, citygym):
                self.citygym = citygym

            def next(self, state, action):
                env = CityLearnEnv(schema=Constants.schema_path)

            def __getitem__(self, state):
                # TODO limit the optional actions according to the state
                return {action: self.next(state, action) for action in self.citygym.action_space}

        self.P = P(self)

        # self.P = {
        #     state: optional_actions for state in
        # }

        # print(self.step([[0, 0, 0, 0] for i in range(self.num_buildings)]))

    def render(self):
        return self.env.render()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        result = self.env.step(action)
        print("##############")
        return result

    def initial_state(self):
        return tuple([tuple(self.env.observation_space[0].sample()) for i in range(self.num_buildings)])


city_gym = CityGym()
g = GymProblem(city_gym, city_gym.initial_state())
best_value, best_node, best_plan, explored_count, ex_terminated = \
    breadth_first_search(problem=g, log=True, log_file=None, iter_limit=defs.NA, time_limit=defs.NA)
