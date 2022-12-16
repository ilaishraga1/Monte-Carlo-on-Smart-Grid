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


def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }


def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict


# env = CityLearnEnv(schema=Constants.schema_path)

# print(f' OBSERVATION SPACES {env.observation_space}')
# print(f' OBSERVATION SPACE for Builiding ONE is {env.observation_space[0]}')
#
# for building in range(5):
#     print(f' SAMPLE OBSERVATION SPACE for Builiding ONE >>> {len(env.observation_space[building].sample()), env.observation_space[building].sample()}')
#
# print(f' ACTION SPACES {env.action_space}')
# print(f' ACTION SPACE for Builiding ONE is {env.action_space[0]}')
#
# # sample some actions
# for action in range(5):
#     print(f' SAMPLE ACTION SPACE for Builiding ONE >>> {env.action_space[1].sample()}')


def aaaa(box):
    high = box.high if box.dtype.kind == "f" else box.high.astype("int64") + 1
    sample = np.empty(box.shape)

    # Masking arrays which classify the coordinates according to interval
    # type
    unbounded = ~box.bounded_below & ~box.bounded_above
    upp_bounded = ~box.bounded_below & box.bounded_above
    low_bounded = box.bounded_below & ~box.bounded_above
    bounded = box.bounded_below & box.bounded_above

    ranges = []
    print(box.shape)
    # for i in range(box.shape):
    #     pass

    # Vectorized sampling by interval type
    sample[unbounded] = box.np_random.normal(size=unbounded[unbounded].shape)

    sample[low_bounded] = (
        box.np_random.exponential(size=low_bounded[low_bounded].shape)
        + box.low[low_bounded]
    )

    sample[upp_bounded] = (
        -box.np_random.exponential(size=upp_bounded[upp_bounded].shape)
        + box.high[upp_bounded]
    )

    sample[bounded] = box.np_random.uniform(
        low=box.low[bounded], high=high[bounded], size=bounded[bounded].shape
    )


class CityGym(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = CityLearnEnv(schema=Constants.schema_path)
        # num_states = 1000000

        # actions_box = self.env.action_space[0]

        # print(len(self.env.action_space))

        self.action_space = np.arange(-1.0, 1.1, 0.1)
        self.num_buildings = 1

        low, high = self.env.observation_space[0].low, self.env.observation_space[0].high
        print("\n".join([str(x) for x in zip(low, high)]))
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
