import gym
from ai_dm.Environments.gym_envs.gym_problem import GymProblem
from ai_dm.Search.best_first_search import best_first_search, breadth_first_search, depth_first_search, a_star, depth_first_search_l
import ai_dm.Search.utils as utils
import ai_dm.Search.defs as defs
import ai_dm.Search.heuristic as heuristic


def create_env():
    # define the environment
    taxi_env = gym.make("Taxi-v3", render_mode='ansi').env
    taxi_env.reset()
    init_state = taxi_env.encode(0, 3, 4, 1)  # (taxi row, taxi column, passenger index, destination index)
    taxi_row, taxi_col, pass_idx, dest_idx = taxi_env.decode(init_state)
    print(taxi_row)
    taxi_env.unwrapped.s = init_state
    print("State:", init_state)
    print(taxi_env.render())
    return taxi_env 
    
    
taxi_env = create_env()

# create a wrapper of the environment to the search
taxi_p = GymProblem(taxi_env, taxi_env.unwrapped.s)


# perform BFS
[best_value, best_node, best_plan, explored_count, ex_terminated] = breadth_first_search(problem=taxi_p,
                                                                                         log=True,
                                                                                         log_file=None,
                                                                                         iter_limit=defs.NA,
                                                                                         time_limit=defs.NA,
                                                                                        )
