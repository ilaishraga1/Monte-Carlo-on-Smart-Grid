from enum import Enum
from ai_dm.Search.best_first_search import breadth_first_search, a_star, \
    greedy_best_first_search, depth_first_search_l
from mc_heuristic import heuristic_montecarlo
from mcts import mcts
from grid_infrastructure import CityProblem, reward_heuristic


class Algorithms(Enum):
    BFS = "BreadthFirstSearch"
    DFS = "DepthFirstSearch"
    A_star = "A*"
    BFS_MC = "BestFirstSearch_MonteCarlo"
    MCTS = "MonteCarloTreeSearch"
    BFS_Reward = "BestFirstSearch_RewardHeuristic"


def run(algorithm, building_indices=(0,), log=False):
    print(algorithm.name, building_indices)

    problem = CityProblem(building_indices)

    if algorithm == Algorithms.BFS:
        breadth_first_search(problem=problem, log=log)
    elif algorithm == Algorithms.DFS:
        depth_first_search_l(problem=problem, max_depth=8, log=log)
    elif algorithm == Algorithms.A_star:
        a_star(problem=problem, log=log)
    elif algorithm == Algorithms.BFS_MC:
        greedy_best_first_search(problem=problem, heuristic_func=heuristic_montecarlo, log=log)
    elif algorithm == Algorithms.MCTS:
        mcts(problem, 1000, log=log)
    elif algorithm == Algorithms.BFS_Reward:
        greedy_best_first_search(problem=problem, heuristic_func=reward_heuristic, log=log)
    else:
        assert False, "Invalid algorithm"

    result = problem.infrastructure.best_result, problem.infrastructure.discovered_states, \
             problem.infrastructure.deepest_depth
    print(result)
    return result


if __name__ == "__main__":
    run(Algorithms.MCTS, log=True)
