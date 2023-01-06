from enum import Enum
from ai_dm.Search.best_first_search import breadth_first_search, a_star, depth_first_search, greedy_best_first_search
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


def run(algorithm):
    city_gym = CityProblem()

    if algorithm == Algorithms.BFS:
        result = breadth_first_search(problem=city_gym, log=True)
    elif algorithm == Algorithms.DFS:
        result = depth_first_search(problem=city_gym, log=True)
    elif algorithm == Algorithms.A_star:
        result = a_star(problem=city_gym, log=True)
    elif algorithm == Algorithms.BFS_MC:
        result = greedy_best_first_search(problem=city_gym, heuristic_func=heuristic_montecarlo, log=True)
    elif algorithm == Algorithms.MCTS:
        result = mcts(city_gym, 1000)
    elif algorithm == Algorithms.BFS_Reward:
        result = greedy_best_first_search(problem=city_gym, heuristic_func=reward_heuristic, log=True)
    else:
        assert False, "Invalid algorithm"

    print(result)


if __name__ == "__main__":
    run(Algorithms.MCTS)
