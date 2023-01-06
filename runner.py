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


def run(algorithm, log=False):
    print(algorithm.name)

    city_gym = CityProblem()

    if algorithm == Algorithms.BFS:
        result = breadth_first_search(problem=city_gym, log=log)
    elif algorithm == Algorithms.DFS:
        result = depth_first_search(problem=city_gym, log=log)
    elif algorithm == Algorithms.A_star:
        result = a_star(problem=city_gym, log=log)
    elif algorithm == Algorithms.BFS_MC:
        result = greedy_best_first_search(problem=city_gym, heuristic_func=heuristic_montecarlo, log=log)
    elif algorithm == Algorithms.MCTS:
        result = mcts(city_gym, 1000, log=log)
    elif algorithm == Algorithms.BFS_Reward:
        result = greedy_best_first_search(problem=city_gym, heuristic_func=reward_heuristic, log=log)
    else:
        assert False, "Invalid algorithm"

    print(result)
    return result


def tester(num=5):
    for algorithm in Algorithms:
        for i in range(num):
            run(algorithm)


if __name__ == "__main__":
    # tester(5)
    run(Algorithms.MCTS)
