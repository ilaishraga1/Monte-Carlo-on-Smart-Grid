import random


N = 2  # Number of montecarlo simulations
STEPS = 3  # Number of montecarlo recursive steps


def set_parameters(n, steps):
    global N, STEPS
    N = n
    STEPS = steps


def montecarlo_aux(state, left_steps):
    if left_steps == 0:
        return 0

    actions = state.get_applicable_actions()
    checked = N if N <= len(actions) else len(actions)

    if checked == 0:
        return 0

    rewards = 0
    for action in random.sample(actions, checked):
        neighbor = state.successor(action)
        rewards += neighbor.rewards[-1] + montecarlo_aux(neighbor, left_steps - 1)

    return rewards / checked


def heuristic_montecarlo(node):
    state = node.state.get_key()
    reward = state.rewards[-1] + montecarlo_aux(state, STEPS)
    return -reward
