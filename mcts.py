import ai_dm.Search.utils as utils
import numpy as np
from copy import deepcopy


class MCTSNode(utils.Node):
    def __init__(self, state, applicable_actions, parent=None, action=None, path_cost=0, info=None, parent_action=None):
        super().__init__(state, parent, action, path_cost, info)
        self.parent_action = parent_action
        # todo: support settings with continuous spaces
        self.children = []
        self._number_of_visits = 0
        self._results = []
        self._untried_actions = applicable_actions
        self._applicable_actions = applicable_actions
        return

    def is_leaf(self):
        # return self._untried_actions == []
        return self.children == []

    def expand(self, policy):
        action = self._untried_actions.pop()
        next_state = self.state.get_key().successor(action)
        child_node = MCTSNode(next_state, self._applicable_actions, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results.append(result)
        if self.parent:
            self.parent.backpropagate(result)


def mcts(problem, comp_resources, selection_policy, expansion_policy, rollout_policy):
    # initialize the search tree
    root_node = MCTSNode(problem.get_current_state(), problem.get_applicable_actions_at_state(problem.get_current_state()),  None, None, 0, None, None)

    # perform the search
    for i in range(comp_resources):
        # use selection (tree) policy to choose the next leaf node to expand
        leaf = select(root_node, selection_policy)
        # choose which child of the selected leaf to expand (i.e. perform a simulation from)
        expanded_child = leaf.expand(expansion_policy)
        # perform a single simulation according to the rollout_policy from the expanded child node
        simulation_result = simulate(expanded_child, problem, rollout_policy)
        # update the tree with the current values
        expanded_child.backpropagate(simulation_result)

    return max(root_node._results)


# recursively traverse the tree until a leaf node (of the current MCTS tree) is reached.
def select(mcts_node, selection_policy):
    if mcts_node.is_leaf():
        return mcts_node
    return select(selection_policy(mcts_node), selection_policy)


def simulate(init_node, problem, rollout_policy):
    current_rollout_state = deepcopy(init_node.state)

    i = 0
    while not current_rollout_state.is_done() and i < 10:
        possible_moves = problem.get_applicable_actions_at_state(current_rollout_state)
        action = rollout_policy(possible_moves)
        current_rollout_state = current_rollout_state.move(action)
        i += 1

    return current_rollout_state.result()


def default_rollout_policy(possible_moves):
    return possible_moves[np.random.randint(len(possible_moves))]


def default_selection_policy(node):
    index = np.argmax([np.mean(c.results) for c in node.children])
    return node.children[index]


def default_expansion_policy(node):
    current_node = node
    while not current_node.is_terminal_node():

        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
    return current_node
