import ai_dm.Search.utils as utils
import numpy as np


def default_rollout_policy(possible_moves):
    return possible_moves[np.random.randint(len(possible_moves))]


def default_selection_policy(node):
    # index = np.argmax([np.max(c._results) for c in node.children])
    # return node.children[index]

    child = node.children[np.random.randint(len(node.children))]
    return child

    # probabilities = [np.max(x._results) + 1 for x in node.children]
    # probabilities = [x / sum(probabilities) for x in probabilities]
    # return np.random.choice(node.children, p=probabilities)


def default_expansion_policy(node):
    action = node._untried_actions[np.random.randint(len(node._untried_actions))]
    return action


class MCTSNode(utils.Node):
    def __init__(self, problem, state, parent=None, action=None, path_cost=0, info=None):
        super().__init__(state, parent, action, path_cost, info)
        self.problem = problem
        self.children = []
        self._number_of_visits = 0
        self._results = []
        self._untried_actions = problem.get_applicable_actions_at_state(state)

    def is_leaf(self):
        return self._untried_actions != []

    def expand(self, policy):
        action = policy(self)
        self._untried_actions.remove(action)
        next_state = self.state.get_key().successor(action)
        next_state = utils.State(next_state, next_state.is_done())
        child_node = MCTSNode(self.problem, next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results.append(result)
        if self.parent:
            self.parent.backpropagate(result)


def mcts(problem, comp_resources, selection_policy=default_selection_policy,
         expansion_policy=default_expansion_policy, rollout_policy=default_rollout_policy):
    # initialize the search tree
    root_node = MCTSNode(problem, problem.get_current_state())

    # perform the search
    for i in range(comp_resources):
        # use selection (tree) policy to choose the next leaf node to expand
        leaf = select(root_node, selection_policy)
        # choose which child of the selected leaf to expand (i.e. perform a simulation from)
        expanded_child = leaf.expand(expansion_policy)
        # perform a single simulation according to the rollout_policy from the expanded child node
        simulation_result = simulate(expanded_child, rollout_policy)
        # update the tree with the current values
        expanded_child.backpropagate(simulation_result)

        if i % 50 == 0:
            print(max(root_node._results))

    return max(root_node._results)


# recursively traverse the tree until a leaf node (of the current MCTS tree) is reached.
def select(mcts_node, selection_policy):
    if mcts_node.is_leaf():
        return mcts_node
    return select(selection_policy(mcts_node), selection_policy)


def simulate(init_node, rollout_policy, simulation_depth=10):
    current_rollout_state = init_node.state.get_key()

    i = 0
    while (not current_rollout_state.is_done()) and i < simulation_depth:
        possible_moves = current_rollout_state.get_applicable_actions()
        if len(possible_moves) == 0:
            break
        action = rollout_policy(possible_moves)
        current_rollout_state = current_rollout_state.successor(action)
        i += 1

    return current_rollout_state.result()
