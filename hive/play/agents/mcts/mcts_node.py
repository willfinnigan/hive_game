from __future__ import annotations

import math
from typing import List

from hive.game_engine.game_functions import get_winner, current_turn_colour, opposite_colour
from hive.game_engine.player_functions import get_players_possible_moves_or_placements


class MCTSNode:
    def __init__(self, model_callable: callable, move=None, game=None, parent=None, prior=0.0):
        """MCTS node.
        game: the current game state
        move: the move that leads to this node (None if this is the root node)
        colour: the colour of the player whose turn it is at this node
        value: value from the perspective of the player whose turn it is
        """

        self.parent = parent
        self.move = move  # for lazy evaluation just pass in move
        self.game = game  # can get to move from game
        self.children: List[MCTSNode] = []  # children nodes (MCTSNode)
        self.prior = prior  # P(s,a) from policy network
        self.model_callable = model_callable  # function to get moves and value

        self.value_sum = 0  # N(s)
        self.visits = 0  # W(s)

        if self.game is None:
            self.colour = opposite_colour(self.move.colour)  # because move lead to this node, so now opposite turn
        elif self.move is None:
            self.colour = current_turn_colour(self.game)
        else:
            raise ValueError("Node must have either a game or a move to determine colour.")

    def q_value(self):
        """Average value of this node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def exploration(self, c_puct):
        return c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)

    def ucb_score(self, c_puct=1):
        return self.q_value() + self.exploration(c_puct)

    def is_leaf(self):
        """Check if the node is a leaf node (no children)."""
        return len(self.children) == 0

    def select(self) -> MCTSNode:
        """Walk down the tree selecting nodes using UCB until we reach a leaf node."""
        current = self
        while not current.is_leaf():
            current = max(current.children, key=lambda child: child.ucb_score())
        return current

    def evaluate(self) -> float:
        assert self.visits == 0, "Node must not be visited before evaluation."

        if self.game is None:
            assert self.move is not None, "Node must have a move or game to evaluate."
            assert self.parent.game is not None, "Parent node must have a game to evaluate."
            self.game = self.move.play(self.parent.game) if self.parent else None

        move_probs, value = self.model_callable(self.game, self.colour)
        moves = get_players_possible_moves_or_placements(self.colour, self.game)

        for move, prob in zip(moves, move_probs):
            #print(f"Prob: {prob:.4f} for move: {move}")
            # create a new child node for each possible move
            #new_game = move.play(self.game)
            child_node = MCTSNode(move=move, model_callable=self.model_callable, parent=self, prior=prob)
            self.children.append(child_node)

        # set value and visits
        return value

    def backpropagate(self, value):
        value = -value  # invert the sign for correct perspective when selecting children
        self.value_sum += value
        self.visits += 1
        if self.parent is not None:
            self.parent.backpropagate(value)

    def is_terminal(self) -> bool:
        """Check if the game is over."""
        if get_winner(self.game) is None:
            return False
        else:
            return True