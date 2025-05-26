
from __future__ import annotations

import math
import time
from typing import Union, List

import numpy as np

from hive.game_engine.game_functions import get_winner, current_turn_colour, opposite_colour
from hive.game_engine.game_state import Colour, Game, WHITE, BLACK, create_reduced_pieces
from hive.game_engine.moves import Move, NoMove
from hive.game_engine.player_functions import get_players_possible_moves_or_placements
from hive.play.agents.board_score.simple_board_score import score_board_queens
from hive.play.agents.random_ai import RandomAI
from hive.play.agents.scored_moves_based_ai import prioritise_moves, score_moves_simple
from hive.play.play_game import play
from hive.play.player import Player


class MCTS_AI(Player):
    """Monte Carlo Tree Search AI for Hive game."""

    def __init__(self, 
                 colour: Colour,
                 game: Game,
                 model_callable: callable,
                 search_time: float = 20,
                 iterations: int = 5000):
        
        super().__init__(colour)

        self.search_time = search_time
        self.root = MCTSNode(model_callable=model_callable, game=game)
        self.root.evaluate()
        self.iterations = iterations


    def get_move(self, game: Game) -> Union[Move|NoMove]:
        """This is the function called by all player classes to get their next move"""

        # Check if the game state matches the root node - update root if necessary
        if game.parent is not None:
            if isinstance(game.move, NoMove):
                print("Warning: NoMove detected when updating root.")

            if self.root.move != game.move:
                # If the root node does not match the parent move, update it
                self.update_root(game.move)

        # Perform MCTS
        best_move = self.run(game)
        return best_move


    def update_root(self, move: Move):
        """Look at the children of root, find the child that matches the move, and set it as the new root."""
        for child in self.root.children:
            assert child != self.root, "Root node should not be a child of itself."
            if child.move == move:
                self.root = child
                return

        raise ValueError("Move not found in root's children. Ensure the move is valid and corresponds to the current game state.")

    def run(self, game: Game) -> Move:
        """Perform MCTS to find the best move."""
        # assert that the root node is the current game state
        if self.root.game is not None:
            assert self.root.game == game, "Root node must match the current game state."
        else:
            assert self.root.move == game.move, "Root node must match the current game state based on parent move."

        # Search for time allowed
        t0 = time.time()
        count = 0
        while (time.time() - t0 < self.search_time) and count < self.iterations:
            node = self.root.select()  # selection - find a node which has not been evaluated
            value = node.evaluate()  # evaluation and expansion - get the value of the node using the model, and expand
            node.backpropagate(value)  # backpropagation - update the values up the tree
            count += 1

        # once time is up, pick best move from the root
        sorted_children = sorted(self.root.children, key=lambda child: child.visits, reverse=True)
        best_child = sorted_children[0] if sorted_children else None

        print(f"MCTS completed {count} iterations in {time.time() - t0:.2f} seconds. Best move: {best_child.move} with visits: {best_child.visits} and value: {best_child.q_value()}")
        print([(child.q_value(), child.visits, child.prior) for child in sorted_children])  # print all children with their values and visits

        self.update_root(best_child.move)  # update the root to the best child

        return best_child.game.move


def mock_model_callable(game: Game, colour: Colour) -> tuple[list[float], float]:
    possible_moves = get_players_possible_moves_or_placements(colour, game)

    #scored_moves = prioritise_moves(possible_moves, game)

    # assign a random value to every move
    #scores = [np.random.rand() for _ in possible_moves]
    scores = [1] * len(possible_moves)  # for testing, give all moves the same score
    scored_moves = list(zip(scores, possible_moves))

    #scored_moves = score_moves_simple(possible_moves, game)

    scores = [score for score, _ in scored_moves]
    total_score = sum(scores)

    # make scores sum to 1
    if total_score > 0:
        probabilities = [score / total_score for score in scores]
    else:
        # If total score is 0, assign equal probabilities
        probabilities = [0] * len(possible_moves)

    # get value using board score
    value = score_board_queens(game, colour)

    return probabilities, value


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


if __name__ == '__main__':
    from hive.game_engine.game_state import initial_game

    game = initial_game(pieces_function=create_reduced_pieces)

    mctsai = MCTS_AI(WHITE, game, model_callable=mock_model_callable)
    randomai = RandomAI(BLACK)

    winner = play(mctsai, randomai, game=game, max_turns=500)
    