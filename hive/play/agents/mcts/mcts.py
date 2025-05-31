
from __future__ import annotations

import time
from typing import Union

from hive.game_engine.game_state import Colour, Game, WHITE, BLACK, create_reduced_pieces
from hive.game_engine.moves import Move, NoMove
from hive.play.agents.mcts.mcts_node import MCTSNode
from hive.play.agents.mcts.model_callable import mock_model_callable
from hive.play.agents.random_ai import RandomAI
from hive.play.play_game import play
from hive.play.player import Player


class MCTS_AI(Player):
    """Monte Carlo Tree Search AI for Hive game."""

    def __init__(self, 
                 colour: Colour,
                 game: Game,
                 model_callable: callable,
                 workers: int = 4,
                 search_time: float = 5,
                 iterations: int = 1000):
        
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







if __name__ == '__main__':
    from hive.game_engine.game_state import initial_game

    game = initial_game(pieces_function=create_reduced_pieces)

    mctsai = MCTS_AI(WHITE, game, model_callable=mock_model_callable)
    randomai = RandomAI(BLACK)

    winner = play(mctsai, randomai, game=game, max_turns=500)
    