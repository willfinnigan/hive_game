import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Union, List

from v1.hive.game.game import Game
from v1.hive.game.grid_functions import pieces_around_location
from v1.hive.game.types_and_errors import Colour, WHITE, BLACK
from v1.hive.play.move import NoMove, Move
from v1.hive.play.player import Player

COLOURS = [WHITE, BLACK]

@dataclass
class BoardScores:
    per_queen_surrounded: int = -5
    per_enemy_queen_surrounded: int = 5

    def score(self, game: Game, colour: Colour) -> int:
        score = 0
        queen = game.queens.get(colour)
        other_colour = [c for c in COLOURS if c != colour][0]
        enemy_queen = game.queens[other_colour]

        if queen is not None:
            pieces_around_queen = pieces_around_location(game.grid, queen.location)
            score += len(pieces_around_queen) * self.per_queen_surrounded

        if enemy_queen is not None:
            pieces_around_enemy_queen = pieces_around_location(game.grid, enemy_queen.location)
            score += len(pieces_around_enemy_queen) * self.per_enemy_queen_surrounded

        return score





class ScoreBoardStateAI(Player):
    """ Plays moves according which will give the best board state"""

    def __init__(self, colour: Colour, scores: BoardScores = None):
        super().__init__(colour)
        self.scores = scores or BoardScores()

    def get_move(self, game) -> Union[Move|NoMove]:
        possible_moves = self.possible_moves(game)
        if len(possible_moves) == 0:
            return NoMove(self.colour)

        # score the moves
        scored_moves = self.score_future_board_states(game, possible_moves)
        return scored_moves[0][1]

    def score_future_board_states(self, game: Game, moves: List[Move]):

        move_scores = []
        for move in moves:

            tmp_game = deepcopy(game)

            # play the move
            move.play(tmp_game)

            # score the board state
            score = self.scores.score(tmp_game, self.colour)
            move_scores.append((score, move))


        # shuffle then sort scored_moves
        random.shuffle(move_scores)
        move_scores = sorted(move_scores, key=lambda x: x[0], reverse=True)
        return move_scores

