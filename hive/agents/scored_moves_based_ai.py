import random
from dataclasses import dataclass
from typing import Union, Callable, List, Tuple

from hive.game.game import Game
from hive.game.grid_functions import pieces_around_location
from hive.game.pieces.ant import Ant
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.queen import Queen
from hive.game.pieces.spider import Spider
from hive.game.types_and_errors import Colour
from hive.play.move import Move, NoMove
from hive.play.player import Player

Strategy = Callable[[List[Move], Game], List[Tuple[int, Move]]]

@dataclass
class MoveScores:
    move_to_queen: int = 5
    piece_already_at_queen: int = -5
    move_away_from_queen: int = 3

    play_ant: int = 4
    play_beetle: int = 1
    play_grasshopper: int = 1
    play_spider: int = 1
    play_queen: int = 3

def score_queen(move, around_current_location, around_move, scores) -> int:
    opposing_queens = [piece for piece in around_current_location if piece.colour != move.piece.colour and isinstance(piece, Queen)]
    if len(opposing_queens) > 0:
        return scores.piece_already_at_queen  # already next to a queen - don't move!
    opposing_queens = [piece for piece in around_move if piece.colour != move.piece.colour and isinstance(piece, Queen)]
    if len(opposing_queens) > 0:
        return scores.move_to_queen  # move next to enemy queen

    allied_queens_current = [piece for piece in around_current_location if piece.colour == move.piece.colour and isinstance(piece, Queen)]
    if len(allied_queens_current) > 0:
        allied_queens_new = [piece for piece in around_move if piece.colour == move.piece.colour and isinstance(piece, Queen)]
        if len(allied_queens_new) == 0:
            return scores.move_away_from_queen

    return 0

def score_play_piece(move, scores) -> int:
    if move.piece.location is not None:
        return 0

    if isinstance(move.piece, Ant):
        return scores.play_ant
    if isinstance(move.piece, Beetle):
        return scores.play_beetle
    if isinstance(move.piece, GrassHopper):
        return scores.play_grasshopper
    if isinstance(move.piece, Spider):
        return scores.play_spider
    if isinstance(move.piece, Queen):
        return scores.play_queen

    return 0



def prioritise_moves(moves: List[Move], game: Game, scores: MoveScores = None) -> List[Tuple[int, Move]]:

    scores = scores or MoveScores()

    scored_moves = []
    for move in moves:
        score = 0
        if move.piece.location is not None:
            around_current_location = pieces_around_location(game.grid, move.piece.location)
            around_move = pieces_around_location(game.grid, move.location)

            # score the move
            score += score_queen(move, around_current_location, around_move, scores)

        scored_moves.append((score, move))

    # shuffle then sort scored_moves
    random.shuffle(scored_moves)
    scored_moves = sorted(scored_moves, key=lambda x: x[0], reverse=True)
    return scored_moves

class ScoreMovesAI(Player):
    """ Plays moves according to a move scoring policy
    Still pretty poor AI, but better than random.
    """

    def __init__(self, colour: Colour, scores: MoveScores = None):
        super().__init__(colour)
        self.scores = scores or MoveScores()

    def get_move(self, game) -> Union[Move|NoMove]:
        possible_moves = self.possible_moves(game)
        if len(possible_moves) == 0:
            return NoMove(self.colour)

        # score the moves
        scored_moves = prioritise_moves(possible_moves, game, self.scores)

        # return the best move
        return scored_moves[0][1]
