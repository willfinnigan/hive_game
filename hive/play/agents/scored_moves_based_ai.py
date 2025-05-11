import random
from dataclasses import dataclass
from typing import Union, Callable, List, Tuple

from hive.game_engine import pieces
from hive.game_engine.game_state import Colour, Game
from hive.game_engine.grid_functions import pieces_around_location
from hive.game_engine.moves import Move, NoMove
from hive.game_engine.player_functions import get_players_possible_moves_or_placements
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

def score_move_by_queen(move, 
                        locs_around_current_which_contain_pieces, 
                        locs_around_move_which_contain_pieces, 
                        scores: MoveScores) -> int:

    # look at current location - are we already attacking enemy queen?
    for loc in locs_around_current_which_contain_pieces:
        stack = move.game.grid.get(loc, ())
        piece = stack[-1] if stack else None
        if piece.name == pieces.QUEEN and piece.colour != move.piece.colour:
            return scores.piece_already_at_queen  # already next to a enemy queen - don't move!

        
    # look at new location - are we attacking enemy queen?
    for loc in locs_around_move_which_contain_pieces:
        stack = move.game.grid.get(loc, ())
        piece = stack[-1] if stack else None
        if piece is not None and piece.name == pieces.QUEEN and piece.colour != move.piece.colour:
            return scores.move_to_queen  # move next to enemy queen
        

    # look at current location - are there allied queens we should move away from
    for loc in locs_around_move_which_contain_pieces:
        stack = move.game.grid.get(loc, ())
        piece = stack[-1] if stack else None
        if piece is not None and piece.name == pieces.QUEEN and piece.colour == move.piece.colour:
            return scores.move_away_from_queen
    
    # otherwise return 0
    return 0
    

def score_play_piece(move, scores) -> int:
    if move.current_location is not None:
        raise ValueError("Move must be a placement move to score it as placement")

    if move.piece.name == pieces.ANT:
        return scores.play_ant
    if move.piece.name == pieces.BEETLE:
        return scores.play_beetle
    if move.piece.name == pieces.GRASSHOPPER:
        return scores.play_grasshopper
    if move.piece.name == pieces.SPIDER:
        return scores.play_spider
    if move.piece.name == pieces.QUEEN:
        return scores.play_queen
    return 0



def prioritise_moves(moves: List[Move], game: Game, scores: MoveScores = None) -> List[Tuple[int, Move]]:

    scores = scores or MoveScores()

    scored_moves = []
    for move in moves:
        score = 0
        if move.current_location is not None:
            locs_around_current_which_contain_pieces = pieces_around_location(game.grid, move.current_location)
            locs_around_move_which_contain_pieces = pieces_around_location(game.grid, move.new_location)

            # score the move
            score += score_move_by_queen(move, locs_around_current_which_contain_pieces, locs_around_move_which_contain_pieces, scores)
        else:
            # score the piece being played
            score += score_play_piece(move, scores)

        scored_moves.append((score, move))

    # shuffle then sort scored_moves
    random.shuffle(scored_moves)
    scored_moves = sorted(scored_moves, key=lambda x: x[0], reverse=True)
    return scored_moves



class ScoreMovesAI(Player):
    """ Plays moves according to a move scoring policy
    """

    def __init__(self, colour: Colour, scores: MoveScores = None):
        super().__init__(colour)
        self.scores = scores or MoveScores()

    def get_move(self, game) -> Union[Move|NoMove]:
        possible_moves = get_players_possible_moves_or_placements(self.colour, game)
        if len(possible_moves) == 0:
            return NoMove(self.colour)

        # score the moves
        scored_moves = prioritise_moves(possible_moves, game, self.scores)

        # return the best move
        return scored_moves[0][1]

