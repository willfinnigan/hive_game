from typing import List

from hive.game_engine.game_state import Game, Colour

from hive.game_engine.errors import NoQueenError
from hive.game_engine.game_functions import check_queen_timely_placement
from hive.game_engine.game_state import Colour, Game
from hive.game_engine.grid_functions import get_placeable_locations
from hive.game_engine.moves import Move, get_possible_moves
from hive.game_engine.pieces import QUEEN


def get_players_possible_moves_or_placements(colour: Colour, game: Game) -> List[Move]:

    if _must_play_queen(colour, game):
        return _play_queen_placements(colour, game)

    moves = _all_placements(colour, game)  # can always place pieces
    if game.queens.get(colour) is not None:  # but can only move once queen is placed
        moves += _placed_piece_moves(colour, game)
    return moves

def _must_play_queen(colour: Colour, game: Game) -> bool:
    try:
        check_queen_timely_placement(game, colour, moves_to_queen=3)
    except NoQueenError:
        return True
    return False

def _play_queen_placements(colour: Colour, game: Game) -> List[Move]:
    """ If the player must play a queen, return all possible placements """

    unplayed_pieces = game.unplayed_pieces[colour]
    queen = [piece for piece in unplayed_pieces if piece.name == QUEEN][0]
    placeable_locations = get_placeable_locations(game.grid, colour)

    possible_moves = []
    for location in placeable_locations:
        possible_moves.append(Move(piece=queen,
                                   current_location=None,
                                   current_stack_idx=None,
                                   new_location=location,
                                   new_stack_idx=0))
    return possible_moves

def _all_placements(colour: Colour, game: Game) -> List[Move]:
    """ Return all possible placements of pieces """

    # all placeable locations * all pieces
    possible_moves = []
    placeable_locations = get_placeable_locations(game.grid, colour)
    for piece in game.unplayed_pieces[colour]:
        for location in placeable_locations:
            possible_moves.append(Move(piece=piece,
                                       current_location=None,
                                       current_stack_idx=None,
                                       new_location=location,
                                       new_stack_idx=0))
    return possible_moves

def _placed_piece_moves(colour: Colour, game: Game) -> List[Move]:

    moveable_pieces = []
    for loc, stack in game.grid.items():
        if stack and stack[-1].colour == colour:
            piece = stack[-1]
            moveable_pieces.append((piece, loc, len(stack)-1))

    possible_moves = []
    for piece, current_location, stack_idx in moveable_pieces:
        possible_moves += get_possible_moves(game.grid, current_location, stack_idx)
    return possible_moves

