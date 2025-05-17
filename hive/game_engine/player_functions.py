from typing import List

from hive.game_engine.game_state import Game, Colour

from hive.game_engine.errors import NoQueenError
from hive.game_engine.game_functions import check_queen_timely_placement
from hive.game_engine.game_state import Colour, Game
from hive.game_engine.grid_functions import get_placeable_locations
from hive.game_engine.moves import Move, get_possible_moves
from hive.game_engine.pieces import QUEEN, PILLBUG


def get_players_possible_moves_or_placements(colour: Colour, game: Game) -> List[Move]:

    if _must_play_queen(colour, game):
        return _play_queen_placements(colour, game)

    moves = _all_placements(colour, game)  # can always place pieces
    if game.queens.get(colour) is not None:  # but can only move once queen is placed
        moves += get_players_moves(colour, game)
    return moves

def get_players_moves(colour: Colour, game: Game) -> List[Move]:

    moveable_pieces = []
    for loc, stack in game.grid.items():
        if stack and stack[-1].colour == colour:
            piece = stack[-1]
            moveable_pieces.append((piece, loc, len(stack)-1))

    possible_moves = []
    for piece, current_location, stack_idx in moveable_pieces:
        piece_possible_moves = get_possible_moves(game.grid, current_location, stack_idx)
        piece_possible_moves = _filter_pillbug_moves(game, piece_possible_moves, piece)
        possible_moves += piece_possible_moves

    return possible_moves

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




def _filter_pillbug_moves(game, possible_moves, piece):
    """Filter out moves disallowed because of pillbug"""

    previous_pillbug_move_piece = None
    if game.move is not None:
        if game.move.pillbug_moved_other_piece is True:
            previous_pillbug_move_piece = game.move.piece

    moves = []
    for mv in possible_moves:

        # first - is the move a pillbug move - two different types of check we need to do
        if mv.pillbug_moved_other_piece == True:
            # the pillbug is moving another piece, and can't move this if it was moved last turn
            if game.piece_moved_last_turn == mv.piece:
                continue

            # was the piece performing the pillbug move moved last turn by a pillbug? if so it cant move other pieces
            if piece == previous_pillbug_move_piece:
                continue

        else:
            # this is a normal move - if piece was moved *by pillbug* last turn, then can't move it
            if game.piece_moved_last_turn == mv.piece:
                # piece was played last move - but was it by a pillbug?
                if game.move.pillbug_moved_other_piece == True:
                    # the piece was moved by a pillbug - so we can't move it
                    continue

        # we passed all checks so keep the move
        moves.append(mv)

    return moves



