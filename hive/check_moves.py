from __future__ import annotations

from typing import TYPE_CHECKING

from hive.errors import InvalidPlacementError, InvalidMoveError, InvalidLocationError, BreaksConnectionError

from hive.grid_functions import pieces_around_location, is_position_connected, can_remove_piece
from hive.game_types import Grid, Location, Colour


def check_is_valid_placement(grid: Grid, loc: Location, colour: Colour):
    """Place a piece for the first time can not be touching a piece from opposite colour."""

    # first piece can be anywhere
    if len(grid) == 0:
        return

    # piece must be placed next to another piece
    piece_locations = pieces_around_location(grid, loc)
    if len(piece_locations) == 0:
        raise InvalidPlacementError(f"Not connected")

    # if this is the second piece, can go anywhere
    if len(grid) == 1:
        return

    # if not the first piece of that colour, can only go next to pieces of the same colour
    stacks = [grid.get(loc) for loc in piece_locations]
    colours_surrounding = set([p[-1].colour for p in stacks])
    if len(colours_surrounding) > 1 or colour not in colours_surrounding:
        raise InvalidPlacementError(
            f"{colour} piece can not be placed next to pieces of other color: {colours_surrounding}")


def check_is_valid_move(grid, current_loc, loc: Location):

    # are all the positions around the current location still connected if the piece is removed?
    stack = grid.get(current_loc)
    if len(stack) >= 2:
        pass
    elif can_remove_piece(grid, current_loc) == False:
        raise BreaksConnectionError

    # check piece will be connected to the hive once moved
    piece_locations_new = pieces_around_location(grid, loc)
    if len(stack) == 1:
        piece_locations_new = [loc for loc in piece_locations_new if loc != current_loc]
    if len(piece_locations_new) == 0:
        raise InvalidPlacementError(f"Not connected")

def check_is_valid_location(loc: Location):
    """Sum of locations must be even"""
    if sum(loc) % 2 != 0:
        raise InvalidLocationError(f"Invalid location - {loc} does not sum to even")
