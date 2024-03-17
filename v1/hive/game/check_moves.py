from __future__ import annotations
from typing import TYPE_CHECKING
from v1.hive.game.grid_functions import pieces_around_location
from v1.hive.game.types_and_errors import Location, InvalidPlacementError, InvalidMoveError, InvalidLocationError, Grid

if TYPE_CHECKING:
    from v1.hive.game.pieces.piece_base_class import Piece


def exception_if_invalid_placement(grid: Grid, loc: Location, piece: Piece):
    """PLace a piece for the first time can not be touching a piece from opposite colour."""

    if len(grid) == 0:
        return

    pieces = pieces_around_location(grid, loc)
    if len(pieces) == 0:
        raise InvalidPlacementError(f"Not connected")

    # if this is the first piece of that colour, can go anywhere
    if len([p for p in grid.values() if p.colour == piece.colour]) == 0:
        return

    # if not the first piece of that colour, can only go next to pieces of the same colour
    colours_surrounding = set([p.colour for p in pieces])
    if len(colours_surrounding) > 1 or piece.colour not in colours_surrounding:
        raise InvalidPlacementError(
            f"{piece.colour} piece can not be placed next to pieces of other color: {colours_surrounding}")


def exception_if_invalid_move(grid, loc: Location, piece: Piece, ):
    if grid.get(loc) is not None and piece.can_move_on_top is False:
        raise InvalidMoveError(f"Invalid move for {piece.__class__} - {loc} is not empty")

    pieces = pieces_around_location(grid, loc)
    if len(pieces) == 0:
        raise InvalidPlacementError(f"Not connected")


def exception_if_invalid_location(loc: Location):
    """Sum of locations must be even"""
    if sum(loc) % 2 != 0:
        raise InvalidLocationError(f"Invalid location - {loc} does not sum to even")