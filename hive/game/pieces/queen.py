from __future__ import annotations

from copy import deepcopy
from typing import List

from hive.game.types_and_errors import Location, Colour, Grid
from hive.game.grid_functions import one_move_away, can_remove_piece
from hive.game.pieces.piece_base_class import Piece



class Queen(Piece):

        def __init__(self, colour: Colour):
            super().__init__(colour)
            self.piece_letter = 'Q'

        def get_possible_moves(self, grid: Grid) -> List[Location]:
            """Queen can move 1 space in any direction."""
            if self.on_top is not None:
                return []
            if can_remove_piece(grid, self) == False:
                return []  # if we can not move the piece, there are no possible moves

            # temporarily remove piece from grid
            loc = self.location
            grid.pop(self.location)
            self.location = None

            # get possible moves
            possible_moves = one_move_away(grid, loc)

            # revert grid
            self.location = loc
            grid[loc] = self
            return possible_moves
