from __future__ import annotations

from typing import List

from v1.hive.game.types_and_errors import Location, Colour, Grid
from v1.hive.game.grid_functions import one_move_away, can_remove_piece
from v1.hive.game.pieces.piece_base_class import Piece



class Queen(Piece):

        def __init__(self, colour: Colour):
            super().__init__(colour)
            self.piece_letter = 'Q'

        def get_moves(self, grid: Grid, loc: Location) -> List[Location]:
            """Get possible moves for a piece"""
            return one_move_away(grid, loc)
