from __future__ import annotations

from typing import List

from hive.game.types_and_errors import Location, Colour, Grid
from hive.game.grid_functions import is_position_connected, can_remove_piece
from hive.game.pieces.piece_base_class import Piece


class GrassHopper(Piece):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.piece_letter = 'G'



    def get_possible_moves(self, grid: Grid) -> List[Location]:
        """Grasshopper can jump over pieces in a straight line"""
        if self.on_top is not None:
            return []
        if can_remove_piece(grid, self) == False:
            return []

        line_transformations = [(-1, -1), (+1, -1), (-2, 0), (+2, 0), (-1, +1), (+1, +1)]

        jumps = []
        for transform in line_transformations:
            pieces_jumped = 0
            pos = (self.location[0] + transform[0], self.location[1] + transform[1])
            while grid.get(pos) is not None:
                pieces_jumped += 1
                pos = (pos[0]+transform[0], pos[1]+transform[1])

            if pieces_jumped == 0:
                continue
            elif is_position_connected(grid, pos) == True:
                jumps.append(pos)

        return jumps
