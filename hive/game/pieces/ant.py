from __future__ import annotations

from copy import deepcopy
from typing import List

from hive.game.types_and_errors import Location, Colour, Grid
from hive.game.grid_functions import one_move_away, can_remove_piece
from hive.game.pieces.piece_base_class import Piece


class Ant(Piece):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.piece_letter = 'A'

    def get_moves(self, grid: Grid, loc: Location) -> List[Location]:
        # move 1 piece at a time clockwise round the board
        locations = [loc]
        possible_moves = one_move_away(grid, locations[-1])
        unexplored = [move for move in possible_moves if move not in locations]
        while len(unexplored) > 0:
            locations.append(unexplored[0])
            possible_moves = one_move_away(grid, locations[-1])
            unexplored = [move for move in possible_moves if move not in locations]

        locations.remove(loc)
        return locations



