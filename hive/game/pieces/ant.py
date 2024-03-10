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




    def get_possible_moves(self, grid: Grid) -> List[Location]:
        """Ant can move to any space, provided there is a linear path to it"""
        if self.on_top is not None:
            print('Ant can not move as piece is on top')
            return []
        if can_remove_piece(grid, self) == False:
            print(f'{self.colour} ant cant disconnect hive')
            return []  # if we can not move the piece, there are no possible moves

        tmp_grid = deepcopy(grid)
        tmp_grid.pop(self.location)

        # move 1 piece at a time clockwise round the board
        locations = [self.location]
        possible_moves = one_move_away(tmp_grid, locations[-1])
        unexplored = [move for move in possible_moves if move not in locations]
        while len(unexplored) > 0:
            locations.append(unexplored[0])
            possible_moves = one_move_away(tmp_grid, locations[-1])
            unexplored = [move for move in possible_moves if move not in locations]

        locations.remove(self.location)
        return locations
