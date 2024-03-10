from __future__ import annotations

from copy import deepcopy
from typing import List

from hive.game.types_and_errors import Location, Colour, Grid
from hive.game.grid_functions import one_move_away, can_remove_piece
from hive.game.pieces.piece_base_class import Piece



class Spider(Piece):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.piece_letter = 'S'

    def get_possible_moves(self, grid: Grid) -> List[Location]:
        """Spider must move 3 spaces"""
        if self.on_top is not None:
            return []
        if can_remove_piece(grid, self) == False:
            return []

        tmp_grid = deepcopy(grid)
        tmp_grid.pop(self.location)
        locations = one_move_away(tmp_grid, self.location)


        # first step
        walks = [[self.location]]
        new_walks = []
        for walk in walks:
            for move in one_move_away(tmp_grid, walk[-1]):
                if move not in walk:
                    new_walks.append(walk + [move])

        # second step
        walks = new_walks
        new_walks = []
        for walk in walks:
            for move in one_move_away(tmp_grid, walk[-1]):
                if move not in walk:
                    new_walks.append(walk + [move])

        # third step
        walks = new_walks
        new_walks = []
        for walk in walks:
            for move in one_move_away(tmp_grid, walk[-1]):
                if move not in walk:
                    new_walks.append(walk + [move])

        return [walk[-1] for walk in new_walks]
