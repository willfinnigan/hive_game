from __future__ import annotations

from typing import List

from hive.game.types_and_errors import Location, Colour, Grid
from hive.game.grid_functions import one_move_away, can_remove_piece
from hive.game.pieces.piece_base_class import Piece



class Spider(Piece):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.piece_letter = 'S'

    def get_moves(self, grid: Grid, loc: Location) -> List[Location]:
        # first step
        walks = [[loc]]
        new_walks = []
        for walk in walks:
            for move in one_move_away(grid, walk[-1]):
                if move not in walk:
                    new_walks.append(walk + [move])

        # second step
        walks = new_walks
        new_walks = []
        for walk in walks:
            for move in one_move_away(grid, walk[-1]):
                if move not in walk:
                    new_walks.append(walk + [move])

        # third step
        walks = new_walks
        new_walks = []
        for walk in walks:
            for move in one_move_away(grid, walk[-1]):
                if move not in walk:
                    new_walks.append(walk + [move])

        return [walk[-1] for walk in new_walks]



