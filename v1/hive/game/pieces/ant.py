from __future__ import annotations
from typing import List

from v1.hive.game.types_and_errors import Location, Colour, Grid
from v1.hive.game.grid_functions import one_move_away, can_remove_piece
from v1.hive.game.pieces.piece_base_class import Piece


class Ant(Piece):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.piece_letter = 'A'

    def _clockwise_moves(self, grid, loc):
        locations = [loc]
        possible_moves = one_move_away(grid, locations[-1])
        unexplored = [move for move in possible_moves if move not in locations]
        while len(unexplored) > 0:
            locations.append(unexplored[0])
            possible_moves = one_move_away(grid, locations[-1])
            unexplored = [move for move in possible_moves if move not in locations]

        locations.remove(loc)
        return locations

    def _anticlockwise_moves(self, grid, loc):
        locations = [loc]
        possible_moves = one_move_away(grid, locations[0])
        unexplored = [move for move in possible_moves if move not in locations]
        while len(unexplored) > 0:
            locations.append(unexplored[0])
            possible_moves = one_move_away(grid, locations[0])
            unexplored = [move for move in possible_moves if move not in locations]

        locations.remove(loc)
        return locations


    def get_moves(self, grid: Grid, loc: Location) -> List[Location]:
        # move 1 piece at a time clockwise and anticlockwise round the board
        clockwise = self._anticlockwise_moves(grid, loc)
        anticlockwise = self._clockwise_moves(grid, loc)

        moves = clockwise + anticlockwise
        return list(set(moves))



