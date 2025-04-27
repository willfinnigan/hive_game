from __future__ import annotations

from typing import Optional, List

from hive.game.types_and_errors import Location, Colour, Grid
from hive.game.grid_functions import beetle_one_move_away, can_remove_piece
from hive.game.pieces.piece_base_class import Piece



class Beetle(Piece):

    def __init__(self, colour: Colour, number: int = 0):
        super().__init__(colour)
        self.piece_letter = 'B'
        self.number = number  # number of this piece on the board
        self.sitting_on: Optional[Piece] = None
        self.can_move_on_top = True

    def place(self, grid, location, new_piece):
        self._check_is_valid_placement(grid, location, new_piece)

        assert self.sitting_on is None
        assert self.on_top is None
        self.location = None

        if grid.get(location) is not None:
            self._place_on_top_of_another_piece(grid, location)
        else:
            self._place_on_new_location(grid, location)

    def get_moves(self, grid: Grid, loc: Location) -> List[Location]:
        """Get possible moves for a piece"""
        return beetle_one_move_away(grid, loc)

    def get_possible_moves(self, grid: Grid) -> List[Location]:
        """Beetle can move 1 space in any direction, or climb on top of another piece"""
        # check if beetle is on top of another piece

        if self.on_top is not None:
            return []
        if can_remove_piece(grid, self) == False:
            return []

        loc = self.location
        if self.sitting_on is None:
            grid.pop(self.location)
        self.location = None

        locations = self.get_moves(grid, loc)

        self.location = loc
        if self.sitting_on is None:
            grid[loc] = self

        return locations
