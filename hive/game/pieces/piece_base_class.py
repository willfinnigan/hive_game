from __future__ import annotations
from typing import Optional, List

from hive.game.check_moves import exception_if_invalid_location, exception_if_invalid_placement, exception_if_invalid_move
from hive.game.types_and_errors import InvalidMoveError, BreaksConnectionError, Location, Colour, Grid, WHITE, BLACK
from hive.game.grid_functions import get_empty_locations,  can_remove_piece


white = "\033[37m"
black = "\033[30m"
darkblue = "\033[36m"
yellow = "\033[33m"
reset = "\033[0m"


class Piece:
    def __init__(self, colour: Colour, number: int):
        self.colour: Colour = colour
        self.piece_letter: str = 'P'
        self.location: Location = None
        self.number: int = number  # number of this piece on the board

        self.on_top: Optional[Piece] = None  # if there is a piece on top of this one
        self.sitting_on: Optional[Piece] = None  # if this piece is om top of another one

        self.can_move_on_top = False  # basically is a beetle at this point

    def __post_init__(self):
        # number must be single digit (negative or positive)
        if self.number > 9 or self.number < -9:
            raise ValueError("Piece number must be a single digit")

    def as_text_colour(self):
        if self.colour == "W":
            return f"{yellow}{self.piece_letter}{reset}"
        else:
            return f"{darkblue}{self.piece_letter}{reset}"

    def as_text(self):
        if self.number == 0:
            return f"{self.colour} {self.piece_letter} "
        else:
            return f"{self.colour}{self.piece_letter}{self.number}"

    def __repr__(self):
        return self.as_text()

    def place(self, grid: Grid, location: Location, new_piece: bool):
        """Place this piece on the board, either for the first time or as a move"""
        self._check_is_valid_placement(grid, location, new_piece)
        self._place_on_new_location(grid, location)

    def remove(self, grid: Grid):
        if self.on_top is not None:
            raise InvalidMoveError(f"{self.__class__} can not move - it is carrying other pieces.")

        if self.sitting_on is not None:
            self.sitting_on.on_top = None  # beetle no longer on top of the piece
            self.sitting_on = None
            self.location = None
            return

        if can_remove_piece(grid, self) == False:
            raise BreaksConnectionError("Removing piece breaks the graph")

        grid.pop(self.location)  # remove piece from grid
        self.location = None  # update piece location

    def get_moves(self, grid: Grid, loc: Location) -> List[Location]:
        """Get possible moves for a piece"""
        return get_empty_locations(grid)

    def get_possible_moves(self, grid: Grid) -> List[Location]:
        """Base piece can move anywhere"""
        if self.on_top is not None:
            return []
        if can_remove_piece(grid, self) == False:
            return []  # if we can not move the piece, there are no possible moves

        # temporarily remove piece from grid
        loc = self.location
        grid.pop(loc)
        self.location = None

        locations = self.get_moves(grid, loc)

        # revert grid
        self.location = loc
        grid[loc] = self

        return locations


    def _check_is_valid_placement(self, grid: Grid, location: Location, new_piece: bool):
        """Check if a placement is valid - raises exception if not"""
        exception_if_invalid_location(location)
        if new_piece == True:
            exception_if_invalid_placement(grid, location, self)
        else:
            exception_if_invalid_move(grid, location, self)

    def _place_on_new_location(self, grid: Grid, location: Location):
        grid[location] = self  # add piece to grid
        self.location = location  # update piece location

    def _place_on_top_of_another_piece(self, grid: Grid, location: Location):

        # if was previously on top of another piece, update that piece
        if self.sitting_on is not None:
            self.sitting_on.on_top = None

        # get the top piece by starting at the bottom and moving up
        bottom_piece = grid[location]
        top_piece = bottom_piece
        while top_piece.on_top is not None:
            top_piece = top_piece.on_top

        top_piece.on_top = self  # add beetle on top of the stack
        self.sitting_on = top_piece  # beetle is sitting on top of the top piece
        self.location = location
