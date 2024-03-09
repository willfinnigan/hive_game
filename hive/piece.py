from __future__ import annotations
from copy import deepcopy
from typing import List, Optional

from hive.graph_functions import can_remove_piece
from hive.grid_functions import is_position_connected, get_empty_locations, positions_around_location, one_move_away, \
    beetle_one_move_away

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hive.board import Board, Colour, Location


class Piece:

    def __init__(self, colour: Colour):
        self.colour: Colour = colour
        self.piece_letter: str = 'P'
        self.location: Location = None
        self.on_top: Optional[Piece] = None

    def as_text(self):
        letter = self.colour[0].lower()
        return f"{letter}{self.piece_letter}"

    def can_move(self):
        if self.on_top is not None:
            return False
        return True

    def get_possible_moves(self, board: Board) -> List[Location]:
        """Can move anywhere"""
        if self.can_move == False:
            return []
        if can_remove_piece(board.graph, self) == False:
            return []  # if we can not move the piece, there are no possible moves

        locations = get_empty_locations(board.grid)
        return locations


class Queen(Piece):

        def __init__(self, colour: Colour):
            super().__init__(colour)
            self.piece_letter = 'Q'

        def get_possible_moves(self, board: Board) -> List[Location]:
            """Queen can move 1 space in any direction."""
            if self.can_move == False:
                return []
            if can_remove_piece(board.graph, self) == False:
                return []  # if we can not move the piece, there are no possible moves
            tmp_grid = deepcopy(board.grid)
            tmp_grid.pop(self.location)
            return one_move_away(tmp_grid, self.location)


class Ant(Piece):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.piece_letter = 'A'

    def get_possible_moves(self, board: Board) -> List[Location]:
        """Ant can move to any space, provided there is a linear path to it"""
        if self.can_move == False:
            return []
        if can_remove_piece(board.graph, self) == False:
            return []  # if we can not move the piece, there are no possible moves

        tmp_grid = deepcopy(board.grid)
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

class GrassHopper(Piece):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.piece_letter = 'G'

    def get_possible_moves(self, board: Board) -> List[Location]:
        """Grasshopper can jump over pieces in a straight line"""
        if self.can_move == False:
            return []
        if can_remove_piece(board.graph, self) == False:
            return []

        line_transformations = [(-1, -1), (+1, -1), (-2, 0), (+2, 0), (-1, +1), (+1, +1)]

        jumps = []
        for transform in line_transformations:
            pieces_jumped = 0
            pos = (self.location[0] + transform[0], self.location[1] + transform[1])
            while board.get_piece(pos) is not None:
                pieces_jumped += 1
                pos = (pos[0]+transform[0], pos[1]+transform[1])

            if pieces_jumped == 0:
                continue
            elif is_position_connected(board.grid, pos) == True:
                jumps.append(pos)

        return jumps


class Spider(Piece):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.piece_letter = 'S'

    def get_possible_moves(self, board: Board) -> List[Location]:
        """Spider must move 3 spaces"""
        if self.can_move == False:
            return []
        if can_remove_piece(board.graph, self) == False:
            return []

        tmp_grid = deepcopy(board.grid)
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

class Beetle(Piece):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.piece_letter = 'B'
        self.sitting_on: Optional[Piece] = None

    def get_possible_moves(self, board: Board) -> List[Location]:
        """Beetle can move 1 space in any direction, or climb on top of another piece"""
        if self.can_move == False:
            return []
        if can_remove_piece(board.graph, self) == False:
            return []

        tmp_grid = deepcopy(board.grid)
        tmp_grid.pop(self.location)
        locations = beetle_one_move_away(tmp_grid, self.location)
        return locations















