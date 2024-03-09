from hive.graph_functions import can_remove_piece
from hive.grid_functions import pieces_around_location
from hive.custom_types import Location, Colour
import networkx as nx
from hive.errors import InvalidPlacementError, BreaksConnectionError, InvalidLocationError, InvalidMoveError
from hive.piece import Piece, Beetle


class Board:
    def __init__(self):
        self.grid = {}
        self.graph = nx.Graph()

    def get_piece(self, loc: Location) -> Piece:
        self._check_valid_location(loc)
        return self.grid.get(loc)

    def place_piece(self, piece: Piece, loc: Location, new_piece: bool):
        self._check_valid_location(loc)

        if new_piece == True:
            self._is_valid_placement(loc, piece)
        else:
            self._is_valid_move(loc, piece)

        if isinstance(piece, Beetle) is True and self.grid.get(loc) is not None:
            # get the top piece by starting at the bottom and moving up
            bottom_piece = self.grid[loc]
            top_piece = bottom_piece
            while top_piece.on_top is not None:
                top_piece = top_piece.on_top

            top_piece.on_top = piece  # add beetle on top of the top piece
            piece.sitting_on = top_piece  # beetle is sitting on top of the top piece
            piece.location = loc
            return

        self.grid[loc] = piece  # add piece to grid
        self.graph.add_node(piece)  # add piece to graph
        for p in pieces_around_location(self.grid, loc):  # add edges to pieces around it
            self.graph.add_edge(piece, p)

        piece.location = loc  # update piece location

    def remove_piece(self, piece: Piece):
        if piece.can_move() == False:
            raise InvalidMoveError(f"{piece.__class__} can not move - it is carrying other pieces.")

        if isinstance(piece, Beetle) is True:
            if piece.sitting_on is not None:
                piece.sitting_on.on_top = None  # beetle no longer on top of the piece
                piece.sitting_on = None
                piece.location = None
                return

        if can_remove_piece(self.graph, piece) == False:
            raise BreaksConnectionError("Removing piece breaks the graph")

        piece = self.grid.pop(piece.location)  # remove piece from grid
        self.graph.remove_node(piece)  # remove piece from graph
        piece.location = None  # update piece location

    def _is_valid_placement(self, loc: Location, piece: Piece):
        """PLace a piece for the first time can not be touching a piece from opposite colour."""

        if len(self.grid) == 0:
            return

        pieces = pieces_around_location(self.grid, loc)
        if len(pieces) == 0:
            raise InvalidPlacementError(f"Not connected")

        # if this is the first piece of that colour, can go anywhere
        if len([p for p in self.grid.values() if p.colour == piece.colour]) == 0:
            return

        # if not the first piece of that colour, can only go next to pieces of the same colour
        colours_surrounding = set([p.colour for p in pieces])
        if len(colours_surrounding) > 1 or piece.colour not in colours_surrounding:
            raise InvalidPlacementError(
                f"{piece.colour} piece can not be placed next to pieces of other color: {colours_surrounding}")

    def _is_valid_move(self, loc: Location, piece: Piece,):
        if self.get_piece(loc) is not None:
            if isinstance(piece, Beetle) is False:
                raise InvalidMoveError(f"Invalid move for {piece.__class__} - {loc} is not empty")

        pieces = pieces_around_location(self.grid, loc)
        if len(pieces) == 0:
            raise InvalidPlacementError(f"Not connected")

    def _check_valid_location(self, loc: Location):
        """Sum of locations must be even"""
        if sum(loc) % 2 != 0:
            raise InvalidLocationError(f"Invalid location - {loc} does not sum to even")
