from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List

from hive.game.types_and_errors import InvalidMoveError, PlayerHasLost, NoQueenError, Location, Colour, WHITE, BLACK, \
    Grid
from hive.game.grid_functions import pieces_around_location
from hive.game.pieces.queen import Queen

if TYPE_CHECKING:
    from hive.game.pieces.piece_base_class import Piece

class Game:

    def __init__(self,
                 grid: Grid = None,
                 player_colours: List[Colour] = None):
        colours = player_colours or [WHITE, BLACK]
        self.grid = grid or {}
        self.player_turns = {colour: 0 for colour in colours}
        self.queens = {colour: None for colour in colours}

    def get_piece(self, loc: Location) -> Piece:
        return self.grid.get(loc)

    def place_piece(self, piece: Piece, loc: Location):
        if self.has_player_lost(piece.colour):
            raise PlayerHasLost(f"{piece.colour} has lost")

        piece.place(grid=self.grid, location=loc, new_piece=True)

        self.player_turns[piece.colour] += 1
        if isinstance(piece, Queen):
            self.queens[piece.colour] = piece

        self.check_queen_exists(piece.colour)

    def move_piece(self, piece: Piece, new_loc: Location):
        if self.has_player_lost(piece.colour):
            raise PlayerHasLost(f"{piece.colour} has lost")

        possible_moves = piece.get_possible_moves(self.grid)
        if new_loc not in possible_moves:
            raise InvalidMoveError(f"Invalid move for {piece.__class__} - {new_loc} not in {possible_moves}")

        piece.remove(self.grid)
        piece.place(self.grid, new_loc, False)
        self.player_turns[piece.colour] += 1
        self.check_queen_exists(piece.colour)

    def check_queen_exists(self, colour: Colour, moves_to_queen=4):
        if self.player_turns[colour] < moves_to_queen:
            return

        if not self.queens.get(colour, False):
            raise NoQueenError(f"No Queen found for {colour}")

    def has_player_lost(self, colour: Colour) -> bool:
        """Player has lost if their queen is surrounded"""
        queen = self.queens.get(colour)
        if queen is not None:
            around_queen = pieces_around_location(self.grid, queen.location)
            if len(around_queen) == 6:
                return True
        return False

    def get_winner(self) -> Optional[Colour]:
        """If there is only one queen left, that player has won"""
        colours_still_in_game = []
        for queen_colour in self.queens:
            if self.has_player_lost(queen_colour) == False:
                colours_still_in_game.append(queen_colour)

        if len(colours_still_in_game) == 1:
            return colours_still_in_game[0]
        return None















