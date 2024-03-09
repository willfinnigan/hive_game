from collections import defaultdict
from hive.board import Board
from hive.custom_types import Location, Colour
from hive.errors import NoQueenError, InvalidMoveError, PlayerHasLost
from hive.grid_functions import pieces_around_location
from hive.piece import Piece, Queen


class Game:

    def __init__(self,
                 board: Board = None):

        self.board = board or Board()
        self.player_turns = defaultdict(int)
        self.queens = {}

    def place_piece(self, piece: Piece, loc: Location):
        if self.has_player_lost(piece.colour):
            raise PlayerHasLost(f"{piece.colour} has lost")

        piece.place(board, location)

        self.board.place_piece(piece, loc, new_piece=True)
        self.player_turns[piece.colour] += 1
        if isinstance(piece, Queen):
            self.queens[piece.colour] = piece

        self._check_queen_exists(piece.colour)

    def move_piece(self, piece: Piece, new_loc: Location):
        if self.has_player_lost(piece.colour):
            raise PlayerHasLost(f"{piece.colour} has lost")

        possible_moves = piece.get_possible_moves(self.board)
        if new_loc not in possible_moves:
            raise InvalidMoveError(f"Invalid move for {piece.__class__} - {new_loc} not in {possible_moves}")

        self.board.remove_piece(piece)
        self.board.place_piece(piece, new_loc, new_piece=False)
        self.player_turns[piece.colour] += 1
        self._check_queen_exists(piece.colour)

    def _check_queen_exists(self, colour: Colour, moves_to_queen=4):
        if self.player_turns[colour] < moves_to_queen:
            return

        if not self.queens.get(colour, False):
            raise NoQueenError(f"No Queen found for {colour}")

    def has_player_lost(self, colour: Colour) -> bool:
        queen = self.queens.get(colour)
        if queen is not None:
            around_queen = pieces_around_location(self.board.grid, queen.location)
            if len(around_queen) == 6:
                return True
        return False

    def has_player_won(self, colour: Colour) -> bool:
        """ Player has won if all queens are surrounded"""
        for queen_colour in self.queens:
            if queen_colour == colour:
                continue
            if self.has_player_lost(queen_colour) == False:
                return False
        return True














