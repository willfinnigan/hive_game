from collections import defaultdict
from typing import Dict, Optional

from hive.check_moves import check_is_valid_location, check_is_valid_placement, check_is_valid_move
from hive.errors import NoQueenError
from hive.grid_functions import pieces_around_location
from hive.game_types import Grid, WHITE, BLACK, Piece, Location, Colour, PieceName


class Game:

    def __init__(self, grid: Grid = None):
        self.grid: Grid = grid or defaultdict(list)
        self.player_turns: Dict[Colour, int] = {WHITE: 0, BLACK: 0}
        self.queens: Dict[Colour, Location] = {}

    def place_piece(self, piece: Piece, location: Location):
        check_is_valid_location(location)
        check_is_valid_placement(self.grid, location, piece.colour)
        self.grid[location].append(piece)

        if piece.name == PieceName.QUEEN:
            self.queens[piece.colour] = location

        self.player_turns[piece.colour] += 1
        self.check_queen_exists(piece.colour)

    def move_piece(self, current_location: Location, location: Location):
        check_is_valid_location(location)
        check_is_valid_move(self.grid, current_location, location)

        # remove piece from current location
        piece = self.grid[current_location].pop(-1)
        if len(self.grid[current_location]) == 0:
            del self.grid[current_location]

        # place piece in new location
        self.grid[location].append(piece)

        if piece.name == PieceName.QUEEN:  # keep queen location updated
            self.queens[piece.colour] = location

        self.player_turns[piece.colour] += 1
        self.check_queen_exists(piece.colour)

    def check_queen_exists(self, colour: Colour, moves_to_queen=4):
        if self.player_turns[colour] < moves_to_queen:
            return

        if not self.queens.get(colour, False):
            raise NoQueenError(f"No Queen found for {colour}")

    def has_player_lost(self, colour: Colour) -> bool:
        """Player has lost if their queen is surrounded"""
        queen_location = self.queens.get(colour)
        if queen_location is not None:
            around_queen = pieces_around_location(self.grid, queen_location)
            if len(around_queen) == 6:
                return True
        return False

    def get_winner(self) -> Optional[Colour]:
        """If there is only one queen left, that player has won"""
        colours_still_in_game = []
        for colour in self.player_turns.keys():
            if self.has_player_lost(colour) == False:
                colours_still_in_game.append(colour)

        if len(colours_still_in_game) == 1:
            return colours_still_in_game[0]
        return None
