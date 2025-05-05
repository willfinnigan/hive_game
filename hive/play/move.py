from copy import deepcopy
from dataclasses import dataclass

from hive.game.game import Game
from hive.game.pieces.piece_base_class import Piece
from hive.game.types_and_errors import Location, Colour


@dataclass
class Move():
    piece: Piece
    location: Location
    place: bool

    def __post_init__(self):
        self.original_location = self.piece.location

    def play(self, game: Game):
        if self.place == True:
            game.place_piece(self.piece, self.location)
        else:
            game.move_piece(self.piece, self.location)

    def undo(self, game: Game):
        """Undo the move by moving the piece back to its original location"""
        if self.place == True:
            game.remove_piece(self.piece)
        else:
            game.move_piece(self.piece, self.original_location)

    def play_branch(self, game: Game) -> Game:
        """Returns a new game with the move applied"""
        new_game = deepcopy(game)
        new_game.player_turns[self.colour] += 1
        if self.place == True:
            new_game.place_piece(self.piece, self.location)
        else:
            new_game.move_piece(self.piece, self.location)
        return new_game



@dataclass
class NoMove:
    colour: Colour

    # need to manually update the player_turns as no piece is moved or placed

    def play(self, game):
        game.player_turns[self.colour] += 1

    def undo(self, game):
        game.player_turns[self.colour] -= 1

    def play_branch(self, game: Game) -> Game:
        new_game = deepcopy(game)
        new_game.player_turns[self.colour] += 1
        return new_game


