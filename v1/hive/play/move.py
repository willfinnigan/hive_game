from dataclasses import dataclass

from v1.hive.game.pieces.piece_base_class import Piece
from v1.hive.game.types_and_errors import Location, Colour


@dataclass
class Move():
    piece: Piece
    location: Location
    place: bool

    def __post_init__(self):
        self.original_location = self.piece.location

    def play(self, game):
        if self.place == True:
            game.place_piece(self.piece, self.location)
        else:
            game.move_piece(self.piece, self.location)



@dataclass
class NoMove:
    colour: Colour
    def play(self, game):
        game.player_turns[self.colour] += 1


