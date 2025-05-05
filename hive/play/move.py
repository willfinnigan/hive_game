from dataclasses import dataclass
from typing import Optional

from hive.game_types import Piece, Location, Colour


@dataclass
class Move:
    piece: Piece
    current_location: Optional[Location]
    new_location: Location

    def play(self, game):
        if self.current_location is None:
            game.place_piece(self.piece, self.new_location)
        else:
            game.move_piece(self.current_location, self.new_location)

    def __repr__(self):
        piece_name = f"{self.piece.colour}_{self.piece.name.name}_{self.piece.number}"
        if self.current_location is None:
            return f"Place({piece_name} at {self.new_location})"
        return f"Move({piece_name} from {self.current_location} to {self.new_location}"



@dataclass
class NoMove:
    colour: Colour
    def play(self, game):
        game.player_turns[self.colour] += 1


