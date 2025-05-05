from dataclasses import dataclass
from typing import Optional

from hive.game_engine.game_moves import move_piece, pass_move, place_piece
from hive.game_engine.game_state import Game, Piece, Location, Colour


@dataclass
class Move:
    piece: Piece
    current_location: Optional[Location]
    new_location: Location

    def play(self, game: Game) -> Game:
        if self.current_location is None:
            game = place_piece(game, self.piece, self.new_location)
        else:
            game = move_piece(game, self.current_location, self.new_location)
        return game

    def __repr__(self):
        piece_name = f"{self.piece.colour}_{self.piece.name}_{self.piece.number}"
        if self.current_location is None:
            return f"Place({piece_name} at {self.new_location})"
        return f"Move ({piece_name} from {self.current_location} to {self.new_location}"


@dataclass
class NoMove:
    colour: Colour

    def play(self, game: Game) -> Game:
        game = pass_move(game, self.colour)
        return game
    
    def __repr__(self):
        return f"Pass ({self.colour})"


