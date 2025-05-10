from dataclasses import dataclass
from typing import Optional

from hive.game_engine.game_moves import move_piece, pass_move, place_piece
from hive.game_engine.game_state import Game, Piece, Location, Colour


@dataclass
class Move:
    piece: Piece
    current_location: Optional[Location]
    new_location: Location
    game: Game
    mv_game: Optional[Game] = None


    def play(self) -> Game:
        if self.mv_game is None:
            if self.current_location is None:
                self.mv_game = place_piece(self.game, self.piece, self.new_location)
            else:
                self.mv_game = move_piece(self.game, self.current_location, self.new_location)

        return self.mv_game

    def __repr__(self):
        piece_name = f"{self.piece.colour}_{self.piece.name}_{self.piece.number}"
        if self.current_location is None:
            return f"Place({piece_name} at {self.new_location})"
        return f"Move ({piece_name} from {self.current_location} to {self.new_location}"


@dataclass
class NoMove:
    colour: Colour
    game: Game
    mv_game: Optional[Game] = None

    def play(self) -> Game:
        if self.mv_game is None:
            # Pass the turn
            self.mv_game = pass_move(self.game, self.colour)
        return self.mv_game
    
    def __repr__(self):
        return f"Pass ({self.colour})"


