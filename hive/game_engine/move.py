from dataclasses import dataclass
from typing import Optional, List

from hive.game_engine.game_moves import move_piece, pass_move, place_piece
from hive.game_engine.game_state import Game, Piece, Location, Colour

from hive.game_engine.errors import NoQueenError
from hive.game_engine.game_moves import check_queen_timely_placement
from hive.game_engine.game_state import Colour, Game
from hive.game_engine.grid_functions import get_placeable_locations
from hive.game_engine.piece_logic import get_possible_moves
from hive.game_engine.pieces import QUEEN

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
    


def get_players_possible_moves_or_placements(colour: Colour, game: Game) -> List[Move]:

    if _must_play_queen(colour, game):
        return _play_queen_placements(colour, game)

    moves = _all_placements(colour, game)  # can always place pieces
    if game.queens.get(colour) is not None:  # but can only move once queen is placed
        moves += _placed_piece_moves(colour, game)
    return moves

def _must_play_queen(colour: Colour, game: Game) -> bool:
    try:
        check_queen_timely_placement(game, colour, moves_to_queen=3)
    except NoQueenError:
        return True
    return False

def _play_queen_placements(colour: Colour, game: Game) -> List[Move]:
    """ If the player must play a queen, return all possible placements """

    unplayed_pieces = game.unplayed_pieces[colour]
    queen = [piece for piece in unplayed_pieces if piece.name == QUEEN][0]
    placeable_locations = get_placeable_locations(game.grid, colour)

    possible_moves = []
    for location in placeable_locations:
        possible_moves.append(Move(queen, None, location, game))
    return possible_moves

def _all_placements(colour: Colour, game: Game) -> List[Move]:
    """ Return all possible placements of pieces """

    # all placeable locations * all pieces
    possible_moves = []
    placeable_locations = get_placeable_locations(game.grid, colour)
    for piece in game.unplayed_pieces[colour]:
        for location in placeable_locations:
            possible_moves.append(Move(piece, None, location, game))
    return possible_moves

def _placed_piece_moves(colour: Colour, game: Game) -> List[Move]:

    moveable_pieces = []
    for loc, stack in game.grid.items():
        if stack and stack[-1].colour == colour:
            piece = stack[-1]
            moveable_pieces.append((piece, loc))

    possible_moves = []
    for piece, current_location in moveable_pieces:
        move_locations = get_possible_moves(game.grid, current_location)
        possible_moves += [Move(piece, current_location, loc, game) for loc in move_locations]
    return possible_moves

