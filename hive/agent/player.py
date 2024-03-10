from __future__ import annotations
import random
from typing import List, Optional

from hive.agent.move import Move, NoMove
from hive.game.grid_functions import get_placeable_locations, is_piece_connected_to_hive, all_connected
from hive.game.pieces.ant import Ant
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.queen import Queen
from hive.game.pieces.spider import Spider
from hive.game.types_and_errors import Colour, NoQueenError
from hive.render.python_output import game_as_text


class Player():
    def __init__(self, colour: Colour):
        self.colour = colour
        self.pieces = [Queen(colour),
                       Spider(colour), Spider(colour),
                       Beetle(colour), Beetle(colour),
                       GrassHopper(colour), GrassHopper(colour), GrassHopper(colour),
                       Ant(colour), Ant(colour), Ant(colour)]

    def get_move(self, game) -> Move:
        """ AI or Human selection of move """
        pass

    def possible_moves(self, game) -> List[Move]:

        if self._must_play_queen(game):
            return self._queen_placements(game)

        moves = self._all_placements(game)  # can always place pieces
        if game.queens.get(self.colour) is not None:  # but can only move once queen is placed
            moves += self._placed_piece_moves(game)
        return moves

    def _must_play_queen(self, game) -> bool:
        try:
            game.check_queen_exists(self.colour, moves_to_queen=3)
        except NoQueenError:
            return True
        return False

    def _queen_placements(self, game) -> List[Move]:
        possible_moves = []
        queen = [piece for piece in self.pieces if isinstance(piece, Queen)][0]
        placeable_locations = get_placeable_locations(game.grid, self.colour)
        for location in placeable_locations:
            possible_moves.append(Move(queen, location, True))
        return possible_moves

    def _all_placements(self, game) -> List[Move]:

        # all placeable locations * all pieces
        possible_moves = []
        placeable_locations = get_placeable_locations(game.grid, self.colour)
        unplaced_pieces = [piece for piece in self.pieces if piece.location is None]
        for piece in unplaced_pieces:
            for location in placeable_locations:
                possible_moves.append(Move(piece, location, True))

        return possible_moves

    def _placed_piece_moves(self, game) -> List[Move]:
        # placed_pieces where can they move
        possible_moves = []
        placed_pieces = [piece for piece in self.pieces if piece.location is not None]
        for piece in placed_pieces:
            locations = piece.get_possible_moves(game.grid)
            possible_moves += [Move(piece, location, False) for location in locations]
        return possible_moves

class RandomAI(Player):
    """ Plays moves completely randomly """
    def get_move(self, game) -> Move:
        possible_moves = self.possible_moves(game)
        if len(possible_moves) == 0:
            return NoMove(self.colour)

        return random.choice(possible_moves)









