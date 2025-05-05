from __future__ import annotations
from typing import List

from hive.game.grid_functions import get_placeable_locations
from hive.game.pieces.ant import Ant
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.queen import Queen
from hive.game.pieces.spider import Spider
from hive.game.types_and_errors import Colour, NoQueenError
from hive.play.move import Move


class Player():
    def __init__(self, colour: Colour):
        self.colour = colour
        self.pieces = [Queen(colour, 1),
                       Spider(colour, 1), Spider(colour, 2),
                       Beetle(colour, 1), Beetle(colour, 2),
                       GrassHopper(colour, 1), GrassHopper(colour, 2), GrassHopper(colour, 3),
                       Ant(colour, 1), Ant(colour, 2), Ant(colour, 3)]

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



if __name__ == "__main__":
    from hive.game.types_and_errors import Grid, Colour, WHITE
    from hive.play.move import Move
    from hive.game.game import Game

    game = Game()
    player = Player(WHITE)
    move = Move(player.pieces[0], (0, 0), True)
    move.play(game)
    print(game.grid)





