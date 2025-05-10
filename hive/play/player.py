from __future__ import annotations
from typing import List

from hive.game_engine.errors import NoQueenError
from hive.game_engine.grid_functions import get_placeable_locations
from hive.game_engine.piece_logic import get_possible_moves
from hive.play.move import Move
from hive.game_engine.game_state import Colour, Piece, Game
from hive.game_engine.game_moves import check_queen_timely_placement
from hive.game_engine import pieces


class Player():
    def __init__(self, colour: Colour):
        self.colour = colour
        self.pieces = [Piece(colour, pieces.QUEEN, 1),
                       Piece(colour, pieces.ANT, 1),
                       Piece(colour, pieces.ANT, 2),
                       Piece(colour, pieces.ANT, 3),
                       Piece(colour, pieces.BEETLE, 1),
                       Piece(colour, pieces.BEETLE, 2),
                       Piece(colour, pieces.GRASSHOPPER, 1),
                       Piece(colour, pieces.GRASSHOPPER, 2),
                       Piece(colour, pieces.GRASSHOPPER, 3),
                       Piece(colour, pieces.SPIDER, 1),
                       Piece(colour, pieces.SPIDER, 2)]

    def get_move(self, game) -> Move:
        """ AI or Human selection of move """
        pass

    def possible_moves(self, game: Game) -> List[Move]:

        if self._must_play_queen(game):
            return self._queen_placements(game)

        moves = self._all_placements(game)  # can always place pieces
        if game.queens.get(self.colour) is not None:  # but can only move once queen is placed
            moves += self._placed_piece_moves(game)
        return moves

    def _must_play_queen(self, game: Game) -> bool:
        try:
            check_queen_timely_placement(game, self.colour, moves_to_queen=3)
        except NoQueenError:
            return True
        return False

    def _queen_placements(self, game: Game) -> List[Move]:
        possible_moves = []
        queen = [piece for piece in self.pieces if piece.name == pieces.QUEEN][0]
        placeable_locations = get_placeable_locations(game.grid, self.colour)
        for location in placeable_locations:
            possible_moves.append(Move(queen, None, location, game))
        return possible_moves

    def _all_placements(self, game: Game) -> List[Move]:

        # all placeable locations * all pieces
        possible_moves = []
        placeable_locations = get_placeable_locations(game.grid, self.colour)

        placed_pieces = []
        for loc, stack in game.grid.items():
            for piece in stack:
                if piece.colour == self.colour:
                    placed_pieces.append(piece)

        unplaced_pieces = [piece for piece in self.pieces if piece not in placed_pieces]

        for piece in unplaced_pieces:
            for location in placeable_locations:
                possible_moves.append(Move(piece, None, location, game))

        return possible_moves

    def _placed_piece_moves(self, game: Game) -> List[Move]:

        moveable_pieces = []
        for loc, stack in game.grid.items():
            if stack and stack[-1].colour == self.colour:
                piece = stack[-1]
                moveable_pieces.append((piece, loc))

        possible_moves = []
        for piece, current_location in moveable_pieces:
            move_locations = get_possible_moves(game.grid, current_location)
            possible_moves += [Move(piece, current_location, loc, game) for loc in move_locations]
        return possible_moves
