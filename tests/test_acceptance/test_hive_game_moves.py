from hive import pieces
from hive.game import Game, place_piece
from hive.game_types import Piece, WHITE, BLACK


def test_can_run_through_the_first_four_moves():

    game = Game()

    piece = Piece(WHITE, pieces.ANT, 1)
    game = place_piece(game, piece, (0, 0))

    piece = Piece(BLACK, pieces.ANT, 1)
    game = place_piece(game, piece, (2, 0))

    piece = Piece(WHITE, pieces.ANT, 2)
    game = place_piece(game, piece, (-2, 0))

    piece = Piece(BLACK, pieces.ANT, 2)
    game = place_piece(game, piece, (4, 0))






