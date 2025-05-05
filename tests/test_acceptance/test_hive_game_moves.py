from hive.game import Game
from hive.game_types import Piece, WHITE, BLACK


def test_can_run_through_the_first_four_moves():

    game = Game()

    piece = Piece(WHITE, 'ANT', 1)
    game.place_piece(piece, (0, 0))

    piece = Piece(BLACK, 'ANT', 1)
    game.place_piece(piece, (2, 0))

    piece = Piece(WHITE, 'ANT', 2)
    game.place_piece(piece, (-2, 0))

    piece = Piece(BLACK, 'ANT', 2)
    game.place_piece(piece, (4, 0))






