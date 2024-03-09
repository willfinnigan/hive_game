from hive.custom_types import WHITE, BLACK
from hive.game import Game
from hive.piece import Piece
from hive.render.python_output import game_as_text


def test_can_run_through_the_first_four_moves():

    game = Game()

    piece = Piece(colour=BLACK)
    game.place_piece(piece, (0, 0))

    piece = Piece(colour=WHITE)
    game.place_piece(piece, (2, 0))

    piece = Piece(colour=BLACK)
    game.place_piece(piece, (-2, 0))

    piece = Piece(colour=WHITE)
    game.place_piece(piece, (4, 0))

    game_text = game_as_text(game)
    print()
    print(game_text)






