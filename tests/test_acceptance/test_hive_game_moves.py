from hive.game.types_and_errors import WHITE, BLACK
from hive.game.game import Game
from hive.game.pieces.piece_base_class import Piece
from hive.render.text.python_output import game_as_text


def test_can_run_through_the_first_four_moves():

    game = Game()

    piece = Piece(colour=BLACK, number=1)
    game.place_piece(piece, (0, 0))

    piece = Piece(colour=WHITE, number=1)
    game.place_piece(piece, (2, 0))

    piece = Piece(colour=BLACK, number=1)
    game.place_piece(piece, (-2, 0))

    piece = Piece(colour=WHITE, number=1)
    game.place_piece(piece, (4, 0))

    game_text = game_as_text(game)
    print()
    print(game_text)






