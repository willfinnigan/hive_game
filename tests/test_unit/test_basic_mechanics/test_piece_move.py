import pytest

from hive import pieces
from hive.errors import InvalidMoveError, BreaksConnectionError
from hive.game import place_piece, move_piece
from hive.render.to_text import game_to_text
from hive.game_types import Piece, WHITE, BLACK
from tests.test_acceptance.test_hive_game_moves import Game

def test_can_move_a_piece():
    game = Game()
    piece = Piece(WHITE, pieces.ANT, 1)
    game = place_piece(game, piece, (0, 0))
    piece = Piece(BLACK, pieces.ANT, 1)
    game = place_piece(game, piece, (2, 0))

    piece_to_move = Piece(WHITE, pieces.ANT, 2)
    game = place_piece(game, piece_to_move, (-2, 0))
    piece = Piece(BLACK, pieces.ANT, 2)
    game = place_piece(game, piece, (4, 0))

    print()
    print(game_to_text(game, highlight_piece_at=(-2, 0)))

    new_game = move_piece(game, (-2, 0), (1, 1))

    assert new_game.grid.get((-2, 0), ()) == ()
    assert new_game.grid.get((1, 1)) == (piece_to_move,)


def test_can_not_move_a_piece_which_breaks_connection():
    game = Game()
    piece_to_move = Piece(WHITE, pieces.QUEEN, 1)
    game = place_piece(game, piece_to_move, (0, 0))
    piece = Piece(BLACK, pieces.QUEEN, 1)
    game = place_piece(game, piece, (2, 0))
    piece = Piece(WHITE, pieces.ANT, 1)
    game = place_piece(game, piece, (-2, 0))
    piece = Piece(BLACK, pieces.ANT, 1)
    game = place_piece(game, piece, (4, 0))

    with pytest.raises(BreaksConnectionError):
        move_piece(game, (0, 0), (6, 0))