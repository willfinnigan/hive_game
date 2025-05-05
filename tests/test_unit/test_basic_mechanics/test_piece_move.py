import pytest

from hive.errors import InvalidMoveError, BreaksConnectionError
from hive.render.to_text import game_to_text
from hive.game_types import Piece, WHITE, BLACK, PieceName
from tests.test_acceptance.test_hive_game_moves import Game

def test_can_move_a_piece():
    game = Game()
    piece = Piece(WHITE, PieceName.ANT, 1)
    game.place_piece(piece, (0, 0))
    piece = Piece(BLACK, PieceName.ANT, 1)
    game.place_piece(piece, (2, 0))

    piece_to_move = Piece(WHITE, PieceName.ANT, 2)
    game.place_piece(piece_to_move, (-2, 0))
    piece = Piece(BLACK, PieceName.ANT, 2)
    game.place_piece(piece, (4, 0))

    print()
    print(game_to_text(game, highlight_piece_at=(-2, 0)))

    game.move_piece((-2, 0), (1, 1))

    assert game.grid.get((-2, 0), []) == []
    assert game.grid.get((1, 1)) == [piece_to_move]


def test_can_not_move_a_piece_which_breaks_connection():
    game = Game()
    piece_to_move = Piece(WHITE, 'QUEEN', 1)
    game.place_piece(piece_to_move, (0, 0))
    piece = Piece(BLACK, 'QUEEN', 1)
    game.place_piece(piece, (2, 0))
    piece = Piece(WHITE, 'ANT', 1)
    game.place_piece(piece, (-2, 0))
    piece = Piece(BLACK, 'ANT', 1)
    game.place_piece(piece, (4, 0))

    with pytest.raises(BreaksConnectionError):
        game.move_piece((0, 0), (6, 0))