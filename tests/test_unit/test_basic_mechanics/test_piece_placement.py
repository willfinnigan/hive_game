import pytest

from hive.game_engine import pieces
from hive.game_engine.game_moves import place_piece
from hive.game_engine.game_state import Piece, WHITE, BLACK
from tests.test_acceptance.test_hive_game_moves import Game
from hive.game_engine.errors import InvalidPlacementError


def test_can_place_first_piece():
    game = Game()
    piece = Piece(WHITE, pieces.ANT, 1)
    new_game = place_piece(game, piece, (0, 0))
    assert new_game.grid[(0, 0)] == (piece,)

def test_second_piece_raised_exception_if_not_connected_to_first():
    game = Game()
    piece = Piece(WHITE, pieces.ANT, 1)
    new_game = place_piece(game, piece, (0, 0))

    piece = Piece(BLACK, pieces.ANT, 1)
    with pytest.raises(InvalidPlacementError):
        place_piece(new_game, piece, (0, 2))


def test_cannot_place_piece_on_occupied_location():
    game = Game()
    piece = Piece(BLACK, pieces.ANT, 1)
    game = place_piece(game, piece, (0, 0))

    piece = Piece(BLACK, pieces.ANT, 1)
    with pytest.raises(InvalidPlacementError):
        game = place_piece(game, piece, (0, 0))


def test_can_not_place_piece_touching_opposing_player_except_first_placement():
    game = Game()
    piece = Piece(WHITE, pieces.ANT, 1)
    game = place_piece(game, piece, (0, 0))

    piece = Piece(BLACK, pieces.ANT, 1)
    game = place_piece(game, piece, (2, 0))

    piece = Piece(WHITE, pieces.ANT, 2)
    with pytest.raises(InvalidPlacementError):
        game = place_piece(game, piece, (4, 0))