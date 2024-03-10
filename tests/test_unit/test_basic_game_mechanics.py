import pytest

from hive.game.pieces.queen import Queen
from hive.game.types_and_errors import InvalidPlacementError, InvalidMoveError, WHITE, BLACK
from hive.game.game import Game
from hive.game.pieces.piece_base_class import Piece


def test_can_place_first_piece():
    game = Game()
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (0, 0))
    assert game.grid[(0, 0)] == piece


def test_second_piece_raised_exception_if_not_connected_to_first():
    game = Game()
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (0, 0))

    piece = Piece(colour=BLACK)
    with pytest.raises(InvalidPlacementError):
        game.place_piece(piece, (0, 2))


def test_cannot_place_piece_on_occupied_location():
    game = Game()
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (0, 0))
    piece = Piece(colour=BLACK)

    with pytest.raises(InvalidPlacementError):
        game.place_piece(piece, (0, 0))


def test_can_not_place_piece_touching_opposing_player_except_first_placement():
    game = Game()
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (0, 0))

    piece = Piece(colour=BLACK)
    game.place_piece(piece, (2, 0))

    piece = Piece(colour=WHITE)
    with pytest.raises(InvalidPlacementError):
        game.place_piece(piece, (4, 0))


def test_can_move_a_piece():
    game = Game()
    piece = Queen(colour=WHITE)
    game.place_piece(piece, (0, 0))
    piece = Queen(colour=BLACK)
    game.place_piece(piece, (2, 0))
    piece_to_move = Piece(colour=WHITE)
    game.place_piece(piece_to_move, (-2, 0))
    piece = Piece(colour=BLACK)
    game.place_piece(piece, (4, 0))

    game.move_piece(piece_to_move, (1, 1))

    assert game.grid.get((-2, 0)) is None
    assert game.grid.get((1, 1)) == piece_to_move


def test_can_not_move_a_piece_which_breaks_connection():
    game = Game()
    piece_to_move = Queen(colour=WHITE)
    game.place_piece(piece_to_move, (0, 0))
    piece = Queen(colour=BLACK)
    game.place_piece(piece, (2, 0))
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (-2, 0))
    piece = Piece(colour=BLACK)
    game.place_piece(piece, (4, 0))

    with pytest.raises(InvalidMoveError):
        game.move_piece(piece_to_move, (6, 0))


