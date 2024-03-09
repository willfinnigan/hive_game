import pytest
from hive.board import Board, InvalidPlacementError, BreaksConnectionError
from hive.custom_types import WHITE, BLACK
from hive.errors import InvalidMoveError
from hive.game import Game
from hive.piece import Piece
from hive.render.python_output import game_as_text


def test_can_place_first_piece():
    board = Board()
    game = Game(board=board)
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (0, 0))
    assert board.grid[(0, 0)] == piece


def test_second_piece_raised_exception_if_not_connected_to_first():
    board = Board()
    game = Game(board=board)
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (0, 0))

    piece = Piece(colour=BLACK)
    with pytest.raises(InvalidPlacementError):
        game.place_piece(piece, (0, 2))


def test_cannot_place_piece_on_occupied_location():
    board = Board()
    game = Game(board=board)
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (0, 0))
    piece = Piece(colour=BLACK)

    with pytest.raises(InvalidPlacementError):
        game.place_piece(piece, (0, 0))


def test_can_not_place_piece_touching_opposing_player_except_first_placement():
    board = Board()
    game = Game(board=board)
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (0, 0))

    piece = Piece(colour=BLACK)
    game.place_piece(piece, (2, 0))

    piece = Piece(colour=WHITE)
    with pytest.raises(InvalidPlacementError):
        game.place_piece(piece, (4, 0))


def test_can_move_a_piece():
    board = Board()
    game = Game(board=board)
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (0, 0))
    piece = Piece(colour=BLACK)
    game.place_piece(piece, (2, 0))
    piece_to_move = Piece(colour=WHITE)
    game.place_piece(piece_to_move, (-2, 0))
    piece = Piece(colour=BLACK)
    game.place_piece(piece, (4, 0))

    game.move_piece(piece_to_move, (1, 1))

    assert board.grid.get((-2, 0)) is None
    assert board.grid.get((1, 1)) == piece_to_move


def test_can_not_move_a_piece_which_breaks_connection():
    board = Board()
    game = Game(board=board)
    piece_to_move = Piece(colour=WHITE)
    game.place_piece(piece_to_move, (0, 0))
    piece = Piece(colour=BLACK)
    game.place_piece(piece, (2, 0))
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (-2, 0))
    piece = Piece(colour=BLACK)
    game.place_piece(piece, (4, 0))

    with pytest.raises(InvalidMoveError):
        game.move_piece(piece_to_move, (6, 0))

def test_player_can_lose():
    game = Game()


