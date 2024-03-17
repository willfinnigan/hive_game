import pytest

from hive.errors import NoQueenError
from hive.game import Game
from hive.piece_logic import get_possible_moves
from hive.types import Piece, WHITE, PieceName, BLACK

def initial_3_move_game() -> Game:
    game = Game()

    # white 1
    piece = Piece(WHITE, PieceName.BLANK, 1)
    game.place_piece(piece, (0, 0))

    # black 1
    piece = Piece(BLACK, PieceName.BLANK, 1)
    game.place_piece(piece, (2, 0))

    # white 2
    piece = Piece(WHITE, PieceName.BLANK, 2)
    game.place_piece(piece, (-2, 0))

    # black 2
    piece = Piece(BLACK, PieceName.BLANK, 2)
    game.place_piece(piece, (4, 0))

    # white 3
    piece = Piece(WHITE, PieceName.BLANK, 3)
    game.place_piece(piece, (-4, 0))

    # black 3
    piece = Piece(BLACK, PieceName.BLANK, 3)
    game.place_piece(piece, (6, 0))

    return game

def initial_5_move_game():
    game = initial_3_move_game()

    # white 4
    white_queen = Piece(WHITE, PieceName.QUEEN, 3)
    game.place_piece(white_queen, (-6, 0))

    # black 4
    black_queen = Piece(BLACK, PieceName.QUEEN, 3)
    game.place_piece(black_queen, (8, 0))

    return game


def test_no_queen_on_move_4_raises_exception():
    game = initial_3_move_game()

    piece = Piece(WHITE, PieceName.BLANK, 4)
    with pytest.raises(NoQueenError):
        game.place_piece(piece, (-6, 0))


def test_queen_possible_moves_are_1_away():
    game = initial_3_move_game()
    queen = Piece(WHITE, PieceName.QUEEN, 3)
    game.place_piece(queen, (-6, 0))

    possible_moves = get_possible_moves(game.grid, (-6, 0))
    assert possible_moves == [(-5, -1), (-5, 1)]

def test_queen_can_not_jump_a_piece():
    queen = Piece(WHITE, PieceName.QUEEN, 3)
    grid = {(6, 2): [queen],
            (8, 2): [Piece(BLACK, PieceName.QUEEN, 3)],
            (9, 3): [Piece(WHITE, PieceName.ANT, 4)],
            (8, 4): [Piece(BLACK, PieceName.ANT, 4)],
            (6, 4): [Piece(WHITE, PieceName.ANT, 5)]}

    moves = get_possible_moves(grid, (6, 2))

    assert (5, 3) not in moves   # this shouldn't be possible because it breaks the connection
    assert len(moves) == 2


def test_ant_can_move_anywhere_on_a_line():
    game = initial_5_move_game()

    ant = Piece(WHITE, PieceName.ANT, 5)
    game.place_piece(ant, (-8, 0))

    possible_moves = get_possible_moves(game.grid, (-8, 0))
    assert len(possible_moves) == 19


def test_grasshoper_can_jump_over_pieces():
    game = initial_5_move_game()

    grass_hopper = Piece(WHITE, PieceName.GRASSHOPPER, 5)
    game.place_piece(grass_hopper, (-8, 0))

    possible_moves = get_possible_moves(game.grid, (-8, 0))
    assert possible_moves == [(10, 0)]

def test_spider_moves_3_moves():
    game = initial_5_move_game()

    spider = Piece(WHITE, PieceName.SPIDER, 5)
    game.place_piece(spider, (-8, 0))

    possible_moves = get_possible_moves(game.grid, (-8, 0))
    assert possible_moves == [(-3, -1), (-3, 1)]   # 3 moves along the straight line on either side


def test_beetle_can_get_possible_moves_move_1_space_and_on_to_another_piece():
    game = Game()
    queen = Piece(WHITE, PieceName.QUEEN, 1)
    game.place_piece(queen, (0, 0))

    beetle = Piece(WHITE, PieceName.BEETLE, 2)
    game.place_piece(beetle, (2, 0))

    possible_moves = get_possible_moves(game.grid, (2, 0))
    assert possible_moves == [(1, -1), (1, 1), (0, 0)]

def test_beetle_can_move_one_top_of_another_piece():
    game = Game()
    queen = Piece(WHITE, PieceName.QUEEN, 1)
    game.place_piece(queen, (0, 0))

    ant = Piece(BLACK, PieceName.ANT, 1)
    game.place_piece(ant, (-2, 0))

    beetle = Piece(WHITE, PieceName.BEETLE, 2)
    game.place_piece(beetle, (2, 0))

    game.move_piece((2, 0), (0, 0))
    assert game.grid.get((0,0)) == [queen, beetle]

    game.move_piece((0, 0), (2, 0))
    assert game.grid.get((0, 0)) == [queen]
    assert game.grid.get((2, 0)) == [beetle]


def test_ant_can_move_around_2_piece():
    grid = {(0, 0): [Piece(WHITE, PieceName.ANT, 1)],
            (1, -1): [Piece(BLACK, PieceName.ANT, 1)],
            (2, 0): [Piece(WHITE, PieceName.ANT, 2)]}

    possible_moves = get_possible_moves(grid, (0, 0))
    assert len(possible_moves) == 7
