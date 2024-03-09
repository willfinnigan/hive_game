import pytest

from hive.custom_types import  WHITE, BLACK
from hive.errors import NoQueenError
from hive.game import Game
from hive.piece import Piece, Queen, Ant, GrassHopper, Spider, Beetle
from hive.render.python_output import game_as_text


def initial_3_move_game() -> Game:
    game = Game()

    # white 1
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (0, 0))

    # black 1
    piece = Piece(colour=BLACK)
    game.place_piece(piece, (2, 0))

    # white 2
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (-2, 0))

    # black 2
    piece = Piece(colour=BLACK)
    game.place_piece(piece, (4, 0))

    # white 3
    piece = Piece(colour=WHITE)
    game.place_piece(piece, (-4, 0))

    # black 3
    piece = Piece(colour=BLACK)
    game.place_piece(piece, (6, 0))

    return game

def initial_5_move_game():
    game = initial_3_move_game()

    # white 4
    white_queen = Queen(colour=WHITE)
    game.place_piece(white_queen, (-6, 0))

    # black 4
    black_queen = Queen(colour=BLACK)
    game.place_piece(black_queen, (8, 0))

    return game


def test_no_queen_on_move_4_raises_exception():
    game = initial_3_move_game()

    piece = Piece(colour=WHITE)
    with pytest.raises(NoQueenError):
        game.place_piece(piece, (-6, 0))


def test_queen_possible_moves_are_1_away():
    game = initial_3_move_game()
    queen = Queen(colour=WHITE)
    game.place_piece(queen, (-6, 0))

    print('\nGame board')
    print(game_as_text(game))

    possible_moves = queen.get_possible_moves(game.board)
    assert possible_moves == [(-5, -1), (-5, 1)]

def test_ant_can_move_anywhere_on_a_line():
    game = initial_5_move_game()

    ant = Ant(colour=WHITE)
    game.place_piece(ant, (-8, 0))

    possible_moves = ant.get_possible_moves(game.board)
    assert len(possible_moves) == 19


def test_grasshoper_can_jump_over_pieces():
    game = initial_5_move_game()

    grass_hopper = GrassHopper(colour=WHITE)
    game.place_piece(grass_hopper, (-8, 0))

    possible_moves = grass_hopper.get_possible_moves(game.board)
    assert possible_moves == [(10, 0)]

def test_spider_moves_3_moves():
    game = initial_5_move_game()

    spider = Spider(colour=WHITE)
    game.place_piece(spider, (-8, 0))

    possible_moves = spider.get_possible_moves(game.board)
    assert possible_moves == [(-3, -1), (-3, 1)]   # 3 moves along the straight line on either side



def test_beetle_can_get_possible_moves_move_1_space_and_on_to_another_piece():
    game = Game()
    queen = Queen(colour=WHITE)
    game.place_piece(queen, (0, 0))

    beetle = Beetle(colour=BLACK)
    game.place_piece(beetle, (2, 0))

    possible_moves = beetle.get_possible_moves(game.board)
    assert possible_moves == [(1, -1), (1, 1), (0, 0)]

def test_beetle_can_move_one_top_of_another_piece():
    game = Game()
    queen = Queen(colour=WHITE)
    game.place_piece(queen, (0, 0))

    ant = Ant(colour=BLACK)
    game.place_piece(ant, (-2, 0))

    beetle = Beetle(colour=WHITE)
    game.place_piece(beetle, (2, 0))

    game.move_piece(beetle, (0, 0))
    assert game.board.get_piece((0,0)) == queen
    assert beetle.location == (0, 0)
    assert beetle.sitting_on == queen
    assert queen.on_top == beetle




