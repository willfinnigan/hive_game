import pytest

from hive.game.types_and_errors import NoQueenError, WHITE, BLACK
from hive.game.game import Game
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.spider import Spider
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.ant import Ant
from hive.game.pieces.queen import Queen
from hive.game.pieces.piece_base_class import Piece
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
    print(game.grid)
    possible_moves = queen.get_possible_moves(game.grid)
    assert possible_moves == [(-5, -1), (-5, 1)]

def test_queen_can_not_jump_a_piece():
    queen = Queen(WHITE)
    grid = {(6, 2): queen, (8, 2): Queen(BLACK),
            (9, 3): Ant(WHITE), (8, 4): Ant(BLACK),
            (6, 4): Ant(WHITE)}

    # update piece positions from the grid
    for loc, piece in grid.items():
        piece.location = loc

    moves = queen.get_possible_moves(grid)

    print(moves)

    # this shouldn't be possible because it breaks the connection
    assert (5, 3) not in moves
    assert len(moves) == 2





def test_ant_can_move_anywhere_on_a_line():
    game = initial_5_move_game()

    ant = Ant(colour=WHITE)
    game.place_piece(ant, (-8, 0))

    possible_moves = ant.get_possible_moves(game.grid)
    assert len(possible_moves) == 19


def test_grasshoper_can_jump_over_pieces():
    game = initial_5_move_game()

    grass_hopper = GrassHopper(colour=WHITE)
    game.place_piece(grass_hopper, (-8, 0))

    possible_moves = grass_hopper.get_possible_moves(game.grid)
    assert possible_moves == [(10, 0)]

def test_spider_moves_3_moves():
    game = initial_5_move_game()

    spider = Spider(colour=WHITE)
    game.place_piece(spider, (-8, 0))

    possible_moves = spider.get_possible_moves(game.grid)
    assert possible_moves == [(-3, -1), (-3, 1)]   # 3 moves along the straight line on either side



def test_beetle_can_get_possible_moves_move_1_space_and_on_to_another_piece():
    game = Game()
    queen = Queen(colour=WHITE)
    game.place_piece(queen, (0, 0))

    beetle = Beetle(colour=BLACK)
    game.place_piece(beetle, (2, 0))

    possible_moves = beetle.get_possible_moves(game.grid)
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
    assert game.grid.get((0,0)) == queen
    assert beetle.location == (0, 0)
    assert beetle.sitting_on == queen
    assert queen.on_top == beetle

def test_ant_can_move_around_2_piece():
    ant1 = Ant(WHITE)
    ant2 = Ant(BLACK)
    ant3 = Ant(WHITE)
    ant1.location = (0, 0)
    ant2.location = (1, 1)
    ant3.location = (2, 0)
    grid = {(0, 0): ant1, (1, -1): ant2, (2, 0): ant3}
    assert len(ant1.get_possible_moves(grid)) == 7
