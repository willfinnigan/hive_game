

import pytest
from hive.game_engine import pieces
from hive.game_engine.errors import NoQueenError
from hive.game_engine.game_functions import move_piece, place_piece
from hive.game_engine.game_state import BLACK, WHITE, Game, Piece, create_immutable_grid, initial_game
from hive.game_engine.moves import get_possible_moves


def initial_3_move_game() -> Game:
    game = initial_game()

    # white 1
    piece = Piece(WHITE, pieces.BLANK, 1)
    game = place_piece(game, piece, (0, 0))

    # black 1
    piece = Piece(BLACK, pieces.BLANK, 1)
    game = place_piece(game, piece, (2, 0))

    # white 2
    piece = Piece(WHITE, pieces.BLANK, 2)
    game = place_piece(game, piece, (-2, 0))

    # black 2
    piece = Piece(BLACK, pieces.BLANK, 2)
    game = place_piece(game, piece, (4, 0))

    # white 3
    piece = Piece(WHITE, pieces.BLANK, 3)
    game = place_piece(game, piece, (-4, 0))

    # black 3
    piece = Piece(BLACK, pieces.BLANK, 3)
    game = place_piece(game, piece, (6, 0))

    return game

def initial_5_move_game():
    game = initial_3_move_game()

    # white 4
    white_queen = Piece(WHITE, pieces.QUEEN, 3)
    game = place_piece(game, white_queen, (-6, 0))

    # black 4
    black_queen = Piece(BLACK, pieces.QUEEN, 3)
    game = place_piece(game, black_queen, (8, 0))

    return game


def test_no_queen_on_move_4_raises_exception():
    game = initial_3_move_game()

    piece = Piece(WHITE, pieces.BLANK, 4)
    with pytest.raises(NoQueenError):
        game = place_piece(game, piece, (-6, 0))


def test_queen_possible_moves_are_1_away():
    game = initial_3_move_game()
    queen = Piece(WHITE, pieces.QUEEN, 3)
    game = place_piece(game, queen, (-6, 0))

    possible_moves = get_possible_moves(game.grid, (-6, 0), 0)
    move_locations = [move.new_location for move in possible_moves]
    print(possible_moves)

    assert move_locations == [(-5, -1), (-5, 1)]

def test_queen_can_not_jump_a_piece():
    queen = Piece(WHITE, pieces.QUEEN, 3)

    grid = {(6, 2): (queen,),
            (8, 2): (Piece(BLACK, pieces.QUEEN, 3),),
            (9, 3): (Piece(WHITE, pieces.ANT, 4),),
            (8, 4): (Piece(BLACK, pieces.ANT, 4),),
            (6, 4): (Piece(WHITE, pieces.ANT, 5),)}
    
    grid = create_immutable_grid(grid)

    moves = get_possible_moves(grid, (6, 2), 0)

    assert (5, 3) not in moves   # this shouldn't be possible because it breaks the connection
    assert len(moves) == 2


def test_ant_can_move_anywhere_on_a_line():
    game = initial_5_move_game()

    ant = Piece(WHITE, pieces.ANT, 5)
    game = place_piece(game, ant, (-8, 0))

    possible_moves = get_possible_moves(game.grid, (-8, 0), 0)
    assert len(possible_moves) == 19


def test_grasshoper_can_jump_over_pieces():
    game = initial_5_move_game()

    grass_hopper = Piece(WHITE, pieces.GRASSHOPPER, 5)
    game = place_piece(game, grass_hopper, (-8, 0))

    possible_moves = get_possible_moves(game.grid, (-8, 0), 0)

    move_locations = [move.new_location for move in possible_moves]
    print(possible_moves)

    assert move_locations == [(10, 0)]

def test_spider_moves_3_moves():
    game = initial_5_move_game()

    spider = Piece(WHITE, pieces.SPIDER, 5)
    game = place_piece(game, spider, (-8, 0))

    possible_moves = get_possible_moves(game.grid, (-8, 0), 0)

    move_locations = [move.new_location for move in possible_moves]

    assert move_locations == [(-3, -1), (-3, 1)]   # 3 moves along the straight line on either side


def test_beetle_can_get_possible_moves_move_1_space_and_on_to_another_piece():
    game = Game()
    queen = Piece(WHITE, pieces.QUEEN, 1)
    game = place_piece(game, queen, (0, 0))

    beetle = Piece(WHITE, pieces.BEETLE, 2)
    game = place_piece(game, beetle, (2, 0))

    possible_moves = get_possible_moves(game.grid, (2, 0), 0)

    move_locations = [move.new_location for move in possible_moves]

    assert move_locations == [(1, -1), (1, 1), (0, 0)]

def test_beetle_can_move_one_top_of_another_piece():
    game = Game()
    queen = Piece(WHITE, pieces.QUEEN, 1)
    game = place_piece(game, queen, (0, 0))

    ant = Piece(BLACK, pieces.ANT, 1)
    game = place_piece(game, ant, (-2, 0))

    beetle = Piece(WHITE, pieces.BEETLE, 2)
    game = place_piece(game, beetle, (2, 0))

    new_game = move_piece(game, (2, 0), (0, 0), colour=WHITE)
    assert new_game.grid.get((0,0)) == (queen, beetle)

    new_game2 = move_piece(new_game, (0, 0), (2, 0), colour=WHITE)
    assert new_game2.grid.get((0, 0)) == (queen,)
    assert new_game2.grid.get((2, 0)) == (beetle,)


def test_ant_can_move_around_2_piece():
    grid = {(0, 0): (Piece(WHITE, pieces.ANT, 1),),
            (1, -1): (Piece(BLACK, pieces.ANT, 1),),
            (2, 0): (Piece(WHITE, pieces.ANT, 2),)}
    
    grid = create_immutable_grid(grid)

    possible_moves = get_possible_moves(grid, (0, 0), 0)
    assert len(possible_moves) == 7

