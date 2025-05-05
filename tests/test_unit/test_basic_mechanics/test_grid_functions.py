from hive import pieces
from hive.game import Game
from hive.grid_functions import get_placeable_locations, can_remove_piece, is_piece_connected_to_hive
from hive.render.to_text import game_to_text
from hive.game_types import WHITE, Piece, BLACK


def test_can_identify_all_6_positions_around_a_single_white_piece():
    piece = Piece(WHITE, pieces.ANT, 1)
    grid = {(0, 0): (piece,)}
    locations = get_placeable_locations(grid, WHITE)
    assert len(locations) == 6

def test_can_identify_all_3_positions_around_a_white_piece_plus_a_black_piece():
    grid = {(0, 0):  (Piece(WHITE, pieces.ANT, 1),),
            (1, -1):  (Piece(BLACK, pieces.ANT, 1),)}
    locations = get_placeable_locations(grid, WHITE)
    assert len(locations) == 3

def test_can_remove_piece_next_to_1_other_piece():
    grid = {(0, 0): (Piece(WHITE, pieces.ANT, 1),),
            (2, 0): (Piece(WHITE, pieces.ANT, 2),),
            (1, -1): (Piece(BLACK, pieces.ANT, 2),)}
    assert can_remove_piece(grid, (0, 0)) == True

def test_can_remove_piece_next_to_2_other_piece():
    grid = {(0, 0): (Piece(WHITE, pieces.ANT, 1),),
            (2, 0): (Piece(WHITE, pieces.ANT, 2),),
            (1, 1): (Piece(BLACK, pieces.ANT, 2),)}
    assert can_remove_piece(grid, (0, 0)) == True

def test_cant_remove_piece_in_line():
    grid = {(0, 0): (Piece(WHITE, pieces.ANT, 1),),
            (2, 0): (Piece(WHITE, pieces.ANT, 2),),
            (4, 9): (Piece(BLACK, pieces.ANT, 2),)}
    assert can_remove_piece(grid, (2, 0)) == False

def test_cant_remove_piece_if_disconnects_hive():
    grid = {(6, 2): (Piece(WHITE, pieces.ANT, 1)),
            (5, 1): (Piece(WHITE, pieces.ANT, 2)),
            (4, 2): (Piece(WHITE, pieces.ANT, 3)),
            (3, 1): (Piece(WHITE, pieces.ANT, 4)),
            (8, 2): (Piece(WHITE, pieces.ANT, 5)),
            (10, 2): (Piece(WHITE, pieces.ANT, 6))}

    game = Game(grid=grid)
    print()
    print(game_to_text(game, highlight_piece_at=(6, 2)))

    assert can_remove_piece(grid, (6, 2)) == False


def test_piece_is_connected_to_all_others():
    grid = {(0, 0): (Piece(WHITE, pieces.ANT, 1),),
            (1, -1): (Piece(WHITE, pieces.ANT, 2),),
            (1, 1): (Piece(BLACK, pieces.ANT, 2),),
            (0, 2): (Piece(BLACK, pieces.ANT, 2),)}
    assert is_piece_connected_to_hive(grid, (0, 2)) == True

    grid.pop((1, 1))
    assert is_piece_connected_to_hive(grid, (0, 2)) == False





