from hive.game.grid_functions import get_placeable_locations, can_remove_piece, is_piece_connected_to_hive
from hive.game.pieces.ant import Ant
from hive.game.types_and_errors import WHITE, BLACK

def test_can_identify_all_6_positions_around_a_single_white_piece():
    grid = {(0, 0): Ant(WHITE)}
    locations = get_placeable_locations(grid, WHITE)
    assert len(locations) == 6

def test_can_identify_all_3_positions_around_a_white_piece_plus_a_black_piece():
    grid = {(0, 0): Ant(WHITE), (1, -1): Ant(BLACK)}
    locations = get_placeable_locations(grid, WHITE)
    assert len(locations) == 3

def test_can_remove_piece_next_to_1_other_piece():
    ant1 = Ant(WHITE)
    ant2 = Ant(BLACK)
    ant1.location = (0, 0)
    ant2.location = (1, -1)
    grid = {(0, 0): ant1, (1, -1): ant2}
    assert can_remove_piece(grid, ant1) == True

def test_can_remove_piece_next_to_2_other_piece():
    ant1 = Ant(WHITE)
    ant2 = Ant(BLACK)
    ant3 = Ant(WHITE)
    ant1.location = (0, 0)
    ant2.location = (1, 1)
    ant3.location = (2, 0)
    grid = {(0, 0): ant1, (1, -1): ant2, (2, 0): ant3}
    assert can_remove_piece(grid, ant3) == True

def test_cant_remove_piece_in_line():
    ant1 = Ant(WHITE)
    ant2 = Ant(BLACK)
    ant3 = Ant(WHITE)
    ant1.location = (0, 0)
    ant2.location = (2, 0)
    ant3.location = (4, 0)
    grid = {(0, 0): ant1, (2, 0): ant2, (4, 0): ant3}
    assert can_remove_piece(grid, ant2) == False

def test_piece_is_connected_to_all_others():
    ant1 = Ant(WHITE)
    ant2 = Ant(BLACK)
    ant3 = Ant(WHITE)
    ant4 = Ant(BLACK)
    ant1.location = (0, 0)
    ant2.location = (1, -1)
    ant3.location = (1, 1)
    ant4.location = (0, 2)
    grid = {(0, 0): ant1, (1, -1): ant2, (1, 1): ant3, (0, 2): ant4}
    assert is_piece_connected_to_hive(grid, ant4) == True

    ant3.location = None
    grid.pop((1, 1))
    assert is_piece_connected_to_hive(grid, ant4) == False





