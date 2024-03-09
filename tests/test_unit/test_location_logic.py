from copy import deepcopy
from hive.grid_functions import one_move_away

grid = {(4, 2): True,
        (5, 1): True,
        (7, 1): True,
        (8, 2): True,
        (7, 3): True,
        (5, 3): True}

def test_freedom_of_movement_4_2():
    pos = (4, 2)
    tmp_grid = deepcopy(grid)
    tmp_grid.pop(pos)
    positions = one_move_away(tmp_grid, pos)
    assert (6,2) not in positions
    assert len(positions) == 2  # should only be two places to move

def test_freedom_of_movement_5_1():
    pos = (5, 1)
    tmp_grid = deepcopy(grid)
    tmp_grid.pop(pos)
    positions = one_move_away(tmp_grid, pos)
    assert (6, 2) not in positions
    assert len(positions) == 2  # should only be two places to move

def test_freedom_of_movement_7_1():
    pos = (7, 1)
    tmp_grid = deepcopy(grid)
    tmp_grid.pop(pos)
    positions = one_move_away(tmp_grid, pos)
    assert (6, 2) not in positions
    assert len(positions) == 2  # should only be two places to move

def test_freedom_of_movement_8_2():
    pos = (8, 2)
    tmp_grid = deepcopy(grid)
    tmp_grid.pop(pos)
    positions = one_move_away(tmp_grid, pos)
    assert (6, 2) not in positions
    assert len(positions) == 2  # should only be two places to move

def test_freedom_of_movement_7_3():
    pos = (7, 3)
    tmp_grid = deepcopy(grid)
    tmp_grid.pop(pos)
    positions = one_move_away(tmp_grid, pos)
    assert (6, 2) not in positions
    assert len(positions) == 2  # should only be two places to move

def test_freedom_of_movement_5_3():
    pos = (5, 3)
    tmp_grid = deepcopy(grid)
    tmp_grid.pop(pos)
    positions = one_move_away(tmp_grid, pos)
    assert (6, 2) not in positions
    assert len(positions) == 2  # should only be two places to move




