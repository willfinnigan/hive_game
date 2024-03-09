from typing import List
from hive.custom_types import Location

def positions_around_location(loc: Location) -> list:
    """Returns the position around a location in a clockwise order"""

    """
         [q-1, r-1] [q+1, r-1]
    [q-2, r  ][q  , r  ][q+2, r  ]
         [q-1, r+1] [q+1, r+1]
    """
    # Doubled coordinates - double width - https://www.redblobgames.com/grids/hexagons/
    q, r = loc
    return [(q-1, r-1), (q+1, r-1), (q+2, r), (q+1, r+1), (q-1, r+1), (q-2, r)]

def pieces_around_location(grid, loc: Location) -> list:
    positions = positions_around_location(loc)
    pieces = [grid.get(pos) for pos in positions]
    pieces = [p for p in pieces if p is not None]
    return pieces

def is_position_connected(grid, loc: Location):
    pieces = pieces_around_location(grid, loc)
    if len(pieces) == 0:
        return False
    return True

def get_empty_locations(grid):
    """Return all empty locations 1 move away from any piece"""
    empty = []
    for loc in grid.keys():
        for a_loc in positions_around_location(loc):
            if grid.get(a_loc) is None:
                empty.append(a_loc)
    empty = list(set(empty))
    return empty

def one_move_away(grid, loc: Location) -> List[Location]:
    """Return all connected empty locations 1 move away from location"""

    one_move_away = positions_around_location(loc)
    one_move_away = [loc for loc in one_move_away if grid.get(loc, None) is None]
    connected_spaces = []
    for space in one_move_away:
        if is_position_connected(grid, space) == False:
            continue
        elif check_can_slide_to(grid, loc, space) == False:
            continue
        else:
            connected_spaces.append(space)
    return connected_spaces

def beetle_one_move_away(grid, loc: Location) -> List[Location]:
    """Return all connected empty locations 1 move away from location"""

    one_move_away = positions_around_location(loc)
    connected_spaces = []
    for space in one_move_away:
        if is_position_connected(grid, space) == False and grid.get(space) is None:
            continue
        else:
            connected_spaces.append(space)
    return connected_spaces



def check_can_slide_to(grid, loc: Location, to_loc: Location):
    """Check if a piece can slide to a location"""
    #  eg (5,3) -> (6,2) is not possible because of pieces at (4,2) and (7, 3)

    if loc[1] == to_loc[1]:  # same row
        if loc[0] > to_loc[0]:  # moving left to right
            pos_1 = (to_loc[0]+1, to_loc[1]-1)
            pos_2 = (to_loc[0]+1, to_loc[1]+1)
        elif loc[0] < to_loc[0]:  # moving right to left
            pos_1 = (to_loc[0]-1, to_loc[1]-1)
            pos_2 = (to_loc[0]-1, to_loc[1]+1)
        else:
            raise Exception("This should never happen - critical logic error")

    elif loc[1] > to_loc[1]:  # moving up
        if loc[0] > to_loc[0]:  # moving up+left
            pos_1 = (to_loc[0]-1, to_loc[1]+1)
            pos_2 = (to_loc[0]+2, to_loc[1])
        elif loc[0] < to_loc[0]:  # moving up+right
            pos_1 = (to_loc[0]-2, to_loc[1])
            pos_2 = (to_loc[0]+1, to_loc[1]+1)
        else:
            raise Exception("This should never happen - critical logic error")

    elif loc[1] < to_loc[1]:  # moving down
        if loc[0] > to_loc[0]:  # moving down+left
            pos_1 = (to_loc[0]-1, to_loc[1]-1)
            pos_2 = (to_loc[0]+2, to_loc[1])
        elif loc[0] < to_loc[0]:  # moving down+right
            pos_1 = (to_loc[0]-2, to_loc[1])
            pos_2 = (to_loc[0]+1, to_loc[1]-1)
        else:
            raise Exception("This should never happen - critical logic error")

    else:
        raise Exception("This should never happen - critical logic error")

    # if there are pieces at both these locations, then the piece can't slide to to_loc
    if grid.get(pos_1) is not None and grid.get(pos_2) is not None:
        return False

    return True

"""
                 / b   \
         / b   \ \ A   /
 /    wA  \ \ Q   /
 \-10,-10 / / w   \
         \ Q   /
"""


"""
0          /      \
1 /      \ \      /
2 \      / /      \
3          \      / 
"""


def draw_grid():
    top_grid    = "  /      \\"
    bottom_grid = "  \      /"
    empty_space = "          "


    num_cols = 10
    num_rows = 10

    grid = "\033[31m"


    # top row
    grid += empty_space
    for c in range(int(num_cols/2)):
        grid += top_grid
        grid += empty_space
    grid += "\n"

    # all other rows
    for r in range(num_rows):
        for c in range(num_cols):
            if r % 2 == 1:
                if c % 2 == 0:
                    grid += bottom_grid
                elif r == num_rows-1:
                    grid += empty_space
                else:
                    grid += top_grid
            else:
                if c % 2 == 0:
                    grid += top_grid
                else:
                    grid += bottom_grid
        grid += "\n"

    return grid

def draw_grid2():
    top_grid    = "  /      \\"
    bottom_grid = "  \      /"
    empty_space = "          "
    num_cols = 10
    num_rows = 10

    red = "\033[31m"
    blue = "\033[34m"
    lightblue = "\033[94m"
    darkblue = "\033[36m"

    grid = lightblue

    # top row
    for c in range(int(num_cols / 2)):
        grid += top_grid
        grid += empty_space
    grid += "\n"

    for r in range(-1, num_rows):
        for c in range(num_cols):
            if r % 2 == 1:
                if c % 2 == 0:
                    grid += lightblue
                    grid += bottom_grid
                else:
                    grid += darkblue
                    grid += top_grid
            else:
                if c % 2 == 0:
                    grid += lightblue
                    grid += top_grid
                else:
                    grid += darkblue
                    grid += bottom_grid
        grid += "\n"

    # bottom row
    grid += darkblue
    for c in range(int(num_cols / 2)):
        grid += empty_space
        grid += bottom_grid

    grid += "\n"

    return grid


if __name__ == '__main__':
    grid = draw_grid2()
    print(grid)