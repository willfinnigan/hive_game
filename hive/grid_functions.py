from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple, NamedTuple
from hive.game_types import Location, Grid, Colour, Piece, GridLocation


@lru_cache(maxsize=None)
def positions_around_location(loc: Location) -> Tuple[Location, Location, Location, Location, Location, Location]:
    """
    Returns the position around a location in a clockwise order

         [q-1, r-1] [q+1, r-1]
    [q-2, r  ][q  , r  ][q+2, r  ]
         [q-1, r+1] [q+1, r+1]

    Doubled coordinates - double width - https://www.redblobgames.com/grids/hexagons/
    """

    q, r = loc
    return ((q - 1, r - 1), (q + 1, r - 1), (q + 2, r), (q + 1, r + 1), (q - 1, r + 1), (q - 2, r))


def pieces_around_location(grid: Grid, loc: Location) -> Tuple[Location, ...]:
    return tuple(pos for pos in positions_around_location(loc) if pos in grid)

def is_position_connected(grid: Grid, loc: Location, positions_to_ignore: List[Location] = None) -> bool:
    piece_locations = pieces_around_location(grid, loc)

    # if there are positions to ignore, remove them from grid_locations
    if positions_to_ignore is not None:
        piece_locations = [loc for loc in piece_locations if loc not in positions_to_ignore]

    if len(piece_locations) == 0:
        return False
    return True


def get_empty_locations(grid: Grid):
    """Return all empty locations 1 move away from any piece"""
    empty = []
    for loc in grid.keys():
        for a_loc in positions_around_location(loc):
            if grid.get(a_loc, []) == []:
                empty.append(a_loc)
    empty = list(set(empty))
    return empty


def one_move_away(grid: Grid, loc: Location, positions_to_ignore: List[Location] = None) -> List[Location]:
    """Return all connected empty locations 1 move away from location
    But must be able to slide to that location without breaking connection
    """

    one_away = positions_around_location(loc)
    empty_one_move_away = (loc for loc in one_away if grid.get(loc, None) is None)

    connected_spaces = []
    for space in empty_one_move_away:

        # is there a piece next to this position, which is directly next to the original location
        adjacent_spaces_with_piece = tuple(
            p for p in positions_around_location(space) if p in one_away and grid.get(p, None) is not None)
        if len(adjacent_spaces_with_piece) == 0:
            continue
        elif is_position_connected(grid, space, positions_to_ignore=positions_to_ignore) == False:
            continue
        elif check_can_slide_to(grid, loc, space) == False:
            continue
        else:
            connected_spaces.append(space)
    return connected_spaces


def beetle_one_move_away(grid: Grid, loc: Location, positions_to_ignore: List[Location] = None) -> List[Location]:
    one_move_away = positions_around_location(loc)
    connected_spaces = []
    for space in one_move_away:
        if is_position_connected(grid, space, positions_to_ignore=positions_to_ignore) == False and grid.get(space) is None:
            continue
        else:
            connected_spaces.append(space)
    return connected_spaces


def check_can_slide_to(grid: Grid, loc: Location, to_loc: Location):
    """Check if a piece can slide to a location"""
    #  eg (5,3) -> (6,2) is not possible because of pieces at (4,2) and (7, 3)

    if loc[1] == to_loc[1]:  # same row
        if loc[0] > to_loc[0]:  # moving left to right
            pos_1 = (to_loc[0] + 1, to_loc[1] - 1)
            pos_2 = (to_loc[0] + 1, to_loc[1] + 1)
        elif loc[0] < to_loc[0]:  # moving right to left
            pos_1 = (to_loc[0] - 1, to_loc[1] - 1)
            pos_2 = (to_loc[0] - 1, to_loc[1] + 1)
        else:
            raise Exception("This should never happen - critical logic error")

    elif loc[1] > to_loc[1]:  # moving up
        if loc[0] > to_loc[0]:  # moving up+left
            pos_1 = (to_loc[0] - 1, to_loc[1] + 1)
            pos_2 = (to_loc[0] + 2, to_loc[1])
        elif loc[0] < to_loc[0]:  # moving up+right
            pos_1 = (to_loc[0] - 2, to_loc[1])
            pos_2 = (to_loc[0] + 1, to_loc[1] + 1)
        else:
            raise Exception("This should never happen - critical logic error")

    elif loc[1] < to_loc[1]:  # moving down
        if loc[0] > to_loc[0]:  # moving down+left
            pos_1 = (to_loc[0] - 1, to_loc[1] - 1)
            pos_2 = (to_loc[0] + 2, to_loc[1])
        elif loc[0] < to_loc[0]:  # moving down+right
            pos_1 = (to_loc[0] - 2, to_loc[1])
            pos_2 = (to_loc[0] + 1, to_loc[1] - 1)
        else:
            raise Exception("This should never happen - critical logic error")

    else:
        raise Exception("This should never happen - critical logic error")

    # if there are pieces at both these locations, then the piece can't slide to to_loc
    if grid.get(pos_1) is not None and grid.get(pos_2) is not None:
        return False

    return True


def is_placeable_location(grid: Grid, loc: Location, colour: Colour) -> bool:
    piece_locations = pieces_around_location(grid, loc)
    stacks = [grid[loc] for loc in piece_locations]
    colours_surrounding = set([stack[0].colour for stack in stacks])
    if len(colours_surrounding) > 1 or colour not in colours_surrounding:
        return False  # colour would be next to another colour
    return True  # colour would be next to the same colour


def get_placeable_locations(grid: Grid, colour: Colour) -> List[Location]:
    """Return all locations where a piece of the given colour can be placed"""
    if len(grid) == 0:
        return [(0, 0)]

    empty = get_empty_locations(grid)

    if len(grid) == 1:
        return empty

    allowed = []
    for pos in empty:
        if is_placeable_location(grid, pos, colour) == True:
            allowed.append(pos)
    return allowed

def can_remove_piece(grid: Grid, loc: Location) -> bool:
    """ does removing piece break the hive? """
    if len(grid) <= 2:
        return True  # if less than 2 pieces in the hive, can always remove a piece

    # are all the positions around the current location still connected if the piece is removed?
    piece_locations_current = pieces_around_location(grid, loc)
    for piece_loc in piece_locations_current:
        # would removing this piece disconnect the hive?
        if is_piece_connected_to_hive(grid, piece_loc, ignore_positions=[loc]) == False:
            return False
    return True

def all_connected(grid: Grid, loc: Location, ignore_positions: List[Location] = None):
    """Get all the pieces connected to a piece (should be entire hive)"""
    if ignore_positions is None:
        ignore_positions = []
    connected = set()
    to_visit = {loc}
    while to_visit:
        p = to_visit.pop()
        if p in connected:
            continue
        connected.add(p)
        piece_locations = pieces_around_location(grid, p)
        for piece_loc in piece_locations:
            if piece_loc not in ignore_positions and piece_loc not in connected:
                to_visit.add(piece_loc)
    return connected

def is_piece_connected_to_hive(grid: Grid, loc: Location, ignore_positions: List[Location] = None) -> bool:
    """Can this piece reach all other pieces in the hive?"""
    if ignore_positions is None:
        ignore_positions = []

    connected = all_connected(grid, loc, ignore_positions=ignore_positions)
    if len(connected) == len(grid)-len(ignore_positions):
        return True
    return False

