from __future__ import annotations

from copy import deepcopy
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from hive.game.pieces.piece_base_class import Piece
    from hive.game.types_and_errors import Location, Grid, Colour


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

def pieces_around_location(grid: Grid, loc: Location) -> list:
    positions = positions_around_location(loc)
    pieces = [grid.get(pos) for pos in positions]
    pieces = [p for p in pieces if p is not None]
    return pieces

def is_position_connected(grid: Grid, loc: Location):
    pieces = pieces_around_location(grid, loc)
    if len(pieces) == 0:
        return False
    return True

def get_empty_locations(grid: Grid):
    """Return all empty locations 1 move away from any piece"""
    empty = []
    for loc in grid.keys():
        for a_loc in positions_around_location(loc):
            if grid.get(a_loc) is None:
                empty.append(a_loc)
    empty = list(set(empty))
    return empty

def one_move_away(grid: Grid, loc: Location) -> List[Location]:
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

def beetle_one_move_away(grid: Grid, loc: Location) -> List[Location]:

    one_move_away = positions_around_location(loc)
    connected_spaces = []
    for space in one_move_away:
        if is_position_connected(grid, space) == False and grid.get(space) is None:
            continue
        else:
            connected_spaces.append(space)
    return connected_spaces


def check_can_slide_to(grid: Grid, loc: Location, to_loc: Location):
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


def can_remove_piece(grid: Grid, piece: Piece) -> bool:
    """ does removing piece break the hive? """

    loc = piece.location
    surrounding = pieces_around_location(grid, loc)  # up to 6 surrounding pieces

    tmp_grid = deepcopy(grid)
    try:
        tmp_grid.pop(loc)
    except:
        print(f"Error removing piece {piece} {piece.location} from grid at {loc}")
        print(grid)
        raise Exception()

    # if I remove this piece, are can the pieces directly around still connected to each other?
    for piece in surrounding:
        if is_piece_connected_to_hive(tmp_grid, piece) == False:
            return False
    return True


def is_piece_connected_to_hive(grid: Grid, piece: Piece) -> bool:
    """Can this piece reach all other pieces in the hive?"""
    connected = all_connected(piece, grid)
    if len(connected) == len(grid):
        return True
    return False

def all_connected(piece: Piece, grid: Grid):
    """Get all the pieces connected to a piece (should be entire hive)"""
    connected = set()
    to_visit = {piece.location}
    while to_visit:
        p = to_visit.pop()
        if p in connected:
            continue
        connected.add(p)
        around = pieces_around_location(grid, p)
        for piece in around:
            to_visit.add(piece.location)
    return connected


def is_placeable_location(grid: Grid, loc: Location, colour: Colour) -> bool:
    pieces = pieces_around_location(grid, loc)

    colours_surrounding = set([p.colour for p in pieces])
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
