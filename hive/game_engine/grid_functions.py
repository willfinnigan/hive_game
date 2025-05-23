from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple, NamedTuple
from typing import TYPE_CHECKING
from hive.game_engine.errors import BreaksConnectionError, InvalidLocationError, InvalidMoveError, InvalidPlacementError
from hive.game_engine.game_state import Location, Grid, Colour, Piece, GridLocation


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


@lru_cache(maxsize=None)
def pieces_around_location(grid: Grid, loc: Location) -> Tuple[Location, ...]:
    """Return all positions around a location that contain pieces"""
    # Pre-compute the positions once
    positions = positions_around_location(loc)
    
    # Use a list comprehension for better performance
    # This is faster than a generator expression when we need all results
    return tuple(pos for pos in positions if pos in grid)

@lru_cache(maxsize=None)
def is_position_connected(grid: Grid, loc: Location, positions_to_ignore: Tuple[Location] = None) -> bool:
    """Check if a position is connected to at least one piece in the grid"""
    piece_locations = pieces_around_location(grid, loc)

    # Fast path: if no positions to ignore, just check if there are any pieces around
    if positions_to_ignore is None:
        return len(piece_locations) > 0
        
    # Only filter if we actually have positions to ignore
    if positions_to_ignore:
        # Use a set for faster lookups when filtering
        ignore_set = set(positions_to_ignore)
        piece_locations = [loc for loc in piece_locations if loc not in ignore_set]

    # Simplified return statement
    return len(piece_locations) > 0


def get_empty_locations(grid: Grid):
    """Return all empty locations 1 move away from any piece"""
    empty = []
    for loc in grid.keys():
        for a_loc in positions_around_location(loc):
            if not grid.get(a_loc):
                empty.append(a_loc)
    empty = list(set(empty))
    return empty


@lru_cache(maxsize=None)  # Increased cache size for better performance
def one_move_away(grid: Grid, loc: Location, positions_to_ignore: Tuple[Location] = None) -> List[Location]:
    """Return all connected empty locations 1 move away from location
    But must be able to slide to that location without breaking connection
    """
    # Pre-compute positions around location once
    one_away = positions_around_location(loc)
    # Convert to list immediately instead of using a generator for better performance
    empty_one_move_away = [pos for pos in one_away if grid.get(pos, None) is None]

    connected_spaces = []
    for space in empty_one_move_away:
        # Check if there's at least one adjacent piece first
        has_adjacent_piece = False
        for p in positions_around_location(space):
            if p in one_away and grid.get(p, None) is not None:
                has_adjacent_piece = True
                break
                
        if not has_adjacent_piece:
            continue
            
        # Only check these more expensive conditions if the first check passes
        if not is_position_connected(grid, space, positions_to_ignore=positions_to_ignore):
            continue
            
        if not check_can_slide_to(grid, loc, space, positions_to_ignore=positions_to_ignore):
            continue
            
        connected_spaces.append(space)
    return connected_spaces

@lru_cache(maxsize=None)  # Increased cache size for better performance
def beetle_one_move_away(grid: Grid, loc: Location, positions_to_ignore: Tuple[Location] = None) -> List[Location]:
    """Return all connected locations 1 move away from location for a beetle
    Beetles can move on top of other pieces, so we check differently than other pieces
    """
    # Pre-compute positions around location once
    positions = positions_around_location(loc)

    connected_spaces = []
    for space in positions:
        stack = grid.get(space, ())
        if stack == ():
            connected = is_position_connected(grid, space, positions_to_ignore=positions_to_ignore)
            if connected == False:
                continue

        # otherwise add to connected
        connected_spaces.append(space)

    return connected_spaces


@lru_cache(maxsize=None)  # Cache the results for better performance
def check_can_slide_to(grid: Grid, loc: Location, to_loc: Location, positions_to_ignore: Tuple[Location] = None) -> bool:
    """Check if a piece can slide to a location"""
    #  eg (5,3) -> (6,2) is not possible because of pieces at (4,2) and (7, 3)
    
    # Calculate direction vector
    dq = to_loc[0] - loc[0]
    dr = to_loc[1] - loc[1]
    
    # Use a lookup table for the positions to check based on direction
    # This replaces the complex nested conditionals with a simple lookup
    if (dq, dr) == (2, 0):  # moving right
        pos_1, pos_2 = (to_loc[0] - 1, to_loc[1] - 1), (to_loc[0] - 1, to_loc[1] + 1)
    elif (dq, dr) == (-2, 0):  # moving left
        pos_1, pos_2 = (to_loc[0] + 1, to_loc[1] - 1), (to_loc[0] + 1, to_loc[1] + 1)
    elif (dq, dr) == (1, -1):  # moving up+right
        pos_1, pos_2 = (to_loc[0] - 2, to_loc[1]), (to_loc[0] + 1, to_loc[1] + 1)
    elif (dq, dr) == (-1, -1):  # moving up+left
        pos_1, pos_2 = (to_loc[0] - 1, to_loc[1] + 1), (to_loc[0] + 2, to_loc[1])
    elif (dq, dr) == (1, 1):  # moving down+right
        pos_1, pos_2 = (to_loc[0] - 2, to_loc[1]), (to_loc[0] + 1, to_loc[1] - 1)
    elif (dq, dr) == (-1, 1):  # moving down+left
        pos_1, pos_2 = (to_loc[0] - 1, to_loc[1] - 1), (to_loc[0] + 2, to_loc[1])
    else:
        raise Exception(f"Invalid move direction: from {loc} to {to_loc}, delta: ({dq}, {dr})")

    # Check if positions to ignore are provided and exclude them
    if positions_to_ignore:
        ignore_set = set(positions_to_ignore)
        if pos_1 in ignore_set or pos_2 in ignore_set:
            return True  # Ignore these positions for the sliding check

    return not (grid.get(pos_1) is not None and grid.get(pos_2) is not None)


def is_placeable_location(grid: Grid, loc: Location, colour: Colour) -> bool:
    """Check if a piece can be placed at a location"""

    piece_locations = pieces_around_location(grid, loc)
    stacks = [grid[loc] for loc in piece_locations]

    colours_surrounding = set()

    # we only look at the top of the stack - if there are pieces below, they are not relevant
    for stack in stacks:
        if stack is None or len(stack) == 0:
            continue
        piece = stack[-1]
        colours_surrounding.add(piece.colour)

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

def are_hexes_adjacent(hex1: Location, hex2: Location) -> bool:
    """Check if two hexes are adjacent"""
    q1, r1 = hex1
    q2, r2 = hex2
    return (abs(q1 - q2) == 1 and r1 == r2) or (abs(r1 - r2) == 1 and q1 == q2) or (abs(q1 - q2) == 0 and abs(r1 - r2) == 0)

def check_is_valid_placement(grid: Grid, loc: Location, colour: Colour):
    """Place a piece for the first time can not be touching a piece from opposite colour."""

    # first piece can be anywhere
    if len(grid) == 0:
        return

    # piece must be placed next to another piece
    piece_locations = pieces_around_location(grid, loc)
    if len(piece_locations) == 0:
        raise InvalidPlacementError(f"Not connected")

    # if this is the second piece, can go anywhere
    if len(grid) == 1:
        return

    # if not the first piece of that colour, can only go next to pieces of the same colour
    stacks = [grid.get(loc) for loc in piece_locations]
    colours_surrounding = set([p[-1].colour for p in stacks])
    if len(colours_surrounding) > 1 or colour not in colours_surrounding:
        raise InvalidPlacementError(
            f"{colour} piece can not be placed next to pieces of other color: {colours_surrounding}")


def check_is_valid_move(grid, current_loc, loc: Location):

    # are all the positions around the current location still connected if the piece is removed?
    stack = grid.get(current_loc)
    if stack is None:
        raise InvalidMoveError(f"No piece at location {current_loc}")
    elif len(stack) >= 2:
        pass
    elif can_remove_piece(grid, current_loc) == False:
        raise BreaksConnectionError

    # if the piece is on top of another piece, it is connected - if loc is already occupied
    if len(grid.get(loc, ())) > 0:
        return

    # check piece will be connected to the hive once moved
    piece_locations_new = pieces_around_location(grid, loc)
    if len(stack) == 1:
        piece_locations_new = [loc for loc in piece_locations_new if loc != current_loc]
    if len(piece_locations_new) == 0:
        raise InvalidPlacementError(f"Not connected")

def check_is_valid_location(loc: Location):
    """Sum of locations must be even"""
    if sum(loc) % 2 != 0:
        raise InvalidLocationError(f"Invalid location - {loc} does not sum to even")
