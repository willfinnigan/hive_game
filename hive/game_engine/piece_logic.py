from typing import List
from hive.game_engine.game_state import Grid, Location
from hive.game_engine.grid_functions import one_move_away, is_position_connected, beetle_one_move_away, can_remove_piece, pieces_around_location, positions_around_location
from hive.game_engine import pieces


def get_ant_moves(grid: Grid, loc: Location) -> List[Location]:
    """Get all possible moves for an ant piece
    
    Ants can move to any empty space connected to the hive.
    We use a breadth-first search to find all possible moves.
    """
    # Early return if the piece can't be removed
    if not can_remove_piece(grid, loc):
        return []
        
    # Use sets for faster lookups and to avoid duplicates
    visited = {loc}
    to_visit = set()
    
    # Initial moves
    initial_moves = one_move_away(grid, loc)
    to_visit.update(initial_moves)
    
    # Breadth-first search to find all possible moves
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
            
        visited.add(current)
        
        # Get next possible moves
        next_moves = one_move_away(grid, current, positions_to_ignore=(loc,))
        for move in next_moves:
            if move not in visited:
                to_visit.add(move)
    
    # Remove the starting location
    visited.remove(loc)
    
    return list(visited)

def get_beetle_moves(grid: Grid, loc: Location) -> List[Location]:
    """Get all possible moves for a beetle piece
    
    Beetles can move one space in any direction, including on top of other pieces.
    """
    # Early return if the piece can't be removed
    if not can_remove_piece(grid, loc):
        return []
    return beetle_one_move_away(grid, loc, positions_to_ignore=(loc,))

def get_grasshopper_moves(grid: Grid, loc: Location) -> List[Location]:
    if can_remove_piece(grid, loc) == False:
        return []

    line_transformations = [(-1, -1), (+1, -1), (-2, 0), (+2, 0), (-1, +1), (+1, +1)]

    jumps = []
    for transform in line_transformations:
        pieces_jumped = 0
        pos = (loc[0] + transform[0], loc[1] + transform[1])
        while grid.get(pos) is not None:
            pieces_jumped += 1
            pos = (pos[0] + transform[0], pos[1] + transform[1])

        if pieces_jumped == 0:
            continue
        elif is_position_connected(grid, pos, positions_to_ignore=(loc,)) == True:
            jumps.append(pos)

    return jumps

def get_queen_moves(grid: Grid, loc: Location) -> List[Location]:
    """Get all possible moves for a queen piece
    
    Queens can move one space in any direction, but cannot climb on top of other pieces.
    """
    # Early return if the piece can't be removed
    if not can_remove_piece(grid, loc):
        return []
    return one_move_away(grid, loc)

def get_spider_moves(grid: Grid, loc: Location) -> List[Location]:
    """Get all possible moves for a spider piece
    
    Spiders must move exactly 3 steps around the hive.
    """
    # Early return if the piece can't be removed
    if not can_remove_piece(grid, loc):
        return []

    # Use a more efficient approach with a single function for all steps
    def get_next_steps(current_paths, positions_to_ignore):
        next_paths = []
        for path in current_paths:
            current_pos = path[-1]
            # Get all possible next positions
            next_positions = one_move_away(grid, current_pos, positions_to_ignore=positions_to_ignore)
            # Only consider positions not already in the path
            for next_pos in next_positions:
                if next_pos not in path:
                    next_paths.append(path + [next_pos])
        return next_paths

    # Start with the initial position
    paths = [[loc]]
    
    # Perform exactly 3 steps
    paths = get_next_steps(paths, (loc,))
    paths = get_next_steps(paths, (loc,))
    paths = get_next_steps(paths, (loc,))
    
    # Extract the final positions from each valid path
    return [path[-1] for path in paths]

def get_mosquito_moves(grid: Grid, loc: Location) -> List[Location]:
    """Get all possible moves for a mosquito piece
    
    The Mosquito mimics the movement ability of any piece it's touching.
    For example, if touching a Beetle, it can move like a Beetle.
    If touching a Spider, it can move like a Spider.
    
    The Mosquito must be touching the piece at the start of its turn to copy its movement.
    If the Mosquito is on top of the hive (like a Beetle would be), it can only move like a Beetle,
    regardless of what other pieces it's touching.
    """
    # Early return if the piece can't be removed
    if not can_remove_piece(grid, loc):
        return []
    
    stack = grid.get(loc)
    if not stack:
        return []
    
    # If mosquito is on top of another piece, it can only move like a beetle
    if len(stack) > 1:
        return get_beetle_moves(grid, loc)
    
    # Get all adjacent pieces
    adjacent_positions = pieces_around_location(grid, loc)
    
    # If no adjacent pieces, return empty list
    if not adjacent_positions:
        return []
    
    # Collect all possible moves from adjacent pieces
    all_moves = []
    
    for adj_pos in adjacent_positions:
        adj_stack = grid.get(adj_pos)
        if not adj_stack:
            continue
            
        # Get the top piece at this position
        adj_piece = adj_stack[-1]
        
        # Skip if it's another mosquito to prevent infinite recursion
        if adj_piece.name == pieces.MOSQUITO:
            continue
            
        # Get moves based on the adjacent piece type
        if adj_piece.name == pieces.ANT:
            moves = get_ant_moves(grid, loc)
            all_moves.extend(moves)
        elif adj_piece.name == pieces.BEETLE:
            moves = get_beetle_moves(grid, loc)
            all_moves.extend(moves)
        elif adj_piece.name == pieces.GRASSHOPPER:
            moves = get_grasshopper_moves(grid, loc)
            all_moves.extend(moves)
        elif adj_piece.name == pieces.QUEEN:
            moves = get_queen_moves(grid, loc)
            all_moves.extend(moves)
        elif adj_piece.name == pieces.SPIDER:
            moves = get_spider_moves(grid, loc)
            all_moves.extend(moves)
        elif adj_piece.name == pieces.MOSQUITO:
            # If the adjacent piece is a mosquito, we can ignore it
            pass
        elif adj_piece.name == pieces.PILLBUG:
            moves = get_pillbug_moves(grid, loc)
            all_moves.extend(moves)
        elif adj_piece.name == pieces.LADYBUG:
            moves = get_ladybug_moves(grid, loc)
            all_moves.extend(moves)
    
    # Remove duplicates and return
    return list(set(all_moves))

def get_pillbug_moves(grid: Grid, loc: Location) -> List[Location]:
    """Get all possible moves for a pillbug piece
    
    The Pillbug has two abilities:
        - It can move one space around the hive like the Queen Bee.
        - It has a special ability to move other pieces: 
            Once per turn, instead of moving, the Pillbug can move an adjacent unstacked piece (friendly or opposing) 
            to an empty space adjacent to itself, provided the move doesn't break the hive. This special ability cannot 
            be used on a piece that was moved in the opponent's last turn, and cannot move the Queen if it would 
            break the hive. (so pillbug can not move beetles on top of other pieces).
    """
    pass

def get_ladybug_moves(grid: Grid, loc: Location) -> List[Location]:
    """Get all possible moves for a ladybug piece
    
    The Ladybug moves exactly three spaces:
        two on top of the hive followed by one down to the ground level.
        
    The Ladybug must move all three spaces and cannot remain on top of the hive.
    Unlike the Beetle, it cannot stay on top of other pieces.

    The Ladybug's unique movement pattern allows it to jump gaps and potentially
    reach positions that would be inaccessible to other pieces.
    """
    
    # Early return if the piece can't be removed
    if not can_remove_piece(grid, loc):
        return []
    
    # First steps - get all the positions containing a piece around the ladybug
    first_steps = set()
    for pos in pieces_around_location(grid, loc):
        first_steps.add(pos)

    # If no valid first steps, return empty list
    if len(first_steps) == 0:
        return []
    
    # Second step: Move on top of the hive again (like a beetle)
    second_steps = set()
    for first_pos in first_steps:
        for pos in pieces_around_location(grid, first_pos):
            second_steps.add(pos)

    # Second step can not be the starting position
    second_steps.discard(loc)

    # If no valid second steps, return empty list
    if len(second_steps) == 0:
        return []
    
    # Finially must move to an empty ground space one move away from the second step
    possible_final_positions = set()
    for second_pos in second_steps:
        for pos in positions_around_location(second_pos):
            possible_final_positions.add(pos)
    
    final_positions = set()
    for pos in possible_final_positions:
        # if space is not empty, can't move there
        if len(grid.get(pos, ())) != 0:
            continue

        # space must be connected to the hive
        if is_position_connected(grid, pos, positions_to_ignore=(loc,)):
            final_positions.add(pos)

    return list(final_positions)

move_functions = {pieces.ANT: get_ant_moves,
                  pieces.BEETLE: get_beetle_moves,
                  pieces.GRASSHOPPER: get_grasshopper_moves,
                  pieces.QUEEN: get_queen_moves,
                  pieces.SPIDER: get_spider_moves,
                  pieces.MOSQUITO: get_mosquito_moves,
                  pieces.PILLBUG: get_pillbug_moves,
                  pieces.LADYBUG: get_ladybug_moves,
                  }

def get_possible_moves(grid: Grid, location: Location) -> List[Location]:
    stack = grid.get(location)
    if not stack:
        return []

    piece = stack[-1]
    return move_functions[piece.name](grid, location)


