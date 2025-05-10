from typing import List
from hive.game_engine.game_state import Grid, Location
from hive.game_engine.grid_functions import one_move_away, is_position_connected, beetle_one_move_away, can_remove_piece
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



move_functions = {pieces.ANT: get_ant_moves,
                  pieces.BEETLE: get_beetle_moves,
                  pieces.GRASSHOPPER: get_grasshopper_moves,
                  pieces.QUEEN: get_queen_moves,
                  pieces.SPIDER: get_spider_moves
                  }

def get_possible_moves(grid: Grid, location: Location) -> List[Location]:
    stack = grid.get(location)
    if not stack:
        return []

    piece = stack[-1]
    return move_functions[piece.name](grid, location)
