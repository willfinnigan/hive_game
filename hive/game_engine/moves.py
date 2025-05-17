from dataclasses import dataclass
from typing import List, Optional
from hive.game_engine.game_functions import move_piece, pass_move, place_piece
from hive.game_engine.game_state import Colour, Game, Grid, Location, Piece
from hive.game_engine.grid_functions import one_move_away, is_position_connected, beetle_one_move_away, can_remove_piece, pieces_around_location, positions_around_location
from hive.game_engine import pieces

@dataclass
class Move:
    piece: Piece
    current_location: Optional[Location]
    current_stack_idx: Optional[int]
    new_location: Location
    new_stack_idx: int
    colour: Colour = None
    pillbug_moved_other_piece: bool = False

    def __post_init__(self):
        # For pillbugs, we pass in the colour
        if self.colour is None:
            self.colour = self.piece.colour

    def play(self, game) -> Game:
        if self.current_location is None:
            new_game = place_piece(game, self.piece, self.new_location, move=self)
        else:
            new_game = move_piece(game, self.current_location, self.new_location, self.colour, move=self)

        return new_game
    
    def get_colour(self):
        return self.piece.colour

    def _move_string(self):
        piece_name = f"{self.piece.colour}_{self.piece.name}_{self.piece.number}"
        if self.current_location is None:
            return f"Place({piece_name} at {self.new_location})"
        return f"Move ({piece_name} from {self.current_location}_{self.current_stack_idx} to {self.new_location}_{self.new_stack_idx})"

    def __repr__(self):
        move_string = self._move_string()
        pb = ""
        if self.pillbug_moved_other_piece:
            pb = "PB "
        return f"{pb}{move_string}_s"

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False

        # did not include pillbug because can't get this information from replay game strings.
        return (self.piece == other.piece and
                self.current_location == other.current_location and
                self.new_location == other.new_location and
                self.current_stack_idx == other.current_stack_idx and
                self.new_stack_idx == other.new_stack_idx and
                self.colour == other.colour)


    def __hash__(self):
        return hash(str(self._move_string()))


@dataclass
class NoMove:
    colour: Colour
    pillbug_moved_other_piece: bool = False

    def play(self, game) -> Game:
        new_game = pass_move(game, self.colour, move=self)
        return new_game
    
    def get_colour(self):
        return self.colour
    
    def __repr__(self):
        return f"Pass ({self.colour})"
    
    def __hash__(self):
        return hash(str(self))

def get_ant_moves(grid: Grid, loc: Location, stack_idx: int) -> List[Move]:
    """Get all possible moves for an ant piece
    
    Ants can move to any empty space connected to the hive.
    We use a breadth-first search to find all possible moves.
    """
    stack = grid.get(loc, ())
    if len(stack) == 0:
        raise ValueError("No pieces at the given location")
    
    piece = stack[stack_idx]

    # if theres a piece on top, cant move
    if len(stack) > 1:
        return []

    # Early return if the piece can't be removed
    if not can_remove_piece(grid, loc):
        return []

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
    
    visited = list(visited)

    moves = []
    for new_loc in visited:
        mv = Move(piece=piece, 
                  current_location=loc, 
                  current_stack_idx=stack_idx, 
                  new_location=new_loc,
                  new_stack_idx=0)
        moves.append(mv)
    return moves



def get_beetle_moves(grid: Grid, loc: Location, stack_idx: int) -> List[Move]:
    """Get all possible moves for a beetle piece
    
    Beetles can move one space in any direction, including on top of other pieces.
    """
    stack = grid.get(loc, ())
    if len(stack) == 0:
        raise ValueError("No pieces at the given location")
    
    piece = stack[stack_idx]

    # if theres a piece on top, cant move
    # if stack of 3, and beetle is idx 1, then beetle is in the middle, but cant move
    # if stack of 3, and beetle is idx 2, then beetle is on top, and can move
    if stack_idx < len(stack) - 1:
        return []

    # Early return if the piece can't be removed
    if stack_idx == 0:
        if not can_remove_piece(grid, loc):
            return []

    # we're currently on top of another piece, so ignore the current location
    if len(stack) > 1:
        positions_to_ignore = ()
    else:
        positions_to_ignore = (loc,)
    
    locations = beetle_one_move_away(grid, loc, positions_to_ignore=positions_to_ignore)

    moves = []
    for new_loc in locations:
        # get the stack height at each location - beetle can move to the top of a stack
        stack_height = len(grid.get(new_loc, ()))
        mv = Move(piece=piece,
                  current_location=loc, 
                  current_stack_idx=stack_idx, 
                  new_location=new_loc,
                  new_stack_idx=stack_height)
        moves.append(mv)

    return moves

def get_grasshopper_moves(grid: Grid, loc: Location, stack_idx: int) -> List[Move]:
    if can_remove_piece(grid, loc) == False:
        return []
    
    stack = grid.get(loc, ())
    if len(stack) == 0:
        raise ValueError("No pieces at the given location")
    
    piece = stack[stack_idx]

    line_transformations = [(-1, -1), (+1, -1), (-2, 0), (+2, 0), (-1, +1), (+1, +1)]

    jumps = set()
    for transform in line_transformations:
        pieces_jumped = 0
        pos = (loc[0] + transform[0], loc[1] + transform[1])
        while grid.get(pos) is not None:
            pieces_jumped += 1
            pos = (pos[0] + transform[0], pos[1] + transform[1])

        if pieces_jumped == 0:
            continue
        elif is_position_connected(grid, pos, positions_to_ignore=(loc,)) == True:
            jumps.add(pos)

    moves = []
    for jump in jumps:
        mv = Move(piece=piece,
                  current_location=loc, 
                  current_stack_idx=stack_idx,
                  new_location=jump,
                  new_stack_idx=0)
        moves.append(mv)
    
    return moves

def get_queen_moves(grid: Grid, loc: Location, stack_idx: int) -> List[Move]:
    """Get all possible moves for a queen piece
    
    Queens can move one space in any direction, but cannot climb on top of other pieces.
    """
    # Early return if the piece can't be removed
    if not can_remove_piece(grid, loc):
        return []
    
    stack = grid.get(loc, ())
    if len(stack) == 0:
        raise ValueError("No pieces at the given location")
    
    piece = stack[stack_idx]

    locs = one_move_away(grid, loc)

    moves = []
    for new_loc in locs:
        mv = Move(piece=piece,
                  current_location=loc, 
                  current_stack_idx=stack_idx, 
                  new_location=new_loc,
                  new_stack_idx=0)
        moves.append(mv)
    return moves

def get_spider_moves(grid: Grid, loc: Location, stack_idx: int) -> List[Move]:
    """Get all possible moves for a spider piece
    
    Spiders must move exactly 3 steps around the hive.
    """
    # Early return if the piece can't be removed
    if not can_remove_piece(grid, loc):
        return []
    
    stack = grid.get(loc, ())
    if len(stack) == 0:
        raise ValueError("No pieces at the given location")
    
    piece = stack[stack_idx]

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

    locs = [path[-1] for path in paths]

    moves = []
    for new_loc in locs:
        mv = Move(piece=piece,
                  current_location=loc, 
                  current_stack_idx=stack_idx, 
                  new_location=new_loc,
                  new_stack_idx=0)
        moves.append(mv)
    return moves

def get_mosquito_moves(grid: Grid, loc: Location, stack_idx: int) -> List[Move]:
    """Get all possible moves for a mosquito piece
    
    The Mosquito mimics the movement ability of any piece it's touching.
    For example, if touching a Beetle, it can move like a Beetle.
    If touching a Spider, it can move like a Spider.
    
    The Mosquito must be touching the piece at the start of its turn to copy its movement.
    If the Mosquito is on top of the hive (like a Beetle would be), it can only move like a Beetle,
    regardless of what other pieces it's touching.

    If a mosquito is touching another mosquito at the start of its turn, it can not move at all

    For stacks use the top piece
    """
    # Early return if the piece can't be removed


    stack = grid.get(loc, ())
    if len(stack) == 0:
        raise ValueError("No pieces at the given location")
    
    piece = stack[stack_idx]

    # Get all adjacent pieces
    adjacent_positions = pieces_around_location(grid, loc)

    # are any of the surrounding pieces mosquitoes?
    for adj_pos in adjacent_positions:
        adj_stack = grid.get(adj_pos, ())
        if len(adj_stack) == 0:
            continue

        if stack_idx >= len(adj_stack):
            continue  # mosquito is higher than the adjacent stack, so no adjacency

        # if the adjacent piece is under another piece, then we can ignore it
        if len(adj_stack) > stack_idx:
            continue  # adjacent piece is under another piece

        adj_piece = adj_stack[stack_idx]
        if adj_piece.name == pieces.MOSQUITO:
            # If the adjacent piece is a mosquito, we can not move
            print("Can't move, adjacent mosquito")
            return []

    # If mosquito is on top of another piece, it can only move like a beetle
    if len(stack) > 1:
        return get_beetle_moves(grid, loc, stack_idx)

    # Collect all possible moves from adjacent pieces

    adjacent_piece_types = set()
    for adj_pos in adjacent_positions:
        adj_stack = grid.get(adj_pos)
        if not adj_stack:
            continue

        # Get the top piece at this position
        adj_piece = adj_stack[-1]

        if adj_piece.name == pieces.ANT:
            adjacent_piece_types.add(pieces.ANT)
        elif adj_piece.name == pieces.BEETLE:
            adjacent_piece_types.add(pieces.BEETLE)
        elif adj_piece.name == pieces.GRASSHOPPER:
            adjacent_piece_types.add(pieces.GRASSHOPPER)
        elif adj_piece.name == pieces.QUEEN:
            adjacent_piece_types.add(pieces.QUEEN)
        elif adj_piece.name == pieces.SPIDER:
            adjacent_piece_types.add(pieces.SPIDER)
        elif adj_piece.name == pieces.PILLBUG:
            adjacent_piece_types.add(pieces.PILLBUG)
        elif adj_piece.name == pieces.LADYBUG:
            adjacent_piece_types.add(pieces.LADYBUG)


    all_moves = []
    for piece_type in adjacent_piece_types:
        all_moves += move_functions[piece_type](grid, loc, stack_idx)

    return all_moves

def check_can_slide_with_height(grid: Grid, from_loc: Location, to_loc: Location, height_threshold: int) -> bool:
    """
    Check if a piece can slide from one location to another considering stack heights.
    
    Args:
        grid: The game grid
        from_loc: The starting location
        to_loc: The destination location
        height_threshold: The height threshold - stacks taller than this will block movement
        
    Returns:
        bool: True if the piece can slide, False if blocked by stacks
    """
    # Calculate direction vector
    dq = to_loc[0] - from_loc[0]
    dr = to_loc[1] - from_loc[1]
    
    # Determine positions that would create a narrow gap based on direction
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
        raise Exception(f"Invalid move direction: from {from_loc} to {to_loc}, delta: ({dq}, {dr})")
    
    # Check if either position has a stack taller than the height threshold
    stack1 = grid.get(pos_1, ())
    stack2 = grid.get(pos_2, ())
    
    # Movement is only blocked if BOTH positions have stacks AND both stacks are taller than the threshold
    if stack1 and stack2:
        if len(stack1) > height_threshold and len(stack2) > height_threshold:
            return False
    
    return True

def get_pillbug_moves(grid: Grid, loc: Location, stack_idx: int) -> List[Move]:
    """Get all possible moves for a pillbug piece
    
    The Pillbug has two abilities:
        - It can move one space around the hive like the Queen Bee.
        - It has a special ability to move other pieces:
            Once per turn, instead of moving, the Pillbug can move an adjacent unstacked piece (friendly or opposing)
            up onto itself, and then down onto an empty space adjacent to itself,
            provided the move doesn't break the hive.

          This special ability cannot be used on a piece that was moved in the opponent's last turn.
          The pillbug can not move pieces through narrow gaps.
    """
    moves = []

    stack = grid.get(loc, ())
    if len(stack) == 0:
        raise ValueError("No pieces at the given location")

    # If beetle on top of pillbug, no moves
    if len(stack) > 1:
        return []

    pillbug_piece = stack[stack_idx]
    pillbug_height = len(stack)  # Height of the pillbug stack

    # Moves the pillbug can make - 1 move away like queen
    pillbug_moves = get_queen_moves(grid, loc, stack_idx)

    # Now moving adjacent pieces.
    # First lets get all the positions a piece on top of the pillbug, could move to
    # ..we can do this by pretending to be a beetle on top of the pillbug
    locations = beetle_one_move_away(grid, loc, positions_to_ignore=None)

    # But we can only move to empty spaces
    locations = [loc for loc in locations if len(grid.get(loc, ())) == 0]

    # Now lets get the adjacent pieces we could move
    adjacent_piece_locations = pieces_around_location(grid, loc)
    adjacent_pieces = []
    for pos in adjacent_piece_locations:
        adj_stack = grid.get(pos, ())
        if len(adj_stack) == 0:
            continue
        if len(adj_stack) > 1:
            continue
        adj_piece = adj_stack[0]

        # check we can remove this piece
        if can_remove_piece(grid, pos) == True:
            adjacent_pieces.append((adj_piece, pos))

    # Ok so now we have pieces we could move, and possible locations
    # (we know these are connected because only 1 move away)

    moves = []
    for adj_piece, pos in adjacent_pieces:
        for move_loc in locations:
            # Check if the piece can slide to the pillbug (first step)
            # Any stack higher than the pillbug's height will block
            if not check_can_slide_with_height(grid, pos, loc, pillbug_height):
                continue
                
            # Check if the piece can slide from the pillbug to the destination (second step)
            # Any stack higher than the pillbug's height will block
            if not check_can_slide_with_height(grid, loc, move_loc, pillbug_height):
                continue
                
            # If both checks pass, add the move
            move = Move(piece=adj_piece,
                        current_stack_idx=0,
                        current_location=pos,
                        new_stack_idx=0,
                        new_location=move_loc,
                        colour=pillbug_piece.colour,
                        pillbug_moved_other_piece=True)
            moves.append(move)

    return pillbug_moves + moves


def get_ladybug_moves(grid: Grid, loc: Location, stack_idx: int) -> List[Move]:
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
    
    stack = grid.get(loc, ())
    if len(stack) == 0:
        raise ValueError("No pieces at the given location")
    
    piece = stack[stack_idx]
    
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

    moves = []
    for new_loc in final_positions:
        mv = Move(piece=piece,
                  current_location=loc, 
                  current_stack_idx=stack_idx, 
                  new_location=new_loc,
                  new_stack_idx=0)
        moves.append(mv)

    return moves

move_functions = {pieces.ANT: get_ant_moves,
                  pieces.BEETLE: get_beetle_moves,
                  pieces.GRASSHOPPER: get_grasshopper_moves,
                  pieces.QUEEN: get_queen_moves,
                  pieces.SPIDER: get_spider_moves,
                  pieces.MOSQUITO: get_mosquito_moves,
                  pieces.PILLBUG: get_pillbug_moves,
                  pieces.LADYBUG: get_ladybug_moves,
                  }
def get_possible_moves(grid: Grid, location: Location, stack_idx: int) -> List[Move]:
    stack = grid.get(location)
    if not stack:
        return []

    piece = stack[-1]
    return move_functions[piece.name](grid, location, stack_idx)


def is_pillbug_move(game, move):
    """
    Determine if a move was made using a pillbug's special ability.
    
    This function analyzes a move in the context of a game state to determine
    if the move was likely made by a pillbug. It uses a two-step process:
    1. Check if there's a pillbug adjacent to both the start and end positions
    2. Check if the piece could have moved on its own without the pillbug
    
    Args:
        game: The game state after the move was made
        move: The move to analyze
        
    Returns:
        bool: True if the move was likely made by a pillbug, False otherwise
    """
    # If it's a placement or a pass, it's not a pillbug move
    if move.current_location is None or isinstance(move, NoMove):
        return False
    
    # First, check if the move is already marked as a pillbug move
    if move.pillbug_moved_other_piece:
        return True
        
    # Step 1: Check if there's a pillbug adjacent to both the start and end positions
    start_loc = move.current_location
    end_loc = move.new_location
    
    # Find pillbugs adjacent to the start location
    pillbugs_adjacent_to_start = []
    for adj_loc in positions_around_location(start_loc):
        stack = game.grid.get(adj_loc, ())
        for i, piece in enumerate(stack):
            if piece.name == pieces.PILLBUG and i == len(stack) - 1:  # Must be on top
                pillbugs_adjacent_to_start.append((piece, adj_loc))
    
    # If no pillbugs adjacent to start, it's not a pillbug move
    if not pillbugs_adjacent_to_start:
        return False
    
    # Check if any of these pillbugs are also adjacent to the end location
    pillbugs_adjacent_to_both = []
    for pillbug, pillbug_loc in pillbugs_adjacent_to_start:
        if pillbug_loc in positions_around_location(end_loc):
            pillbugs_adjacent_to_both.append((pillbug, pillbug_loc))
    
    # If no pillbugs adjacent to both positions, it's not a pillbug move
    if not pillbugs_adjacent_to_both:
        return False
    
    # Step 2: Check if this move could have been made WITHOUT the pillbug
    
    # Case 1: Different color - definitely a pillbug move
    if game.parent is None:
        return False
        
    current_player_color = game.parent.current_turn  # Color of player who just moved
    if move.piece.colour != current_player_color:
        return True
    
    # Case 2: Same color - check if the piece could have moved on its own
    # Get all possible moves for this piece
    
    # We need to check the game state BEFORE the move was made
    parent_game = game.parent
    possible_moves = get_possible_moves(parent_game.grid, start_loc, move.current_stack_idx)
    
    # Check if the actual move is in the list of possible moves
    # (excluding pillbug-assisted moves)
    for possible_move in possible_moves:
        if (possible_move.new_location == end_loc and
            not possible_move.pillbug_moved_other_piece):
            # The piece could have moved there on its own
            return False
    
    # If we get here, the piece couldn't have moved there on its own
    # So it must be a pillbug move
    return True


