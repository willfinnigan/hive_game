"""
BoardSpace notation module for Hive game trajectories.

This module provides functionality to convert between internal move representation
and the BoardSpace notation format, as well as save and load complete game trajectories.

BoardSpace notation format:
1. Piece Identification (Short Name):
   - Color: `w` or `b`
   - Type: `Q` (Queen), `S` (Spider), `B` (Beetle), `G` (Grasshopper), `A` (Ant), etc.
   - Number (for pieces with multiple instances): 1, 2, or 3 indicating the order it was placed

2. Move Types:
   - First Move: Just the piece short name (e.g., `wS1`)
   - Placing a New Piece: `PieceShortName RelativePosition` (e.g., `bA1 wG1/`)
   - Moving an Existing Piece: Same format as placing (e.g., `wG1 bA1\`)
   - Moving Onto the Hive (Beetle): `MovingPieceShortName TargetPieceShortName` (e.g., `wB1 bA1`)
   - Passing: `pass`

3. Relative Positioning:
   - Direction indicators (-, /, \) combined with a reference piece
   - The indicator comes before the reference piece for Left (-), Bottom Left (/), Top Left (\)
   - The indicator comes after the reference piece for Right (-), Bottom Right (\), Top Right (/)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import re
import os

from hive.game_engine.game_state import Game, Piece, Location, Colour, WHITE, BLACK
from hive.game_engine import pieces
from hive.game_engine.grid_functions import positions_around_location
from hive.game_engine.moves import Move, NoMove


# Mapping between internal color representation and BoardSpace notation
COLOR_TO_BOARDSPACE = {
    WHITE: 'w',
    BLACK: 'b'
}

BOARDSPACE_TO_COLOR = {
    'w': WHITE,
    'b': BLACK
}

# Mapping between piece types and BoardSpace notation
PIECE_TYPE_TO_LETTER = {
    pieces.QUEEN: 'Q',
    pieces.ANT: 'A',
    pieces.BEETLE: 'B',
    pieces.GRASSHOPPER: 'G',
    pieces.SPIDER: 'S',
    pieces.LADYBUG: 'L',
    pieces.PILLBUG: 'P',
    pieces.MOSQUITO: 'M'
}

LETTER_TO_PIECE_TYPE = {
    'Q': pieces.QUEEN,
    'A': pieces.ANT,
    'B': pieces.BEETLE,
    'G': pieces.GRASSHOPPER,
    'S': pieces.SPIDER,
    'L': pieces.LADYBUG,
    'P': pieces.PILLBUG,
    'M': pieces.MOSQUITO
}

# Direction mapping for relative positioning based on doubled coordinates (double width)
# The keys represent the direction from the reference piece
# (0,0) is the reference piece
# (-2,0) is left, (2,0) is right
# (-1,-1) is top-left, (1,-1) is top-right
# (-1,1) is bottom-left, (1,1) is bottom-right
DIRECTION_TO_NOTATION = {
    (-2, 0): '-',   # Left
    (2, 0): '-',    # Right
    (-1, -1): '\\', # Top-left
    (1, -1): '/',   # Top-right
    (-1, 1): '/',   # Bottom-left
    (1, 1): '\\'    # Bottom-right
}

# Whether the direction indicator comes before or after the reference piece
DIRECTION_INDICATOR_BEFORE = {
    (-2, 0): True,   # Left: indicator before
    (2, 0): False,   # Right: indicator after
    (-1, -1): True,  # Top-left: indicator before
    (1, -1): False,  # Top-right: indicator after
    (-1, 1): True,   # Bottom-left: indicator before
    (1, 1): False    # Bottom-right: indicator after
}


@dataclass
class MoveString:
    """
    Class for parsing and generating BoardSpace move notation strings.
    
    Attributes:
        raw_string: The raw BoardSpace notation string
        piece_id: The piece identifier (e.g., 'wQ1')
        reference_piece_id: The reference piece identifier (e.g., 'bA1')
        direction_indicator: The direction indicator (-, /, or \)
        is_pass: Whether this move is a pass
        colour: The colour of the player whose turn it is (not necessarily the piece colour)
    """
    raw_string: str
    piece_id: Optional[str] = None
    reference_piece_id: Optional[str] = None
    direction_indicator: Optional[str] = None
    is_pass: bool = False
    colour: Optional[str] = None
    
    def __post_init__(self):
        """Parse the raw string into components if not already set."""
        if self.raw_string.lower() == 'pass' or self.raw_string.lower() == 'pass\n':
            self.is_pass = True
            return
            
        if not self.piece_id:
            self._parse_raw_string()
    
    def _parse_raw_string(self):
        """Parse the raw BoardSpace notation string into components."""
        parts = self.raw_string.strip().split()

        
        # First part is always the piece ID
        self.piece_id = parts[0]
        
        if len(parts) == 1:
            # First move of the game, just the piece ID
            return
        
        # Check for beetle moving onto another piece (no direction indicator)
        if len(parts) == 2 and not any(c in parts[1] for c in ['-', '/', '\\']):
            self.reference_piece_id = parts[1]
            return
            
        # Otherwise, we have a direction indicator and reference piece
        second_part = parts[1]
        
        # Find the direction indicator and reference piece
        if second_part.startswith('-'):
            self.direction_indicator = '-'
            self.reference_piece_id = second_part[1:]
        elif second_part.startswith('/'):
            self.direction_indicator = '/'
            self.reference_piece_id = second_part[1:]
        elif second_part.startswith('\\'):
            self.direction_indicator = '\\'
            self.reference_piece_id = second_part[1:]
        elif second_part.endswith('-'):
            self.direction_indicator = '-'
            self.reference_piece_id = second_part[:-1]
        elif second_part.endswith('/'):
            self.direction_indicator = '/'
            self.reference_piece_id = second_part[:-1]
        elif second_part.endswith('\\'):
            self.direction_indicator = '\\'
            self.reference_piece_id = second_part[:-1]
        else:
            raise ValueError(f"Invalid BoardSpace notation: {self.raw_string}")
    
    @classmethod
    def from_components(cls, piece_id: str, reference_piece_id: Optional[str] = None,
                       direction_indicator: Optional[str] = None, colour: Optional[str] = None):
        """Create a MoveString from components."""
        if reference_piece_id is None:
            # First move
            return cls(piece_id, colour=colour)
        
        if direction_indicator is None:
            # Beetle moving onto another piece
            return cls(f"{piece_id} {reference_piece_id}", colour=colour)
        
        # Determine if direction indicator should be before or after reference piece
        # For simplicity, we'll use the same logic as in the original code
        # In a real implementation, this would be determined based on the actual positions
        if direction_indicator == '-' and piece_id < reference_piece_id:
            # Left: indicator before
            return cls(f"{piece_id} {direction_indicator}{reference_piece_id}", colour=colour)
        else:
            # Right, Top-right, Bottom-right: indicator after
            return cls(f"{piece_id} {reference_piece_id}{direction_indicator}", colour=colour)
    
    @classmethod
    def pass_move(cls, colour: Optional[str] = None):
        """Create a pass move."""
        return cls("pass", is_pass=True, colour=colour)
    
    def __str__(self):
        """Return the raw string representation."""
        return self.raw_string


def get_piece_id(piece: Piece) -> str:
    """
    Get the BoardSpace piece identifier for a piece.
    
    Args:
        piece: The piece to get the identifier for
        
    Returns:
        str: The BoardSpace piece identifier (e.g., 'wQ1')
    """
    color = COLOR_TO_BOARDSPACE[piece.colour]
    piece_type = PIECE_TYPE_TO_LETTER.get(piece.name, 'P')  # Default to 'P' for generic pieces
    return f"{color}{piece_type}{piece.number}"


def find_piece_by_id(game: Game, piece_id: str) -> Optional[Tuple[Piece, Location]]:
    """
    Find a piece in the game by its BoardSpace identifier.
    
    Args:
        game: The game to search in
        piece_id: The BoardSpace piece identifier (e.g., 'wQ1')
        
    Returns:
        Optional[Tuple[Piece, Location]]: The piece and its location if found, None otherwise
    """
    color = BOARDSPACE_TO_COLOR[piece_id[0]]
    piece_type = LETTER_TO_PIECE_TYPE.get(piece_id[1])

    if len(piece_id) < 3:
        piece_num = 1
    else:
        piece_num = int(piece_id[2:])
    
    for loc, stack in game.grid.items():
        for piece in stack:
            if (piece.colour == color and 
                piece.name == piece_type and 
                piece.number == piece_num):
                return (piece, loc)
    
    return None


def calculate_relative_direction(from_loc: Location, to_loc: Location) -> Tuple[int, int]:
    """
    Calculate the relative direction from one location to another using doubled coordinates.
    
    Args:
        from_loc: The starting location
        to_loc: The ending location
        
    Returns:
        Tuple[int, int]: The relative direction as (q_diff, r_diff)
    """
    # Get all six adjacent positions
    adjacent_positions = positions_around_location(from_loc)
    
    # Find which adjacent position matches or is closest to to_loc
    if to_loc in adjacent_positions:
        # Direct match - return the exact direction
        return (to_loc[0] - from_loc[0], to_loc[1] - from_loc[1])
    
    # If no direct match, find the closest direction
    # This is a simplified approach - in a real implementation, you might need
    # more sophisticated logic to determine the best direction
    q_diff = to_loc[0] - from_loc[0]
    r_diff = to_loc[1] - from_loc[1]
    
    # Map to one of the six standard directions
    if abs(q_diff) >= abs(r_diff):
        # Primarily horizontal movement
        if q_diff > 0:
            return (2, 0)  # Right
        else:
            return (-2, 0)  # Left
    else:
        # Diagonal movement
        if q_diff >= 0 and r_diff < 0:
            return (1, -1)  # Top-right
        elif q_diff < 0 and r_diff < 0:
            return (-1, -1)  # Top-left
        elif q_diff >= 0 and r_diff > 0:
            return (1, 1)   # Bottom-right
        else:
            return (-1, 1)  # Bottom-left


def find_reference_piece(game: Game, target_loc: Location) -> Tuple[Optional[Tuple[Piece, Location]], Optional[str]]:
    """
    Find a suitable reference piece for a move to the target location.
    
    Args:
        game: The game state
        target_loc: The target location for the move
        
    Returns:
        Tuple[Optional[Tuple[Piece, Location]], Optional[str]]: The reference piece with its location and direction indicator
    """
    # Get all pieces adjacent to the target location
    adjacent_locations = positions_around_location(target_loc)
    
    # Find the first adjacent location that has a piece
    for adj_loc in adjacent_locations:
        stack = game.grid.get(adj_loc)
        if stack:
            # Use the top piece in the stack as the reference
            ref_piece = stack[-1]
            
            # Calculate the direction from the reference piece to the target location
            direction = calculate_relative_direction(adj_loc, target_loc)
            
            # Get the direction indicator
            indicator = DIRECTION_TO_NOTATION.get(direction)
            
            if indicator is None:
                # If we couldn't find a direct mapping, try to find the closest standard direction
                for std_dir, notation in DIRECTION_TO_NOTATION.items():
                    if (direction[0] * std_dir[0] > 0 and direction[1] * std_dir[1] >= 0) or \
                       (direction[1] * std_dir[1] > 0 and direction[0] * std_dir[0] >= 0):
                        indicator = notation
                        direction = std_dir
                        break
            
            return (ref_piece, adj_loc), indicator
    
    return None, None


def move_to_boardspace(game: Game, move: Union[Move, NoMove]) -> MoveString:
    """
    Convert an internal move to BoardSpace notation.
    
    Args:
        game: The current game state
        move: The move to convert
        
    Returns:
        MoveString: The BoardSpace notation for the move
    """
    # Get the colour of the player making the move
    move_colour = move.colour
    
    if isinstance(move, NoMove):
        move_str = MoveString.pass_move(colour=move_colour)
        # Store the color for later use
        move_str.pass_color = move_colour
        return move_str
    
    piece_id = get_piece_id(move.piece)
    
    # First move of the game
    if len(game.grid) == 0:
        return MoveString(piece_id, colour=move_colour)
    
    # Check if this is a beetle moving onto another piece
    if move.piece.name == pieces.BEETLE and game.grid.get(move.new_location):
        target_stack = game.grid.get(move.new_location)
        target_piece = target_stack[-1]
        target_piece_id = get_piece_id(target_piece)
        return MoveString(f"{piece_id} {target_piece_id}", colour=move_colour)
    
    # Find a reference piece and direction
    ref_piece_info, direction = find_reference_piece(game, move.new_location)
    
    if ref_piece_info is None:
        raise ValueError(f"Could not find a reference piece for move: {move}")
    
    ref_piece, ref_loc = ref_piece_info
    ref_piece_id = get_piece_id(ref_piece)
    
    # Determine if the direction indicator comes before or after the reference piece
    if DIRECTION_INDICATOR_BEFORE.get(calculate_relative_direction(ref_loc, move.new_location), False):
        return MoveString(f"{piece_id} {direction}{ref_piece_id}", colour=move_colour)
    else:
        return MoveString(f"{piece_id} {ref_piece_id}{direction}", colour=move_colour)


def boardspace_to_move(game: Game, move_str: MoveString) -> Union[Move, NoMove]:
    """
    Convert a BoardSpace notation move to an internal move.
    
    Args:
        game: The current game state
        move_str: The BoardSpace notation move
        
    Returns:
        Union[Move, NoMove]: The internal move representation
        
    Note:
        The color of the move is determined by the MoveString's colour attribute if available,
        otherwise by the current player's turn in replay_trajectory,
        not by the piece color. This is especially important for pillbug moves.
    """
    if move_str.is_pass:
        # For pass moves, use the MoveString's colour attribute if available
        if move_str.colour is not None:
            return NoMove(move_str.colour)
        # Otherwise, use pass_color if available
        elif hasattr(move_str, 'pass_color') and move_str.pass_color:
            return NoMove(move_str.pass_color)
        # As a last resort, determine the color of the player whose turn it is
        else:
            colors = list(game.player_turns.keys())
            current_color = min(colors, key=lambda c: game.player_turns[c])
            return NoMove(current_color)
    
    # Parse the piece ID
    piece_id = move_str.piece_id
    color = BOARDSPACE_TO_COLOR[piece_id[0]]
    piece_type = LETTER_TO_PIECE_TYPE.get(piece_id[1])

    if len(piece_id) < 3:
        piece_num = 1
    else:
        piece_num = int(piece_id[2:])
    
    # Create the piece
    piece = Piece(colour=color, name=piece_type, number=piece_num)
    
    # First move of the game
    if move_str.reference_piece_id is None:
        # Note: The actual color will be set in replay_trajectory based on player turn
        return Move(piece=piece, current_location=None, current_stack_idx=None, new_location=(0, 0), new_stack_idx=0)
    
    # Find the moving piece in the game
    piece_info = find_piece_by_id(game, piece_id)
    
    # If the piece doesn't exist in the game, it's a placement
    if piece_info is None:
        current_location = None
        current_stack_idx = None
    else:
        _, current_location = piece_info
        current_stack_idx = len(game.grid[current_location]) - 1
    
    # Beetle moving onto another piece
    if move_str.direction_indicator is None:
        ref_piece_info = find_piece_by_id(game, move_str.reference_piece_id)
        if ref_piece_info is None:
            raise ValueError(f"Reference piece not found: {move_str.reference_piece_id}")
        _, ref_loc = ref_piece_info
        # Use the MoveString's colour attribute if available
        move_colour = move_str.colour if move_str.colour is not None else None
        return Move(piece=piece,
                     current_location=current_location, current_stack_idx=current_stack_idx,
                     new_location=ref_loc, new_stack_idx=len(game.grid[ref_loc]),
                     colour=move_colour)
    
    # Find the reference piece
    ref_piece_info = find_piece_by_id(game, move_str.reference_piece_id)
    if ref_piece_info is None:
        raise ValueError(f"Reference piece not found: {move_str.reference_piece_id}")
    
    _, ref_loc = ref_piece_info
    
    # Determine the target location based on the reference piece and direction
    direction_indicator = move_str.direction_indicator
    
    # Map the direction indicator and its position to a target location
    if direction_indicator == '-':
        if move_str.raw_string.find('-') < move_str.raw_string.find(move_str.reference_piece_id):
            # Indicator before: Left
            target_loc = (ref_loc[0] - 2, ref_loc[1])
        else:
            # Indicator after: Right
            target_loc = (ref_loc[0] + 2, ref_loc[1])
    elif direction_indicator == '/':
        if move_str.raw_string.find('/') < move_str.raw_string.find(move_str.reference_piece_id):
            # Indicator before: Bottom-left
            target_loc = (ref_loc[0] - 1, ref_loc[1] + 1)
        else:
            # Indicator after: Top-right
            target_loc = (ref_loc[0] + 1, ref_loc[1] - 1)
    elif direction_indicator == '\\':
        if move_str.raw_string.find('\\') < move_str.raw_string.find(move_str.reference_piece_id):
            # Indicator before: Top-left
            target_loc = (ref_loc[0] - 1, ref_loc[1] - 1)
        else:
            # Indicator after: Bottom-right
            target_loc = (ref_loc[0] + 1, ref_loc[1] + 1)
    else:
        raise ValueError(f"Invalid direction indicator: {direction_indicator}")

    new_stack_height = len(game.grid.get(target_loc, []))

    # Use the MoveString's colour attribute if available
    move_colour = move_str.colour if move_str.colour is not None else None
    return Move(piece=piece,
                 current_location=current_location, current_stack_idx=current_stack_idx,
                 new_location=target_loc, new_stack_idx=new_stack_height,
                 colour=move_colour)


def save_trajectory(moves: List[MoveString], filename: str):
    """
    Save a list of moves to a file.
    
    Args:
        moves: The list of moves to save
        filename: The filename to save to
    """
    with open(filename, 'w') as f:
        for move in moves:
            f.write(f"{move.raw_string}\n")


def load_trajectory(filename: str) -> List[MoveString]:
    """
    Load a list of moves from a file.
    
    Args:
        filename: The filename to load from
        
    Returns:
        List[MoveString]: The list of moves
    """
    moves = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                moves.append(MoveString(line))
    return moves


def replay_trajectory(moves: List[MoveString], turn_info: Optional[str] = None) -> Game:
    """
    Replay a trajectory to get the final game state.
    
    Args:
        moves: The list of moves to replay
        turn_info: Optional turn information string (e.g., "Black[18]")
        
    Returns:
        Game: The final game state
    """
    from hive.game_engine.game_state import initial_game
    from hive.game_engine.moves import is_pillbug_move
    
    game = initial_game()
    
    # Track the current player's turn (WHITE starts in Hive)
    current_player = WHITE
    
    # Set the colour attribute for each move based on whose turn it is
    for i, move_str in enumerate(moves):
        # Set the colour attribute of the MoveString
        move_str.colour = current_player
        
        move = boardspace_to_move(game, move_str)
        
        # Check if this is likely a pillbug move based on piece color vs current player
        is_likely_pillbug_move = False
        if isinstance(move, Move) and move.current_location is not None:
            # If the piece color is different from the current player's color,
            # it's definitely a pillbug move
            if move.piece.colour != current_player:
                is_likely_pillbug_move = True
        
        # Set the move color based on the current player's turn, not the piece color
        # This is especially important for pillbug moves that can move opponent pieces
        if isinstance(move, Move):
            # If it's likely a pillbug move, set the flag
            if is_likely_pillbug_move:
                move = Move(
                    piece=move.piece,
                    current_location=move.current_location,
                    current_stack_idx=move.current_stack_idx,
                    new_location=move.new_location,
                    new_stack_idx=move.new_stack_idx,
                    colour=current_player,
                    pillbug_moved_other_piece=True
                )
            else:
                move.colour = current_player
        elif isinstance(move, NoMove):
            move.colour = current_player
            
        # Play the move
        game = move.play(game)
        
        # If we didn't already mark it as a pillbug move, check using the standard function
        if not is_likely_pillbug_move and isinstance(game.move, Move) and is_pillbug_move(game, game.move):
            # Create a new move with the pillbug_moved_other_piece flag set to True
            updated_move = Move(
                piece=game.move.piece,
                current_location=game.move.current_location,
                current_stack_idx=game.move.current_stack_idx,
                new_location=game.move.new_location,
                new_stack_idx=game.move.new_stack_idx,
                colour=game.move.colour,
                pillbug_moved_other_piece=True
            )
            # Update the game's move
            game = game.set("move", updated_move)
        
        # Switch to the other player's turn
        current_player = BLACK if current_player == WHITE else WHITE
    
    # If turn information is provided, set the current turn accordingly
    if turn_info:
        # Parse turn information (e.g., "Black[18]")
        import re
        match = re.match(r"(White|Black)\[(\d+)\]", turn_info)
        if match:
            turn_color = match.group(1)
            turn_number = int(match.group(2))
            
            # Set the current turn
            if turn_color == "White":
                game = game.set("current_turn", WHITE)
            elif turn_color == "Black":
                game = game.set("current_turn", BLACK)
    
    return game


