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

from hive.game.game import Game
from hive.game.types_and_errors import Location, Colour, WHITE, BLACK
from hive.game.pieces.piece_base_class import Piece
from hive.game.pieces.ant import Ant
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.queen import Queen
from hive.game.pieces.spider import Spider
from hive.play.move import Move, NoMove


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
    Queen: 'Q',
    Ant: 'A',
    Beetle: 'B',
    GrassHopper: 'G',
    Spider: 'S'
}

LETTER_TO_PIECE_TYPE = {
    'Q': Queen,
    'A': Ant,
    'B': Beetle,
    'G': GrassHopper,
    'S': Spider
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
    """
    raw_string: str
    piece_id: Optional[str] = None
    reference_piece_id: Optional[str] = None
    direction_indicator: Optional[str] = None
    is_pass: bool = False
    
    def __post_init__(self):
        """Parse the raw string into components if not already set."""
        if self.raw_string.lower() == 'pass':
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
                       direction_indicator: Optional[str] = None):
        """Create a MoveString from components."""
        if reference_piece_id is None:
            # First move
            return cls(piece_id)
        
        if direction_indicator is None:
            # Beetle moving onto another piece
            return cls(f"{piece_id} {reference_piece_id}")
        
        # Check if direction indicator should be before or after reference piece
        color = piece_id[0]
        piece_type = piece_id[1]
        piece_num = piece_id[2:]
        
        ref_color = reference_piece_id[0]
        ref_piece_type = reference_piece_id[1]
        ref_piece_num = reference_piece_id[2:]
        
        # Determine the relative direction based on the game state
        # This is a placeholder - in a real implementation, you would need to 
        # calculate this based on the actual positions of the pieces
        # For now, we'll just put the indicator after the reference piece
        if direction_indicator == '-' and piece_id < reference_piece_id:
            # Left: indicator before
            return cls(f"{piece_id} {direction_indicator}{reference_piece_id}")
        else:
            # Right, Top-right, Bottom-right: indicator after
            return cls(f"{piece_id} {reference_piece_id}{direction_indicator}")
    
    @classmethod
    def pass_move(cls):
        """Create a pass move."""
        return cls("pass", is_pass=True)
    
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
    piece_type = PIECE_TYPE_TO_LETTER.get(piece.__class__, 'P')  # Default to 'P' for generic pieces
    return f"{color}{piece_type}{piece.number}"


def find_piece_by_id(game: Game, piece_id: str) -> Optional[Piece]:
    """
    Find a piece in the game by its BoardSpace identifier.
    
    Args:
        game: The game to search in
        piece_id: The BoardSpace piece identifier (e.g., 'wQ1')
        
    Returns:
        Optional[Piece]: The piece if found, None otherwise
    """
    color = BOARDSPACE_TO_COLOR[piece_id[0]]
    piece_type = LETTER_TO_PIECE_TYPE.get(piece_id[1])
    piece_num = int(piece_id[2:])
    
    for loc, piece in game.grid.items():
        if (piece.colour == color and 
            piece.__class__ == piece_type and 
            piece.number == piece_num):
            return piece
    
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
    # Get the positions around the from_loc using the game's grid function
    from hive.game.grid_functions import positions_around_location
    
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


def find_reference_piece(game: Game, target_loc: Location) -> Tuple[Optional[Piece], Optional[str]]:
    """
    Find a suitable reference piece for a move to the target location.
    
    Args:
        game: The game state
        target_loc: The target location for the move
        
    Returns:
        Tuple[Optional[Piece], Optional[str]]: The reference piece and direction indicator
    """
    # Get all pieces adjacent to the target location
    from hive.game.grid_functions import pieces_around_location
    adjacent_pieces = pieces_around_location(game.grid, target_loc)
    
    if not adjacent_pieces:
        return None, None
    
    # Choose the first adjacent piece as the reference
    ref_piece = adjacent_pieces[0]
    
    # Calculate the direction from the reference piece to the target location
    direction = calculate_relative_direction(ref_piece.location, target_loc)
    
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
    
    return ref_piece, indicator


def move_to_boardspace(game: Game, move: Union[Move, NoMove]) -> MoveString:
    """
    Convert an internal move to BoardSpace notation.
    
    Args:
        game: The current game state
        move: The move to convert
        
    Returns:
        MoveString: The BoardSpace notation for the move
    """
    if isinstance(move, NoMove):
        return MoveString.pass_move()
    
    piece_id = get_piece_id(move.piece)
    
    # First move of the game or first move for a player
    if len(game.grid) == 0 or (move.place and move.piece.location is None):
        # Apply the move to update the game state temporarily
        original_location = move.piece.location
        game.grid[move.location] = move.piece
        move.piece.location = move.location
        
        # If this is the very first move of the game
        if len(game.grid) == 1:
            # Revert the game state
            game.grid.pop(move.location)
            move.piece.location = original_location
            return MoveString(piece_id)
        
        # Find a reference piece now that we've temporarily placed the piece
        ref_piece, direction = find_reference_piece(game, move.location)
        
        # Revert the game state
        game.grid.pop(move.location)
        move.piece.location = original_location
        
        if ref_piece is None:
            # This should not happen since we've placed the piece
            return MoveString(piece_id)
        
        ref_piece_id = get_piece_id(ref_piece)
        
        # Determine if the direction indicator comes before or after the reference piece
        if DIRECTION_INDICATOR_BEFORE.get(calculate_relative_direction(ref_piece.location, move.location), False):
            return MoveString(f"{piece_id} {direction}{ref_piece_id}")
        else:
            return MoveString(f"{piece_id} {ref_piece_id}{direction}")
    
    # Check if this is a beetle moving onto another piece
    if isinstance(move.piece, Beetle) and game.get_piece(move.location) is not None:
        target_piece = game.get_piece(move.location)
        target_piece_id = get_piece_id(target_piece)
        return MoveString(f"{piece_id} {target_piece_id}")
    
    # Find a reference piece and direction
    ref_piece, direction = find_reference_piece(game, move.location)
    
    if ref_piece is None:
        raise ValueError(f"Could not find a reference piece for move: {move}")
    
    ref_piece_id = get_piece_id(ref_piece)
    
    # Determine if the direction indicator comes before or after the reference piece
    if DIRECTION_INDICATOR_BEFORE.get(calculate_relative_direction(ref_piece.location, move.location), False):
        return MoveString(f"{piece_id} {direction}{ref_piece_id}")
    else:
        return MoveString(f"{piece_id} {ref_piece_id}{direction}")


def boardspace_to_move(game: Game, move_str: MoveString) -> Union[Move, NoMove]:
    """
    Convert a BoardSpace notation move to an internal move.
    
    Args:
        game: The current game state
        move_str: The BoardSpace notation move
        
    Returns:
        Union[Move, NoMove]: The internal move representation
    """
    if move_str.is_pass:
        # Determine the color of the player whose turn it is
        colors = list(game.player_turns.keys())
        current_color = min(colors, key=lambda c: game.player_turns[c])
        return NoMove(current_color)
    
    # Parse the piece ID
    piece_id = move_str.piece_id
    color = BOARDSPACE_TO_COLOR[piece_id[0]]
    piece_type = LETTER_TO_PIECE_TYPE.get(piece_id[1])
    piece_num = int(piece_id[2:])
    
    # First move of the game
    if move_str.reference_piece_id is None:
        # Create a new piece
        piece = piece_type(color, piece_num)
        return Move(piece=piece, location=(0, 0), place=True)
    
    # Find the moving piece
    piece = find_piece_by_id(game, piece_id)
    
    # If the piece doesn't exist, create it
    if piece is None:
        piece = piece_type(color, piece_num)
        place = True
    else:
        place = False
    
    # Beetle moving onto another piece
    if move_str.direction_indicator is None:
        ref_piece = find_piece_by_id(game, move_str.reference_piece_id)
        if ref_piece is None:
            raise ValueError(f"Reference piece not found: {move_str.reference_piece_id}")
        return Move(piece=piece, location=ref_piece.location, place=place)
    
    # Find the reference piece
    ref_piece = find_piece_by_id(game, move_str.reference_piece_id)
    if ref_piece is None:
        raise ValueError(f"Reference piece not found: {move_str.reference_piece_id}")
    
    # Determine the target location based on the reference piece and direction
    ref_loc = ref_piece.location
    
    # Determine the direction based on the indicator and whether it's before or after
    direction_indicator = move_str.direction_indicator
    
    # Use the positions_around_location function to get the exact adjacent positions
    from hive.game.grid_functions import positions_around_location
    adjacent_positions = positions_around_location(ref_loc)
    
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
    
    return Move(piece=piece, location=target_loc, place=place)


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


def replay_trajectory(moves: List[MoveString]) -> Game:
    """
    Replay a trajectory to get the final game state.
    
    Args:
        moves: The list of moves to replay
        
    Returns:
        Game: The final game state
    """
    game = Game()
    
    for move_str in moves:
        move = boardspace_to_move(game, move_str)
        move.play(game)
    
    return game


def record_game(game_controller, filename: str):
    """
    Record a game as it's being played and save the trajectory.
    
    Args:
        game_controller: The game controller
        filename: The filename to save to
    """
    original_play = game_controller.play
    moves = []
    
    def record_play():
        while game_controller.game.get_winner() is None:
            player = game_controller.get_next_player()
            move = player.get_move(game_controller.game)
            print(f"Turn {game_controller.game.player_turns[player.colour]}: {player.colour} - {move}")
            
            # Convert the move to BoardSpace notation and record it
            move_str = move_to_boardspace(game_controller.game, move)
            moves.append(move_str)
            print(f"BoardSpace notation: {move_str.raw_string}")
            
            move.play(game_controller.game)
            from hive.render.text.small_print import game_to_text
            print(game_to_text(game_controller.game))
        
        winner = game_controller.game.get_winner()
        print(f"{winner} wins!")
        
        # Save the trajectory
        save_trajectory(moves, filename)
        print(f"Game trajectory saved to {filename}")
        
        return winner
    
    game_controller.play = record_play
    return game_controller