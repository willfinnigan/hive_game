

from typing import Callable, List, Optional

from hive.game_engine import pieces
from hive.game_engine.game_functions import get_queen_location, opposite_colour
from hive.game_engine.game_state import Colour, Grid, Location, Piece
from hive.game_engine.moves import get_possible_moves

NodeFeatureMethod = Callable[[Optional[Piece], Location, int, Colour, Grid], List[float|int]]

# One hot encoding for piece types
# eg [0, 1, 0, 0, 0]
PIECE_TYPES = {pieces.QUEEN: 0, 
               pieces.ANT: 1, 
               pieces.SPIDER: 2, 
               pieces.BEETLE: 3, 
               pieces.GRASSHOPPER: 4,
               pieces.PILLBUG: 5,
               pieces.MOSQUITO: 6,
               pieces.LADYBUG: 7}

# One hot encoding for piece colours
AFFILIATION = {"CURRENT_PLAYER": 0, "OPPONENT_PLAYER": 1}

def featurise_piece_type(piece: Optional[Piece], loc: Location, i: int, colour: Colour, grid: Grid) -> List[float|int]:
    """Featurise the piece type.  If the piece is None, then all values are 0 which represents an empty location"""
    one_hot = [0] * len(PIECE_TYPES)
    if piece is not None:
        one_hot[PIECE_TYPES[piece.name]] = 1
    return one_hot

def featurise_piece_affiliation(piece: Optional[Piece], loc: Location, i: int, colour: Colour, grid: Grid) -> List[float|int]:
    """Featurise the piece colour.  If the piece is None, then all values are 0 which represents an empty location"""
    one_hot = [0] * len(AFFILIATION)
    if piece is None:
        return one_hot
    
    if piece.colour == colour:
        one_hot[AFFILIATION["CURRENT_PLAYER"]] = 1
    else:
        one_hot[AFFILIATION["OPPONENT_PLAYER"]] = 1
    return one_hot

def featurise_stack_height(piece: Optional[Piece], loc: Location, i: int, colour: Colour, grid: Grid) -> List[float|int]:
    """Featurise the stack height.  If the piece is None, then all values are 0 which represents an empty location"""
    return [i+1]  # Stack height is the index of the piece in the stack

def featurise_moves(piece: Optional[Piece], loc: Location, i: int, colour: Colour, grid: Grid) -> List[float|int]:
    """Featurise moves
    - has legal moves (0 or 1)
    - can piece move to opponent queen (0 or 1)
    - can piece move to our queen (0 or 1)
    """
    # if is not at the top of the stack, then no moves
    # check if piece is at the top of the stack by looking at the grid
    stack = grid.get(loc)
    if stack is None or len(stack) == 0 or stack[-1] != piece:
        return [0, 0, 0]

    legal_moves = get_possible_moves(grid, loc, len(stack)-1)  # returns a list of locations
    has_legal_moves = 1 if len(legal_moves) > 0 else 0

    enemy_colour = opposite_colour(colour)
    current_player_queen_loc = get_queen_location(grid, colour)
    opponent_player_queen_loc = get_queen_location(grid, enemy_colour)

    can_move_to_our_queen = 1 if current_player_queen_loc in legal_moves else 0
    can_move_to_opponent_queen = 1 if opponent_player_queen_loc in legal_moves else 0

    return [has_legal_moves, can_move_to_our_queen, can_move_to_opponent_queen]


all_node_feature_methods = [
    featurise_piece_type,
    featurise_piece_affiliation,
    featurise_stack_height,
    featurise_moves
]






