

from typing import Callable, List, Optional

from hive.game_engine import pieces
from hive.game_engine.game_functions import get_queen_location, opposite_colour
from hive.game_engine.game_state import Colour, Grid, Location, Piece
from hive.game_engine.moves import get_possible_moves

NodeFeatureMethod = Callable[[Optional[Piece], Location, int, Colour, Grid], List[float|int]]

# One hot encoding for piece types
# eg [0, 1, 0, 0, 0, 0, 0, 0]
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
    if piece is None:
        return [0]
    else:
        return [i+1]  # Stack height is the index of the piece in the stack


all_node_feature_methods = [
    featurise_piece_type,
    featurise_piece_affiliation,
    featurise_stack_height,
]






