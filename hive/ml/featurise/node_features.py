

from typing import Callable, List, Optional

from hive.game_engine import pieces
from hive.game_engine.game_functions import get_queen_location, opposite_colour
from hive.game_engine.game_state import Colour, Grid, Location, Piece
from hive.game_engine.grid_functions import can_remove_piece
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

def is_played_piece(piece: Optional[Piece], loc: Location, i: int, colour: Colour, grid: Grid) -> List[float|int]:
    """Featurise whether the piece is played.  If the piece is None, then all values are 0 which represents an empty location"""
    # if no location, then its not a played piece
    if loc is None:
        return [0]
    elif piece is None:
        return [0]  # Not played piece
    else:
        return [1]  # Played piece

def is_empty_space(piece: Optional[Piece], loc: Location, i: int, colour: Colour, grid: Grid) -> List[float|int]:
    """Featurise whether the location is empty.  If the piece is None, then all values are 0 which represents an empty location"""
    # if no location, then its not an empty space
    if loc is None:
        return [0]
    elif piece is None:
        return [1]  # Empty space
    else:
        return [0]  # Not empty space

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

def is_piece_pinned(piece: Optional[Piece], loc: Location, i: int, colour: Colour, grid: Grid) -> List[float|int]:
    """Featurise whether the piece is pinned. """
    if piece is None:
        return [0]  # Not pinned
    elif loc is None:
        return [0]
    
    # is piece top of stack?
    stack = grid.get(loc, None)
    if stack is None or len(stack) == 0:
        print(f"Warning: When featurising is_piece_pinned, no stack at location {loc} for piece {piece}.")
        return [0]  # Not pinned if no stack
    
    if i != len(stack) - 1:
        return [1]  # is pinned if not top of stack
    
    can_remove = can_remove_piece(grid, loc)
    if can_remove == True:
        return [0]
    else:
        return [1]
    
    
def is_pass_node(piece: Optional[Piece], loc: Location, i: int, colour: Colour, grid: Grid) -> List[float|int]:
    """Featurise whether the node is a pass node.  
       If piece and loc both None, this is a pass node."""
    if piece is None and loc is None:
        return [1]  # If grid is None, then this is a pass node
    else:
        return [0]


all_node_feature_methods = [
    featurise_piece_type,
    featurise_piece_affiliation,
    is_piece_pinned,
    featurise_stack_height,
    is_played_piece,
    is_empty_space,
    is_pass_node
]

NUM_NODE_FEATS = 0
for method in all_node_feature_methods:
    NUM_NODE_FEATS += len(method(None, None, None, None, None))  # Use dummy values to get the length

FEATURE_LABELS = {featurise_piece_type: list(PIECE_TYPES.keys()),
                  featurise_piece_affiliation: list(AFFILIATION.keys()),
                  featurise_stack_height: ["stack_height"],
                  is_empty_space: ["is_empty_space"],
                  is_piece_pinned: ["is_piece_pinned"],
                  is_played_piece: ["is_played_piece"],
                  is_pass_node: ["is_pass_node"]}

def get_feature_label_vector(feature_methods: List[NodeFeatureMethod]) -> List[str]:
    """Given a list of feature methods, return a list of feature labels that will result from applying these methods."""

    feature_labels = []
    for method in feature_methods:
        if method in FEATURE_LABELS:
            feature_labels.extend(FEATURE_LABELS[method])
        else:
            raise ValueError(f"Feature method {method} not found in FEATURE_LABELS")
    return feature_labels

if __name__ == '__main__':
    print(f"Number of node features: {NUM_NODE_FEATS}")
    print(get_feature_label_vector(all_node_feature_methods))





