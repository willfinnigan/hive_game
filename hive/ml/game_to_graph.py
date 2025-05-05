from hive.game.game import Game
from torch_geometric.data import Data
import torch
from hive.game.grid_functions import positions_around_location, get_empty_locations
from hive.game.types_and_errors import WHITE, BLACK
from hive.game.pieces.queen import Queen
from hive.game.pieces.ant import Ant
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.spider import Spider


def game_to_graph(game: Game) -> Data:
    """From a game state, output a pyTorch Geometric graph
    
    This function converts a Hive game state into a PyTorch Geometric graph representation.
    
    Node features include:
    - One-hot encoding of piece type (Queen, Ant, Beetle, Grasshopper, Spider)
    - Color (White or Black)
    - Is piece stacked (sitting on another piece)
    - Stack level (how many pieces are below this one)
    - Is piece pinned (cannot move because it would break the hive)
    - Is empty space (1 for empty spaces, 0 for pieces)
    
    Edge types include:
    - 0: Adjacent (pieces next to each other)
    - 1: Stacked_On (piece is stacked on another)
    - 2: Covered_By (piece is covered by another)
    
    Args:
        game: The current game state
        
    Returns:
        Data: PyTorch Geometric Data object representing the game state
    """
    # Define piece type mapping for one-hot encoding
    piece_type_map = {
        Queen: 0,
        Ant: 1,
        Beetle: 2,
        GrassHopper: 3,
        Spider: 4
    }
    
    # Initialize lists to store node and edge information
    node_features = []
    edge_index = [[], []]  # [source_nodes, target_nodes]
    edge_attrs = []  # Edge types
    
    # Map from location to node index
    loc_to_idx = {}
    
    # Process each piece on the board
    for idx, (loc, piece) in enumerate(game.grid.items()):
        loc_to_idx[loc] = idx
        
        # Create one-hot encoding for piece type
        piece_type_one_hot = [0] * len(piece_type_map)
        piece_type_one_hot[piece_type_map.get(piece.__class__, 0)] = 1
        
        # Color feature (0 for white, 1 for black)
        color_feature = [1, 0] if piece.colour == WHITE else [0, 1]
        
        # Stacking features
        is_stacked = 1 if piece.sitting_on is not None else 0
        
        # Calculate stack level (how many pieces are below this one)
        stack_level = 0
        current_piece = piece
        while current_piece.sitting_on is not None:
            stack_level += 1
            current_piece = current_piece.sitting_on
        
        # Check if piece is pinned (cannot move)
        from hive.game.grid_functions import can_remove_piece
        is_pinned = 0 if can_remove_piece(game.grid, piece) else 1
        
        # Combine all features (including is_empty=0 for pieces)
        features = piece_type_one_hot + color_feature + [is_stacked, stack_level, is_pinned, 0]
        node_features.append(features)
        
        # Add edges for adjacent pieces
        for adj_loc in positions_around_location(loc):
            if adj_loc in game.grid:
                adj_idx = loc_to_idx.get(adj_loc)
                if adj_idx is not None:  # This check is needed because we're iterating through the grid
                    edge_index[0].append(idx)
                    edge_index[1].append(adj_idx)
                    edge_attrs.append(0)  # Edge type 0: Adjacent
        
        # Add edges for stacked pieces
        if piece.sitting_on is not None:
            # Find the index of the piece below
            for other_loc, other_piece in game.grid.items():
                if other_piece == piece.sitting_on:
                    other_idx = loc_to_idx[other_loc]
                    # Add edge from current piece to piece below
                    edge_index[0].append(idx)
                    edge_index[1].append(other_idx)
                    edge_attrs.append(1)  # Edge type 1: Stacked_On
                    
                    # Add edge from piece below to current piece
                    edge_index[0].append(other_idx)
                    edge_index[1].append(idx)
                    edge_attrs.append(2)  # Edge type 2: Covered_By
                    break
    
    # Get all empty locations adjacent to any piece
    empty_locations = get_empty_locations(game.grid)
    
    # Process each empty location
    for empty_loc in empty_locations:
        # Add empty location to the location-to-index map
        empty_idx = len(node_features)
        loc_to_idx[empty_loc] = empty_idx
        
        # Create features for empty space
        # All zeros for piece type, color, stacking, etc., but 1 for is_empty
        empty_features = [0] * len(piece_type_map) + [0, 0] + [0, 0, 0, 1]  # piece_type + color + stacking + is_empty
        node_features.append(empty_features)
        
        # Add edges between this empty location and adjacent pieces
        for adj_loc in positions_around_location(empty_loc):
            if adj_loc in game.grid:
                adj_idx = loc_to_idx[adj_loc]
                # Edge from empty space to piece
                edge_index[0].append(empty_idx)
                edge_index[1].append(adj_idx)
                edge_attrs.append(0)  # Edge type 0: Adjacent
                
                # Edge from piece to empty space
                edge_index[0].append(adj_idx)
                edge_index[1].append(empty_idx)
                edge_attrs.append(0)  # Edge type 0: Adjacent
    
    # Convert lists to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.long)
    
    # Create and return the Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attrs)
    return data