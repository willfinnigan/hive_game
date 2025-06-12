from pathlib import Path
import torch
from torch_geometric.data import Data
import gc
import os

from typing import List
from hive.game_engine.game_functions import get_winner
from hive.game_engine.game_state import Game
from hive.game_engine.moves import NoMove
from hive.game_engine.player_functions import get_players_possible_moves_or_placements
from hive.ml.featurise.game_to_graph import Graph
from hive.ml.featurise.graph_to_pyg import game_to_pytorch, graph_to_pytorch
from hive.trajectory.game_dataloader import GameDataLoader


def get_move_index(graph: Graph, expert_move, game: Game) -> int:
    """
    Create a one-hot encoded label vector for the expert move.
    
    Args:
        graph: The Graph object containing nodes and edges
        expert_move: The move that was actually played in the expert game
        
    Returns:
        A tensor of shape [num_candidate_moves] where the index 
        corresponding to the expert move is 1 and all others are 0
    """

    
    # If pass, then select the first move (which is the pass edge)
    if isinstance(expert_move, NoMove) is True:
        if len(graph.edge_moves) != 1:
            print()
            print(f"Expert move is a pass but there are {len(graph.edge_moves)} moves available")
            print(dict(game.grid))
            print()
        return 0  # Pass is always the first move in the list

    for i, mv in enumerate(graph.edge_moves):
        if mv == hash(expert_move):
            return i

    print()
    print(f'Expert move {expert_move} not found in available moves: {graph.edge_moves}')
    print(dict(game.grid))
    print()
    print("Returning a -100 move which will be skipped")
    return -100


def add_move_y_labels(data: Data, graph: Graph, expert_move, game: Game):
    move_index = get_move_index(graph, expert_move, game)

    if move_index is not None:
        data.policy = torch.tensor(move_index, dtype=torch.long)
    else:
        data.policy = torch.tensor(-100, dtype=torch.long)

    num_moves = len(graph.edge_moves)
    data.move_batch_idx = torch.full((num_moves,), fill_value=0, dtype=torch.long)

    # can now drop data.edge_moves as it is not needed anymore
    data.edge_moves = None

    '''The magic happens in the collate function of the PyG DataLoader. 
    When you create a batch from a list of Data objects, the DataLoader does something special for attributes 
    that end in _idx or _index (and for the batch attribute specifically).'''

    return data


def add_value_y_label(data: Data, game: Game, winner, step: int, total_length: int, value_discount):
    # Determine winner value
    if winner == None:
        value = 0
    elif winner == game.current_turn:
        value = 1
    else:
        value = -1

    # apply discount
    if step > 0:
        game_progress = (step / total_length)
        value = value * (value_discount ** game_progress)
    else:
        game_progress = 0

    # Store value label in the Data object
    data.value = torch.tensor(value, dtype=torch.float)
    data.game_progress = 1 - game_progress
    
    # store game step but starting from 0
    data.step = total_length - step

    return data

def add_mobile_pieces_count(data: Data, game: Game, graph: Graph):
    """Get the number of mobile pieces for the current player"""

    moves = get_players_possible_moves_or_placements(game.current_turn, game)

    # only count moves that are not placements, and only count 1 move per piece
    seen_pieces = set()
    count = 0
    for move in moves:
        if isinstance(move, NoMove):
            continue

        if move.current_location is not None and move.piece.name not in seen_pieces:
            seen_pieces.add(move.piece.name)
            count += 1
    
    data.mobile_pieces = torch.tensor(count, dtype=torch.float)
    return data



def add_auxilary_labels(data: Data, game: Game, graph: Graph):
    """Add auxiliary labels to the Data object
    I mostly want to use these for debugging and analysis, 
    can the GNN predict simple things effectively?
    """
    data = add_mobile_pieces_count(data, game, graph)
    # First, does edge move go next to enemy queen?
    return data


def process_endgame(game: Game,
                    include_moves=True,
                    include_value=True,
                    include_auxiliary=True,
                    value_discount=0.75) -> List[Data]:
    """Taking a game in endgame state, return Data objects with move labels and winner information"""
    winner = get_winner(game)

    # Generate a list of games, and the move that was player (which will come from the game one step head)
    all_data = []

    # Calculate total length
    total_length = 0
    tmp_game = game
    while tmp_game.move is not None:
        total_length += 1
        tmp_game = tmp_game.parent
    
    '''Not processing the terminal state because there is no move made'''
    # Process terminal state (no moves)
    # graph = Graph(game)
    # data = graph_to_pytorch(graph)
    
    # if include_moves:
    #     # No move was made in the terminal state
    #     data = add_move_y_labels(data, graph, None, game)
    # if include_value:
    #     # The value is likelihood of winning for the current player
    #     data = add_value_y_label(data, game, winner, 0, total_length, value_discount)

    # all_data.append(data)  # append to the data list

    # Walk through game states
    steps = 0
    current_game = game  # Use a separate variable to avoid modifying the original reference
    
    while current_game.move is not None:
        # Create a data object for the game state before the move was made
        parent_game = current_game.parent
        graph = Graph(parent_game)
        data = graph_to_pytorch(graph)

        # Add labels
        if include_moves:
            # Which move was made in this game state for the current player
            move = current_game.move  # this is the move that was made *from* parent_game *to* current_game
            data = add_move_y_labels(data, graph, move, game)
        if include_value:
            # The value is likelihood of winning for the current player
            data = add_value_y_label(data, parent_game, winner, steps, total_length, value_discount)
        
        if include_auxiliary:
            # Add auxiliary labels
            data = add_auxilary_labels(data, parent_game, graph)

        all_data.append(data)  # append to the data list
        

        # Move to the previous game state
        next_parent = parent_game.parent
        current_game = parent_game
        steps += 1

    # remove any data objects with -100 move index
    all_data = [data for data in all_data if data.policy.item() != -100]

    # Return data for training
    return all_data


if __name__ == '__main__':
    filepath = f"{Path(__file__).parents[3]}/game_strings/combined.txt"
    batch_size = 10
    loader = GameDataLoader(filepath, batch_size=batch_size)
    total_batches = (len(loader) + batch_size - 1) // batch_size

    # load first game
    game = loader.get_game(10)

    # process the endgame
    all_data = process_endgame(game, include_moves=True, include_value=True)

    print(f"Processed {len(all_data)} data objects from the endgame")
    for i, data in enumerate(all_data):
        print(f"Data object {i}:")
        print(f"  Move idx: {data.policy}")
        print(f"  Mobile piece count: {data.mobile_pieces}")
        print(f"  Step: {data.step}")
        print(f"  Batch index: {data.move_batch_idx}")
        print(f"  Value: {data.value}")
        print(f"  Game progress: {data.game_progress}")
        print(f"  Number of nodes: {data.num_nodes}")
        print(f"  Edge index shape: {data.edge_index.shape}")
        print(f"  Edge attributes shape: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")

        # number of current turn nodes - feature number 8 is current turn
        current_turn_nodes = data.x[data.x[:, 8] == 1]
        print(f"  Number of current turn nodes: {current_turn_nodes.shape[0]}")

        # number of opponent turn nodes - feature number 9 is current turn
        opponent_turn_nodes = data.x[data.x[:, 9] == 1]
        print(f"  Number of opponent turn nodes: {opponent_turn_nodes.shape[0]}")
        print()
        


