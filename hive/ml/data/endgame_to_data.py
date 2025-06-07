from pathlib import Path
import torch
from torch_geometric.data import Data
import gc
import os

from typing import List
from hive.game_engine.game_functions import get_winner
from hive.game_engine.game_state import Game
from hive.game_engine.moves import NoMove
from hive.ml.featurise.game_to_graph import Graph
from hive.ml.featurise.graph_to_pyg import game_to_pytorch, graph_to_pytorch
from hive.trajectory.game_dataloader import GameDataLoader


def create_move_labels(graph: Graph, expert_move, game: Game):
    """
    Create a one-hot encoded label vector for the expert move.
    
    Args:
        graph: The Graph object containing nodes and edges
        expert_move: The move that was actually played in the expert game
        
    Returns:
        A tensor of shape [num_candidate_moves] where the index 
        corresponding to the expert move is 1 and all others are 0
    """
    # Initialize label tensor with zeros
    labels = []

    for mv in graph.edge_moves:
        if mv == hash(expert_move):
            labels.append(1)
        else:
            labels.append(0)

    if expert_move is None:
        return labels

    # are all the labels 0?
    # if move is Pass, length of moves should be 0
    if isinstance(expert_move, NoMove) == True and len(graph.edge_moves) != 0:
        # print()
        # print(f"Expert move is a pass but there are {len(graph.edge_moves)} moves available")
        # print(dict(game.grid))
        # print()
        pass
    elif all(label == 0 for label in labels):
        # print()
        # print(f'No valid moves found for expert move: {expert_move} in {len(graph.edge_moves)} available moves')
        # print(dict(game.grid))
        # print()
        pass
        # raise ValueError("All labels are 0, no valid moves found.")

    return labels


def add_move_y_labels(data: Data, graph: Graph, expert_move, game: Game):
    # Create move labels
    moves = create_move_labels(graph, expert_move, game)

    # Store move labels and winner directly in the Data object
    data.move_labels = torch.tensor(moves, dtype=torch.float)
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
    return data


def process_endgame(game: Game,
                    include_moves=True,
                    include_value=True,
                    value_discount=0.5,
                    skip_initial=8) -> List[Data]:
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
    
    # Clean up temporary game reference
    del tmp_game
    
    steps_to_stop_at = total_length - skip_initial

    # Process terminal state (no moves)
    graph = Graph(game)
    data = graph_to_pytorch(graph)
    
    if include_moves:
        # No move was made in the terminal state
        data = add_move_y_labels(data, graph, None, game)
    if include_value:
        # The value is likelihood of winning for the current player
        data = add_value_y_label(data, game, winner, 0, total_length, value_discount)

    all_data.append(data)  # append to the data list

    # Clean up graph after use
    del graph
    
    # Force MPS/GPU memory cleanup if using MPS
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

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

        all_data.append(data)  # append to the data list
        
        # Clean up graph after use
        del graph
        
        # Move to the previous game state
        next_parent = parent_game.parent
        current_game = parent_game

        # Clean up the current game reference
        try:
            del parent_game.move
        except:
            pass

        try:
            del parent_game.grid
        except:
            pass

        del parent_game

        steps += 1

        # Periodically force garbage collection (less frequent to reduce overhead)
        if steps % 20 == 0:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        if steps >= steps_to_stop_at:
            break

    # Clean up remaining references
    del current_game
    del game
    
    # Force final garbage collection
    gc.collect()
    
    # Force MPS/GPU memory cleanup if using MPS
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

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
        print(f"  Move labels: {data.move_labels}")
        print(f"  Value: {data.value}")
        print(f"  Game progress: {data.game_progress}")
        print(f"  Number of nodes: {data.num_nodes}")
        print(f"  Edge index shape: {data.edge_index.shape}")
        print(f"  Edge attributes shape: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
        print()
