from pathlib import Path
import torch
from torch_geometric.data import Data


from typing import List
from hive.game_engine.game_functions import get_winner
from hive.game_engine.game_state import Game
from hive.game_engine.moves import NoMove
from hive.ml.featurise.game_to_graph import Graph
from hive.ml.featurise.graph_to_pyg import game_to_pytorch, graph_to_pytorch
from hive.trajectory.game_dataloader import GameDataLoader


def create_move_labels(graph: Graph, expert_move):
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
        if mv == expert_move:
            labels.append(1)
        else:
            labels.append(0)

    # are all the labels 0?
    if all(label == 0 for label in labels):
        pass
        #print(f'No valid moves found for expert move: {expert_move}')
        #raise ValueError("All labels are 0, no valid moves found.")

    return labels


def add_move_y_labels(data: Data, graph: Graph, expert_move):

    # Create move labels
    moves = create_move_labels(graph, expert_move)

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

    # Store value label in the Data object
    data.value = torch.tensor(value, dtype=torch.float)
    return data



def process_endgame(game: Game,
                    include_moves=True,
                    include_value=True,
                    value_discount=0.5) -> List[Data]:
    """Taking a game in endgame state, return Data objects with move labels and winner information"""
    winner = get_winner(game)

    # Generate a list of games, and the move that was player (which will come from the game one step head)
    all_data = []

    total_length = 0
    tmp_game = game
    while tmp_game.move is not None:
        total_length += 1
        tmp_game = tmp_game.parent

    # add terminal state (no moves)
    graph = Graph(game)
    data = graph_to_pytorch(graph)
    if include_moves == True:
        # No move was made in the terminal state
        data = add_move_y_labels(data, graph, None)
    if include_value == True:
        # The value is liklihood of winning for the current player
        data = add_value_y_label(data, game, winner, 0, total_length, value_discount)

    all_data.append(data)  # append to the data list

    # walk through game states
    steps = 0
    while game.move is not None:
        '''
        game.parent is the game state before the move was made
        game.move is the move that made
        game is the resulting game state after the move
        '''

        # create a data object, and values for training
        # We create a data object for the game state before the move was made
        graph = Graph(game.parent)
        data = graph_to_pytorch(graph)

        # add labels
        if include_moves == True:
            # Which move was made in this game state for the current player
            move = game.move  # this is the move that was made *from* game.parent *to* game
            data = add_move_y_labels(data, graph, move)
        if include_value == True:
            # The value is liklihood of winning for the current player
            data = add_value_y_label(data, game.parent, winner, steps, total_length, value_discount)

        all_data.append(data)  # append to the data list
        game = game.parent  # move to the previous game state
        steps += 1

    # return data for training
    return all_data








if __name__ == '__main__':
    filepath = f"{Path(__file__).parents[3]}/game_strings/combined.txt"
    batch_size = 10
    loader = GameDataLoader(filepath, batch_size=batch_size)
    total_batches = (len(loader) + batch_size - 1) // batch_size

    # load first game
    game = loader.get_game(100)
    print(f"Game loaded with {len(game)} moves")

    # process the endgame
    all_data = process_endgame(game, include_moves=True, include_value=True)

    print(f"Processed {len(all_data)} data objects from the endgame")
    for i, data in enumerate(all_data):
        print(f"Data object {i}:")
        print(f"  Move labels: {data.move_labels}")
        print(f"  Value: {data.value}")
        print(f"  Number of nodes: {data.num_nodes}")
        print(f"  Edge index shape: {data.edge_index.shape}")
        print(f"  Edge attributes shape: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
        print()
