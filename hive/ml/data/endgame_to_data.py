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
        print(f'No valid moves found for expert move: {expert_move}')
        #raise ValueError("All labels are 0, no valid moves found.")

    return labels


def process_endgame(game: Game) -> List[Data]:
    """Taking a game in endgame state, return Data objects with move labels and winner information"""
    winner = get_winner(game)

    # Generate a list of games, and the move that was player (which will come from the game one step head)
    all_data = []
    while game.move is not None:
        '''
        game.parent is the game state before the move was made
        game.move is the move that made
        game is the resulting game state after the move
        '''

        # create a data object, and values for training
        graph = Graph(game.parent)
        data = graph_to_pytorch(graph)

        # Create move labels
        if game.move is None or isinstance(game.move, NoMove) == True:
            moves = []
        else:
            moves = create_move_labels(graph, game.move)
        
        # Determine winner value
        if winner == None:
            current_player_winner = 0
        elif winner == game.parent.current_turn:
            current_player_winner = 1
        else:
            current_player_winner = -1

        # Store move labels and winner directly in the Data object
        data.move_labels = torch.tensor(moves, dtype=torch.float)
        data.winner = torch.tensor(current_player_winner, dtype=torch.float)

        # append to the data list
        all_data.append(data)

        game = game.parent  # move to the previous game state

    # return data for training
    return all_data










if __name__ == '__main__':
    filepath = f"{Path(__file__).parents[3]}/game_strings/combined.txt"
    batch_size = 10
    loader = GameDataLoader(filepath, batch_size=batch_size)
    total_batches = (len(loader) + batch_size - 1) // batch_size

    # load first game
    game = loader.get_game(1)

