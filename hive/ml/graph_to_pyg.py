

import torch
from torch_geometric.data import Data

from hive.game_engine.game_functions import current_turn_colour
from hive.game_engine.game_state import Game, Piece, initial_game
from hive.ml.game_to_graph import Graph


def graph_to_pytorch(graph: Graph) -> Data:
    """
    Convert a graph representation to a PyTorch Geometric Data object.
    """
    # Create edge indices and features
    node_features = []
    edge_indices = []
    edge_features = []

    node_ids = [node.node_id for node in graph.nodes]

    # loop through nodes pulling out nodes and edges
    for i, node in enumerate(graph.nodes):
        # add node features
        node_features.append(node.node_features)

        # add edge idx and features
        for j, n_node in enumerate(node.edges):
            n_node_idx = node_ids.index(n_node.node_id)
            edge_indices.append([i, n_node_idx])
            edge_features.append(node.edge_features[j])

    move_edge_idxs = []
    for i, edge_feats in enumerate(edge_features):

        # if the edge is a candidate move, add it to the list
        if edge_feats[3] == 1:
            # if edge source node is the current player, add it
            if graph.nodes[edge_indices[i][0]].piece.colour == graph.current_colour:
                move_edge_idxs.append(i)


    # Create the Data object
    data = Data(x=torch.tensor(node_features, dtype=torch.float), 
                edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), 
                edge_attr=torch.tensor(edge_features, dtype=torch.float),
                candidate_moves=torch.tensor(edge_indices, dtype=torch.long)[move_edge_idxs])
    
    return data

def game_to_pytorch(game: Game) -> Data:
    """
    Convert a game representation to a PyTorch Geometric Data object.
    """
    # get the current player turn
    graph = Graph(game)
    data = graph_to_pytorch(graph)
    return data

if __name__ == '__main__':
    # Example usage
    grid = {(0, 0): (Piece(colour="WHITE", name="ANT", number=1), 
                     Piece(colour="WHITE", name="BEETLE", number=1),
                     Piece(colour="BLACK", name="BEETLE", number=1)), 
            (1, 1): (Piece(colour="WHITE", name="QUEEN", number=1),),
            (-1, -1): (Piece(colour="WHITE", name="SPIDER", number=1),),}
    game = initial_game(grid=grid)
    
    graph = Graph(game)
    data = graph_to_pytorch(graph)

    print(data)