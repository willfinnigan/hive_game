import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch

from hive.game_engine.game_state import Piece, initial_game
from hive.ml.game_to_graph import Graph
from hive.ml.graph_to_pyg import graph_to_pytorch



class Hive_GNN_Model(nn.Module):
    """
    Graph Neural Network model for the board game Hive using GATv2 layers.
    
    This model predicts:
    1. Valid moves (edges) based on the current board state
    2. A value estimation of the current board position
    
    Each node in the graph represents a piece on the board, with features encoding:
    - Piece type (Queen Bee, Spider, Beetle, Grasshopper, Ant)
    - Owner (player 1 or player 2)
    - Position features
    """
    
    def __init__(self, 
                 node_features=14,        # Number of features per piece/node
                 edge_features=5,         # Number of features per edge (used by GAT layers)
                 hidden_dim=64,           # Hidden dimension size
                 num_gat_layers=3,        # Number of GAT layers
                 heads=4,                 # Number of attention heads
                 dropout=0.1,             # Dropout rate
                 alpha=0.2):              # LeakyReLU negative slope
        super(Hive_GNN_Model, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # GATv2 layers for node embedding
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(GATv2Conv(
            in_channels=node_features,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_features, # GAT layers still use edge_features for existing graph edges
            concat=True
        ))
        
        # Middle layers
        for _ in range(num_gat_layers - 2):
            self.gat_layers.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_features,
                concat=True
            ))
        
        # Last GAT layer
        self.gat_layers.append(GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim, # Output hidden_dim per head, then averaged
            heads=heads,
            dropout=dropout,
            edge_dim=edge_features,
            concat=False  # Average the attention heads
        ))
        
        # Edge prediction layers (move prediction)
        # Input: Concatenation of two node embeddings (hidden_dim each)
        # NO candidate_edge_attr used here.
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), # Adjusted input size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Value head (board evaluation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Value between -1 and 1
        )
        
    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 1. Node embeddings through GAT layers
        for layer in self.gat_layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x) 

        node_embeddings = x # Shape: [num_nodes, hidden_dim]

        # 2. Edge prediction (move prediction)
        # data.candidate_moves: Tensor of shape [num_candidate_moves, 2]
        
        source_node_indices = data.candidate_moves[:, 0]
        target_node_indices = data.candidate_moves[:, 1]
        
        emb_sources = node_embeddings[source_node_indices] # [num_candidates, hidden_dim]
        emb_targets = node_embeddings[target_node_indices] # [num_candidates, hidden_dim]
        
        # Concatenate [source_emb, target_emb]
        # candidate_edge_attr is NOT used here.
        edge_predictor_input = torch.cat([emb_sources, emb_targets], dim=-1)
        # Expected shape: [num_candidates, 2 * hidden_dim]
        
        move_logits = self.edge_predictor(edge_predictor_input) # Shape: [num_candidates, 1]
        
        # 3. Value prediction (board evaluation)
        if hasattr(data, 'batch') and data.batch is not None:
            batch_vector = data.batch
        else:
            batch_vector = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)
        
        graph_embedding = global_mean_pool(node_embeddings, batch_vector)
        
        value = self.value_head(graph_embedding) # Shape: [num_graphs_in_batch, 1]
        
        return move_logits, value
    

if __name__ == '__main__':
    # Example usage
    node_features = 14
    edge_features = 5
    model = Hive_GNN_Model(node_features=node_features, edge_features=edge_features)

    # Example usage
    grid = {(0, 0): (Piece(colour="WHITE", name="ANT", number=1), 
                     Piece(colour="WHITE", name="BEETLE", number=1),
                     Piece(colour="BLACK", name="BEETLE", number=1)), 
            (1, 1): (Piece(colour="WHITE", name="QUEEN", number=1),),
            (-1, -1): (Piece(colour="BLACK", name="SPIDER", number=1),),}
    game = initial_game(grid=grid)
    
    graph = Graph(game)
    data = graph_to_pytorch(graph)

    print(data)

    print(model(data))


