import torch
from torch import nn


class TaskHead(nn.Module):
    """Base class for task-specific heads"""

    def __init__(self, in_channels, hidden_dim, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def forward(self, node_embeddings=None, graph_embedding=None, data=None):
        raise NotImplementedError


class MovePredictor(TaskHead):
    """Head for predicting valid moves (edge-level prediction)"""

    def __init__(self, in_channels, hidden_dim, dropout=0.1):
        super().__init__(in_channels, hidden_dim, dropout)

        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )


    def forward(self, node_embeddings=None, graph_embedding=None, data=None):
        
        # Ensure candidate_moves is a tensor of shape [num_moves, 2] with correct device
        candidate_moves = data.edge_index[:, data.move_edge_idxs].T

        # Extract source and target node embeddings
        source_node_indices = candidate_moves[:, 0]
        target_node_indices = candidate_moves[:, 1]

        emb_sources = node_embeddings[source_node_indices]
        emb_targets = node_embeddings[target_node_indices]

        # Concatenate source and target embeddings
        edge_predictor_input = torch.cat([emb_sources, emb_targets], dim=-1)

        # Predict move logits
        move_logits = self.edge_predictor(edge_predictor_input)

        # Squeeze out the singleton dimension to get a 1D tensor
        move_logits = move_logits.squeeze(-1)

        return move_logits


class ValuePredictor(TaskHead):
    """Head for predicting game position value"""

    def __init__(self, in_channels, hidden_dim, dropout=0.1):
        super().__init__(in_channels, hidden_dim, dropout)

        self.value_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Value between -1 and 1
        )

    def forward(self, node_embeddings=None, graph_embedding=None, data=None):
        if graph_embedding is None:
            raise ValueError("ValuePredictor requires graph_embedding")

        return self.value_head(graph_embedding).squeeze(-1)
