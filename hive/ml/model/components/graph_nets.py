from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
import torch.nn.functional as F


class GraphConvBase(nn.Module):
    """Base class for graph convolution modules"""

    def __init__(self, in_channels, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        raise NotImplementedError


class GATConvNet(GraphConvBase):
    """Graph Attention Network module"""

    def __init__(self, in_channels, hidden_dim, num_layers, heads=4, edge_dim=None, dropout=0.1):
        super().__init__(in_channels, hidden_dim, num_layers, dropout)

        self.conv_layers = nn.ModuleList()

        # First layer
        self.conv_layers.append(GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True
        ))

        # Middle layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=True
            ))

        # Last layer
        self.conv_layers.append(GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=False  # output channels = hidden_dim // heads
        ))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GCNNet(GraphConvBase):
    """Graph Convolutional Network module"""

    def __init__(self, in_channels, hidden_dim, num_layers, dropout=0.1):
        super().__init__(in_channels, hidden_dim, num_layers, dropout)

        self.conv_layers = nn.ModuleList()

        # First layer
        self.conv_layers.append(GCNConv(in_channels, hidden_dim))

        # Middle and last layers
        for _ in range(num_layers-1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class SAGENet(GraphConvBase):
    """GraphSAGE Network module"""

    def __init__(self, in_channels, hidden_dim, num_layers, dropout=0.1):
        super().__init__(in_channels, hidden_dim, num_layers, dropout)

        self.conv_layers = nn.ModuleList()

        # First layer
        self.conv_layers.append(SAGEConv(in_channels, hidden_dim))
        
        # Middle and last layers
        for _ in range(num_layers-1):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x