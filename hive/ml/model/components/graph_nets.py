from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, BatchNorm
import torch.nn.functional as F


class GraphConvBase(nn.Module):
    """Base class for graph convolution modules"""

    def __init__(self, in_channels, hidden_dim, num_layers, dropout=0.1, batch_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        raise NotImplementedError


class GATConvNet(GraphConvBase):
    """Graph Attention Network module"""

    def __init__(self, in_channels, hidden_dim, num_layers, heads=4, edge_dim=None, dropout=0.1, residual=True, batch_norm=False):

        super().__init__(in_channels, hidden_dim, num_layers, dropout, batch_norm)

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        # First layer (can only use residual if in_channels == hidden_dim)
        first_layer_residual = False
        if in_channels == hidden_dim and residual == True:
            first_layer_residual = True

        self.conv_layers.append(GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            residual=first_layer_residual,
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
                residual=residual,
                concat=True
            ))

        # Last layer
        self.conv_layers.append(GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            residual=residual,
            concat=False  # output channels = hidden_dim // heads
        ))

        # Add batch norm layers if requested
        if self.batch_norm:
            for i in range(num_layers):
                if i == num_layers - 1:
                    # Last layer output dimension
                    self.batch_norms.append(BatchNorm(hidden_dim // heads))
                else:
                    # All other layers output dimension
                    self.batch_norms.append(BatchNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GCNNet(GraphConvBase):
    """Graph Convolutional Network module"""

    def __init__(self, in_channels, hidden_dim, num_layers, dropout=0.1, batch_norm=False):
        super().__init__(in_channels, hidden_dim, num_layers, dropout, batch_norm)

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        # First layer
        self.conv_layers.append(GCNConv(in_channels, hidden_dim))

        # Middle and last layers
        for _ in range(num_layers-1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Add batch norm layers if requested
        if self.batch_norm:
            for _ in range(num_layers):
                self.batch_norms.append(BatchNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x, edge_index)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class SAGENet(GraphConvBase):
    """GraphSAGE Network module"""

    def __init__(self, in_channels, hidden_dim, num_layers, dropout=0.1, batch_norm=False):
        super().__init__(in_channels, hidden_dim, num_layers, dropout, batch_norm)

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        # First layer
        self.conv_layers.append(SAGEConv(in_channels, hidden_dim))
        
        # Middle and last layers
        for _ in range(num_layers-1):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Add batch norm layers if requested
        if self.batch_norm:
            for _ in range(num_layers):
                self.batch_norms.append(BatchNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x, edge_index)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x