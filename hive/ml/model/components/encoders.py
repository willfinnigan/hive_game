from torch import nn


class NodeEncoder(nn.Module):
    """Base class for node feature encoders"""

    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

    def forward(self, x):
        raise NotImplementedError


class SimpleNodeEncoder(NodeEncoder):
    """Simple MLP encoder for node features"""

    def __init__(self, in_channels, hidden_dim):
        super().__init__(in_channels, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class ResidualNodeEncoder(NodeEncoder):
    """Node encoder with residual connections"""

    def __init__(self, in_channels, hidden_dim):
        super().__init__(in_channels, hidden_dim)
        self.proj = nn.Linear(in_channels, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(2)
        ])

    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)
        return x