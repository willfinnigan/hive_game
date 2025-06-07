from torch import nn


from hive.ml.featurise.graph_to_pyg import game_to_pytorch
from hive.ml.model.components.encoders import SimpleNodeEncoder
from hive.ml.model.components.graph_nets import GATConvNet
from hive.ml.model.components.model import HiveGNN
from hive.ml.model.components.task_heads import MovePredictor, ValuePredictor

node_feats = 11
edge_feats = 6

hidden_dim = 128
num_layers = 4
heads = 4
dropout = 0.1
residual = True
batch_norm = True

encoder = SimpleNodeEncoder(in_channels=node_feats,
                            hidden_dim=hidden_dim)

gatv2 = GATConvNet(in_channels=hidden_dim,
                   hidden_dim=hidden_dim,
                   edge_dim=edge_feats, 
                   num_layers=num_layers,
                   heads=heads,
                   dropout=dropout,
                   residual=residual,
                   batch_norm=False)

task_heads = nn.ModuleDict({
    "policy": MovePredictor(in_channels=hidden_dim//heads,
                            hidden_dim=hidden_dim),
    "value": ValuePredictor(in_channels=hidden_dim//heads,
                            hidden_dim=hidden_dim)
})

value_only_head = nn.ModuleDict({
    "value": ValuePredictor(in_channels=hidden_dim//heads,
                            hidden_dim=hidden_dim)
})


hive_gatv2 = HiveGNN(encoder=encoder,
                     conv_net=gatv2,
                     task_heads=task_heads,
                     pooling_type="add")

if __name__ == "__main__":
    # Example usage
    from hive.ml.featurise.graph_to_pyg import graph_to_pytorch
    from hive.game_engine.game_state import Game, initial_game
    from hive.game_engine.game_state import Piece

    grid = {(0, 0): (Piece(colour="WHITE", name="ANT", number=1),
                     Piece(colour="WHITE", name="BEETLE", number=1),
                     Piece(colour="BLACK", name="BEETLE", number=1)),
            (1, 1): (Piece(colour="WHITE", name="QUEEN", number=1),),
            (-1, -1): (Piece(colour="BLACK", name="SPIDER", number=1),)}
    game = initial_game(grid=grid)
    
    data = game_to_pytorch(game)

    # Forward pass
    outputs = hive_gatv2(data)

    print(outputs)