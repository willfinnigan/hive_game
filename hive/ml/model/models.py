from torch import nn


from hive.ml.featurise.graph_to_pyg import game_to_pytorch
from hive.ml.model.components.encoders import SimpleNodeEncoder
from hive.ml.model.components.graph_nets import GATConvNet
from hive.ml.model.components.model import HiveGNN
from hive.ml.model.components.task_heads import MovePredictor, ValuePredictor

node_feats = 14
edge_feats = 6

encoder = SimpleNodeEncoder(in_channels=node_feats,
                            hidden_dim=128)

gatv2 = GATConvNet(in_channels=128,
                   hidden_dim=128, 
                   edge_dim=edge_feats, 
                   num_layers=4, 
                   heads=4, 
                   dropout=0.1)

task_heads = nn.ModuleDict({
    "move_predictor": MovePredictor(in_channels=128//4,
                                    hidden_dim=128),
    "value_predictor": ValuePredictor(in_channels=128//4,
                                      hidden_dim=128)
})


hive_gatv2 = HiveGNN(encoder=encoder,
                     conv_net=gatv2,
                     task_heads=task_heads,
                     pooling_type="mean")

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