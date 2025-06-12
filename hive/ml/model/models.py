from typing import List
from torch import nn
import lightning as L

from hive.ml.featurise.graph_to_pyg import game_to_pytorch
from hive.ml.model.components.encoders import SimpleNodeEncoder
from hive.ml.model.components.graph_nets import GATConvNet
from hive.ml.model.components.model import HiveGNN, PoolingType
from hive.ml.model.components.task_heads import MovePredictor, NumMobilePiecesPredictor, ValuePredictor
from hive.ml.featurise.node_features import NUM_NODE_FEATS

NODE_FEATS = NUM_NODE_FEATS
EDGE_FEATS = 6

def create_hive_gatv2_gnn(hidden_dim: int,
                          num_layers: int,
                          heads: int,
                          dropout: float,
                          residual: bool,
                          batch_norm: bool,
                          task_heads: List[str],
                          pool_method: PoolingType
                        ):

    encoder = SimpleNodeEncoder(in_channels=NODE_FEATS,
                                hidden_dim=hidden_dim)

    gatv2 = GATConvNet(in_channels=hidden_dim,
                       hidden_dim=hidden_dim,
                       edge_dim=EDGE_FEATS,
                       num_layers=num_layers,
                       heads=heads,
                       dropout=dropout,
                       residual=residual,
                       batch_norm=batch_norm,
                       )
    
    value_head = ValuePredictor(in_channels=hidden_dim // heads,
                                hidden_dim=hidden_dim,
                                dropout=dropout)
    
    move_head = MovePredictor(in_channels=hidden_dim // heads,
                                hidden_dim=hidden_dim,
                                dropout=dropout)
    
    mobile_pieces_head = NumMobilePiecesPredictor(in_channels=hidden_dim // heads,
                                                  hidden_dim=hidden_dim,
                                                  dropout=dropout)
    

    all_task_heads = {
        "policy": move_head,
        "value": value_head,
        "mobile_pieces": mobile_pieces_head  
    }

    task_head_modules = nn.ModuleDict({name: head for name, head in all_task_heads.items() if name in task_heads})


    return HiveGNN(encoder=encoder,
                   conv_net=gatv2,
                   task_heads=task_head_modules,
                   pooling_type=pool_method)











if __name__ == "__main__":
    # Example usage
    from hive.ml.featurise.graph_to_pyg import graph_to_pytorch
    from hive.game_engine.game_state import Game, initial_game
    from hive.game_engine.game_state import Piece

    model = create_hive_gatv2_gnn(hidden_dim=64,
                                  num_layers=2,
                                  heads=1,
                                  dropout=0.05,
                                  residual=False,
                                  batch_norm=False,
                                  pool_method='add')

    grid = {(0, 0): (Piece(colour="WHITE", name="ANT", number=1),
                     Piece(colour="WHITE", name="BEETLE", number=1),
                     Piece(colour="BLACK", name="BEETLE", number=1)),
            (1, 1): (Piece(colour="WHITE", name="QUEEN", number=1),),
            (-1, -1): (Piece(colour="BLACK", name="SPIDER", number=1),)}
    game = initial_game(grid=grid)
    
    data = game_to_pytorch(game)

    # Forward pass
    outputs = model(data)

    print(outputs)