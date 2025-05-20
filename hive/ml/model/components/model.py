from enum import Enum

import torch
from torch import nn
from torch_geometric.data import Data
from typing import Literal, Dict, Callable, Optional

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from hive.ml.model.components.encoders import NodeEncoder
from hive.ml.model.components.graph_nets import GraphConvBase
from hive.ml.model.components.task_heads import TaskHead

PoolingType = Literal["add", "mean", "max"]
PoolingFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

POOLING_METHODS: Dict[PoolingType, PoolingFunction] = {'add': global_add_pool,
                                                        'mean': global_mean_pool,
                                                        'max': global_max_pool}

class HiveGNN(nn.Module):
    """
    Base class for composable Hive GNN models.
    """
    def __init__(self,
                    conv_net: GraphConvBase,
                    task_heads: nn.ModuleDict,
                    encoder: Optional[NodeEncoder] = None,
                    pooling_type: PoolingType = "mean",
                    ):

        super().__init__()
        self.encoder = encoder
        self.conv_net = conv_net
        self.task_heads = task_heads
        self.pooling_type = pooling_type

    def forward(self, data: Data):
        if self.conv_net is None or self.encoder is None:
            raise ValueError("encoder and conv_net must be set before forward pass")

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 1. Encode node features if encoder is provided
        if self.encoder is not None:
            x = self.encoder(x)

        # 2. Apply graph convolutions
        node_embeddings = self.conv_net(x, edge_index, edge_attr=edge_attr)

        # 3. Global pooling for graph-level tasks
        batch_vector = self._get_batch_vector(data, node_embeddings)
        graph_embedding = POOLING_METHODS[self.pooling_type](node_embeddings, batch_vector)

        # 4. Task-specific predictions
        outputs = {}
        for name, head in self.task_heads.items():
            outputs[name] = head(
                node_embeddings=node_embeddings,
                graph_embedding=graph_embedding,
                data=data
            )

        return outputs

    def _get_batch_vector(self, data, node_embeddings):
        if hasattr(data, 'batch') and data.batch is not None:
            batch_vector = data.batch
        else:
            batch_vector = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)

        return batch_vector