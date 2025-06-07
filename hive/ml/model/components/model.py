from enum import Enum

import torch
from torch import nn, optim
from torch_geometric.data import Data
import torch.nn.functional as F
import lightning as L
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
        if self.conv_net is None:
            raise ValueError("conv_net must be set before forward pass")

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 1. Encode node features if encoder is provided
        # Otherwise use raw node features
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




class HiveLightningModel(L.LightningModule):

    def __init__(self,
                 model,
                 learning_rate=0.01,
                 weight_decay=5e-4):
        super().__init__()
        self.save_hyperparameters()   # Save hyperparameters like learning_rate to the checkpoint
        self.model = model

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        # Calculate loss
        value_preds = outputs["value"]
        value_targets = batch.value
        loss = F.mse_loss(value_preds, value_targets)

        # Calculate accuracy for logging
        acc = ((value_preds > 0) == (value_targets > 0)).float().mean()

        # Log metrics using self.log(). Lightning handles the backend (e.g., MLflow)
        # `on_step=True` logs it per batch, `on_epoch=True` aggregates and logs at epoch end.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(),
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)


