from typing import List, Optional, Callable, Dict
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset

from hive.game_engine.game_state import Game
from hive.ml.data.endgame_to_data import process_endgame
from hive.ml.featurise.game_to_graph import Graph
from hive.ml.featurise.graph_to_pyg import graph_to_pytorch
from hive.trajectory.game_dataloader import GameDataLoader


class HiveLazyGameDataset(Dataset):
    """
    PyTorch Geometric Dataset for Hive games that loads data lazily.
    This is more memory-efficient for large datasets.
    """
    def __init__(
        self,
        filepath: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        batch_size: int = 100,
        max_skip_attempts: int = 100
    ):
        super().__init__(None, transform, pre_transform)
        self.filepath = filepath
        self.batch_size = batch_size
        self.max_skip_attempts = max_skip_attempts
        
        # Initialize the data loader
        self.loader = GameDataLoader(filepath, batch_size=batch_size)
        self.length = len(self.loader)
        
        # Keep track of valid indices
        self.valid_indices = set()
        self.invalid_indices = set()
        self.data = {}

    
    def len(self):
        return self.length
    
    def get(self, idx) -> Optional[List[Data]]:
        """Get a single game by index and convert to PyG Data."""
        game = self.loader.get_game(idx)
        if game is None:
            print(f"Unexpected error: Game at index {idx} is None")
            return None

        all_data = process_endgame(game)
        return all_data


def collate_fn(batch):
    """
    Custom collate function for batching PyG Data objects.
    
    Args:
        batch: A list of lists of PyG Data objects, where each Data object contains:
            - Standard PyG attributes (x, edge_index, edge_attr, etc.)
            - move_labels: Tensor of move labels
            - winner: Tensor indicating if current player won
            
    Returns:
        A Batch object with all attributes properly batched
    """
    # Flatten the batch if needed (since each item might be a list of Data objects)
    flattened_batch = []
    for item in batch:
        if item is None:
            continue
        if isinstance(item, list):
            flattened_batch.extend(item)
        else:
            raise ValueError(f"Expected a list of Data objects, got {type(item)}")
    
    if len(flattened_batch) == 0:
        return None
    
    # Batch the PyG Data objects
    # PyG's Batch.from_data_list will handle all attributes automatically
    batched_data = torch_geometric.data.Batch.from_data_list(flattened_batch)
    
    return batched_data
