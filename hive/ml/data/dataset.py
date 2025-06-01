from typing import List, Optional, Callable, Dict
import torch
import torch_geometric
import os
from torch_geometric.data import Data, Dataset

from hive.game_engine.game_state import Game
from hive.ml.data.endgame_to_data import process_endgame
from hive.ml.featurise.game_to_graph import Graph
from hive.ml.featurise.graph_to_pyg import graph_to_pytorch
from hive.trajectory.game_dataloader import GameDataLoader
from hive.ml.data.cache import SQLiteCache


class HiveLazyGameDataset(Dataset):
    """
    PyTorch Geometric Dataset for Hive games that loads data lazily.
    This is more memory-efficient for large datasets.
    
    With caching enabled, processed data is stored in an SQLite database
    for faster access in subsequent runs.
    """
    def __init__(
        self,
        filepath: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        batch_size: int = 100,
        max_skip_attempts: int = 100,
        cache_path: Optional[str] = None,
        use_cache: bool = True
    ):
        super().__init__(None, transform, pre_transform)
        self.filepath = filepath
        self.batch_size = batch_size
        self.max_skip_attempts = max_skip_attempts
        self.use_cache = use_cache
        
        # Initialize the data loader
        self.loader = GameDataLoader(filepath, batch_size=batch_size)
        self.length = len(self.loader)
        
        # Keep track of valid indices
        self.valid_indices = set()
        self.invalid_indices = set()
        
        # Initialize cache if enabled
        if self.use_cache:
            if cache_path is None:
                # Create cache in the same directory as the data file
                data_dir = os.path.dirname(os.path.abspath(filepath))
                filename = os.path.basename(filepath)
                cache_path = os.path.join(data_dir, f"{filename}.cache.db")
            self.cache = SQLiteCache(cache_path)
            print(f"Using SQLite cache at: {cache_path}")

    
    def len(self):
        return self.length
    
    def get(self, idx) -> Optional[List[Data]]:
        """Get a single game by index and convert to PyG Data."""
        # Try to get from cache first if caching is enabled
        if self.use_cache:
            cache_key = f"{self.filepath}:{idx}"
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # If not in cache or caching disabled, load and process
        game = self.loader.get_game(idx)
        if game is None:
            print(f"Unexpected error: Game at index {idx} is None")
            return None

        all_data = process_endgame(game)
        
        # Store in cache if caching is enabled
        if self.use_cache and all_data is not None:
            cache_key = f"{self.filepath}:{idx}"
            self.cache.set(cache_key, all_data)

        del game # Clear the game object to free memory

        return all_data
    
    def is_cached(self, idx: int) -> bool:
        """
        Check if a specific game index is cached.
        
        Args:
            idx: The game index to check
            
        Returns:
            True if the game is cached, False otherwise
        """
        if not self.use_cache:
            return False
            
        cache_key = f"{self.filepath}:{idx}"
        return self.cache.get(cache_key) is not None
    
    def clear_cache(self) -> None:
        """Clear all entries from the cache."""
        if self.use_cache:
            self.cache.clear()
            print("Cache cleared")
    
    def prefetch_to_cache(self, indices: List[int]) -> None:
        """
        Prefetch and cache multiple games by their indices.
        
        This is useful for preloading commonly accessed games into the cache
        before training starts.
        
        Args:
            indices: List of game indices to prefetch
        """
        if not self.use_cache:
            print("Caching is disabled, cannot prefetch")
            return
            
        print(f"Prefetching {len(indices)} games to cache...")
        for i, idx in enumerate(indices):
            if not self.is_cached(idx):
                self.get(idx)
            
            if (i + 1) % 10 == 0:
                print(f"Prefetched {i + 1}/{len(indices)} games")
        
        print(f"Prefetching complete. Cache now contains {self.cache.get_size()} entries.")


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



