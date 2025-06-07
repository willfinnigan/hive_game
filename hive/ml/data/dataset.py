from typing import List, Optional, Callable, Dict
import torch
import torch_geometric
import os
import gc
import time
import psutil
from torch_geometric.data import Data, Dataset
import torch.backends.mps

from hive.game_engine.game_state import Game
from hive.ml.data.endgame_to_data import process_endgame
from hive.ml.featurise.game_to_graph import Graph
from hive.ml.featurise.graph_to_pyg import graph_to_pytorch
from hive.trajectory.game_dataloader import GameDataLoader
from hive.ml.data.cache import SQLiteCache, ShardedPyTorchCache


class HiveLazyGameDataset(Dataset):
    """
    PyTorch Geometric Dataset for Hive games that loads data lazily.
    This is more memory-efficient for large datasets.
    
    With caching enabled, processed data is stored using either a high-performance
    sharded PyTorch cache (default) or SQLite database for faster access in
    subsequent runs.
    
    This dataset is designed to work with PyTorch's DataLoader multiprocessing
    by properly handling file handles during pickling.
    
    Args:
        filepath: Path to the game data file
        transform: Optional transform to apply to data
        pre_transform: Optional pre-transform to apply to data
        batch_size: Batch size for internal game loading
        max_skip_attempts: Maximum attempts to skip invalid games
        cache_path: Custom cache path (auto-generated if None)
        use_cache: Whether to enable caching
        cache_type: Type of cache to use ("sharded" or "sqlite")
        num_shards: Number of shards for sharded cache (None for auto-calculation)
        migrate_from_sqlite: Whether to migrate from existing SQLite cache
    """
    def __init__(
        self,
        filepath: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        batch_size: int = 100,
        max_skip_attempts: int = 100,
        cache_path: Optional[str] = None,
        use_cache: bool = True,
        cache_type: str = "sharded",  # "sharded" or "sqlite"
        num_shards: Optional[int] = None,
        migrate_from_sqlite: bool = False,
    ):
        super().__init__(None, transform, pre_transform)
        self.filepath = filepath
        self.batch_size = batch_size
        self.max_skip_attempts = max_skip_attempts
        self.use_cache = use_cache
        self.cache_type = cache_type
        self.num_shards = num_shards
        self.migrate_from_sqlite = migrate_from_sqlite
        
        # Initialize the data loader
        self.loader = GameDataLoader(filepath, batch_size=batch_size)
        self.length = len(self.loader)
        
        # Keep track of valid indices
        self.valid_indices = set()
        self.invalid_indices = set()

        self.gc_freq = 100
        
        # Initialize cache if enabled
        if self.use_cache:
            if cache_path is None:
                # Create cache in the same directory as the data file
                data_dir = os.path.dirname(os.path.abspath(filepath))
                filename = os.path.basename(filepath)
                if cache_type == "sharded":
                    cache_path = os.path.join(data_dir, f"{filename}.cache")
                else:
                    cache_path = os.path.join(data_dir, f"{filename}.cache.db")
            
            # Initialize the appropriate cache type
            if cache_type == "sharded":
                self.cache = ShardedPyTorchCache(
                    cache_path,
                    num_shards=num_shards,
                    source_filepath=filepath
                )
                actual_shards = self.cache.num_shards
                print(f"Using ShardedPyTorchCache with {actual_shards} shards at: {cache_path}")
                
                # Migrate from SQLite if requested and SQLite cache exists
                if migrate_from_sqlite:
                    sqlite_cache_path = cache_path.replace(".cache", ".cache.db")
                    if os.path.exists(sqlite_cache_path):
                        print(f"Migrating from SQLite cache: {sqlite_cache_path}")
                        self.cache.migrate_from_sqlite(sqlite_cache_path)
                        
            elif cache_type == "sqlite":
                self.cache = SQLiteCache(cache_path)
                print(f"Using SQLiteCache at: {cache_path}")
            else:
                raise ValueError(f"Unknown cache_type: {cache_type}. Must be 'sharded' or 'sqlite'")

    
    def __getstate__(self):
        """
        Custom method to prepare the object for pickling.
        This is called when the object is being pickled (e.g., for multiprocessing).
        
        We need to close the file handle in the loader before pickling.
        """
        state = self.__dict__.copy()
        
        # Close the file handle in the loader if it exists
        if hasattr(self, 'loader') and self.loader is not None:
            self.loader.close()
            
        return state
    
    def __setstate__(self, state):
        """
        Custom method to restore the object after unpickling.
        This is called when the object is being unpickled in a worker process.
        
        We need to restore the file handle in the loader after unpickling.
        """
        self.__dict__.update(state)
        
        # Ensure the loader's file handle is reopened
        if hasattr(self, 'loader') and self.loader is not None:
            self.loader._ensure_file_open()
    
    def len(self):
        return self.length
    
    def get(self, idx) -> Optional[List[Data]]:
        """Get a single game by index and convert to PyG Data."""
        start_time = time.time()
        
        # Try to get from cache first if caching is enabled
        if self.use_cache:
            cache_key = f"{self.filepath}:{idx}"
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Ensure the loader's file handle is open
        self.loader._ensure_file_open()
        
        # If not in cache or caching disabled, load and process
        game = self.loader.get_game(idx)
        if game is None:
            print(f"Unexpected error: Game at index {idx} is None")
            return None

        # Process the game
        all_data = process_endgame(game)
        
        # Store in cache if caching is enabled
        if self.use_cache and all_data is not None:
            cache_key = f"{self.filepath}:{idx}"
            self.cache.set(cache_key, all_data)

        self.clean_up(game)
        
        return all_data
    
    def clean_up(self, game: Game) -> None:
        """Thoroughly delete all the game data to free memory."""

        while game is not None:
            # Delete the game object and its parent
            try:
                del game.move
            except:
                pass

            parent = game.parent
            del game
            game = parent
        gc.collect()

    
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



def collate_fn(batch):
    """
    Custom collate function for batching PyTorch Geometric Data objects.
    
    This function takes a batch of items from the dataset (each item is a list of
    PyG Data objects representing game states) and combines them into a single
    batched Data object suitable for training.
    
    Args:
        batch: List of items from the dataset, where each item is either:
            - A list of PyG Data objects (game states)
            - None (if the dataset item failed to load)
            
    Returns:
        torch_geometric.data.Batch: Batched PyG Data object with all game states
        combined, or None if no valid data was found
    """
    # Collect all valid Data objects from the batch
    flattened_batch = []
    
    for item in batch:
        if item is None:
            # Skip failed dataset items
            continue
            
        if isinstance(item, list):
            # Extend with all Data objects from this game
            flattened_batch.extend(item)
        else:
            raise ValueError(f"Expected list of Data objects or None, got {type(item)}")
    
    # Return None if no valid data found
    if len(flattened_batch) == 0:
        return None
    
    # Create batched PyG Data object
    try:
        batched_data = torch_geometric.data.Batch.from_data_list(flattened_batch)
    except Exception as e:
        raise RuntimeError(f"Failed to create batch from {len(flattened_batch)} Data objects: {e}")
    
    # Memory cleanup
    flattened_batch.clear()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return batched_data



