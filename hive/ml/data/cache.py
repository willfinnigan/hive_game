import sqlite3
import pickle
import os
import hashlib
import fcntl
import tempfile
import shutil
import threading
from typing import Any, Optional, Dict
from pathlib import Path
import torch
import logging

# Set up logging
logger = logging.getLogger(__name__)


def calculate_optimal_shards(dataset_size: int, target_games_per_shard: int = 1500,
                           min_shards: int = 8, max_shards: int = 512) -> int:
    """
    Calculate optimal number of shards based on dataset size.
    
    Args:
        dataset_size: Number of games in the dataset
        target_games_per_shard: Target number of games per shard (default: 1500)
        min_shards: Minimum number of shards (default: 8)
        max_shards: Maximum number of shards (default: 512)
        
    Returns:
        Optimal number of shards (power of 2 for better hash distribution)
    """
    if dataset_size <= 0:
        return min_shards
    
    # Calculate ideal shard count
    ideal_shards = max(min_shards, dataset_size // target_games_per_shard)
    
    # Round up to next power of 2 for better hash distribution
    power_of_2 = 1
    while power_of_2 < ideal_shards:
        power_of_2 *= 2
    
    # Clamp to reasonable bounds
    optimal_shards = min(max_shards, max(min_shards, power_of_2))
    
    logger.info(f"Dataset size: {dataset_size} games, optimal shards: {optimal_shards} "
                f"(~{dataset_size // optimal_shards} games per shard)")
    
    return optimal_shards


def estimate_dataset_size(filepath: str) -> int:
    """
    Estimate the number of games in a dataset file by counting game entries.
    
    For game string files, each line typically represents one game.
    
    Args:
        filepath: Path to the game data file
        
    Returns:
        Estimated number of games
    """
    try:
        with open(filepath, 'r') as f:
            # Count non-empty lines - each line is typically one game
            game_count = sum(1 for line in f if line.strip())
        
        logger.debug(f"Estimated {game_count} games from {filepath}")
        return game_count
        
    except (FileNotFoundError, IOError) as e:
        logger.warning(f"Could not estimate dataset size for {filepath}: {e}")
        return 1000  # Default fallback estimate


class SQLiteCache:
    """
    A persistent cache using SQLite and pickle for storing processed game data.
    
    This cache stores serialized Python objects in an SQLite database,
    providing persistent storage between runs without consuming memory
    when not in use.
    """
    
    def __init__(self, cache_path: str):
        """
        Initialize the SQLite cache.
        
        Args:
            cache_path: Path to the SQLite database file
        """
        self.cache_path = cache_path
        self._init_db()
        
    def _init_db(self) -> None:
        """Create the cache table if it doesn't exist."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value BLOB
        )
        ''')
        conn.commit()
        conn.close()
        
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found
        """
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        return None
        
    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        pickled_value = pickle.dumps(value)
        cursor.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, pickled_value))
        conn.commit()
        conn.close()
        
    def clear(self) -> None:
        """Clear all entries from the cache."""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        
    def get_size(self) -> int:
        """
        Get the number of entries in the cache.
        
        Returns:
            The number of cached entries
        """
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache")
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else 0


class ShardedPyTorchCache:
    """
    A high-performance sharded cache using PyTorch .pt files for storing processed game data.
    
    This cache distributes data across multiple .pt files (shards) based on hash-based
    partitioning, providing better performance and concurrency than SQLite for ML workloads.
    Each shard is a separate .pt file containing a dictionary of cached data.
    
    Features:
    - Hash-based sharding for even distribution
    - Per-shard file locking for thread safety
    - Atomic write operations using temporary files
    - Native PyTorch serialization for optimal performance
    - Configurable shard count (default: 16)
    - Automatic directory structure creation
    """
    
    def __init__(self, cache_path: str, num_shards: Optional[int] = None,
                 dataset_size: Optional[int] = None, source_filepath: Optional[str] = None):
        """
        Initialize the sharded PyTorch cache.
        
        Args:
            cache_path: Base path for the cache directory
            num_shards: Number of shard files to create. If None, will be calculated automatically
            dataset_size: Size of dataset for automatic shard calculation
            source_filepath: Source data file path for automatic size estimation
        """
        self.cache_path = Path(cache_path)
        
        # Calculate optimal shard count if not provided
        if num_shards is None:
            if dataset_size is not None:
                self.num_shards = calculate_optimal_shards(dataset_size)
            elif source_filepath is not None:
                estimated_size = estimate_dataset_size(source_filepath)
                self.num_shards = calculate_optimal_shards(estimated_size)
            else:
                self.num_shards = 16  # Fallback default
                logger.warning("No dataset size info provided, using default 16 shards")
        else:
            self.num_shards = num_shards
        
        # Create directory structure
        self.shards_dir = self.cache_path / "shards"
        self.lockfiles_dir = self.cache_path / "lockfiles"
        
        self._init_directories()
        
        # In-memory cache for loaded shards to avoid repeated disk I/O
        self._shard_cache: Dict[int, Dict[str, Any]] = {}
        self._init_locks()
        
        # Write batching configuration
        self._write_batch_size = 20  # Number of writes before flushing to disk
        self._dirty_shards = set()   # Track which shards have unflushed changes
        self._write_counts = {i: 0 for i in range(self.num_shards)}  # Track writes per shard
        
        # Track cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'flushes': 0,
            'batched_writes': 0
        }
        
    def _init_locks(self):
        """Initialize thread locks. Called during __init__ and __setstate__."""
        self._shard_locks = {i: threading.RLock() for i in range(self.num_shards)}
    
    def __getstate__(self):
        """Custom method for pickling - exclude non-picklable locks."""
        state = self.__dict__.copy()
        # Remove the unpicklable locks
        del state['_shard_locks']
        return state
    
    def __setstate__(self, state):
        """Custom method for unpickling - recreate locks."""
        self.__dict__.update(state)
        # Recreate the locks
        self._init_locks()
        # Initialize batching attributes if they don't exist (backward compatibility)
        if not hasattr(self, '_write_batch_size'):
            self._write_batch_size = 10
        if not hasattr(self, '_dirty_shards'):
            self._dirty_shards = set()
        if not hasattr(self, '_write_counts'):
            self._write_counts = {i: 0 for i in range(self.num_shards)}
        
    def _init_directories(self) -> None:
        """Create the cache directory structure if it doesn't exist."""
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(exist_ok=True)
        self.lockfiles_dir.mkdir(exist_ok=True)
        
    def _get_shard_id(self, key: str) -> int:
        """
        Get the shard ID for a given key using hash-based partitioning.
        
        Uses the filepath portion of cache keys (format: "filepath:game_index")
        to ensure games from the same file are distributed across shards.
        
        Args:
            key: Cache key in format "filepath:game_index"
            
        Returns:
            Shard ID (0 to num_shards-1)
        """
        # Extract filepath portion for hashing
        filepath = key.split(':')[0] if ':' in key else key
        
        # Use SHA-256 hash for consistent distribution
        hash_obj = hashlib.sha256(filepath.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        
        return hash_int % self.num_shards
    
    def _get_shard_path(self, shard_id: int) -> Path:
        """Get the file path for a shard."""
        return self.shards_dir / f"{shard_id:02d}.pt"
    
    def _get_lockfile_path(self, shard_id: int) -> Path:
        """Get the lockfile path for a shard."""
        return self.lockfiles_dir / f"{shard_id:02d}.lock"
    
    def _load_shard(self, shard_id: int) -> Dict[str, Any]:
        """
        Load a shard from disk into memory.
        
        Args:
            shard_id: The shard ID to load
            
        Returns:
            Dictionary containing the shard data
        """
        shard_path = self._get_shard_path(shard_id)
        
        if not shard_path.exists():
            return {}
        
        try:
            # Use weights_only=False for PyTorch Geometric compatibility
            return torch.load(shard_path, map_location='cpu', weights_only=False)
        except Exception as e:
            logger.warning(f"Failed to load shard {shard_id}: {e}")
            return {}
    
    def _save_shard(self, shard_id: int, shard_data: Dict[str, Any]) -> None:
        """
        Save a shard to disk atomically using temporary file + rename.
        
        Args:
            shard_id: The shard ID to save
            shard_data: The shard data to save
        """
        shard_path = self._get_shard_path(shard_id)
        
        # Use temporary file for atomic write
        with tempfile.NamedTemporaryFile(
            dir=self.shards_dir,
            suffix='.tmp',
            delete=False
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
        
        try:
            # Save to temporary file
            torch.save(shard_data, temp_path)
            
            # Atomic rename
            temp_path.rename(shard_path)
            
        except Exception as e:
            # Clean up temporary file on error
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _acquire_shard_lock(self, shard_id: int):
        """
        Acquire file lock for a shard using fcntl.
        
        Args:
            shard_id: The shard ID to lock
            
        Returns:
            File handle for the lock file
        """
        lockfile_path = self._get_lockfile_path(shard_id)
        
        # Create lockfile if it doesn't exist
        lockfile_path.touch()
        
        # Open and lock the file
        lock_fd = open(lockfile_path, 'w')
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            return lock_fd
        except:
            lock_fd.close()
            raise
    
    def _release_shard_lock(self, lock_fd) -> None:
        """Release file lock for a shard."""
        if lock_fd:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                lock_fd.close()
            except:
                pass
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found
        """
        shard_id = self._get_shard_id(key)
        
        with self._shard_locks[shard_id]:
            # Check in-memory cache first
            if shard_id in self._shard_cache:
                shard_data = self._shard_cache[shard_id]
                if key in shard_data:
                    self._stats['hits'] += 1
                    return shard_data[key]
            
            # Load from disk if not in memory
            lock_fd = None
            try:
                lock_fd = self._acquire_shard_lock(shard_id)
                shard_data = self._load_shard(shard_id)
                
                # Cache in memory for future access
                self._shard_cache[shard_id] = shard_data
                
                if key in shard_data:
                    self._stats['hits'] += 1
                    return shard_data[key]
                else:
                    self._stats['misses'] += 1
                    return None
                    
            finally:
                self._release_shard_lock(lock_fd)
    
    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        shard_id = self._get_shard_id(key)
        
        with self._shard_locks[shard_id]:
            lock_fd = None
            try:
                lock_fd = self._acquire_shard_lock(shard_id)
                
                # Load current shard data if not in memory
                if shard_id not in self._shard_cache:
                    self._shard_cache[shard_id] = self._load_shard(shard_id)
                
                # Update in-memory cache
                self._shard_cache[shard_id][key] = value
                
                # Mark shard as dirty and increment write count
                self._dirty_shards.add(shard_id)
                self._write_counts[shard_id] += 1
                self._stats['writes'] += 1
                self._stats['batched_writes'] += 1
                
                # Flush to disk if batch size reached
                if self._write_counts[shard_id] >= self._write_batch_size:
                    self._flush_shard(shard_id)
                
            finally:
                self._release_shard_lock(lock_fd)
    
    def _flush_shard(self, shard_id: int) -> None:
        """
        Flush a specific shard to disk.
        
        Args:
            shard_id: The shard ID to flush
        """
        if shard_id in self._dirty_shards and shard_id in self._shard_cache:
            self._save_shard(shard_id, self._shard_cache[shard_id])
            self._dirty_shards.discard(shard_id)
            self._write_counts[shard_id] = 0
            self._stats['flushes'] += 1
    
    def flush_all(self) -> None:
        """
        Flush all dirty shards to disk.
        
        This should be called periodically or before process termination
        to ensure all cached data is persisted.
        """
        dirty_shards = list(self._dirty_shards)  # Copy to avoid modification during iteration
        for shard_id in dirty_shards:
            with self._shard_locks[shard_id]:
                lock_fd = None
                try:
                    lock_fd = self._acquire_shard_lock(shard_id)
                    self._flush_shard(shard_id)
                finally:
                    self._release_shard_lock(lock_fd)
    
    def __del__(self):
        """Destructor to ensure all data is flushed before object destruction."""
        try:
            self.flush_all()
        except:
            # Ignore errors during destruction
            pass
    
    def set_batch_size(self, batch_size: int) -> None:
        """
        Set the write batch size for controlling flush frequency.
        
        Args:
            batch_size: Number of writes before automatic flush (default: 10)
        """
        self._write_batch_size = max(1, batch_size)
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        # Flush any pending writes before clearing
        self.flush_all()
        
        # Clear in-memory cache
        self._shard_cache.clear()
        self._dirty_shards.clear()
        self._write_counts = {i: 0 for i in range(self.num_shards)}
        
        # Remove all shard files
        for shard_id in range(self.num_shards):
            with self._shard_locks[shard_id]:
                lock_fd = None
                try:
                    lock_fd = self._acquire_shard_lock(shard_id)
                    shard_path = self._get_shard_path(shard_id)
                    if shard_path.exists():
                        shard_path.unlink()
                finally:
                    self._release_shard_lock(lock_fd)
        
        # Reset statistics
        self._stats = {'hits': 0, 'misses': 0, 'writes': 0}
    
    def get_size(self) -> int:
        """
        Get the total number of entries in the cache.
        
        Returns:
            The number of cached entries across all shards
        """
        total_size = 0
        
        for shard_id in range(self.num_shards):
            with self._shard_locks[shard_id]:
                # Check in-memory cache first
                if shard_id in self._shard_cache:
                    total_size += len(self._shard_cache[shard_id])
                else:
                    # Load from disk to count
                    lock_fd = None
                    try:
                        lock_fd = self._acquire_shard_lock(shard_id)
                        shard_data = self._load_shard(shard_id)
                        total_size += len(shard_data)
                    finally:
                        self._release_shard_lock(lock_fd)
        
        return total_size
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache hit/miss/write statistics
        """
        return self._stats.copy()
    
    def migrate_from_sqlite(self, sqlite_cache_path: str) -> None:
        """
        Migrate data from an existing SQLite cache to the sharded cache.
        
        Args:
            sqlite_cache_path: Path to the existing SQLite cache file
        """
        if not os.path.exists(sqlite_cache_path):
            logger.debug(f"No SQLite cache found at {sqlite_cache_path}, skipping migration")
            return
        
        try:
            logger.info(f"Migrating data from SQLite cache: {sqlite_cache_path}")
            
            # Create temporary SQLite cache to read from
            sqlite_cache = SQLiteCache(sqlite_cache_path)
            
            # Get all keys from SQLite cache
            conn = sqlite3.connect(sqlite_cache_path)
            cursor = conn.cursor()
            
            # Check if the cache table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cache'")
            if not cursor.fetchone():
                logger.info("SQLite cache table does not exist, skipping migration")
                conn.close()
                return
            
            cursor.execute("SELECT key FROM cache")
            keys = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not keys:
                logger.info("SQLite cache is empty, skipping migration")
                return
            
            migrated_count = 0
            for key in keys:
                try:
                    value = sqlite_cache.get(key)
                    if value is not None:
                        self.set(key, value)
                        migrated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to migrate key {key}: {e}")
            
            logger.info(f"Successfully migrated {migrated_count} entries from SQLite cache")
            
        except Exception as e:
            logger.warning(f"Failed to migrate from SQLite cache {sqlite_cache_path}: {e}")
