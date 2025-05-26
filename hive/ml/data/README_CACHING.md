# SQLite Caching for HiveLazyGameDataset

This document explains the SQLite-based caching system implemented for `HiveLazyGameDataset`.

## Overview

The caching system uses SQLite with pickle serialization to store processed game data on disk. This provides:

1. **Persistent storage** between runs
2. **Memory efficiency** - data is only loaded into memory when needed
3. **Significant performance improvements** - ~20x faster access for cached games

## How It Works

1. When a game is requested via `get(idx)`, the system first checks if it exists in the cache
2. If found, it returns the cached data directly (avoiding file I/O and processing)
3. If not found, it loads and processes the game normally, then stores the result in the cache
4. Subsequent requests for the same game will be served from the cache

## Usage

### Basic Usage

```python
from hive.ml.data.dataset import HiveLazyGameDataset

# Create dataset with caching enabled (default)
dataset = HiveLazyGameDataset(
    filepath="game_strings/combined.txt",
    batch_size=100,
    use_cache=True  # Enable caching (default is True)
)

# Access games as normal - caching happens automatically
game_data = dataset.get(0)  # First access will be cached
game_data = dataset.get(0)  # Second access will use cache
```

### Custom Cache Location

By default, the cache is stored in the same directory as the data file with a `.cache.db` extension:

```python
# Custom cache location
dataset = HiveLazyGameDataset(
    filepath="game_strings/combined.txt",
    cache_path="/path/to/custom/cache.db"
)
```

### Cache Management

The dataset provides methods to manage the cache:

```python
# Check if a game is cached
is_cached = dataset.is_cached(10)  # Returns True/False

# Prefetch multiple games into cache (useful before training)
dataset.prefetch_to_cache([10, 11, 12, 13, 14])

# Clear the entire cache
dataset.clear_cache()
```

## Performance

Based on testing, accessing cached games is approximately 20x faster than loading and processing them from scratch. The first time a game is accessed, it will be slightly slower due to the overhead of storing it in the cache, but subsequent accesses will be much faster.

Example performance metrics:
- First access (no cache): ~1.2 seconds for 5 games
- First access (building cache): ~1.7 seconds for 5 games
- Second access (using cache): ~0.06 seconds for 5 games

## Implementation Details

The caching system consists of two main components:

1. **SQLiteCache** (`hive/ml/data/cache.py`) - Handles the low-level SQLite operations
2. **HiveLazyGameDataset** (`hive/ml/data/dataset.py`) - Uses the cache in its `get()` method

The cache stores serialized PyG Data objects using pickle, with keys based on the filepath and game index.

## Testing

A test script is provided to demonstrate the caching functionality:

```bash
python -m hive.ml.data.test_cache
```

This script:
1. Measures performance with and without caching
2. Demonstrates cache management methods
3. Shows the speedup achieved with caching