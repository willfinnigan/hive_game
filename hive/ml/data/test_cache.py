import time
import os
from pathlib import Path
from typing import List
from hive.ml.data.dataset import HiveLazyGameDataset

def test_caching_performance():
    """
    Test the performance improvement from caching.
    
    This script loads a few games with and without caching,
    and measures the time difference to demonstrate the benefits.
    """
    # Get the path to the game strings file
    project_root = Path(__file__).parents[3]
    filepath = os.path.join(project_root, "game_strings", "combined.txt")
    
    if not os.path.exists(filepath):
        print(f"Game strings file not found at: {filepath}")
        print("Please specify a valid filepath to a game strings file.")
        return
    
    # First run without cache
    print("Testing without cache...")
    dataset_no_cache = HiveLazyGameDataset(
        filepath=filepath,
        batch_size=10,
        use_cache=False
    )
    
    # Load a few games and measure time
    indices_to_test = [0, 1, 2, 3, 4]
    
    # First access (no cache)
    start_time = time.time()
    for idx in indices_to_test:
        data = dataset_no_cache.get(idx)
        if data is not None:
            print(f"Game {idx}: {len(data)} data points")
    first_access_time = time.time() - start_time
    print(f"First access (no cache): {first_access_time:.4f} seconds")
    
    # Now with cache
    print("\nTesting with cache...")
    cache_path = os.path.join(project_root, "game_strings", "test_cache.db")
    dataset_with_cache = HiveLazyGameDataset(
        filepath=filepath,
        batch_size=10,
        use_cache=True,
        cache_path=cache_path
    )
    
    # First access (building cache)
    start_time = time.time()
    for idx in indices_to_test:
        data = dataset_with_cache.get(idx)
        if data is not None:
            print(f"Game {idx}: {len(data)} data points")
    cache_build_time = time.time() - start_time
    print(f"First access (building cache): {cache_build_time:.4f} seconds")
    
    # Second access (using cache)
    start_time = time.time()
    for idx in indices_to_test:
        data = dataset_with_cache.get(idx)
        if data is not None:
            print(f"Game {idx}: {len(data)} data points")
    cached_access_time = time.time() - start_time
    print(f"Second access (using cache): {cached_access_time:.4f} seconds")
    
    # Calculate speedup
    if cached_access_time > 0:
        speedup = first_access_time / cached_access_time
        print(f"\nSpeedup from caching: {speedup:.2f}x faster")
    
    # Cache info
    cache_size = dataset_with_cache.cache.get_size()
    print(f"Cache entries: {cache_size}")
    
    print(f"\nCache file location: {cache_path}")
    
    return dataset_with_cache

def test_cache_management():
    """
    Test the cache management methods.
    """
    # Get the path to the game strings file
    project_root = Path(__file__).parents[3]
    filepath = os.path.join(project_root, "game_strings", "combined.txt")
    cache_path = os.path.join(project_root, "game_strings", "management_test_cache.db")
    
    if not os.path.exists(filepath):
        print(f"Game strings file not found at: {filepath}")
        return
    
    # Create dataset with cache
    print("\nTesting cache management...")
    dataset = HiveLazyGameDataset(
        filepath=filepath,
        batch_size=10,
        use_cache=True,
        cache_path=cache_path
    )
    
    # Test is_cached
    indices_to_test = [10, 11, 12, 13, 14]
    
    print("\nChecking cache status before loading:")
    for idx in indices_to_test:
        print(f"Game {idx} is cached: {dataset.is_cached(idx)}")
    
    # Test prefetch_to_cache
    print("\nPrefetching games to cache:")
    dataset.prefetch_to_cache(indices_to_test)
    
    # Check cache status after prefetching
    print("\nChecking cache status after prefetching:")
    for idx in indices_to_test:
        print(f"Game {idx} is cached: {dataset.is_cached(idx)}")
    
    # Test clear_cache
    print("\nClearing cache...")
    dataset.clear_cache()
    
    # Check cache status after clearing
    print("\nChecking cache status after clearing:")
    for idx in indices_to_test:
        print(f"Game {idx} is cached: {dataset.is_cached(idx)}")
    
    print(f"\nCache file location: {cache_path}")

if __name__ == "__main__":
    test_caching_performance()
    test_cache_management()