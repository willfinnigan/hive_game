import gc
import os
import logging
import time
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from hive.ml.data.dataset import HiveLazyGameDataset, collate_fn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import multiprocessing at the top level
import multiprocessing
from multiprocessing import freeze_support


def main(workers=0, max_batches=None, batch_size=32):
    """Main function to run the pre-caching process
    workers - number of processes
    batches - limit number of batches to process
    batch_size - size of batches
    """
    # Set multiprocessing start method inside the main function
    if os.name == 'posix' and 'darwin' in os.uname().sysname.lower():
        # On macOS, use 'spawn' which is safer but has more overhead
        multiprocessing.set_start_method('spawn', force=True)
        logging.info("Using 'spawn' multiprocessing start method on macOS")
    else:
        # On other platforms, 'fork' is fine
        multiprocessing.set_start_method('fork', force=True)
        logging.info("Using 'fork' multiprocessing start method")

    
    folder = Path(__file__).parents[3]
    filepath = f"{folder}/game_strings/combined.txt"
    
    logging.info(f"Creating dataset from {filepath}")
    
    # Create dataset with use_cache=True to enable sharded caching
    train_dataset = HiveLazyGameDataset(filepath, batch_size=batch_size, use_cache=True)
    
    # Configure write batching for better performance with multiprocessing
    if hasattr(train_dataset.cache, 'set_batch_size'):
        train_dataset.cache.set_batch_size(20)  # Batch 20 writes before flushing
        logging.info(f"Configured cache write batching: 20 writes per flush")
    
    # Configure DataLoader with user-specified settings
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,  # Use command-line argument
        collate_fn=collate_fn
    )
    
    # Determine how many batches to process
    total_batches = len(train_loader)
    batches_to_process = total_batches if max_batches is None else min(max_batches, total_batches)
    
    logging.info(f"Total dataset size: {len(train_dataset)} data items")
    logging.info(f"Cache using {train_dataset.cache.num_shards} shards")
    if max_batches is not None:
        logging.info(f"Limited to processing {batches_to_process} batches (max ~{batches_to_process * batch_size} items)")
    else:
        logging.info(f"Processing all {total_batches} batches")
    
    start_time = time.time()
    
    try:
        progress_bar = tqdm(train_loader, total=batches_to_process)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            # Process completed successfully
            progress_bar.set_description(f"Processed batch {batch_idx+1}/{batches_to_process}")
            
            # Report cache size and flush periodically
            if batch_idx % 10 == 0 and batch_idx > 0:
                # Flush any pending writes to ensure data is persisted
                if hasattr(train_dataset.cache, 'flush_all'):
                    train_dataset.cache.flush_all()
                
                cache_size = train_dataset.cache.get_size()
                elapsed = time.time() - start_time
                items_per_sec = (batch_idx * batch_size) / elapsed
                
                # Report batching statistics if available
                if hasattr(train_dataset.cache, 'get_stats'):
                    stats = train_dataset.cache.get_stats()
                    logging.info(f"Batch {batch_idx}: {items_per_sec:.1f} items/sec, "
                               f"cache: {cache_size} entries, "
                               f"flushes: {stats.get('flushes', 0)}, "
                               f"batched writes: {stats.get('batched_writes', 0)}")
            
            # Clean up memory
            del batch_data
            gc.collect()
        
        # Final flush to ensure all data is persisted
        if hasattr(train_dataset.cache, 'flush_all'):
            logging.info("Flushing all pending writes...")
            train_dataset.cache.flush_all()
        
        # Final stats
        elapsed = time.time() - start_time
        cache_size = train_dataset.cache.get_size()
        processed_batches = min(batches_to_process, batch_idx+1) if 'batch_idx' in locals() else 0
        processed_items = processed_batches * batch_size
        items_per_sec = processed_items / elapsed if elapsed > 0 else 0
        
        logging.info(f"Successfully processed {processed_batches} batches ({processed_items} items) in {elapsed:.1f} seconds")
        logging.info(f"Processing speed: {items_per_sec:.1f} items/sec")
        logging.info(f"Cache now contains {cache_size} entries")
        logging.info(f"Cache location: {train_dataset.cache.cache_path}")
        
        # Report final batching statistics
        if hasattr(train_dataset.cache, 'get_stats'):
            stats = train_dataset.cache.get_stats()
            logging.info(f"Write batching stats - Total flushes: {stats.get('flushes', 0)}, "
                        f"Batched writes: {stats.get('batched_writes', 0)}")
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        logging.info(f"Process interrupted by user after {elapsed:.1f} seconds")
        logging.info(f"Cache now contains {train_dataset.cache.get_size()} entries")
        
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    # This is required for multiprocessing with 'spawn' method
    freeze_support()

    main(workers=4, batch_size=4, max_batches=None)

