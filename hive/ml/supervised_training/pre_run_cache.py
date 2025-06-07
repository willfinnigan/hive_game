import gc
import os
import logging
import argparse
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

# Set multiprocessing start method in the main block, not at module level


def main():
    """Main function to run the pre-caching process"""
    # Set multiprocessing start method inside the main function
    if os.name == 'posix' and 'darwin' in os.uname().sysname.lower():
        # On macOS, use 'spawn' which is safer but has more overhead
        multiprocessing.set_start_method('spawn', force=True)
        logging.info("Using 'spawn' multiprocessing start method on macOS")
    else:
        # On other platforms, 'fork' is fine
        multiprocessing.set_start_method('fork', force=True)
        logging.info("Using 'fork' multiprocessing start method")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pre-cache dataset for faster training')
    parser.add_argument('--workers', type=int, default=0, help='Number of worker processes (0 for single-process)')
    parser.add_argument('--batches', type=int, default=None, help='Maximum number of batches to process (None for all)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    folder = Path(__file__).parents[3]
    filepath = f"{folder}/game_strings/combined.txt"
    
    logging.info(f"Creating dataset from {filepath}")
    
    # Create dataset with use_cache=True to enable SQLite caching
    train_dataset = HiveLazyGameDataset(filepath, batch_size=args.batch_size, use_cache=True)
    
    # Configure DataLoader with user-specified settings
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,  # Use command-line argument
        collate_fn=collate_fn
    )
    
    # Determine how many batches to process
    max_batches = args.batches  # None means process all batches
    total_batches = len(train_loader)
    batches_to_process = total_batches if max_batches is None else min(max_batches, total_batches)
    
    logging.info(f"Total dataset size: {len(train_dataset)} items")
    if max_batches is not None:
        logging.info(f"Limited to processing {batches_to_process} batches (max ~{batches_to_process * args.batch_size} items)")
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
            
            # Report cache size periodically
            if batch_idx % 10 == 0 and batch_idx > 0:
                cache_size = train_dataset.cache.get_size()
                elapsed = time.time() - start_time
                items_per_sec = (batch_idx * args.batch_size) / elapsed
                logging.info(f"Progress: {batch_idx}/{batches_to_process} batches, {cache_size} cached items, {items_per_sec:.1f} items/sec")
            
            # Clean up memory
            del batch_data
            gc.collect()
        
        # Final stats
        elapsed = time.time() - start_time
        cache_size = train_dataset.cache.get_size()
        processed_batches = min(batches_to_process, batch_idx+1) if 'batch_idx' in locals() else 0
        processed_items = processed_batches * args.batch_size
        items_per_sec = processed_items / elapsed if elapsed > 0 else 0
        
        logging.info(f"Successfully processed {processed_batches} batches ({processed_items} items) in {elapsed:.1f} seconds")
        logging.info(f"Processing speed: {items_per_sec:.1f} items/sec")
        logging.info(f"Cache now contains {cache_size} entries")
        logging.info(f"Cache location: {train_dataset.cache.cache_path}")
        
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
    main()

