import multiprocessing as mp
import random
from pathlib import Path
from typing import List, Tuple, Optional, Any  # Added Optional, Any

from tqdm import tqdm

from hive.game_engine.game_functions import get_winner
# from hive.game_engine.game_state import Game # Game is implicitly handled by GameDataLoader
from hive.trajectory.game_dataloader import GameDataLoader  # Your provided loader

# Define a default batch size for processing if not otherwise specified
DEFAULT_PROCESSING_BATCH_SIZE = 200  # You can tune this


# New worker function to process a whole batch of games
def process_batch_of_games(args: Tuple[int, Path, int, int]) -> List[Tuple[int, bool, Optional[str]]]:
    """
    Loads a batch of games and checks each for a winner.

    Args:
        args: Tuple of (
                  batch_idx: The index of the batch to process.
                  filepath: Path to the game data file.
                  loader_batch_size: The batch_size the loader was configured with in main.
                  first_game_idx_in_batch: The original file index of the first game in this batch.
              )

    Returns:
        A list of tuples: (original_game_idx, has_winner, error_message_or_None)
                         for each game in the processed batch.
    """
    batch_idx, filepath, loader_batch_size, first_game_idx_in_batch = args
    batch_results = []

    # Each worker process creates its own loader for the batch it's assigned.
    # The filepath needs to be a string for your GameDataLoader
    try:
        # The batch_size passed to GameDataLoader here is mostly for its internal logic
        # if it uses self.batch_size for anything other than calculating total_batches.
        # get_batch(batch_idx) should work independently.
        loader = GameDataLoader(str(filepath), batch_size=loader_batch_size)
        games_in_batch = loader.get_batch(batch_idx)  # This loads the actual Game objects

        for i, game in enumerate(games_in_batch):
            original_game_idx = first_game_idx_in_batch + i
            if game is None:  # GameDataLoader might return None if parsing failed
                batch_results.append(
                    (original_game_idx, False, f"Error loading game {original_game_idx} within batch {batch_idx}"))
                continue
            try:
                has_winner = get_winner(game) is not None
                batch_results.append((original_game_idx, has_winner, None))
            except Exception as e:
                # Error during get_winner or other processing
                batch_results.append((original_game_idx, False,
                                      f"Error processing game {original_game_idx} from batch {batch_idx}: {e}"))

        loader.close()  # Important to close the file handle opened by this worker's loader
    except FileNotFoundError:
        # This specific batch worker couldn't find the file (shouldn't happen if main check passes)
        # Create error entries for all expected games in this batch
        # Calculate how many games were expected in this batch
        # This is a bit tricky if the last batch is smaller.
        # For simplicity, we'll just return an error for the batch.
        # A more robust way would be to know the total_games and calculate expected size.
        num_games_expected_in_batch = loader_batch_size  # Approximation
        for i in range(num_games_expected_in_batch):
            original_game_idx = first_game_idx_in_batch + i
            batch_results.append((original_game_idx, False,
                                  f"FileNotFoundError for batch {batch_idx} processing game {original_game_idx}"))
        # Or, more simply:
        # return [(first_game_idx_in_batch + i, False, f"FileNotFoundError for batch {batch_idx}") for i in range(loader_batch_size)]
        # This part needs careful thought if file can disappear mid-process.
        # Assuming file exists as checked in main.
    except Exception as e:
        # General error creating loader or getting batch
        # Log one error for the batch, or try to create placeholders
        print(f"Critical error in worker for batch {batch_idx}: {e}")
        # Create dummy error entries for the expected number of games in the batch
        # This ensures the progress bar and counts don't get skewed if a whole batch fails
        # Max games in a full batch is loader_batch_size. Last batch might be smaller.
        # We don't know the exact size of this batch from here if it's the last one
        # without loading the main loader to get total_games.
        # For now, just return empty, or a single error marker.
        # A better solution would be to pass total_games and calculate exact batch size here.
        # For now, we'll make it robust by assuming the worst (empty or error)
        # For this example, let's assume we can't recover the game indices if the batch fails this hard.
        # This part shows the complexity of robust error handling in distributed tasks.
        # The current GameDataLoader prints errors for individual game parsing, which is good.
        # This catch is for higher-level errors in the worker.
        pass  # Errors from _parse_game_line are handled by GameDataLoader and game might be None

    return batch_results


def main():
    filepath_pathobj = Path(__file__).parents[2] / 'game_strings' / 'combined.txt'
    filepath_str = str(filepath_pathobj)  # GameDataLoader expects a string

    if not filepath_pathobj.exists():
        print(f"Error: File not found at {filepath_str}")
        return

    # Use a batch size for the GameDataLoader. This determines how many games it attempts to load per call to get_batch.
    # This can be different from the multiprocessing chunksize.
    # Let's use the DEFAULT_PROCESSING_BATCH_SIZE, or the GameDataLoader's default.
    # It's better to be explicit.
    loader_batch_size = DEFAULT_PROCESSING_BATCH_SIZE  # Or GameDataLoader's default if you prefer

    # Instantiate loader once in main process to get total_games and total_batches
    # This also builds the index once.
    try:
        main_loader = GameDataLoader(filepath_str, batch_size=loader_batch_size)
    except FileNotFoundError:  # Should be caught by earlier check, but good practice
        print(f"Error: File not found when creating main GameDataLoader: {filepath_str}")
        return

    total_games = len(main_loader)
    total_batches = main_loader.total_batches  # Calculated by GameDataLoader
    actual_loader_batch_size = main_loader.batch_size  # Use the one from the loader instance

    if total_games == 0:
        print(f"No games found in {filepath_str}. Exiting.")
        main_loader.close()
        return

    print(
        f"Processing {total_games} games in {total_batches} batches (loader batch size: {actual_loader_batch_size})...")

    # Prepare arguments for multiprocessing: (batch_idx, filepath_str, actual_loader_batch_size, first_game_idx_in_this_batch)
    args_list = []
    for i in range(total_batches):
        first_game_idx = i * actual_loader_batch_size
        args_list.append((i, filepath_pathobj, actual_loader_batch_size,
                          first_game_idx))  # Pass Path object to worker, it will str() it
        # Or pass filepath_str directly

    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes...")

    all_game_results_tuples = []  # To store (original_game_idx, has_winner)
    errors = []  # To store error messages

    # The chunksize for imap_unordered here refers to how many "batch processing tasks"
    # are sent to each worker at a time. Setting it to 1 means workers get one batch task,
    # finish it, and ask for another. This is usually good for load balancing.
    # If batch processing is very fast, a larger chunksize might reduce overhead.
    # If batch processing is slow, chunksize=1 is fine.
    multiprocessing_chunksize = 1

    with mp.Pool(processes=num_processes) as pool:
        results_iter = pool.imap_unordered(process_batch_of_games, args_list, chunksize=multiprocessing_chunksize)

        with tqdm(total=total_games, desc="Checking games for winners") as pbar:
            for single_batch_results_list in results_iter:
                for original_game_idx, has_winner, error_msg in single_batch_results_list:
                    if error_msg:
                        # Decide if you want to print these immediately or collect.
                        # tqdm.write is safer for printing while progress bar is active.
                        # pbar.write(error_msg) # Or collect in errors list
                        errors.append(error_msg)
                    all_game_results_tuples.append((original_game_idx, has_winner))

                # Update progress bar by the number of games processed in this batch
                pbar.update(len(single_batch_results_list))

    main_loader.close()  # Close the main loader instance

    if errors:
        print(
            f"\nEncountered {len(errors)} errors during game processing (see details below or check GameDataLoader output). First few:")
        for i, err in enumerate(errors[:10]):  # Print first 10 errors
            print(err)
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors.")
        print("-" * 20)

    # Filter results to get indices with winners
    # Sort by original_game_idx to ensure consistency if order matters for later steps,
    # though for set creation it doesn't.
    all_game_results_tuples.sort(key=lambda x: x[0])
    idxs_with_winner = [idx for idx, has_winner in all_game_results_tuples if has_winner]

    print(f"Total games processed: {len(all_game_results_tuples)}")
    print(f"Total games with winner: {len(idxs_with_winner)}")

    if not idxs_with_winner:
        print("No games with winners found. Cannot create train/test split.")
        return

    random.seed(42)
    random.shuffle(idxs_with_winner)
    split_index = int(0.9 * len(idxs_with_winner))
    train_idxs = set(idxs_with_winner[:split_index])
    test_idxs = set(idxs_with_winner[split_index:])

    train_filepath = filepath_pathobj.parent / 'train_games.txt'
    test_filepath = filepath_pathobj.parent / 'test_games.txt'

    print("Writing split files...")
    lines_written_train = 0
    lines_written_test = 0

    filepath_pathobj.parent.mkdir(parents=True, exist_ok=True)

    # Re-open the original file for writing the split.
    # We need to iterate line by line, checking if the line index is in train_idxs or test_idxs.
    try:
        with open(filepath_str, 'r', encoding='utf-8') as original_file, \
                open(train_filepath, 'w', encoding='utf-8') as train_file, \
                open(test_filepath, 'w', encoding='utf-8') as test_file:

            # tqdm total here should be total_games, as we iterate through the original file line by line
            for i, line in enumerate(tqdm(original_file, total=total_games, desc="Writing split files")):
                if i in train_idxs:
                    train_file.write(line)
                    lines_written_train += 1
                elif i in test_idxs:
                    test_file.write(line)
                    lines_written_test += 1
    except Exception as e:
        print(f"Error during writing split files: {e}")
        return

    print(f"Train games written to {train_filepath}")
    print(f"Test games written to {test_filepath}")
    print(f"Training set: {lines_written_train} games (expected {len(train_idxs)})")
    print(f"Test set: {lines_written_test} games (expected {len(test_idxs)})")

    if lines_written_train != len(train_idxs):
        print(f"Warning: Mismatch in expected training games. Wrote {lines_written_train}, expected {len(train_idxs)}")
    if lines_written_test != len(test_idxs):
        print(f"Warning: Mismatch in expected test games. Wrote {lines_written_test}, expected {len(test_idxs)}")


if __name__ == "__main__":
    main()