# prepare_dataset_flexible_config.py

import json
from pathlib import Path
from multiprocessing import Pool, Manager, freeze_support, cpu_count
from functools import partial
from io import BytesIO
from queue import Empty  # Import Empty exception

import torch
import webdataset as wds
from hive.ml.featurise.endgame_to_data import process_endgame
from hive.trajectory.game_dataloader import GameDataLoader
from tqdm import tqdm


def process_chunk(chunk_info, filepath, output_dir, progress_queue):
    """
    Process a single chunk of games. This function is executed by a worker process.
    It now accepts a progress_queue to report progress back to the main process.
    """
    chunk_id, start_idx, end_idx = chunk_info
    loader = GameDataLoader(filepath)

    samples_written = 0
    pattern = f"{output_dir}/chunk-{chunk_id:03d}-shard-%06d.tar"

    with wds.ShardWriter(pattern, maxcount=10000) as sink:
        actual_end = min(end_idx, len(loader))
        for game_idx in range(start_idx, actual_end):
            game = loader.get_game(game_idx)
            if game is None:
                # Still report progress for this game attempt to keep the bar accurate
                progress_queue.put(1)
                continue

            all_data = process_endgame(game)
            for move_idx, data in enumerate(all_data):
                if data is None:
                    continue
                buffer = BytesIO()
                torch.save(data, buffer)
                pt_bytes = buffer.getvalue()
                sample_key = f"{game_idx}_{move_idx}"
                sample = {"__key__": sample_key, "pt": pt_bytes}
                sink.write(sample)
                samples_written += 1

            # Report that one game has been processed
            progress_queue.put(1)

    return samples_written


def create_webdataset(filepath, output_dir: Path = None, max_games: int = None, num_processes: int = None,
                      total_chunks: int = 100):
    """Loads games and creates a WebDataset with a specific number of chunks, showing a single progress bar."""

    if num_processes is None:
        num_processes = cpu_count()
        print(f"num_processes not specified, automatically using all available cores: {num_processes}")

    # Determine the output directory (cleaned up logic)
    if output_dir is None:
        if max_games is not None and max_games > 0:
            output_dir = Path(f'{filepath}.webdataset_{max_games}_games')
        else:
            output_dir = Path(f'{filepath}.webdataset')

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputting data to: {output_dir}")
    print(f"Using {num_processes} worker processes.")

    loader = GameDataLoader(filepath)
    total_available_games = len(loader)

    if max_games is not None and max_games > 0:
        num_games = min(total_available_games, max_games)
        print(f"Processing the first {num_games} games (out of {total_available_games} available).")
    else:
        num_games = total_available_games
        print(f"Processing all {num_games} available games.")

    if num_games == 0:
        print("No games to process. Exiting.")
        return

    games_per_chunk = (num_games + total_chunks - 1) // total_chunks
    work_items = []
    for i in range(total_chunks):
        start_idx = i * games_per_chunk
        if start_idx >= num_games:
            break
        end_idx = start_idx + games_per_chunk
        work_items.append((i, start_idx, end_idx))

    # Use a Manager to create a shared queue for progress reporting
    with Manager() as manager:
        progress_queue = manager.Queue()

        # Create a partial function with the fixed arguments, including the queue
        process_func = partial(process_chunk,
                               filepath=filepath,
                               output_dir=output_dir,
                               progress_queue=progress_queue)

        print(f"\nDistributing {len(work_items)} chunks across {num_processes} processes...")

        with Pool(processes=num_processes) as pool:
            # Use apply_async to run jobs in a non-blocking way
            result_objects = [pool.apply_async(process_func, (item,)) for item in work_items]

            # Set up the single, unified progress bar
            with tqdm(total=num_games, desc="Overall Progress") as pbar:
                while pbar.n < num_games:
                    try:
                        # Get progress update from the queue, with a timeout to prevent blocking forever
                        update_count = progress_queue.get(timeout=1)
                        pbar.update(update_count)
                    except Empty:
                        # If the queue is empty, check if all worker processes have finished
                        if all(r.ready() for r in result_objects):
                            break

            # Collect results from all workers
            results = [r.get() for r in result_objects]

    total_samples = sum(results)
    print(f"\nCompleted processing. Total samples created: {total_samples}")

    metadata = {"total_samples": total_samples}
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_path}")


if __name__ == '__main__':
    freeze_support()

    MAX_GAMES_TO_PROCESS = 100
    # MAX_GAMES_TO_PROCESS = None  # <--- OR UNCOMMENT THIS LINE FOR A FULL RUN

    FILEPATH = Path(__file__).parents[3] / 'game_strings' / 'combined.txt'
    # Use a smaller number of processes if you have fewer cores
    NUM_PROCESSES = None
    TOTAL_CHUNKS = 1  # Number of chunks to split the dataset into

    create_webdataset(FILEPATH, max_games=MAX_GAMES_TO_PROCESS,
                      num_processes=NUM_PROCESSES, total_chunks=TOTAL_CHUNKS)