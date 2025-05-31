import os
import time
from pathlib import Path
from typing import List, Iterator, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

from hive.game_engine.game_state import Game
from hive.trajectory.game_string import GameString
from hive.trajectory.boardspace import MoveString, replay_trajectory


class GameDataLoader:
    """
    A data loader that keeps a persistent file handle for better performance.

    This version opens the file once and keeps it open for the lifetime of the object,
    which is more efficient for frequent access patterns.
    """

    def __init__(self, filepath: str, batch_size: int = 100):
        """
        Initialize the GameDataLoader.

        Args:
            filepath: Path to the file containing game strings
            batch_size: Number of games to load in each batch
        """
        self.filepath = filepath
        self.batch_size = batch_size
        self.line_positions = []
        self._file = None

        # Validate file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        self._create_index()
        self.total_batches = (len(self.line_positions) + batch_size - 1) // batch_size

        # Open the file once and keep it open
        self._file = open(self.filepath, 'r', encoding='utf-8')

    def _create_index(self) -> None:
        """Create an index of line positions in the file for faster random access."""
        self.line_positions = [0]  # Start with position 0

        with open(self.filepath, 'rb') as f:  # Use binary mode for accurate positioning
            while True:
                line = f.readline()
                if not line:
                    break
                self.line_positions.append(f.tell())

        # Remove the last position (EOF)
        if len(self.line_positions) > 1:
            self.line_positions.pop()

    def __len__(self) -> int:
        """Return the total number of games."""
        return len(self.line_positions)

    def __iter__(self) -> Iterator[List[Game]]:
        """Return an iterator over batches."""
        for batch_idx in range(self.total_batches):
            yield self.get_batch(batch_idx)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the persistent file handle."""
        self.close()

    def __del__(self):
        """Destructor - ensure file is closed when object is garbage collected."""
        self.close()

    def close(self):
        """Explicitly close the file handle."""
        if self._file and not self._file.closed:
            self._file.close()
            self._file = None

    def _ensure_file_open(self):
        """Ensure the file handle is open and valid."""
        if self._file is None or self._file.closed:
            self._file = open(self.filepath, 'r', encoding='utf-8')

    def get_game(self, idx: int) -> Optional[Game]:
        """
        Get a specific game by index.

        Args:
            idx: Index of the game to retrieve

        Returns:
            Game object or None if index is out of range or an error occurs
        """
        if idx < 0 or idx >= len(self.line_positions):
            return None

        self._ensure_file_open()

        try:
            self._file.seek(self.line_positions[idx])
            line = self._file.readline().strip()
            return self._parse_game_line(line, idx)
        except (OSError, IOError) as e:
            print(f"File I/O error at position {idx}: {e}")
            return None

    def _parse_game_line(self, line: str, position: int) -> Optional[Game]:
        """
        Parse a single game line into a Game object.

        Args:
            line: The line content to parse
            position: The position/index for error reporting

        Returns:
            Game object or None if parsing fails
        """
        # Check if line is empty or too short
        if not line or len(line) < 5:
            print(f"Error at position {position}: Line is empty or too short")
            return None

        try:
            parts = line.split(";")

            # Check if we have enough parts
            if len(parts) < 4:  # Need at least units, result, turn, and one move
                print(f"Error at position {position}: Line has insufficient parts ({len(parts)})")
                print(f"Line content: {line[:100]}...")
                return None

            try:
                game_string = GameString(
                    units=parts[0],
                    result=parts[1],
                    turn=parts[2],
                    moves=[MoveString(mv) for mv in parts[3:]]
                )
            except (IndexError, ValueError) as e:
                print(f"Error at position {position}: Error creating GameString: {e}")
                return None

            try:
                # Convert GameString to Game object
                game = replay_trajectory(game_string.moves, game_string.turn)
                if game is None:
                    print(f"Error at position {position}: replay_trajectory returned None")
                    return None
                return game
            except Exception as e:
                print(f"Error replaying game at position {position}: {e}")
                return None

        except Exception as e:
            print(f"Error parsing line at position {position}: {e}")
            print(f"Line content: {line[:100]}...")
            return None

    def get_batch(self, batch_idx: int) -> List[Game]:
        """
        Get a specific batch by index.

        Args:
            batch_idx: Index of the batch to retrieve

        Returns:
            List of Game objects
        """
        if batch_idx < 0 or batch_idx >= self.total_batches:
            return []

        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.line_positions))

        games = []
        errors = 0

        self._ensure_file_open()

        try:
            for i in range(start_idx, end_idx):
                self._file.seek(self.line_positions[i])
                line = self._file.readline().strip()

                game = self._parse_game_line(line, i)
                if game is not None:
                    games.append(game)
                else:
                    errors += 1
        except (OSError, IOError) as e:
            print(f"File I/O error in batch {batch_idx}: {e}")
            return games  # Return whatever we managed to load

        if errors > 0:
            print(f"Batch {batch_idx}: {errors} errors out of {end_idx - start_idx} games")

        return games

    def get_stats(self) -> dict:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            'total_games': len(self.line_positions),
            'total_batches': self.total_batches,
            'batch_size': self.batch_size,
            'file_size_bytes': os.path.getsize(self.filepath),
            'avg_games_per_batch': len(self.line_positions) / max(1, self.total_batches),
            'file_open': self._file is not None and not self._file.closed
        }





if __name__ == "__main__":
    """Example of how to use the GameDataLoader."""
    filepath = Path(__file__).parents[2] / 'game_strings' / 'combined.txt'

    # Using as context manager (recommended)
    with GameDataLoader(filepath, batch_size=50) as loader:
        print(f"Dataset stats: {loader.get_stats()}")

        # Iterate through batches
        for batch_idx, games in enumerate(loader):
            print(f"Batch {batch_idx}: {len(games)} games")
            if batch_idx >= 5:  # Only process first 5 batches for example
                break

        # Get specific game
        game = loader.get_game(0)
        if game:
            print(f"First game loaded successfully")

    print("Performance test: 800 random game accesses")

    # Test persistent file handle approach
    start_time = time.time()
    test_indices = [0, 100, 50, 200, 25, 150, 75, 300] * 100  # 800 operations
    with GameDataLoader(filepath, batch_size=50) as loader:
        games_loaded = 0
        for idx in test_indices:
            if idx < len(loader):
                game = loader.get_game(idx)
                if game:
                    games_loaded += 1

    persistent_time = time.time() - start_time
    print(f"Persistent file handle: {persistent_time:.3f}s, {games_loaded} games loaded = {games_loaded / persistent_time:.2f} games/s")