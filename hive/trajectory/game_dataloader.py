import os
from typing import List, Iterator, Optional, Tuple
from dataclasses import dataclass

from hive.game_engine.game_state import Game
from hive.trajectory.game_string import GameString
from hive.trajectory.boardspace import MoveString, replay_trajectory


class GameDataLoader:
    """
    A data loader that lazily loads games from a file in batches using an index for fast random access.
    
    This class provides an iterator interface to load games in batches,
    which is memory-efficient for large datasets. It always returns Game objects.
    """
    
    def __init__(self, filepath: str, batch_size: int = 100):
        """
        Initialize the GameDataLoader.
        
        Args:
            filepath: Path to the file containing game strings
            batch_size: Number of games to load in each batch
            create_index: Whether to create an index of file positions on initialization
        """
        self.filepath = filepath
        self.batch_size = batch_size
        self.line_positions = []
        
        self._create_index()
        
        self.current_batch = 0
        self.total_batches = (len(self.line_positions) + batch_size - 1) // batch_size
    
    def _create_index(self) -> None:
        """Create an index of line positions in the file for faster random access."""
        with open(self.filepath, 'r') as f:
            pos = 0
            self.line_positions.append(pos)
            
            for line in f:
                pos += len(line)
                self.line_positions.append(pos)
    
    def __len__(self) -> int:
        """Return the total number of games."""
        return len(self.line_positions) - 1  # Subtract 1 because the last position is EOF
    
    def __iter__(self) -> 'GameDataLoader':
        """Return self as iterator."""
        self.current_batch = 0
        return self
    
    def __next__(self) -> List[Game]:
        """Get the next batch of games."""
        if self.current_batch >= self.total_batches:
            raise StopIteration
        
        batch = self.get_batch(self.current_batch)
        self.current_batch += 1
        
        return batch
    
    def get_batch(self, batch_idx: int) -> List[Game]:
        """
        Get a specific batch by index.
        
        Args:
            batch_idx: Index of the batch to retrieve
            
        Returns:
            List of Game objects
        """
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.line_positions) - 1)
        
        if start_idx >= len(self.line_positions) - 1:
            return []
        
        games = []
        
        with open(self.filepath, 'r') as f:
            for i in range(start_idx, end_idx):
                f.seek(self.line_positions[i])
                line = f.readline()
                
                try:
                    parts = line.strip().split(";")
                    game_string = GameString(
                        units=parts[0],
                        result=parts[1],
                        turn=parts[2],
                        moves=[MoveString(mv) for mv in parts[3:]]
                    )
                    
                    try:
                        # Convert GameString to Game object
                        game = replay_trajectory(game_string.moves)
                        if game is not None:
                            games.append(game)
                    except Exception as e:
                        print(f"Error replaying game at position {i}")
                        print(e)
                        continue
                        
                except Exception as e:
                    print(f"Error parsing line at position {i}")
                    print(e)
                    continue
        
        return games
    
    def get_game(self, idx: int) -> Optional[Game]:
        """
        Get a specific game by index.
        
        Args:
            idx: Index of the game to retrieve
            
        Returns:
            Game object or None if index is out of range or an error occurs
        """
        if idx < 0 or idx >= len(self.line_positions) - 1:
            return None
        
        with open(self.filepath, 'r') as f:
            f.seek(self.line_positions[idx])
            line = f.readline()
            
            try:
                parts = line.strip().split(";")
                game_string = GameString(
                    units=parts[0],
                    result=parts[1],
                    turn=parts[2],
                    moves=[MoveString(mv) for mv in parts[3:]]
                )
                
                try:
                    # Convert GameString to Game object
                    return replay_trajectory(game_string.moves)
                except Exception as e:
                    print(f"Error replaying game at position {idx}")
                    print(e)
                    return None
                    
            except Exception as e:
                print(f"Error parsing line at position {idx}")
                print(e)
                return None