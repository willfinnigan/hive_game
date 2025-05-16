import os
from pathlib import Path
import time
from typing import List, Optional, Callable
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from tqdm import tqdm

from hive.game_engine.game_functions import current_turn_colour, get_winner
from hive.game_engine.game_state import Game
from hive.game_engine.player_functions import get_players_possible_moves_or_placements
from hive.ml.game_to_graph import Graph
from hive.ml.graph_to_pyg import graph_to_pytorch
from hive.render.to_text import game_to_text
from hive.trajectory.game_dataloader import GameDataLoader
from hive.trajectory.game_string import GameString

"""
Load games from ../game_strings and learn to make the same moves
"""

class HiveLazyGameDataset(Dataset):
    """
    PyTorch Geometric Dataset for Hive games that loads data lazily.
    This is more memory-efficient for large datasets.
    """
    def __init__(
        self,
        filepath: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        batch_size: int = 100
    ):
        super().__init__(None, transform, pre_transform)
        self.filepath = filepath
        self.batch_size = batch_size
        
        # Initialize the data loader
        self.loader = GameDataLoader(filepath, batch_size=batch_size)
        self.length = len(self.loader)
    
    def game_to_data_fn(self, game: Game) -> Data:
        graph = Graph(game)
        data = graph_to_pytorch(graph)
        return data
    
    def len(self):
        return self.length
    
    def get(self, idx):
        """Get a single game by index and convert to PyG Data."""
        game = self.loader.get_game(idx)
        if game is None:
            raise IndexError(f"Index {idx} out of range")
        
        data = self.game_to_data_fn(game)
        
        if self.transform:
            data = self.transform(data)
        
        return data


def process_games_in_batches(
    filepath: str,
    process_batch_fn: Callable[[List[Game]], None],
    batch_size: int = 100,
    max_batches: Optional[int] = None
):
    """
    Process games from a file in batches.
    
    Args:
        filepath: Path to the file containing game strings
        process_batch_fn: Function to process each batch of games
        batch_size: Number of games to process in each batch
        max_batches: Maximum number of batches to process (for testing)
    """
    # Create the loader
    loader = GameDataLoader(filepath, batch_size=batch_size)
    
    # Process batches with progress bar
    total_batches = (len(loader) + batch_size - 1) // batch_size
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    
    for i, batch in enumerate(tqdm(loader, total=total_batches, desc="Processing game batches")):
        process_batch_fn(batch)
        if max_batches is not None and i + 1 >= max_batches:
            break


if __name__ == '__main__':
    filepath = f"{Path(__file__).parents[2]}/game_strings/combined.txt"
    batch_size = 100
    loader = GameDataLoader(filepath, batch_size=batch_size)
    total_batches = (len(loader) + batch_size - 1) // batch_size
    
    print(f"Total games: {len(loader)}")
    print("Loading first batch...")

    # Create iterator once
    loader_iter = iter(loader)
    
    for i in range(1):
        games = next(loader_iter)
        for j, game in enumerate(games[0:4]):
            print(f"Game {i * batch_size + j + 1}/{len(loader)}")
            # want to check that the moves made in the game match the possible moves identified by the engine
            while game.move is not None:
                print(game.player_turns)
                colour = game.move.get_colour()
                possible_moves = get_players_possible_moves_or_placements(colour, game.parent)

                matches = [mv for mv in possible_moves if mv == game.move]

                # if len(matches) == 0 and possible_moves != []:
                #     print(game.player_turns)
                #     print(game.move)
                #     print(game_to_text(game))
                #     print(game_to_text(game.parent))

                #     for mv in possible_moves:
                #         print(f"{mv} != {game.move}")

                    
                #     print(game.parent.grid)
                #     raise ValueError(f"Move not in possible moves")

                game = game.parent
    


