import os
from pathlib import Path

from hive.game_engine.game_state import Game
from hive.render.to_text import game_to_text
from hive.trajectory.game_dataloader import GameDataLoader

def test_game_dataloader():
    """Test the GameDataLoader functionality with Game objects."""
    # Path to the combined.txt file
    filepath = os.path.join(Path(__file__).parents[1], "game_strings", "combined.txt")
    
    # Create a data loader with batch size of 5
    loader = GameDataLoader(filepath, batch_size=5)
    
    # Print total number of games
    print(f"Total games: {len(loader)}")
    
    # Load and print the first batch
    batch = next(iter(loader))
    print(f"Loaded {len(batch)} games in first batch")
    
    # Print information about the first game
    first_game = batch[0]
    print(f"First game type: {type(first_game).__name__}")
    print(f"Grid locations: {len(first_game.grid)}")
    print(f"Pieces on board: {sum(len(stack) for stack in first_game.grid.values())}")
    print("\nGame board:")
    print(game_to_text(first_game))
    
    # Iterate through the first 2 batches
    print("\nIterating through first 2 batches:")
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        print(f"Batch {i}: {len(batch)} games")

def test_game_dataloader_random_access():
    """Test the GameDataLoader random access functionality."""
    # Path to the combined.txt file
    filepath = os.path.join(Path(__file__).parents[1], "game_strings", "combined.txt")
    
    # Create a data loader
    loader = GameDataLoader(filepath, batch_size=5)
    
    # Print total number of games
    print(f"Total games: {len(loader)}")
    
    # Get a specific game
    game = loader.get_game(5)
    print(f"Game at index 5 type: {type(game).__name__}")
    print(f"Grid locations: {len(game.grid)}")
    print(f"Pieces on board: {sum(len(stack) for stack in game.grid.values())}")
    
    # Get a specific batch
    batch = loader.get_batch(2)
    print(f"Batch 2: {len(batch)} games")

if __name__ == "__main__":
    print("Testing GameDataLoader sequential access:")
    test_game_dataloader()
    
    print("\nTesting GameDataLoader random access:")
    test_game_dataloader_random_access()