from pathlib import Path
from hive.ml.supervised_learning import process_games_in_batches

def test_process_games_in_batches():
    total_games = 0
    total_pieces = 0
    max_batches = 2  # Only process 2 batches for testing
    
    def process_batch(batch):
        nonlocal total_games, total_pieces
        total_games += len(batch)
        
        for game in batch:
            pieces_count = sum(len(stack) for stack in game.grid.values())
            total_pieces += pieces_count
            
        print(f'Batch size: {len(batch)}')
    
    filepath = f"{Path(__file__).parents[1]}/game_strings/combined.txt"
    
    # Process games in batches
    process_games_in_batches(
        filepath=filepath,
        process_batch_fn=process_batch,
        batch_size=5,
        max_batches=max_batches
    )
    
    print(f'Total games processed: {total_games}')
    print(f'Total pieces: {total_pieces}')
    print(f'Average pieces per game: {total_pieces / total_games:.2f}')

if __name__ == "__main__":
    test_process_games_in_batches()