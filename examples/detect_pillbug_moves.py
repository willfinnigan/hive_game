"""
Example script demonstrating how to detect pillbug moves when replaying a game.
"""

from pathlib import Path
from hive.trajectory.boardspace import MoveString, replay_trajectory
from hive.trajectory.game_string import load_replay_game_strings
from hive.render.to_text import game_to_text
from hive.game_engine.moves import is_pillbug_move


def detect_pillbug_moves_in_game(moves):
    """
    Detect and print all pillbug moves in a game.
    
    Args:
        moves: List of MoveString objects representing the game moves
    """
    print("Detecting pillbug moves in game...")
    
    # Replay the game
    game = replay_trajectory(moves)
    
    # Walk back through the game history to analyze each move
    pillbug_moves = []
    regular_moves = []
    placements = []
    
    current_game = game
    while current_game.move is not None:
        move = current_game.move
        
        # Check if it's a placement
        if move.current_location is None:
            placements.append(move)
        # Check if it's a pillbug move
        elif is_pillbug_move(current_game, move):
            pillbug_moves.append(move)
        # Otherwise it's a regular move
        else:
            regular_moves.append(move)
        
        # Move to the previous game state
        current_game = current_game.parent
    
    # Print the results
    print(f"\nFound {len(pillbug_moves)} pillbug moves:")
    for i, move in enumerate(pillbug_moves):
        print(f"{i+1}. {move}")
    
    print(f"\nFound {len(regular_moves)} regular moves:")
    for i, move in enumerate(regular_moves[:5]):  # Just show the first 5 to keep output manageable
        print(f"{i+1}. {move}")
    
    print(f"\nFound {len(placements)} placements:")
    for i, move in enumerate(placements[:5]):  # Just show the first 5 to keep output manageable
        print(f"{i+1}. {move}")


def main():
    """
    Main function demonstrating how to use the pillbug move detection.
    """
    # Load a game from a file
    data_dir = Path(__file__).parents[1] / "tests" / "data"
    filepath = data_dir / "BoardGameArena_Base+MLP+NoBots_20240704_110945.txt"
    
    print(f"Loading game from {filepath}")
    game_strings = load_replay_game_strings(filepath)
    
    if not game_strings:
        print("No games found in the file.")
        return
    
    # Use the first game in the file
    game_string = game_strings[0]
    
    print(f"Game has {len(game_string.moves)} moves")
    
    # Detect pillbug moves
    detect_pillbug_moves_in_game(game_string.moves)


if __name__ == "__main__":
    main()