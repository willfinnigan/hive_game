import os
from pathlib import Path

from hive.game_engine.game_state import WHITE, BLACK
from hive.trajectory.game_string import load_replay_game_strings
from hive.trajectory.boardspace import replay_trajectory

def test_turn_info_usage():
    """Test that turn information is correctly used when replaying games."""
    # Path to the game strings file
    filepath = os.path.join(Path(__file__).parents[0], "data", "BoardGameArena_Base+MLP+NoBots_20240704_110945.txt")
    
    # Load game strings
    game_strings = load_replay_game_strings(filepath)
    
    # Get the first game
    game_string = game_strings[0]
    
    print(f"Game type: {game_string.units}")
    print(f"Result: {game_string.result}")
    print(f"Turn info: {game_string.turn}")
    print(f"Number of moves: {len(game_string.moves)}")
    
    # Replay the game without turn info
    game_without_turn_info = replay_trajectory(game_string.moves)
    
    # Replay the game with turn info
    game_with_turn_info = replay_trajectory(game_string.moves, game_string.turn)
    
    # Check if the current turn is set correctly based on the turn info
    print(f"\nWithout turn info:")
    print(f"Current turn: {game_without_turn_info.current_turn}")
    
    print(f"\nWith turn info:")
    print(f"Current turn: {game_with_turn_info.current_turn}")
    
    # Parse the turn info to check if it matches
    import re
    match = re.match(r"(White|Black)\[(\d+)\]", game_string.turn)
    if match:
        turn_color = match.group(1)
        expected_color = WHITE if turn_color == "White" else BLACK
        print(f"\nExpected turn color from turn info: {expected_color}")
        print(f"Actual turn color in game: {game_with_turn_info.current_turn}")
        print(f"Turn info correctly applied: {expected_color == game_with_turn_info.current_turn}")
    
    # Check the colour attribute of MoveString objects
    print("\nChecking MoveString colour attributes:")
    for i, move in enumerate(game_string.moves[:5]):  # Just check the first 5 moves
        expected_colour = WHITE if i % 2 == 0 else BLACK  # White starts, then alternates
        print(f"Move {i+1}: {move.raw_string}")
        print(f"  Expected colour: {expected_colour}")
        print(f"  Actual colour: {move.colour}")

if __name__ == "__main__":
    test_turn_info_usage()