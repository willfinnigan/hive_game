"""
Examples demonstrating how to use the game string parsing functionality in the LLM-friendly renderer.

This script shows:
1. How to convert an existing game state to a string representation using game_to_string()
2. How to parse a string representation back into a game state using string_to_game()
3. Practical use cases for these functions

Author: Roo
Date: 05/05/2025
"""

import os
from hive.game.game import Game
from hive.game.pieces.ant import Ant
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.queen import Queen
from hive.game.pieces.spider import Spider
from hive.game.types_and_errors import WHITE, BLACK
from hive.render.text.large_print import large_game_to_text, game_to_string, string_to_game


def create_early_game():
    """Create an early game state with just a few pieces placed"""
    # Create pieces
    white_queen = Queen(WHITE, 0)
    black_queen = Queen(BLACK, 0)
    white_ant = Ant(WHITE, 1)
    
    # Set locations
    white_queen.location = (0, 0)
    black_queen.location = (2, 0)
    white_ant.location = (-2, 0)
    
    # Create grid
    grid = {
        (0, 0): white_queen,
        (2, 0): black_queen,
        (-2, 0): white_ant
    }
    
    # Create game
    game = Game(grid=grid)
    game.queens[WHITE] = white_queen
    game.queens[BLACK] = black_queen
    game.player_turns[WHITE] = 2
    game.player_turns[BLACK] = 1
    
    return game


def create_mid_game():
    """Create a mid-game state with more pieces and complex positioning"""
    # Create pieces
    white_queen = Queen(WHITE, 0)
    black_queen = Queen(BLACK, 0)
    white_ant1 = Ant(WHITE, 1)
    black_ant1 = Ant(BLACK, 1)
    white_beetle1 = Beetle(WHITE, 1)
    black_spider1 = Spider(BLACK, 1)
    white_grasshopper1 = GrassHopper(WHITE, 1)
    black_grasshopper1 = GrassHopper(BLACK, 1)
    white_ant2 = Ant(WHITE, 2)
    
    # Set locations
    white_queen.location = (0, 0)
    black_queen.location = (2, 0)
    white_ant1.location = (-2, 0)
    black_ant1.location = (4, 0)
    white_beetle1.location = (1, 1)
    black_spider1.location = (3, -1)
    white_grasshopper1.location = (-1, -1)
    black_grasshopper1.location = (3, 1)
    white_ant2.location = (-1, 1)
    
    # Create grid
    grid = {
        (0, 0): white_queen,
        (2, 0): black_queen,
        (-2, 0): white_ant1,
        (4, 0): black_ant1,
        (1, 1): white_beetle1,
        (3, -1): black_spider1,
        (-1, -1): white_grasshopper1,
        (3, 1): black_grasshopper1,
        (-1, 1): white_ant2
    }
    
    # Create game
    game = Game(grid=grid)
    game.queens[WHITE] = white_queen
    game.queens[BLACK] = black_queen
    game.player_turns[WHITE] = 5
    game.player_turns[BLACK] = 4
    
    return game


def example_1_basic_conversion():
    """Example 1: Basic conversion between game and string"""
    print("\n" + "=" * 50)
    print("EXAMPLE 1: BASIC CONVERSION")
    print("=" * 50)
    
    # Create an early game state
    game = create_early_game()
    
    # Display the original game
    print("\nOriginal Game (Early Game):")
    print(large_game_to_text(game, include_coordinates=True))
    
    # Convert game to string
    game_str = game_to_string(game)
    print("\nGame String Representation:")
    print(game_str)
    
    # Parse string back to game
    parsed_game = string_to_game(game_str)
    print("\nParsed Game (from string):")
    print(large_game_to_text(parsed_game, include_coordinates=True))
    
    # Verify the conversion was successful
    original_render = large_game_to_text(game)
    parsed_render = large_game_to_text(parsed_game)
    if original_render == parsed_render:
        print("\nConversion successful! Original and parsed games match.")
    else:
        print("\nConversion error! Original and parsed games don't match.")


def example_2_save_and_load():
    """Example 2: Saving a game state to a file and loading it back"""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: SAVING AND LOADING GAMES")
    print("=" * 50)
    
    # Create a mid-game state
    game = create_mid_game()
    
    # Display the original game
    print("\nOriginal Game (Mid Game):")
    print(large_game_to_text(game))
    
    # Convert game to string
    game_str = game_to_string(game)
    
    # Save the game string to a file
    filename = "saved_game.txt"
    with open(filename, "w") as f:
        f.write(game_str)
    print(f"\nGame saved to {filename}")
    
    # Load the game string from the file
    with open(filename, "r") as f:
        loaded_str = f.read()
    print(f"\nGame loaded from {filename}")
    print(f"Loaded string: {loaded_str}")
    
    # Parse string back to game
    loaded_game = string_to_game(loaded_str)
    print("\nLoaded Game:")
    print(large_game_to_text(loaded_game))
    
    # Clean up the file
    os.remove(filename)
    print(f"\nRemoved temporary file {filename}")


def example_3_create_specific_scenario():
    """Example 3: Creating a game from a specific string to test scenarios"""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: CREATING SPECIFIC SCENARIOS")
    print("=" * 50)
    
    # Define a specific game scenario as a string
    # This represents a scenario where White is about to win (Black queen is surrounded by 5 pieces)
    scenario_str = "HIVE:turns=W:8,B:8;pieces=(0,0):WQ0,(2,0):BQ0,(1,1):WA1,(3,-1):BA1,(3,1):BS1,(4,0):WB1,(1,-1):WG1"
    
    print("\nScenario String:")
    print(scenario_str)
    
    # Parse the string to create the game
    scenario_game = string_to_game(scenario_str)
    
    # Display the scenario
    print("\nScenario Game:")
    print(large_game_to_text(scenario_game, include_coordinates=True))
    
    # Add one more piece to surround the black queen (demonstrating how to modify a loaded scenario)
    black_ant2 = Ant(BLACK, 2)
    black_ant2.location = (4, -2)
    scenario_game.grid[(4, -2)] = black_ant2
    scenario_game.player_turns[BLACK] += 1
    
    # Display the modified scenario
    print("\nModified Scenario (Black queen now surrounded by 6 pieces):")
    print(large_game_to_text(scenario_game, include_coordinates=True))
    
    # Check if Black has lost
    if scenario_game.has_player_lost(BLACK):
        print("\nBlack has lost! The queen is surrounded.")
    else:
        print("\nBlack has not lost yet.")


def example_4_compare_game_states():
    """Example 4: Comparing two game states by comparing their string representations"""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: COMPARING GAME STATES")
    print("=" * 50)
    
    # Create two slightly different games
    game1 = create_mid_game()
    game2 = create_mid_game()
    
    # Modify game2 slightly (move one piece)
    white_ant2 = game2.grid[(-1, 1)]
    game2.grid.pop((-1, 1))
    white_ant2.location = (-3, 1)
    game2.grid[(-3, 1)] = white_ant2
    
    # Convert both games to strings
    game1_str = game_to_string(game1)
    game2_str = game_to_string(game2)
    
    # Display both games
    print("\nGame 1:")
    print(large_game_to_text(game1, include_coordinates=True))
    print("\nGame 1 String:")
    print(game1_str)
    
    print("\nGame 2 (White Ant 2 moved):")
    print(large_game_to_text(game2, include_coordinates=True))
    print("\nGame 2 String:")
    print(game2_str)
    
    # Compare the games
    print("\nComparing Games:")
    if game1_str == game2_str:
        print("The games are identical.")
    else:
        print("The games are different.")
        
    # Find the differences (simple string comparison for demonstration)
    print("\nDifferences in string representation:")
    parts1 = game1_str.split(',')
    parts2 = game2_str.split(',')
    
    for p1, p2 in zip(parts1, parts2):
        if p1 != p2:
            print(f"- {p1}")
            print(f"+ {p2}")


if __name__ == "__main__":
    print("GAME STRING PARSING EXAMPLES")
    print("===========================")
    print("This script demonstrates how to use the game_to_string() and string_to_game()")
    print("functions from the LLM-friendly renderer.")
    
    # Run all examples
    example_1_basic_conversion()
    example_2_save_and_load()
    example_3_create_specific_scenario()
    example_4_compare_game_states()
    
    print("\n" + "=" * 50)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 50)