#!/usr/bin/env python3
"""
Script to replay a trajectory from a file and display the final game state.
"""

import sys
from hive.game_engine.game_state import WHITE, BLACK
from hive.trajectory.boardspace import load_trajectory, replay_trajectory
from hive.render.to_text import game_to_text

def main():
    """
    Load a trajectory from a file and replay it to display the final game state.
    """
    if len(sys.argv) < 2:
        print("Usage: python replay_trajectory.py <trajectory_file>")
        return
    
    trajectory_file = sys.argv[1]
    
    # Load the trajectory
    print(f"Loading trajectory from {trajectory_file}...")
    moves = load_trajectory(trajectory_file)
    print(f"Loaded {len(moves)} moves")
    
    # Replay the trajectory
    print("Replaying trajectory...")
    final_game = replay_trajectory(moves)
    
    # Display the final game state
    print("\nFinal game state:")
    print(game_to_text(final_game))
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total moves: {len(moves)}")
    print(f"White turns: {final_game.player_turns[WHITE]}")
    print(f"Black turns: {final_game.player_turns[BLACK]}")
    print(f"Pieces on board: {len(final_game.grid)}")
    
if __name__ == "__main__":
    main()