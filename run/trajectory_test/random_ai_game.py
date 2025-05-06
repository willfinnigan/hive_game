#!/usr/bin/env python3
"""
Script to run a game between two random AI players and save the trajectory to a file.
"""

import os
import sys
from datetime import datetime

from hive.play.agents.random_ai import RandomAI
from hive.game_engine.game_state import WHITE, BLACK
from hive.play.play_game import play
from hive.trajectory.boardspace import record_game

def run_random_ai_game(max_turns=100, output_file=None):
    """
    Run a game between two random AI players and save the trajectory to a file.
    
    Args:
        max_turns: Maximum number of turns to play
        output_file: File to save the trajectory to. If None, a default filename is used.
    
    Returns:
        winner: The color of the winning player
    """
    # Create the random AI players
    white_player = RandomAI(WHITE)
    black_player = RandomAI(BLACK)
    
    # If output_file is None, create a default filename
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"random_ai_game_{timestamp}.txt"
    
    # Create a game controller that records the trajectory
    from hive.play.play_game import _get_next_player
    from hive.game_engine.game_state import initial_game
    
    class GameController:
        def __init__(self, player_1, player_2):
            self.player_1 = player_1
            self.player_2 = player_2
            self.game = initial_game()
            
        def get_next_player(self):
            return _get_next_player(self.game, self.player_1, self.player_2)
            
        def play(self):
            # This will be replaced by record_game
            pass
    
    # Create the game controller
    game_controller = GameController(white_player, black_player)
    
    # Record the game
    game_controller = record_game(game_controller, output_file)
    
    # Play the game
    winner = game_controller.play()
    
    return winner

if __name__ == "__main__":
    # Parse command line arguments
    output_file = None
    max_turns = 100
    
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    if len(sys.argv) > 2:
        max_turns = int(sys.argv[2])
    
    # Run the game
    run_random_ai_game(max_turns=max_turns, output_file=output_file)