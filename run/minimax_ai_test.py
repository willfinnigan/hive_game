#!/usr/bin/env python3
"""
Test script for the MinimaxAI implementation.
This script sets up a game between a MinimaxAI and another AI player.
"""

import time
from hive.game_engine.game_state import WHITE, BLACK, initial_game
from hive.play.play_game import play_game
from hive.play.agents import MinimaxAI, RandomAI, ScoreBoardIn1Move_AI
from hive.play.agents.board_score.ai_generated_board_score import (
    score_board_queens_improved,
    score_board_advanced
)
from hive.render.to_text import render_game


def test_minimax_vs_random():
    """Test MinimaxAI against RandomAI"""
    game = initial_game()
    
    # Create players
    white_player = MinimaxAI(
        WHITE,
        max_depth=3,
        eval_function=score_board_advanced,
        use_iterative_deepening=True,
        time_limit=2.0
    )
    black_player = RandomAI(BLACK)
    
    print("Starting game: MinimaxAI (WHITE) vs RandomAI (BLACK)")
    print("Initial board:")
    print(render_game(game))
    
    # Play the game
    start_time = time.time()
    final_game = play_game(game, white_player, black_player, max_turns=30)
    end_time = time.time()
    
    print("\nFinal board after game:")
    print(render_game(final_game))
    print(f"Game completed in {end_time - start_time:.2f} seconds")
    
    # Determine winner
    white_queen_loc = final_game.queens.get(WHITE)
    black_queen_loc = final_game.queens.get(BLACK)
    
    if white_queen_loc is not None and black_queen_loc is not None:
        from hive.game_engine.grid_functions import pieces_around_location
        white_queen_surrounded = len(pieces_around_location(final_game.grid, white_queen_loc)) == 6
        black_queen_surrounded = len(pieces_around_location(final_game.grid, black_queen_loc)) == 6
        
        if white_queen_surrounded and black_queen_surrounded:
            print("Game ended in a draw - both queens surrounded")
        elif white_queen_surrounded:
            print("BLACK wins - WHITE's queen is surrounded")
        elif black_queen_surrounded:
            print("WHITE wins - BLACK's queen is surrounded")
        else:
            print("No winner yet - neither queen is surrounded")
    else:
        print("Game incomplete - one or both queens not placed")


def test_minimax_vs_score_board():
    """Test MinimaxAI against ScoreBoardIn1Move_AI"""
    game = initial_game()
    
    # Create players
    white_player = MinimaxAI(
        WHITE,
        max_depth=3,
        eval_function=score_board_advanced,
        use_iterative_deepening=True,
        time_limit=2.0
    )
    black_player = ScoreBoardIn1Move_AI(BLACK, score_method=score_board_queens_improved)
    
    print("Starting game: MinimaxAI (WHITE) vs ScoreBoardIn1Move_AI (BLACK)")
    print("Initial board:")
    print(render_game(game))
    
    # Play the game
    start_time = time.time()
    final_game = play_game(game, white_player, black_player, max_turns=30)
    end_time = time.time()
    
    print("\nFinal board after game:")
    print(render_game(final_game))
    print(f"Game completed in {end_time - start_time:.2f} seconds")
    
    # Determine winner
    white_queen_loc = final_game.queens.get(WHITE)
    black_queen_loc = final_game.queens.get(BLACK)
    
    if white_queen_loc is not None and black_queen_loc is not None:
        from hive.game_engine.grid_functions import pieces_around_location
        white_queen_surrounded = len(pieces_around_location(final_game.grid, white_queen_loc)) == 6
        black_queen_surrounded = len(pieces_around_location(final_game.grid, black_queen_loc)) == 6
        
        if white_queen_surrounded and black_queen_surrounded:
            print("Game ended in a draw - both queens surrounded")
        elif white_queen_surrounded:
            print("BLACK wins - WHITE's queen is surrounded")
        elif black_queen_surrounded:
            print("WHITE wins - BLACK's queen is surrounded")
        else:
            print("No winner yet - neither queen is surrounded")
    else:
        print("Game incomplete - one or both queens not placed")


def test_minimax_configuration():
    """Test different configurations of MinimaxAI"""
    game = initial_game()
    
    # Create players with different configurations
    white_player = MinimaxAI(
        WHITE,
        max_depth=4,  # Deeper search
        eval_function=score_board_advanced,
        use_iterative_deepening=True,
        time_limit=3.0  # Longer time limit
    )
    black_player = MinimaxAI(
        BLACK,
        max_depth=2,  # Shallower search
        eval_function=score_board_queens_improved,  # Simpler evaluation
        use_iterative_deepening=False,  # No iterative deepening
        time_limit=1.0  # Shorter time limit
    )
    
    print("Starting game: MinimaxAI (depth=4) vs MinimaxAI (depth=2)")
    print("Initial board:")
    print(render_game(game))
    
    # Play the game
    start_time = time.time()
    final_game = play_game(game, white_player, black_player, max_turns=20)
    end_time = time.time()
    
    print("\nFinal board after game:")
    print(render_game(final_game))
    print(f"Game completed in {end_time - start_time:.2f} seconds")
    
    # Print statistics
    print(f"WHITE (depth=4) evaluated {white_player.nodes_evaluated} nodes")
    print(f"BLACK (depth=2) evaluated {black_player.nodes_evaluated} nodes")


if __name__ == "__main__":
    print("=== MinimaxAI vs RandomAI ===")
    test_minimax_vs_random()
    
    print("\n\n=== MinimaxAI vs ScoreBoardIn1Move_AI ===")
    test_minimax_vs_score_board()
    
    print("\n\n=== MinimaxAI Configuration Test ===")
    test_minimax_configuration()