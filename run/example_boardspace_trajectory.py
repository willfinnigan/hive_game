"""
Example script demonstrating how to use the trajectory module for saving and loading
Hive game trajectories using the BoardSpace move notation format.

This script shows:
1. How to record a game and save the trajectory
2. How to load a trajectory and replay it
3. How to convert between internal move representation and BoardSpace notation
"""

from hive.game.game import Game
from hive.game.types_and_errors import WHITE, BLACK
from hive.play.game_controller import GameController
from hive.play.player import Player
from hive.play.move import Move, NoMove
from hive.game.pieces.ant import Ant
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.queen import Queen
from hive.game.pieces.spider import Spider
from hive.render.text.large_print import large_game_to_text
from hive.trajectory.boardspace import (
    MoveString, move_to_boardspace, boardspace_to_move,
    save_trajectory, load_trajectory, replay_trajectory, record_game,
    COLOR_TO_BOARDSPACE, PIECE_TYPE_TO_LETTER
)


def example_1_manual_moves():
    """Example 1: Manually create and convert moves."""
    print("\n" + "=" * 50)
    print("EXAMPLE 1: MANUAL MOVES")
    print("=" * 50)
    
    # Create a new game
    game = Game()
    
    # Create some pieces
    white_queen = Queen(WHITE, 0)
    black_queen = Queen(BLACK, 0)
    white_ant = Ant(WHITE, 1)
    black_ant = Ant(BLACK, 1)
    
    # Create some moves following the game rules
    # 1. First piece can be placed anywhere
    # 2. Second piece must be adjacent to the first
    # 3. After that, pieces must be adjacent to at least one piece of the same color
    moves = [
        Move(white_queen, (0, 0), True),    # First move: place white queen at origin
        Move(black_queen, (2, 0), True),    # Second move: place black queen adjacent to white queen
        Move(white_ant, (-2, 0), True),     # Third move: place white ant adjacent to white queen
        Move(black_ant, (4, 0), True),      # Fourth move: place black ant adjacent to black queen
    ]
    
    # Convert moves to BoardSpace notation
    boardspace_moves = []
    
    for i, move in enumerate(moves):
        print(f"Move: {move}")
        
        # For the first move, we need special handling
        if i == 0:
            # First move is just the piece ID
            piece_id = f"{COLOR_TO_BOARDSPACE[move.piece.colour]}{PIECE_TYPE_TO_LETTER[move.piece.__class__]}{move.piece.number}"
            move_str = MoveString(piece_id)
            boardspace_moves.append(move_str)
            print(f"BoardSpace notation: {move_str.raw_string}")
        else:
            # Convert the move to BoardSpace notation
            move_str = move_to_boardspace(game, move)
            boardspace_moves.append(move_str)
            print(f"BoardSpace notation: {move_str.raw_string}")
        
        # Apply the move to update the game state
        move.play(game)
        print(large_game_to_text(game, include_coordinates=True))
        print()
    
    # Save the trajectory
    filename = "manual_moves.txt"
    save_trajectory(boardspace_moves, filename)
    print(f"Saved trajectory to {filename}")
    
    # Load the trajectory
    loaded_moves = load_trajectory(filename)
    print(f"Loaded {len(loaded_moves)} moves from {filename}")
    
    # Replay the trajectory
    replay_game = replay_trajectory(loaded_moves)
    print("Replayed game:")
    print(large_game_to_text(replay_game, include_coordinates=True))
    
    # Verify the replay matches the original
    original_render = large_game_to_text(game)
    replay_render = large_game_to_text(replay_game)
    if original_render == replay_render:
        print("Replay successful! Original and replayed games match.")
    else:
        print("Replay error! Original and replayed games don't match.")


def example_2_record_ai_game():
    """Example 2: Record a game played by AIs."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: RECORD AI GAME")
    print("=" * 50)
    
    from hive.agents.random_ai import RandomAI
    
    # Create AI players
    white_ai = RandomAI(WHITE)
    black_ai = RandomAI(BLACK)
    
    # Create a new game
    game = Game()
    
    # Create a game controller
    game_controller = GameController(white_ai, black_ai, game)
    
    # Record the game
    filename = "ai_game.txt"
    recorded_controller = record_game(game_controller, filename)
    
    # Play the game (this will record the moves)
    recorded_controller.play()
    
    # Load the recorded trajectory
    loaded_moves = load_trajectory(filename)
    print(f"Loaded {len(loaded_moves)} moves from {filename}")
    
    # Print the moves
    print("Recorded moves:")
    for i, move in enumerate(loaded_moves):
        print(f"{i+1}. {move.raw_string}")


def example_3_parse_boardspace_notation():
    """Example 3: Parse BoardSpace notation strings."""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: PARSE BOARDSPACE NOTATION")
    print("=" * 50)
    
    # Example BoardSpace notation strings
    notation_strings = [
        "wQ1",                  # First move: place white queen
        "bQ1 wQ1/",             # Second move: place black queen next to white queen
        "wS1 -bQ1",             # Third move: place white spider to the left of black queen
        "bA1 wQ1\\",            # Fourth move: place black ant to the bottom-right of white queen
        "wB1 bQ1",              # Fifth move: move white beetle onto black queen
        "pass"                  # Sixth move: pass
    ]
    
    # Parse the strings
    move_strings = [MoveString(s) for s in notation_strings]
    
    # Create a game to convert the moves
    game = Game()
    
    # Convert and apply each move
    for i, move_str in enumerate(move_strings):
        print(f"Move {i+1}: {move_str.raw_string}")
        print(f"  Piece ID: {move_str.piece_id}")
        print(f"  Reference Piece: {move_str.reference_piece_id}")
        print(f"  Direction: {move_str.direction_indicator}")
        print(f"  Is Pass: {move_str.is_pass}")
        
        # Convert to internal move
        try:
            move = boardspace_to_move(game, move_str)
            print(f"  Internal Move: {move}")
            
            # Apply the move
            move.play(game)
            print(large_game_to_text(game, include_coordinates=True))
        except Exception as e:
            print(f"  Error converting move: {e}")
        
        print()


if __name__ == "__main__":
    print("BOARDSPACE TRAJECTORY EXAMPLES")
    print("=============================")
    print("This script demonstrates how to use the trajectory module for saving and loading")
    print("Hive game trajectories using the BoardSpace move notation format.")
    
    # Run all examples
    #example_1_manual_moves()
    example_2_record_ai_game()
    #example_3_parse_boardspace_notation()
    
    print("\n" + "=" * 50)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 50)