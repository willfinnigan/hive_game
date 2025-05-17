from hive.game_engine.game_state import Piece, initial_game
from hive.game_engine.moves import Move, get_possible_moves, is_pillbug_move
from hive.trajectory.boardspace import move_to_boardspace, replay_trajectory, MoveString
from hive.render.to_text import game_to_text
from hive.game_engine.grid_functions import positions_around_location


def test_pillbug_move_detection():
    """Test that is_pillbug_move correctly identifies a pillbug move."""
    # Create a grid with a WHITE spider and a BLACK pillbug adjacent to it
    grid = {(0, 0): (Piece(colour='WHITE', name='SPIDER', number=1),),
            (2, 0): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            }
    
    game = initial_game(grid=grid)
    
    print("Game state:")
    print(game_to_text(game))
    
    # Get all possible moves for the pillbug
    pillbug_moves = get_possible_moves(game.grid, (2, 0), 0)
    
    # Find the move that uses the pillbug's special ability to move the spider
    pillbug_move = None
    for move in pillbug_moves:
        if move.piece.name == 'SPIDER' and move.pillbug_moved_other_piece:
            pillbug_move = move
            break
    
    assert pillbug_move is not None, "No pillbug move found"
    print(f"Pillbug move: {pillbug_move}")
    
    # Play the move
    new_game = pillbug_move.play(game)
    
    # Check that is_pillbug_move correctly identifies it as a pillbug move
    assert is_pillbug_move(new_game, new_game.move), "Failed to identify pillbug move"
    print("Successfully identified pillbug move!")


def test_regular_move_not_detected_as_pillbug():
    """Test that is_pillbug_move correctly identifies a regular move."""
    # Create a grid with a WHITE queen and a BLACK ant
    grid = {(0, 0): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (2, 0): (Piece(colour='BLACK', name='ANT', number=1),),
            }
    
    game = initial_game(grid=grid)
    
    print("Game state:")
    print(game_to_text(game))
    
    # Get moves for the queen
    queen_moves = get_possible_moves(game.grid, (0, 0), 0)
    assert len(queen_moves) > 0, "No queen moves found"
    
    # Get a regular queen move
    regular_move = None
    for move in queen_moves:
        if not move.pillbug_moved_other_piece:
            regular_move = move
            break
    
    assert regular_move is not None, "No regular move found"
    print(f"Regular move: {regular_move}")
    
    # Play the move
    new_game = regular_move.play(game)
    
    # Check that is_pillbug_move correctly identifies it as NOT a pillbug move
    assert not is_pillbug_move(new_game, new_game.move), "Incorrectly identified regular move as pillbug move"
    print("Successfully identified regular move!")


def test_pillbug_move_detection_during_replay():
    """Test that pillbug moves are correctly detected during replay."""
    # We'll create a sequence of moves that will result in a pillbug moving a piece
    
    # First move: Place white spider at origin
    moves = [MoveString("wS1")]
    
    # Second move: Place black pillbug adjacent to spider
    moves.append(MoveString("bP1 wS1-"))
    
    # Replay the first two moves to get the game state before the pillbug move
    setup_game = replay_trajectory(moves)
    
    print("Game state before pillbug move:")
    print(game_to_text(setup_game))
    
    # Now get all possible moves for the black pillbug
    pillbug_loc = None
    for loc, stack in setup_game.grid.items():
        if stack and stack[0].name == 'PILLBUG' and stack[0].colour == 'BLACK':
            pillbug_loc = loc
            break
    
    assert pillbug_loc is not None, "Pillbug not found in game"
    
    # Get all possible moves for the pillbug
    pillbug_moves = get_possible_moves(setup_game.grid, pillbug_loc, 0)
    
    # Find the move that uses the pillbug's special ability to move the spider
    pillbug_move = None
    for move in pillbug_moves:
        if move.piece.name == 'SPIDER' and move.pillbug_moved_other_piece:
            pillbug_move = move
            break
    
    assert pillbug_move is not None, "No pillbug move found"
    print(f"Original pillbug move: {pillbug_move}")
    
    # Convert the move to BoardSpace notation
    move_str = move_to_boardspace(setup_game, pillbug_move)
    print(f"BoardSpace notation: {move_str}")
    
    # Add this move to our sequence
    moves.append(move_str)
    
    # Replay the entire game
    replayed_game = replay_trajectory(moves)
    
    # Get the last move from the replayed game
    last_move = replayed_game.move
    print(f"Replayed move: {last_move}")
    
    # Print the game state for debugging
    print("Game state after pillbug move:")
    print(game_to_text(replayed_game))
    
    # Verify that the pillbug_moved_other_piece flag is set correctly
    assert last_move.pillbug_moved_other_piece, "Pillbug move not detected during replay"
    print("Successfully detected pillbug move during replay!")


def test_pillbug_move_with_piece_that_could_move_itself():
    """
    Test that a move is identified as a pillbug move when:
    1. The piece is moved by the opposing color's pillbug
    2. The piece could have made the same move using its own abilities
    
    This should still register as a pillbug move due to the difference in color.
    """
    # Create a grid with a WHITE queen and a BLACK pillbug adjacent to it
    # The queen can move one space in any direction, and the pillbug can also move the queen
    grid = {(0, 0): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (2, 0): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            }
    
    game = initial_game(grid=grid, current_turn='BLACK')  # Set current turn to BLACK
    
    print("Game state:")
    print(game_to_text(game))
    
    # Get all possible moves for the pillbug
    pillbug_moves = get_possible_moves(game.grid, (2, 0), 0)
    
    # Find the move that uses the pillbug's special ability to move the queen
    pillbug_move = None
    for move in pillbug_moves:
        if move.piece.name == 'QUEEN' and move.pillbug_moved_other_piece:
            pillbug_move = move
            break
    
    assert pillbug_move is not None, "No pillbug move found"
    print(f"Pillbug move: {pillbug_move}")
    
    # Play the move
    new_game = pillbug_move.play(game)
    
    # Check that is_pillbug_move correctly identifies it as a pillbug move
    assert is_pillbug_move(new_game, new_game.move), "Failed to identify pillbug move"
    
    # Now, verify that the queen could have made the same move on its own
    # Get all possible moves for the queen in the original game state
    queen_moves = get_possible_moves(game.grid, (0, 0), 0)
    
    # Check if any of the queen's own moves would have resulted in the same end position
    queen_could_move_itself = False
    for move in queen_moves:
        if move.new_location == pillbug_move.new_location:
            queen_could_move_itself = True
            break
    
    assert queen_could_move_itself, "Queen should be able to make the same move on its own"
    print("Queen could make the same move on its own")
    
    # Despite the queen being able to move itself, it should still be identified as a pillbug move
    # because it was moved by the opposing color's pillbug
    assert is_pillbug_move(new_game, new_game.move), "Failed to identify as pillbug move despite color difference"
    print("Successfully identified as pillbug move due to color difference!")


def test_grasshopper_moved_by_opposing_color_pillbug():
    """
    Test a specific case where a WHITE grasshopper is moved during BLACK's turn.
    This should be identified as a pillbug move.
    
    This test is based on a real game scenario where this type of move was not
    correctly identified as a pillbug move.
    """
    # Create a grid with a WHITE grasshopper and a BLACK pillbug adjacent to it
    grid = {(0, 0): (Piece(colour='WHITE', name='GRASSHOPPER', number=3),),
            (2, 0): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            }
    
    # Create a game with BLACK as the current turn
    game = initial_game(grid=grid, current_turn='BLACK')
    
    print("Game state:")
    print(game_to_text(game))
    
    # Get all possible moves for the pillbug
    pillbug_moves = get_possible_moves(game.grid, (2, 0), 0)
    
    # Find the move that uses the pillbug's special ability to move the grasshopper
    pillbug_move = None
    for move in pillbug_moves:
        if move.piece.name == 'GRASSHOPPER' and move.pillbug_moved_other_piece:
            pillbug_move = move
            break
    
    assert pillbug_move is not None, "No pillbug move found"
    print(f"Pillbug move: {pillbug_move}")
    
    # Play the move
    new_game = pillbug_move.play(game)
    
    # Check that is_pillbug_move correctly identifies it as a pillbug move
    assert is_pillbug_move(new_game, new_game.move), "Failed to identify grasshopper move as pillbug move"
    print("Successfully identified grasshopper move as pillbug move!")
    
    # Print the move details for comparison with the game string
    print(f"Move details: {new_game.move}")
    print(f"Piece moved: {new_game.move.piece}")
    print(f"Move color: {new_game.move.colour}")
    print(f"Pillbug moved other piece: {new_game.move.pillbug_moved_other_piece}")


def test_grasshopper_moved_left_of_ladybug():
    """
    Test a specific case from the game string where a WHITE grasshopper is moved
    to the left of a BLACK ladybug during BLACK's turn.
    
    This should be identified as a pillbug move.
    """
    # Create a grid with a WHITE grasshopper, a BLACK ladybug, and a BLACK pillbug
    grid = {(0, 0): (Piece(colour='WHITE', name='GRASSHOPPER', number=3),),
            (2, 0): (Piece(colour='BLACK', name='LADYBUG', number=1),),
            (1, 1): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            }
    
    # Create a game with BLACK as the current turn
    game = initial_game(grid=grid, current_turn='BLACK')
    
    print("Game state:")
    print(game_to_text(game))
    
    # Get all possible moves for the pillbug
    pillbug_moves = get_possible_moves(game.grid, (1, 1), 0)
    
    # Find the move that uses the pillbug's special ability to move the grasshopper to the left of the ladybug
    pillbug_move = None
    for move in pillbug_moves:
        if (move.piece.name == 'GRASSHOPPER' and
            move.pillbug_moved_other_piece and
            move.new_location[0] < move.current_location[0]):  # Moving left
            pillbug_move = move
            break
    
    assert pillbug_move is not None, "No pillbug move found"
    print(f"Pillbug move: {pillbug_move}")
    
    # Play the move
    new_game = pillbug_move.play(game)
    
    # Check that is_pillbug_move correctly identifies it as a pillbug move
    assert is_pillbug_move(new_game, new_game.move), "Failed to identify grasshopper move as pillbug move"
    print("Successfully identified grasshopper move as pillbug move!")
    
    # Print the move details for comparison with the game string
    print(f"Move details: {new_game.move}")
    print(f"Piece moved: {new_game.move.piece}")
    print(f"Move color: {new_game.move.colour}")
    print(f"Pillbug moved other piece: {new_game.move.pillbug_moved_other_piece}")


if __name__ == "__main__":
    print("\n=== Testing pillbug move detection ===")
    test_pillbug_move_detection()
    
    print("\n=== Testing regular move detection ===")
    test_regular_move_not_detected_as_pillbug()
    
    print("\n=== Testing pillbug move detection during replay ===")
    test_pillbug_move_detection_during_replay()
    
    print("\n=== Testing pillbug move with piece that could move itself ===")
    test_pillbug_move_with_piece_that_could_move_itself()
    
    print("\n=== Testing grasshopper moved by opposing color pillbug ===")
    test_grasshopper_moved_by_opposing_color_pillbug()
    
    print("\n=== Testing grasshopper moved left of ladybug ===")
    test_grasshopper_moved_left_of_ladybug()
    """
    Test a specific case where a WHITE grasshopper is moved during BLACK's turn.
    This should be identified as a pillbug move.
    
    This test is based on a real game scenario where this type of move was not
    correctly identified as a pillbug move.
    """
    # Create a grid with a WHITE grasshopper and a BLACK pillbug adjacent to it
    grid = {(0, 0): (Piece(colour='WHITE', name='GRASSHOPPER', number=3),),
            (2, 0): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            }
    
    # Create a game with BLACK as the current turn
    game = initial_game(grid=grid, current_turn='BLACK')
    
    print("Game state:")
    print(game_to_text(game))
    
    # Get all possible moves for the pillbug
    pillbug_moves = get_possible_moves(game.grid, (2, 0), 0)
    
    # Find the move that uses the pillbug's special ability to move the grasshopper
    pillbug_move = None
    for move in pillbug_moves:
        if move.piece.name == 'GRASSHOPPER' and move.pillbug_moved_other_piece:
            pillbug_move = move
            break
    
    assert pillbug_move is not None, "No pillbug move found"
    print(f"Pillbug move: {pillbug_move}")
    
    # Play the move
    new_game = pillbug_move.play(game)
    
    # Check that is_pillbug_move correctly identifies it as a pillbug move
    assert is_pillbug_move(new_game, new_game.move), "Failed to identify grasshopper move as pillbug move"
    print("Successfully identified grasshopper move as pillbug move!")
    
    # Print the move details for comparison with the game string
    print(f"Move details: {new_game.move}")
    print(f"Piece moved: {new_game.move.piece}")
    print(f"Move color: {new_game.move.colour}")
    print(f"Pillbug moved other piece: {new_game.move.pillbug_moved_other_piece}")