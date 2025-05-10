import pytest
import os
import tempfile
from hive.game_engine.game_state import Game, Piece, initial_game, WHITE, BLACK
from hive.game_engine import pieces
from hive.play.move import Move, NoMove
from hive.trajectory.boardspace import (
    MoveString, 
    get_piece_id, 
    find_piece_by_id, 
    move_to_boardspace, 
    boardspace_to_move,
    save_trajectory,
    load_trajectory,
    replay_trajectory
)


def test_move_string_parsing():
    """Test parsing of BoardSpace notation strings."""
    # Test first move
    move_str = MoveString("wQ1")
    assert move_str.piece_id == "wQ1"
    assert move_str.reference_piece_id is None
    assert move_str.direction_indicator is None
    assert move_str.is_pass is False
    
    # Test placing a piece with direction
    move_str = MoveString("bA1 wQ1-")
    assert move_str.piece_id == "bA1"
    assert move_str.reference_piece_id == "wQ1"
    assert move_str.direction_indicator == "-"
    
    # Test placing a piece with direction before reference
    move_str = MoveString("bS1 -wQ1")
    assert move_str.piece_id == "bS1"
    assert move_str.reference_piece_id == "wQ1"
    assert move_str.direction_indicator == "-"
    
    # Test beetle moving onto another piece
    move_str = MoveString("wB1 bA1")
    assert move_str.piece_id == "wB1"
    assert move_str.reference_piece_id == "bA1"
    assert move_str.direction_indicator is None
    
    # Test pass move
    move_str = MoveString("pass")
    assert move_str.is_pass is True


def test_get_piece_id():
    """Test converting a piece to BoardSpace identifier."""
    queen = Piece(colour=WHITE, name=pieces.QUEEN, number=1)
    ant = Piece(colour=BLACK, name=pieces.ANT, number=2)
    
    assert get_piece_id(queen) == "wQ1"
    assert get_piece_id(ant) == "bA2"


def test_move_conversion_simple_game():
    """Test converting between internal moves and BoardSpace notation in a simple game."""
    # Create a simple game with a few moves
    game = initial_game()
    
    # First move: Place white queen
    white_queen = Piece(colour=WHITE, name=pieces.QUEEN, number=1)
    move1 = Move(piece=white_queen, current_location=None, new_location=(0, 0), game=game)
    
    # Convert to BoardSpace notation
    move_str1 = move_to_boardspace(game, move1)
    assert move_str1.raw_string == "wQ1"
    
    # Apply the move
    game = move1.play()
    
    # Second move: Place black queen next to white queen
    black_queen = Piece(colour=BLACK, name=pieces.QUEEN, number=1)
    move2 = Move(piece=black_queen, current_location=None, new_location=(2, 0), game=game)
    
    # Convert to BoardSpace notation
    move_str2 = move_to_boardspace(game, move2)
    assert move_str2.raw_string == "bQ1 wQ1-" or move_str2.raw_string == "bQ1 wQ1/"
    
    # Convert back to internal move
    move2_converted = boardspace_to_move(game, move_str2)
    assert move2_converted.piece.colour == BLACK
    assert move2_converted.piece.name == pieces.QUEEN
    assert move2_converted.piece.number == 1
    assert move2_converted.current_location is None  # Placement
    assert move2_converted.new_location == (2, 0)


def test_pass_move_conversion():
    """Test converting a pass move to and from BoardSpace notation."""
    game = initial_game()
    
    # Create a pass move
    pass_move = NoMove(WHITE, game)
    
    # Convert to BoardSpace notation
    move_str = move_to_boardspace(game, pass_move)
    assert move_str.raw_string == "pass"
    assert move_str.is_pass is True
    
    # Convert back to internal move
    move_converted = boardspace_to_move(game, move_str)
    assert isinstance(move_converted, NoMove)
    assert move_converted.colour == WHITE


def test_save_and_load_trajectory():
    """Test saving and loading a trajectory to/from a file."""
    # Create a few move strings
    moves = [
        MoveString("wQ1"),
        MoveString("bQ1 wQ1-"),
        MoveString("wA1 wQ1/"),
        MoveString("pass")
    ]
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp_filename = temp.name
    
    try:
        save_trajectory(moves, temp_filename)
        
        # Load the trajectory
        loaded_moves = load_trajectory(temp_filename)
        
        # Check that the loaded moves match the original moves
        assert len(loaded_moves) == len(moves)
        for i, move in enumerate(moves):
            assert loaded_moves[i].raw_string == move.raw_string
    finally:
        # Clean up
        os.unlink(temp_filename)


def test_replay_trajectory():
    """Test replaying a trajectory to get the final game state."""
    # Create a simple trajectory with just the first two moves
    # This avoids the placement rule restrictions
    moves = [
        MoveString("wQ1"),                # White queen at (0,0)
        MoveString("bQ1 wQ1-"),           # Black queen at (2,0)
    ]
    
    # Replay the trajectory
    final_game = replay_trajectory(moves)
    
    # Check the final game state
    assert len(final_game.grid) == 2  # 2 pieces on the board
    
    # Check that the queens are in the right places
    assert WHITE in final_game.queens
    assert BLACK in final_game.queens
    
    # Check turn counts
    assert final_game.player_turns[WHITE] == 1  # White played 1 move
    assert final_game.player_turns[BLACK] == 1  # Black played 1 move