from hive.game_engine.game_state import Piece, initial_game
from hive.game_engine.moves import Move, is_pillbug_move, get_possible_moves
from hive.game_engine.game_functions import move_piece
from hive.render.to_text import game_to_text


def test_is_pillbug_move_different_color():
    """Test that is_pillbug_move correctly identifies a pillbug move with a different color piece."""
    # Create a grid with a WHITE queen and a BLACK pillbug adjacent to it
    grid = {(0, 0): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (2, 0): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            }
    
    game = initial_game(grid=grid)
    
    # Get all possible moves for the pillbug
    pillbug_moves = get_possible_moves(game.grid, (2, 0), 0)
    
    # Find the move that uses the pillbug's special ability to move the queen
    pillbug_move = None
    for move in pillbug_moves:
        if move.piece.name == 'QUEEN' and move.pillbug_moved_other_piece:
            pillbug_move = move
            break
    
    assert pillbug_move is not None, "No pillbug move found"
    
    # Play the move
    new_game = pillbug_move.play(game)
    
    # Check that is_pillbug_move correctly identifies it as a pillbug move
    assert is_pillbug_move(new_game, new_game.move), "Failed to identify pillbug move with different color piece"


def test_is_pillbug_move_same_color():
    """Test that is_pillbug_move correctly identifies a pillbug move with a same color piece."""
    # Create a grid with a WHITE queen and a WHITE pillbug adjacent to it
    grid = {(0, 0): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (2, 0): (Piece(colour='WHITE', name='PILLBUG', number=1),),
            }
    
    game = initial_game(grid=grid)
    
    # Get all possible moves for the pillbug
    pillbug_moves = get_possible_moves(game.grid, (2, 0), 0)
    
    # Find the move that uses the pillbug's special ability to move the queen
    pillbug_move = None
    for move in pillbug_moves:
        if move.piece.name == 'QUEEN' and move.pillbug_moved_other_piece:
            pillbug_move = move
            break
    
    assert pillbug_move is not None, "No pillbug move found"
    
    # Play the move
    new_game = pillbug_move.play(game)
    
    # Check that is_pillbug_move correctly identifies it as a pillbug move
    assert is_pillbug_move(new_game, new_game.move), "Failed to identify pillbug move with same color piece"


def test_not_pillbug_move():
    """Test that is_pillbug_move correctly identifies a regular move."""
    # Create a grid with a WHITE queen and a BLACK ant
    grid = {(0, 0): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (2, 0): (Piece(colour='BLACK', name='ANT', number=1),),
            }
    
    game = initial_game(grid=grid)
    
    # Create a regular move for the queen
    queen_moves = get_possible_moves(game.grid, (0, 0), 0)
    assert len(queen_moves) > 0, "No queen moves found"
    
    # Play the move
    new_game = queen_moves[0].play(game)
    
    # Check that is_pillbug_move correctly identifies it as NOT a pillbug move
    assert not is_pillbug_move(new_game, new_game.move), "Incorrectly identified regular move as pillbug move"


def test_pillbug_regular_move():
    """Test that is_pillbug_move correctly identifies a regular move by a pillbug."""
    # Create a grid with a WHITE pillbug
    grid = {(0, 0): (Piece(colour='WHITE', name='PILLBUG', number=1),),
            (2, 0): (Piece(colour='BLACK', name='ANT', number=1),),
            }
    
    game = initial_game(grid=grid)
    
    # Get regular moves for the pillbug (not using its special ability)
    pillbug_moves = get_possible_moves(game.grid, (0, 0), 0)
    
    # Find a move that is a regular move (not using the special ability)
    regular_move = None
    for move in pillbug_moves:
        if move.piece.name == 'PILLBUG' and not move.pillbug_moved_other_piece:
            regular_move = move
            break
    
    assert regular_move is not None, "No regular pillbug move found"
    
    # Play the move
    new_game = regular_move.play(game)
    
    # Check that is_pillbug_move correctly identifies it as NOT a pillbug move
    assert not is_pillbug_move(new_game, new_game.move), "Incorrectly identified pillbug's regular move as pillbug special move"

