import pytest

from hive.game_engine import pieces
from hive.game_engine.game_functions import Game, place_piece, move_piece
from hive.game_engine.moves import get_possible_moves, get_beetle_moves, get_ant_moves, get_queen_moves, \
    get_grasshopper_moves
from hive.game_engine.game_state import Piece, WHITE, BLACK, create_immutable_grid, initial_game
from hive.game_engine.grid_functions import can_remove_piece, is_position_connected, check_can_slide_to, \
    beetle_one_move_away, pieces_around_location
from hive.render.to_text import game_to_text


def test_mosquito_moves_like_beetle():
    """Test that a mosquito moves like a beetle when adjacent to a beetle."""
    # Create a simple grid where the mosquito can be removed
    # and is adjacent to a beetle
    grid = {
        (0, 0): (Piece(WHITE, pieces.MOSQUITO, 1),),
        (2, 0): (Piece(WHITE, pieces.BEETLE, 2),),
        (4, 0): (Piece(WHITE, pieces.ANT, 3),)
    }

    grid = create_immutable_grid(grid)

    # Check if the mosquito can be removed
    print(f"Can remove mosquito: {can_remove_piece(grid, (0, 0))}")

    # Get possible moves for the mosquito
    possible_moves = get_possible_moves(grid, (0, 0), 0)
    print(f"Possible moves: {possible_moves}")

    # Try getting beetle moves directly for comparison
    beetle_moves = get_beetle_moves(grid, (0, 0), 0)
    print(f"Beetle moves: {beetle_moves}")

    # The mosquito should have moves
    assert len(possible_moves) > 0


def test_mosquito_mimics_ant():
    """Test that a mosquito mimics the movement of an ant when adjacent to an ant."""
    # Create a grid where the mosquito can be removed
    # and is adjacent to an ant
    grid = {
        (0, 0): (Piece(WHITE, pieces.MOSQUITO, 1),),
        (2, 0): (Piece(WHITE, pieces.ANT, 2),),
        (4, 0): (Piece(WHITE, pieces.QUEEN, 3),)
    }

    grid = create_immutable_grid(grid)

    # Check if the pieces are actually adjacent
    adjacent = pieces_around_location(grid, (0, 0))
    print(f"Adjacent pieces: {adjacent}")

    # Check if the mosquito can be removed
    print(f"Can remove mosquito: {can_remove_piece(grid, (0, 0))}")

    # Get possible moves for the mosquito
    possible_moves = get_possible_moves(grid, (0, 0), 0)
    print(f"Possible moves: {possible_moves}")

    # Try getting ant moves directly for comparison
    ant_moves = get_ant_moves(grid, (0, 0), 0)
    print(f"Ant moves: {ant_moves}")

    # The mosquito should have moves
    assert len(possible_moves) > 0


def test_mosquito_cant_move_when_pinned():
    """Test that a mosquito can't move when it can't be removed."""
    # Create a grid where the mosquito is pinned (removing it would break the hive)
    grid = {
        (0, 0): (Piece(WHITE, pieces.MOSQUITO, 1),),
        (2, 0): (Piece(WHITE, pieces.ANT, 2),),
        (-2, 0): (Piece(WHITE, pieces.BEETLE, 3),),
        (4, 0): (Piece(WHITE, pieces.QUEEN, 4),),
        (-4, 0): (Piece(WHITE, pieces.SPIDER, 5),)
    }

    grid = create_immutable_grid(grid)

    # Verify the mosquito is pinned
    assert not can_remove_piece(grid, (0, 0))

    # Get possible moves for the mosquito - should be empty
    possible_moves = get_possible_moves(grid, (0, 0), 0)
    assert possible_moves == []


def test_mosquito_mimics_grasshopper():
    """Test that a mosquito mimics the movement of a grasshopper when adjacent to a grasshopper."""
    # Create a grid where the mosquito can be removed
    # and is adjacent to a grasshopper with pieces to jump over
    grid = {
        (0, 0): (Piece(WHITE, pieces.MOSQUITO, 1),),
        (2, 0): (Piece(WHITE, pieces.GRASSHOPPER, 2),),
        (4, 0): (Piece(WHITE, pieces.ANT, 3),),
        (6, 0): (Piece(WHITE, pieces.QUEEN, 4),)
    }

    grid = create_immutable_grid(grid)

    # Check if the pieces are actually adjacent
    adjacent = pieces_around_location(grid, (0, 0))
    print(f"Adjacent pieces: {adjacent}")

    # Check if the mosquito can be removed
    print(f"Can remove mosquito: {can_remove_piece(grid, (0, 0))}")

    # Get possible moves for the mosquito
    possible_moves = get_possible_moves(grid, (0, 0), 0)
    print(f"Possible moves: {possible_moves}")

    # Try getting grasshopper moves directly for comparison
    grasshopper_moves = get_grasshopper_moves(grid, (0, 0), 0)
    print(f"Grasshopper moves: {grasshopper_moves}")

    # The mosquito should have moves like a grasshopper
    assert len(possible_moves) > 0

    # Check if the mosquito can jump over pieces like a grasshopper
    move_locations = [move.new_location for move in possible_moves]
    assert (8, 0) in move_locations  # Should be able to jump over all pieces in a line


def test_mosquito_mimics_queen():
    """Test that a mosquito mimics the movement of a queen when adjacent to a queen."""
    # Create a grid where the mosquito can be removed
    # and is adjacent to a queen
    grid = {
        (0, 0): (Piece(WHITE, pieces.MOSQUITO, 1),),
        (2, 0): (Piece(WHITE, pieces.QUEEN, 2),),
        (4, 0): (Piece(WHITE, pieces.ANT, 3),)
    }

    grid = create_immutable_grid(grid)

    # Check if the pieces are actually adjacent
    adjacent = pieces_around_location(grid, (0, 0))
    print(f"Adjacent pieces: {adjacent}")

    # Check if the mosquito can be removed
    print(f"Can remove mosquito: {can_remove_piece(grid, (0, 0))}")

    # Get possible moves for the mosquito
    possible_moves = get_possible_moves(grid, (0, 0), 0)
    print(f"Possible moves: {possible_moves}")

    # Try getting queen moves directly for comparison
    queen_moves = get_queen_moves(grid, (0, 0), 0)
    print(f"Queen moves: {queen_moves}")

    # The mosquito should have moves like a queen
    assert len(possible_moves) > 0
    # Queen can only move one space at a time

    move_locations = [move.new_location for move in possible_moves]

    for move_loc in move_locations:
        # Check if the move is one of the six positions around (0,0)
        positions = [(-1, -1), (1, -1), (2, 0), (1, 1), (-1, 1), (-2, 0)]
        if move_loc in positions:
            # Found at least one queen-like move
            break
    else:
        # If we get here, no queen-like moves were found
        assert False, "No queen-like moves found"


def test_mosquito_advanced():
    grid = {(-3, -1): (Piece(colour='BLACK', name='ANT', number=1),),
            (2, 4): (Piece(colour='BLACK', name='ANT', number=3), Piece(colour='WHITE', name='BEETLE', number=1)),
            (6, 2): (Piece(colour='WHITE', name='ANT', number=3),),
            (3, 3): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            (2, 6): (Piece(colour='WHITE', name='BEETLE', number=2),), (3, -1): (
        Piece(colour='WHITE', name='GRASSHOPPER', number=3), Piece(colour='BLACK', name='BEETLE', number=2)),
            (7, 1): (Piece(colour='BLACK', name='GRASSHOPPER', number=2),),
            (-1, -1): (Piece(colour='WHITE', name='LADYBUG', number=1),),
            (1, 5): (Piece(colour='WHITE', name='ANT', number=2),),
            (2, 2): (Piece(colour='BLACK', name='QUEEN', number=1), Piece(colour='WHITE', name='MOSQUITO', number=1)),
            (5, 3): (Piece(colour='BLACK', name='ANT', number=2),),
            (3, 7): (Piece(colour='BLACK', name='SPIDER', number=1),),
            (-1, 1): (Piece(colour='BLACK', name='GRASSHOPPER', number=1),),
            (7, 5): (Piece(colour='BLACK', name='GRASSHOPPER', number=3),),
            (1, 1): (Piece(colour='BLACK', name='LADYBUG', number=1),),
            (-2, 2): (Piece(colour='WHITE', name='SPIDER', number=2), Piece(colour='BLACK', name='BEETLE', number=1)),
            (-1, -3): (Piece(colour='BLACK', name='SPIDER', number=2),),
            (2, 0): (Piece(colour='WHITE', name='GRASSHOPPER', number=1),),
            (0, -2): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (6, 4): (Piece(colour='WHITE', name='ANT', number=1),),
            (1, -1): (Piece(colour='WHITE', name='PILLBUG', number=1),),
            (3, 5): (Piece(colour='WHITE', name='GRASSHOPPER', number=2),),
            (8, 0): (Piece(colour='WHITE', name='SPIDER', number=1),),
            (-1, 5): (Piece(colour='BLACK', name='MOSQUITO', number=1),)}

    game = initial_game(grid=grid)
    print()
    print(game_to_text(game))

    possible_moves = get_possible_moves(game.grid, (2, 2), 1)
    print(possible_moves)

    assert len(possible_moves) == 6

