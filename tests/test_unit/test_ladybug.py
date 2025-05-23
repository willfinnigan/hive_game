import pytest

from hive.game_engine import pieces
from hive.game_engine.game_functions import Game, move_piece, place_piece
from hive.game_engine.moves import get_possible_moves
from hive.game_engine.game_state import Piece, WHITE, BLACK, create_immutable_grid, initial_game
from hive.render.to_text import game_to_text


def test_ladybug_basic_movement():
    """Test that a ladybug can move in a simple scenario.
    
    The ladybug moves exactly three spaces: two on top of the hive followed by one down.
    This tests a basic movement pattern where the ladybug can reach several positions.
    """
    # Create a grid with a ladybug and several pieces forming a hive
    grid = {
        (0, 0): (Piece(WHITE, pieces.LADYBUG, 1),),
        (2, 0): (Piece(WHITE, pieces.ANT, 2),),
        (4, 0): (Piece(WHITE, pieces.QUEEN, 3),),
    }

    game = initial_game(grid=grid)

    # display the grid for debugging
    print(game_to_text(game))

    # Get possible moves for the ladybug
    possible_moves = get_possible_moves(game.grid, (0, 0), 0)

    print("Possible moves for ladybug:", possible_moves)

    # Make each move and display the grid
    for move in possible_moves:
        new_game = move.play(game)
        print(game_to_text(new_game))

    move_locations = [mv.new_location for mv in possible_moves]
    assert sorted(move_locations) == sorted(
        [(3, -1), (3, 1), (5, -1), (5, 1), (6, 0)]), f"Possible moves dont match: {sorted(possible_moves)}"


def test_ladybug_cant_move_when_pinned():
    """Test that a ladybug can't move when it can't be removed from the hive."""
    # Create a grid where the ladybug is pinned (removing it would break the hive)
    grid = {
        (0, 0): (Piece(WHITE, pieces.LADYBUG, 1),),
        (2, 0): (Piece(WHITE, pieces.ANT, 2),),
        (-2, 0): (Piece(WHITE, pieces.BEETLE, 3),),
        (4, 0): (Piece(WHITE, pieces.QUEEN, 4),),
        (-4, 0): (Piece(WHITE, pieces.SPIDER, 5),)
    }

    game = initial_game(grid=grid)

    # Get possible moves for the ladybug - should be empty because it's pinned
    possible_moves = get_possible_moves(game.grid, (0, 0), 0)

    print("Possible moves for pinned ladybug:", possible_moves)

    assert possible_moves == []


def test_ladybug_advanced():
    grid = {(3, -5): (Piece(colour='BLACK', name='ANT', number=1),),
            (6, 2): (Piece(colour='BLACK', name='LADYBUG', number=1),),
            (2, 4): (Piece(colour='BLACK', name='GRASSHOPPER', number=2),),
            (-3, -1): (Piece(colour='BLACK', name='ANT', number=2),),
            (4, 0): (Piece(colour='BLACK', name='ANT', number=3),),
            (0, 4): (Piece(colour='BLACK', name='SPIDER', number=2),),
            (1, -3): (Piece(colour='WHITE', name='SPIDER', number=2),),
            (-5, -3): (Piece(colour='BLACK', name='SPIDER', number=1),),
            (-2, -2): (Piece(colour='WHITE', name='BEETLE', number=2), Piece(colour='WHITE', name='BEETLE', number=1), Piece(colour='BLACK', name='MOSQUITO', number=1)),
            (2, -4): (Piece(colour='WHITE', name='ANT', number=1),),
            (2, 2): (Piece(colour='WHITE', name='PILLBUG', number=1),),
            (0, 0): (Piece(colour='WHITE', name='LADYBUG', number=1),),
            (3, 1): (Piece(colour='WHITE', name='GRASSHOPPER', number=3),),
            (-3, -3): (Piece(colour='WHITE', name='GRASSHOPPER', number=2),),
            (-2, -4): (Piece(colour='WHITE', name='GRASSHOPPER', number=1),),
            (1, 1): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (1, 3): (Piece(colour='WHITE', name='ANT', number=3),),
            (-1, -3): (Piece(colour='WHITE', name='ANT', number=2),), (0, -2): (
        Piece(colour='BLACK', name='QUEEN', number=1), Piece(colour='WHITE', name='MOSQUITO', number=1),
        Piece(colour='BLACK', name='BEETLE', number=1)), (1, -1): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            (-3, -5): (Piece(colour='BLACK', name='GRASSHOPPER', number=1),),
            (4, 2): (Piece(colour='WHITE', name='SPIDER', number=1), Piece(colour='BLACK', name='BEETLE', number=2))}

    game = initial_game(grid=grid)

    print()
    for colour, pieces in game.unplayed_pieces.items():
        for piece in pieces:
            print(piece)

    print(game_to_text(game))
