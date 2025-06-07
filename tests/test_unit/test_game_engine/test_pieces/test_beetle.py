import pytest
from hive.game_engine import pieces
from hive.game_engine.game_functions import move_piece, place_piece
from hive.game_engine.game_state import BLACK, WHITE, Game, Piece, create_immutable_grid, initial_game
from hive.game_engine.grid_functions import positions_around_location
from hive.game_engine.moves import get_possible_moves
from hive.game_engine.player_functions import get_players_possible_moves_or_placements, get_players_moves
from hive.render.to_text import game_to_text


def test_beetle_can_move_to_empty_adjacent_space():
    """Test that a beetle can move to an empty adjacent space"""

    grid = {(0, 0): (Piece(WHITE, pieces.QUEEN, 1),),
            (2, 0): (Piece(WHITE, pieces.BEETLE, 2),)}

    game = initial_game(grid=grid)

    print(game_to_text(game))

    # Get possible moves for the beetle
    possible_moves = get_possible_moves(game.grid, (2, 0), 0)
    move_locations = [move.new_location for move in possible_moves]

    print(move_locations)

    # Beetle should be able to move to empty spaces around it
    assert (0, 0) in move_locations  # Can move onto the queen
    assert (1, -1) in move_locations  # Can move to empty space
    assert (1, 1) in move_locations  # Can move to empty space


def test_beetle_can_move_on_top_of_another_piece():
    """Test that a beetle can move on top of another piece"""
    grid = {(0, 0): (Piece(WHITE, pieces.QUEEN, 1),),
            (2, 0): (Piece(WHITE, pieces.BEETLE, 2),)}

    game = initial_game(grid=grid)

    print(game_to_text(game))

    possible_moves = get_possible_moves(game.grid, (2, 0), 0)

    moves_on_top = [move for move in possible_moves if move.new_stack_idx > 0]
    print(moves_on_top)

    new_game = moves_on_top[0].play(game)

    print(game_to_text(new_game))

    assert new_game.grid.get((0, 0)) == (Piece(WHITE, pieces.QUEEN, 1), Piece(WHITE, pieces.BEETLE, 2))


def test_beetle_on_top_of_piece_can_move():
    """Test that a beetle that is already on top of another piece can move"""
    grid = {(0, 0): (Piece(WHITE, pieces.QUEEN, 1), Piece(WHITE, pieces.BEETLE, 1)),
            (2, 0): (Piece(BLACK, pieces.ANT, 1),)}

    game = initial_game(grid=grid)

    print(game_to_text(game))

    possible_moves = get_possible_moves(game.grid, (0, 0), 1)
    for mv in possible_moves:
        print(mv)
        new_game = mv.play(game)
        print(game_to_text(new_game))

    # -2, 0 should be a possible move
    assert (-2, 0) in [move.new_location for move in possible_moves]


def test_beetle_advanced():

    grid = {(3, -5): (Piece(colour='BLACK', name='ANT', number=1),),
            (6, 2): (Piece(colour='BLACK', name='LADYBUG', number=1),),
            (2, 4): (Piece(colour='BLACK', name='GRASSHOPPER', number=2),),
            (-3, -1): (Piece(colour='BLACK', name='ANT', number=2),),
            (4, 0): (Piece(colour='BLACK', name='ANT', number=3),),
            (0, 4): (Piece(colour='BLACK', name='SPIDER', number=2),),
            (1, -3): (Piece(colour='WHITE', name='SPIDER', number=2),),
            (-5, -3): (Piece(colour='BLACK', name='SPIDER', number=1),), (-2, -2): (
        Piece(colour='WHITE', name='BEETLE', number=2), Piece(colour='WHITE', name='BEETLE', number=1),
        Piece(colour='BLACK', name='MOSQUITO', number=1)), (2, -4): (Piece(colour='WHITE', name='ANT', number=1),),
            (2, 2): (Piece(colour='WHITE', name='PILLBUG', number=1), Piece(colour='BLACK', name='BEETLE', number=2)),
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
            (4, 2): (Piece(colour='WHITE', name='SPIDER', number=1),)}

    game = initial_game(grid=grid)
    print()
    print(game_to_text(game))

    # should be able to move Black Beetle 2 to any position, because its on top
    possible_moves = get_possible_moves(game.grid, (2, 2), 1)

    print(possible_moves)
    move_locations = sorted([move.new_location for move in possible_moves])
    locs_around = sorted(positions_around_location((2, 2)))

    assert move_locations == locs_around


def test_can_move_beetle_on_double_stack():
    grid = {(0, 0): (Piece(colour="WHITE", name="ANT", number=1),
                     Piece(colour="WHITE", name="BEETLE", number=1),
                     Piece(colour="BLACK", name="BEETLE", number=1)),
            (1, 1): (Piece(colour="WHITE", name="QUEEN", number=1),),
            (-1, -1): (Piece(colour="BLACK", name="SPIDER", number=1),), }
    game = initial_game(grid=grid)

    print()
    print(game_to_text(game))

    # Get possible moves for the beetle
    black_moves = get_players_moves('BLACK', game)
    print(black_moves)

    assert len(black_moves) > 5








