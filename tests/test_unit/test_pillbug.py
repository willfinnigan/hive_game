from hive.game_engine.game_state import Piece, initial_game
from hive.game_engine.moves import get_possible_moves
from hive.game_engine.player_functions import get_players_moves
from hive.render.to_text import game_to_text


def test_simple_pillbug():
    grid = {(0, 0): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (2, 0): (Piece(colour='WHITE', name='PILLBUG', number=1),),
            }

    game = initial_game(grid=grid)

    print()
    print(game_to_text(game))

    # check that the pillbug can move the queen into any adjacent space
    moves = get_possible_moves(game.grid, (2, 0), 0)
    for mv in moves:
        print(mv)
        new_game = mv.play(game)
        print(game_to_text(new_game))

    move_locations = [mv.new_location for mv in moves]
    assert (4, 0) in move_locations


def test_mosquito_copies_pillbug_move():
    grid = {(0, 2): (Piece(colour='BLACK', name='BEETLE', number=1),),
            (2, 4): (Piece(colour='BLACK', name='ANT', number=3),),
            (6, 2): (Piece(colour='WHITE', name='ANT', number=3),),
            (3, 3): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            (7, 1): (Piece(colour='BLACK', name='GRASSHOPPER', number=2),),
            (-2, -2): (Piece(colour='BLACK', name='ANT', number=1),),
            (-1, -1): (Piece(colour='WHITE', name='MOSQUITO', number=1),),
            (1, 5): (Piece(colour='WHITE', name='ANT', number=2),),
            (2, 2): (Piece(colour='BLACK', name='QUEEN', number=1),),
            (0, 0): (Piece(colour='WHITE', name='GRASSHOPPER', number=3),),
            (3, 1): (Piece(colour='WHITE', name='SPIDER', number=1),),
            (5, 3): (Piece(colour='BLACK', name='ANT', number=2),),
            (-1, 1): (Piece(colour='BLACK', name='GRASSHOPPER', number=1),),
            (-3, -3): (Piece(colour='WHITE', name='LADYBUG', number=1),),
            (1, 1): (Piece(colour='BLACK', name='LADYBUG', number=1),),
            (-2, 2): (Piece(colour='WHITE', name='SPIDER', number=2),),
            (1, 3): (Piece(colour='BLACK', name='MOSQUITO', number=1),),
            (0, -2): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (-3, -1): (Piece(colour='WHITE', name='PILLBUG', number=1),)}

    game = initial_game(grid=grid)
    print()
    print(game_to_text(game))

    moves = get_possible_moves(game.grid, (-1, -1), 0)

    print(moves)
    for mv in moves:
        print(mv)

    move_locations = [mv.new_location for mv in moves]
    assert (1, -1) in move_locations


def test_pillbug_gap():
    grid = {(-2, 0): (Piece(colour='BLACK', name='SPIDER', number=1),),
            (2, 4): (Piece(colour='WHITE', name='ANT', number=2),),
            (0, 2): (Piece(colour='BLACK', name='GRASSHOPPER', number=1),),
            (-1, 3): (Piece(colour='BLACK', name='BEETLE', number=1),),
            (3, -1): (Piece(colour='BLACK', name='ANT', number=2),),
            (1, -3): (Piece(colour='BLACK', name='PILLBUG', number=1), Piece(colour='WHITE', name='BEETLE', number=2)),
            (-1, -1): (Piece(colour='BLACK', name='LADYBUG', number=1),),
            (-3, 1): (Piece(colour='BLACK', name='ANT', number=1),),
            (-2, 4): (Piece(colour='WHITE', name='BEETLE', number=1),),
            (2, 2): (Piece(colour='BLACK', name='BEETLE', number=2),),
            (0, 0): (Piece(colour='WHITE', name='PILLBUG', number=1),),
            (5, -1): (Piece(colour='WHITE', name='SPIDER', number=1),), (-1, 1): (
            Piece(colour='WHITE', name='QUEEN', number=1), Piece(colour='BLACK', name='MOSQUITO', number=1),
            Piece(colour='WHITE', name='MOSQUITO', number=1)),
            (1, 1): (Piece(colour='BLACK', name='GRASSHOPPER', number=2),),
            (-4, 0): (Piece(colour='BLACK', name='SPIDER', number=2),),
            (1, 3): (Piece(colour='BLACK', name='ANT', number=3),),
            (2, 0): (Piece(colour='WHITE', name='ANT', number=3),),
            (0, -2): (Piece(colour='BLACK', name='QUEEN', number=1),),
            (4, 2): (Piece(colour='WHITE', name='ANT', number=1),)}

    game = initial_game(grid=grid)

    # Move (BLACK_SPIDER_1 from (-2, 0)_0 to (1, -1)_0)

    print()
    print(game_to_text(game))

    moves = get_possible_moves(game.grid, (0, 0), 0)

    for mv in moves:
        print(mv)
        new_game = mv.play(game)
        print(game_to_text(new_game))


def test_pillbug_can_not_move_piece_moved_last_turn():
    grid = {(0, 0): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (2, 0): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            }

    game = initial_game(grid=grid)

    queen_move = get_possible_moves(game.grid, (0, 0), 0)[0]
    move_loc = queen_move.new_location
    queen_piece = queen_move.piece
    game = queen_move.play(game)

    # check that the pillbug can not move the queen at all
    moves = get_players_moves('BLACK', game)
    move_pieces = [mv.piece for mv in moves]
    assert queen_piece not in move_pieces


def test_piece_cannot_move_after_being_moved_by_pillbug():
    grid = {(0, 0): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (2, 0): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            }

    game = initial_game(grid=grid)

    pillbug_moves = get_possible_moves(game.grid, (2, 0), 0)

    # get move for queen
    for move in pillbug_moves:
        if move.piece.name == 'QUEEN':
            game = move.play(game)
            break

    # now check that the queen can not move
    moves = get_players_moves('WHITE', game)
    assert len(moves) == 0


def test_pillbug_slide():
    grid = {
        (-3, -1): (Piece(colour='BLACK', name='BEETLE', number=1), Piece(colour='WHITE', name='MOSQUITO', number=1)),
        (-2, 0): (Piece(colour='BLACK', name='LADYBUG', number=1),),
        (-1, -1): (Piece(colour='BLACK', name='PILLBUG', number=1),),
        (0, -2): (Piece(colour='BLACK', name='QUEEN', number=1), Piece(colour='WHITE', name='BEETLE', number=2)),
        (1, -1): (Piece(colour='WHITE', name='EMPTY', number=1),),
        (0, 0): (Piece(colour='WHITE', name='EMPTY', number=1),),
    }

    game = initial_game(grid=grid)

    print()
    print(game_to_text(game))

    possible_moves = get_possible_moves(game.grid, (-1, -1), 0)

    # nothing should be able to move to (-2, -2) because of stacked pieces either side of pillbug blocking movement
    for mv in possible_moves:
        print(mv)
        new_game = mv.play(game)
        print(game_to_text(new_game))

    move_locations = [mv.new_location for mv in possible_moves]
    assert (-2, -2) not in move_locations, "Pillbug should not be able to slide to (-2, -2) due to blocking pieces"


def test_pillbug_slide_2():
    spider_1 = Piece(colour='BLACK', name='SPIDER', number=1)
    grid = {(-2, 0): (spider_1,),
            (2, 4): (Piece(colour='WHITE', name='ANT', number=2),),
            (0, 2): (Piece(colour='BLACK', name='GRASSHOPPER', number=1),),
            (-1, 3): (Piece(colour='BLACK', name='BEETLE', number=1),),
            (3, -1): (Piece(colour='BLACK', name='ANT', number=2),),
            (1, -3): (Piece(colour='BLACK', name='PILLBUG', number=1), Piece(colour='WHITE', name='BEETLE', number=2)),
            (-1, -1): (Piece(colour='BLACK', name='LADYBUG', number=1),),
            (-3, 1): (Piece(colour='BLACK', name='ANT', number=1),),
            (-2, 4): (Piece(colour='WHITE', name='BEETLE', number=1),),
            (2, 2): (Piece(colour='BLACK', name='BEETLE', number=2),),
            (0, 0): (Piece(colour='WHITE', name='PILLBUG', number=1),),
            (5, -1): (Piece(colour='WHITE', name='SPIDER', number=1),),
            (-1, 1): (Piece(colour='WHITE', name='QUEEN', number=1), Piece(colour='BLACK', name='MOSQUITO', number=1), Piece(colour='WHITE', name='MOSQUITO', number=1)),
            (1, 1): (Piece(colour='BLACK', name='GRASSHOPPER', number=2),),
            (-4, 0): (Piece(colour='BLACK', name='SPIDER', number=2),),
            (1, 3): (Piece(colour='BLACK', name='ANT', number=3),),
            (2, 0): (Piece(colour='WHITE', name='ANT', number=3),),
            (0, -2): (Piece(colour='BLACK', name='QUEEN', number=1),),
            (4, 2): (Piece(colour='WHITE', name='ANT', number=1),)}

    game = initial_game(grid=grid)

    print()
    print(game_to_text(game))

    possible_moves = get_possible_moves(game.grid, (0, 0), 0)

    # This move should be possible
    ## PB Move (BLACK_SPIDER_1 from (-2, 0)_0 to (1, -1)_0)_s

    for mv in possible_moves:
        print(mv)

    move_locations = [mv.new_location for mv in possible_moves]
    move_pieces = [mv.piece for mv in possible_moves]

    assert spider_1 in move_pieces
    assert (1, -1) in move_locations, "Pillbug should be able to slide spider to (1, -1) due to no blocking pieces"
