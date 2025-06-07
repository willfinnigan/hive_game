from hive.game_engine.game_state import Piece, initial_game
from hive.game_engine.moves import get_possible_moves
from hive.render.to_text import game_to_text


def test_spider_advanced():
    grid = {(-4, -2): (Piece(colour='BLACK', name='ANT', number=1),),
            (2, -2): (Piece(colour='BLACK', name='BEETLE', number=2),),
            (-1, -5): (Piece(colour='BLACK', name='SPIDER', number=1),),
            (-3, -1): (Piece(colour='BLACK', name='PILLBUG', number=1),),
            (-10, -4): (Piece(colour='WHITE', name='ANT', number=1),),
            (-6, -6): (Piece(colour='BLACK', name='LADYBUG', number=1),),
            (1, -3): (Piece(colour='WHITE', name='PILLBUG', number=1),),
            (-5, -3): (Piece(colour='BLACK', name='BEETLE', number=1),), (-1, -1): (
        Piece(colour='BLACK', name='GRASSHOPPER', number=1), Piece(colour='WHITE', name='BEETLE', number=2)),
            (0, 0): (Piece(colour='WHITE', name='LADYBUG', number=1),),
            (1, -5): (Piece(colour='BLACK', name='SPIDER', number=2),),
            (-5, -5): (Piece(colour='WHITE', name='ANT', number=2),),
            (-6, -2): (Piece(colour='WHITE', name='ANT', number=3),),
            (-1, 1): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (1, 1): (Piece(colour='WHITE', name='GRASSHOPPER', number=3),),
            (-8, -4): (Piece(colour='BLACK', name='ANT', number=3),),
            (3, -3): (Piece(colour='BLACK', name='GRASSHOPPER', number=2),), (-1, -3): (
        Piece(colour='WHITE', name='GRASSHOPPER', number=2), Piece(colour='BLACK', name='MOSQUITO', number=1),
        Piece(colour='WHITE', name='MOSQUITO', number=1)), (0, -2): (Piece(colour='BLACK', name='QUEEN', number=1),),
            (-6, -4): (Piece(colour='WHITE', name='SPIDER', number=1),),
            (-3, -5): (Piece(colour='BLACK', name='ANT', number=2),),
            #(0, -4):  (Piece(colour='WHITE', name='EMPTY', number=4),)
            }

    game = initial_game(grid=grid)

    print()
    print(game_to_text(game))

    # Move (BLACK_SPIDER_2 from (1, -5)_0 to (4, -4)_0)
    # One down is x -4
    # first step is (0, -4) does this ok.
    # then we lose it

    possible_moves = get_possible_moves(game.grid, (1, -5), 0)

    for move in possible_moves:
        print(move)
        new_game = move.play(game)
        print(game_to_text(new_game))

    move_locations = [move.new_location for move in possible_moves]
    assert (4, -4) in move_locations
