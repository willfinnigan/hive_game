from hive.game_engine.game_functions import place_piece
from hive.game_engine.game_state import Piece, initial_game
from hive.game_engine.grid_functions import get_placeable_locations
from hive.game_engine.player_functions import get_players_possible_moves_or_placements
from hive.render.to_text import game_to_text


def test_piece_placement_error():
    grid = {(-3, -1): (Piece(colour='BLACK', name='ANT', number=1),),
            (2, 4): (Piece(colour='BLACK', name='ANT', number=3), Piece(colour='WHITE', name='BEETLE', number=1)),
            (6, 2): (Piece(colour='WHITE', name='ANT', number=3),),
            (3, 3): (Piece(colour='BLACK', name='PILLBUG', number=1), Piece(colour='WHITE', name='MOSQUITO', number=1)),
            (2, 6): (Piece(colour='WHITE', name='BEETLE', number=2),), (3, -1): (
        Piece(colour='WHITE', name='GRASSHOPPER', number=3), Piece(colour='BLACK', name='BEETLE', number=2)),
            (7, 1): (Piece(colour='BLACK', name='GRASSHOPPER', number=2),),
            (-1, -1): (Piece(colour='WHITE', name='LADYBUG', number=1),),
            (1, 5): (Piece(colour='WHITE', name='ANT', number=2),),
            (2, 2): (Piece(colour='BLACK', name='QUEEN', number=1),),
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
            #(3, 5): (Piece(colour='WHITE', name='GRASSHOPPER', number=2),),
            (8, 0): (Piece(colour='WHITE', name='SPIDER', number=1),),
            (0, 6): (Piece(colour='BLACK', name='MOSQUITO', number=1),)}

    game = initial_game(grid=grid)

    print()

    placeable_locations = get_placeable_locations(game.grid, 'WHITE')
    print(f"Placeable locations: {placeable_locations}")

    assert (3, 5) in placeable_locations




