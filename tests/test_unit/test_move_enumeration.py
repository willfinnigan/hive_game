from hive.game_engine.game_state import Piece, initial_game
from hive.game_engine.moves import get_possible_moves
from hive.ml.featurise.game_to_graph import Graph
from hive.render.to_text import game_to_text


def test_troublesome_grid1():
    grid = {(-2, 0): (Piece(colour='BLACK', name='GRASSHOPPER', number=1),),
            (-4, -2): (Piece(colour='WHITE', name='ANT', number=1),),
            (-5, 1): (Piece(colour='WHITE', name='LADYBUG', number=1),),
            (-7, -5): (Piece(colour='BLACK', name='BEETLE', number=2),),
            (-3, -1): (Piece(colour='BLACK', name='MOSQUITO', number=1),),
            (-5, -3): (Piece(colour='WHITE', name='SPIDER', number=2),),
            (-1, 3): (Piece(colour='BLACK', name='LADYBUG', number=1),),
            (-8, -2): (Piece(colour='BLACK', name='ANT', number=1),),
            (-3, 1): (Piece(colour='BLACK', name='ANT', number=3),),
            (0, 0): (Piece(colour='BLACK', name='GRASSHOPPER', number=2),),
            (-12, -2): (Piece(colour='BLACK', name='SPIDER', number=1),),
            (-3, -3): (Piece(colour='WHITE', name='BEETLE', number=1),),
            (-6, -2): (Piece(colour='BLACK', name='QUEEN', number=1), Piece(colour='WHITE', name='MOSQUITO', number=1)),
            (-8, -4): (Piece(colour='WHITE', name='PILLBUG', number=1),),
            (-2, -4): (Piece(colour='WHITE', name='GRASSHOPPER', number=2),),
            (-2, 2): (Piece(colour='WHITE', name='ANT', number=3),),
            (-4, 0): (Piece(colour='BLACK', name='PILLBUG', number=1), Piece(colour='WHITE', name='BEETLE', number=2), Piece(colour='BLACK', name='BEETLE', number=1)),
            (2, 0): (Piece(colour='WHITE', name='QUEEN', number=1),),
            (0, -2): (Piece(colour='WHITE', name='SPIDER', number=1),),
            (-1, -3): (Piece(colour='BLACK', name='ANT', number=2),),
            (-9, -5): (Piece(colour='BLACK', name='SPIDER', number=2),),
            (-6, -4): (Piece(colour='BLACK', name='GRASSHOPPER', number=3),),
            (-10, -2): (Piece(colour='WHITE', name='ANT', number=2),)}

    game = initial_game(grid=grid)

    print(game_to_text(game))

    graph = Graph(game)

    # check if there is a -6, -4, 1 node
    assert graph.nodes_by_location.get((-6, -4), 1) is not None


    moves = get_possible_moves(game.grid, (-7, -5), 0)

    print(moves)


    # ValueError: Trying to create a move edge to ((-6, -4), 1), but its not in the graph nodes by location