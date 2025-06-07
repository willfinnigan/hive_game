import pytest

from hive.game_engine import pieces
from hive.game_engine.game_state import Piece, WHITE, BLACK, initial_game
from hive.game_engine.grid_functions import positions_around_location
from hive.ml.featurise.game_to_graph import Graph

@pytest.fixture
def two_piece_game():
    # Create a game with some pieces on the board

    # Note the grid coordinate system - x moves by 2, y moves by 1
    ## Doubled coordinates - double width - https://www.redblobgames.com/grids/hexagons/
    ## See grid_functions for example grid

    grid = {
        (0, 0): (Piece(WHITE, pieces.QUEEN, 1),),
        (2, 0): (Piece(BLACK, pieces.BEETLE, 1),),
    }
    game = initial_game(grid=grid)
    graph = Graph(game)

    return game, graph

def test_both_queens_are_present(two_piece_game):
    game, graph = two_piece_game

    ## check in nodes by location
    assert graph.nodes_by_location[((0, 0), 0)].piece == Piece(WHITE, pieces.QUEEN, 1)
    assert graph.nodes_by_location[((2, 0), 0)].piece == Piece(BLACK, pieces.BEETLE, 1)

    # check in nodes
    node_pieces = [node.piece for node in graph.nodes]
    assert Piece(WHITE, pieces.QUEEN, 1) in node_pieces
    assert Piece(BLACK, pieces.BEETLE, 1) in node_pieces

def test_adjacent_edge_between_two_queens(two_piece_game):
    game, graph = two_piece_game

    print(graph.nodes_by_location)

    # Check that there is an edge between the two pieces
    b_node = graph.nodes_by_location[((0, 0), 0)]
    w_node = graph.nodes_by_location[((2, 0), 0)]
    assert (b_node, w_node) in graph.edges
    assert (w_node, b_node) in graph.edges

def test_all_adjacent_empty_spaces_exist(two_piece_game):
    game, graph = two_piece_game

    # Collect all empty spaces adjacent to the two pieces
    expected_empty_spaces = []
    expected_empty_spaces += positions_around_location((0, 0))
    expected_empty_spaces += positions_around_location((2, 0))
    expected_empty_spaces = set(expected_empty_spaces)

    # Remove the occupied spaces
    expected_empty_spaces.discard((0, 0))
    expected_empty_spaces.discard((2, 0))

    # assert all the expected empty spaces are in the graph
    graph_locs = {loc for loc, _ in graph.nodes_by_location.keys()}
    assert expected_empty_spaces.issubset(graph_locs), f"Expected empty spaces {expected_empty_spaces} not found in graph locations {graph_locs}"

def test_one_empty_space_above_queen(two_piece_game):
    game, graph = two_piece_game

    for loc in graph.nodes_by_location.keys():
        print(loc)

    # Check that there is an empty space above the queen - should be ((0, 0), 1)
    assert ((0, 0), 1) in graph.nodes_by_location, "Expected empty space above the queen not found in graph nodes by location"
    assert graph.nodes_by_location[((0, 0), 1)].piece is None, "Expected empty space above the queen should have no piece"









