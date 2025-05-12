
from hive.game_engine.game_state import Game, initial_game
from hive.ml.game_to_graph import Graph


def test_empty_grid_has_expected_edges():
    """Test that an empty grid has edges"""
    game = initial_game()
    graph = Graph(game)

    assert len(graph.nodes) == 17, f"{len(graph.nodes)} nodes"  # 17 nodes (16 unique pieces + 1 empty node)

    empty_node = graph.nodes_by_location[((0, 0), 0)]
    assert len(empty_node.edges) == 16,  f"{len(empty_node.edges)} edges at (0, 0), 0"   # 16 retro edges

    first_piece = graph.nodes[1]
    assert len(first_piece.edges) == 1, f"{len(first_piece.edges)} edges for unplaced piece"  # 1 placement edges (only to 0, 0)



if __name__ == "__main__":
    test_empty_grid_has_expected_edges()
    print("All tests passed!")
    