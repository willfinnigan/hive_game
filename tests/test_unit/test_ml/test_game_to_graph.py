
from hive.game_engine.game_state import initial_game
from hive.ml.featurise.game_to_graph import Graph


def test_empty_grid_has_expected_edges():
    """Test that an empty grid has edges"""
    game = initial_game()
    graph = Graph(game)

    assert len(graph.nodes) == 17, f"{len(graph.nodes)} nodes"  # 17 nodes (16 unique pieces + 1 empty node)
    assert len(graph.edges) == 32, f"{len(graph.edges)} edges"  # 32 edges (16 unique pieces + 16 empty edges)


if __name__ == "__main__":
    test_empty_grid_has_expected_edges()
    print("All tests passed!")
    