from __future__ import annotations
import networkx as nx

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hive.piece import Piece

def can_remove_piece(graph: nx.Graph, piece: Piece) -> bool:
    """ does removing piece break the graph? """
    graph_copy = graph.copy()
    graph_copy.remove_node(piece)
    subgraphs = list(nx.connected_components(graph_copy))
    if len(subgraphs) > 1:
        return False
    return True