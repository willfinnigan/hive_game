from platform import node
from typing import List, Optional

from hive.game_engine.game_functions import current_turn_colour
from hive.game_engine.game_state import WHITE, Colour, Game, Grid, Piece, initial_game
from hive.game_engine.grid_functions import get_placeable_locations, positions_around_location
from hive.game_engine.moves import get_possible_moves
from hive.ml.node_features import NodeFeatureMethod
from hive.ml.node_features import all_node_feature_methods

class Graph():
    """An intermediate representation on the way to a pytorch geometric graph."""

    def __init__(self, game: Game):

        self.game = game
        self.current_colour = current_turn_colour(game)
        
        self.nodes = []
        self.node_dict = {}
        self.nodes_by_location = {}
        
        self._create_nodes_in_play()
        self._create_nodes_unplaced()
    
        for node in self.nodes:
            node.create_edges(self)
            node.featurise(self, all_node_feature_methods)

    def _create_nodes_in_play(self):
        """Create nodes from the grid."""

        # if empty grid, create a single empty node at 0, 0
        if len(self.game.grid) == 0:
            node = Node((0, 0), 0, None)
            self.nodes.append(node)
            self.node_dict[node.node_id] = node
            self.nodes_by_location[node.loc_id] = node
            return

        locations = set(self.game.grid.keys())

        # expand locations to include all locations 1 space away
        for loc in self.game.grid.keys():
            for a_loc in positions_around_location(loc):
                locations.add(a_loc)

        for loc in locations:
            stack = self.game.grid.get(loc, ())

            # if the stack is empty, create a node with no piece
            if len(stack) == 0:
                node = Node(loc, 0, None)
                self.nodes.append(node)
                self.node_dict[node.node_id] = node
                self.nodes_by_location[node.loc_id] = node
            else:
                # create a node for each piece in the stack
                for i, piece in enumerate(stack):
                    node = Node(loc, i+1, piece)
                    self.nodes.append(node)
                    self.node_dict[node.node_id] = node
                    self.nodes_by_location[node.loc_id] = node

        # add an empty node above every stack
        for loc in locations:
            stack = self.game.grid.get(loc, ())
            if len(stack) > 0:
                # add an empty node above the stack
                node = Node(loc, len(stack)+1, None)
                self.nodes.append(node)
                self.node_dict[node.node_id] = node
                self.nodes_by_location[node.loc_id] = node


    def _create_nodes_unplaced(self):

        for colour, unplayed_pieces in self.game.unplayed_pieces.items():

            # one piece of each type
            created = set()
            for piece in unplayed_pieces:
                # if the piece has already been created, skip it
                if piece.name in created:
                    continue

                # create a node for each unplayed piece
                node = Node(None, 0, piece)
                self.nodes.append(node)
                self.node_dict[node.node_id] = node
                created.add(piece.name)

class Node():
    def __init__(self, location: tuple, height: int, piece: Optional[Piece]):
        self.location = location
        self.height = height
        self.piece = piece
        self.node_id = (location, height, piece)
        self.loc_id = (location, height)

        self.node_features = []
        self.edges = []
        self.edge_features = []
        
    def create_edges(self, graph: Graph):
        self._add_adjacent_edges(graph)
        self._add_stack_edges(graph)
        self._add_move_edges(graph)

    def featurise(self, graph: Graph, methods: List[NodeFeatureMethod]) -> List[float|int]:
        """Return a list of features for this node."""

        for method in methods:
            self.node_features.extend(method(self.piece, self.location, self.height, graph.current_colour, graph.game.grid))

    def __repr__(self):
        return f"Node({self.location}, {self.height}, {self.piece})"
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id

    def _add_adjacent_edges(self, graph: Graph):

        # if node has no location, no adjacent edges
        if self.location is None:
            return

        if self.height >= 2:
            # if height is 2 or more, then no adjacent edges
            return

        for location in positions_around_location(self.location):
            # need to make an edge to every location around - just a case of getting if the height is 0 or 1
            stack = graph.game.grid.get(location, ())
            if len(stack) == 0:
                i = 0
            else:
                i = 1
            
            node = graph.nodes_by_location.get((location, i))
            if node is not None:
                self.edges.append(node)
                self.edge_features.append([1, 0, 0, 0, 0]) # adjacent edge
        
    def _add_stack_edges(self, graph: Graph):

        # if height is 0 or 1, then no stack edges
        if self.height < 2:
            return
        
        # get the stack at this nodes location
        stack = graph.game.grid.get(self.location, None)
        if stack is None:
            raise ValueError(f"Node {self.node_id} has no stack in the grid.")
        
        # if theres a piece above this piece add an edge to it.  eg stack is len 2 and height is 1
        if len(stack) > self.height:  # if the total stack size is taller than the current height, then theres a piece above
            above_node = graph.nodes_by_location.get((self.location, self.height + 1))
            if above_node is None:
                raise ValueError(f"Node {self.node_id} has no above node in the graph (stack height={len(stack)}) - but one was expected.")
            self.edges.append(above_node)
            self.edge_features.append([0, 1, 0, 0, 0]) # above edge
        
        # if there is a piece below this piece add an edge to it
        if self.height >= 2:  # if stack height is 2 then there must be a piece below
            below_node = graph.nodes_by_location.get((self.location, self.height - 1))
            if below_node is None:
                raise ValueError(f"Node {self.node_id} has no below node in the graph - but one was expected")
            self.edges.append(below_node)
            self.edge_features.append([0, 0, 1, 0, 0]) # below edge

    def _add_move_edges(self, graph: Graph):

        # if the piece is unplaced, then moves are placements
        if self.location is None:
            placeable_locations = get_placeable_locations(graph.game.grid, self.piece.colour)
            for move in placeable_locations:
                node = graph.nodes_by_location.get((move, 0))
                if node is None:
                    raise ValueError(f"Node {self.node_id} has no move node in the graph - but one was expected")
            
                self.edges.append(node)
                self.edge_features.append([0, 0, 0, 1, 0]) # move edge

                # add the retro move
                node.edges.append(self)
                node.edge_features.append([0, 0, 0, 0, 1]) # retro move edge
            return

        # if height is 0 - there is no piece - so no moves
        if self.height == 0:
            return
        
        # if piece is not top of the stack - then no moves
        stack_height = len(graph.game.grid.get(self.location, ()))
        if self.height != stack_height:
            return

        if self.location is not None:
            # get the moves for this piece
            moves = get_possible_moves(graph.game.grid, self.location, stack_height-1)


            for move in moves:
                move_loc = move.new_location
                move_height = move.new_stack_idx

                # get the node for the move
                node = graph.nodes_by_location.get((move_loc, move_height))
                if node is None:
                    raise ValueError(f"Trying to create a move edge to {(move_loc, move_height)}, but its not in the graph \n nodes by location: {graph.nodes_by_location}")

                self.edges.append(node)
                self.edge_features.append([0, 0, 0, 1, 0]) # move edge

                # add the retro move
                node.edges.append(self)
                node.edge_features.append([0, 0, 0, 0, 1]) # retro move edge



if __name__ == '__main__':
    # Example usage
    grid = {(0, 0): (Piece(colour="WHITE", name="ANT", number=1), 
                     Piece(colour="WHITE", name="BEETLE", number=1),
                     Piece(colour="BLACK", name="BEETLE", number=1)), 
            (1, 1): (Piece(colour="WHITE", name="QUEEN", number=1),),
            (-1, -1): (Piece(colour="BLACK", name="SPIDER", number=1),),}
    game = initial_game(grid=grid)
    print()
    print(game)
    print()
    
    # Convert the game state to a graph representation
    graph = Graph(game)
    print(graph.nodes)
    print()
    for node in graph.nodes:
        print(node)
        print(f"Num edges: {len(node.edges)}")
        print(f"Edges: {node.edges}")
        print()