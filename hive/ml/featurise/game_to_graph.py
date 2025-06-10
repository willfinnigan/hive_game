from typing import List, Optional, Tuple, Dict

from hive.game_engine.game_functions import opposite_colour
from hive.game_engine.game_state import BLACK, WHITE, Game, Piece, initial_game
from hive.game_engine.grid_functions import get_placeable_locations, positions_around_location
from hive.game_engine.moves import NoMove, get_possible_moves
from hive.game_engine.player_functions import get_players_possible_moves_or_placements, get_players_moves
from hive.ml.featurise.node_features import NodeFeatureMethod
from hive.ml.featurise.node_features import all_node_feature_methods

LocID = Tuple[Tuple[int, int], int]  # (location, stack_idx)

# Edge features are defined as follows:
# [adjacent, above, below, forward, retro, is_current_turn]


class Graph():
    """An intermediate representation on the way to a pytorch geometric graph."""

    def __init__(self, game: Game):

        self.current_colour = game.current_turn
        # Add missing current_turn attribute - this was causing the AttributeError
        self.current_turn = game.current_turn

        self.nodes: List[Node] = []
        self.nodes_by_location: Dict[LocID, Node] = {}

        self.edges = []
        self.edge_features = []
        self.edge_moves = []  # store the moves for each edge

        self._create_pass_node()  # creates pass node with a move edge to itself
        self._create_nodes_in_play(game)
        self._create_nodes_unplaced(game)
        self._create_move_edges(game)
        

        for node in self.nodes:
            node.featurise(self, game, all_node_feature_methods)
            edges, edge_features = node.create_edges(self, game)
            self.edges += edges
            self.edge_features += edge_features

    def _create_pass_node(self):
        node = Node(None, 0, None, is_current_turn=self.current_turn == self.current_colour)
        self.nodes.append(node)
        self.nodes_by_location[node.loc_id] = node

        # this edge will be added along with the other edges later
        self.edges.append((node, node))
        self.edge_features.append([0, 0, 0, 1, 0, 1])  # add edge feature for pass node
        self.edge_moves.append(hash(NoMove(colour=self.current_colour,)))  # add the hash of NoMove so we have correct number of edges
        

    def _create_nodes_in_play(self, game):
        """Create nodes from the grid."""

        # if empty grid, create a single empty node at 0, 0
        if len(game.grid) == 0:
            node = Node((0, 0), 0, None, is_current_turn=self.current_turn == self.current_colour)
            self.nodes.append(node)
            self.nodes_by_location[node.loc_id] = node
            return

        locations = set(game.grid.keys())

        # expand grid locations to include all locations 1 space away
        for loc in game.grid.keys():
            for a_loc in positions_around_location(loc):
                locations.add(a_loc)

        # add nodes for each location in the grid
        for loc in locations:
            stack = game.grid.get(loc, ())

            # if the stack is empty, create a node with no piece
            if len(stack) == 0:
                node = Node(loc, 0, None)
                self.nodes.append(node)
                self.nodes_by_location[node.loc_id] = node
            else:
                # create a node for each piece in the stack
                for i, piece in enumerate(stack):
                    node = Node(loc, i, piece)
                    self.nodes.append(node)
                    self.nodes_by_location[node.loc_id] = node


        opponent_colour = opposite_colour(self.current_colour)
        current_moves = get_players_possible_moves_or_placements(self.current_colour, game)
        opponent_moves = get_players_possible_moves_or_placements(opponent_colour, game)

        move_locations_w_stack_idx = set()
        for move in current_moves + opponent_moves:
            if isinstance(move, NoMove):
                continue

            # Add ALL move targets to the set
            move_locations_w_stack_idx.add((move.new_location, move.new_stack_idx))

        # Create empty nodes for move targets that don't already exist
        for loc, stack_idx in move_locations_w_stack_idx:
            loc_id = (loc, stack_idx)
            # Only create the node if it doesn't already exist
            if loc_id not in self.nodes_by_location:
                node = Node(loc, stack_idx, None)
                self.nodes.append(node)
                self.nodes_by_location[node.loc_id] = node

    def _create_nodes_unplaced(self, game):

        for colour, unplayed_pieces in game.unplayed_pieces.items():
            # one piece of each type
            created = set()
            for piece in unplayed_pieces:
                # if the piece has already been created, skip it
                if piece.name in created:
                    continue

                # create a node for each unplayed piece
                node = Node(None, 0, piece, is_current_turn=game.current_turn == self.current_colour)
                self.nodes.append(node)
                self.nodes_by_location[node.loc_id] = node
                created.add(piece.name)

    def _create_move_edges(self, game):
        """Create move edges for all nodes."""
        opponent_colour = opposite_colour(self.current_colour)
        current_moves = get_players_possible_moves_or_placements(self.current_colour, game)
        opponent_moves = get_players_possible_moves_or_placements(opponent_colour, game)

        for move in current_moves:
            # if move is Pass, skip it
            if isinstance(move, NoMove) == True:
                continue

            current_stack_idx = move.current_stack_idx
            if current_stack_idx is None:
                current_stack_idx = 0
            from_node = self.nodes_by_location.get((move.current_location, current_stack_idx))
            to_node = self.nodes_by_location.get((move.new_location, move.new_stack_idx))
            if from_node is None:
                print(game.grid)
                raise ValueError(
                    f"Trying to create a move edge from {(move.current_location, current_stack_idx, move.piece)}, but its not in the graph \n nodes by location: {self.nodes_by_location}")
            if to_node is None:
                print(game.grid)
                raise ValueError(
                    f"Trying to create a move edge to {(move.new_location, move.new_stack_idx)}, but its not in the graph \n nodes by location: {self.nodes_by_location}")

            # forward
            self.edges.append((from_node, to_node))
            self.edge_features.append([0, 0, 0, 1, 0, 1])
            self.edge_moves.append(hash(move))  # store the move for this edge

            # retro
            self.edges.append((to_node, from_node))
            self.edge_features.append([0, 0, 0, 0, 1, 1])

        for move in opponent_moves:
            # if move is Pass, skip it
            if isinstance(move, NoMove) == True:
                continue

            current_stack_idx = move.current_stack_idx
            if current_stack_idx is None:
                current_stack_idx = 0
            from_node = self.nodes_by_location.get((move.current_location, current_stack_idx))
            to_node = self.nodes_by_location.get((move.new_location, move.new_stack_idx))
            if from_node is None:
                print(game.grid)
                raise ValueError(
                    f"Trying to create a move edge from {(move.current_location, current_stack_idx)}, but its not in the graph \n nodes by location: {self.nodes_by_location}")
            if to_node is None:
                print(game.grid)
                raise ValueError(
                    f"Trying to create a move edge to {(move.new_location, move.new_stack_idx)}, but its not in the graph \n nodes by location: {self.nodes_by_location}")

            # forward
            self.edges.append((from_node, to_node))
            self.edge_features.append([0, 0, 0, 1, 0, 0])

            # retro
            self.edges.append((to_node, from_node))
            self.edge_features.append([0, 0, 0, 0, 1, 0])


class Node():
    def __init__(self, location: Tuple[int, int], stack_idx: int, piece: Optional[Piece],
                 is_current_turn: bool = False):
        self.location = location
        self.stack_idx = stack_idx
        self.piece = piece
        self.node_id = (location, stack_idx, piece)
        self.loc_id: LocID = (location, stack_idx)
        self.is_current_turn = is_current_turn
        self.node_features = []


    def create_edges(self, graph: Graph, game: Game):
        (adj_edges, adj_edge_features) = self._add_adjacent_edges(graph, game)
        (stack_edges, stack_edge_features) = self._add_stack_edges(graph, game)
        edges = adj_edges + stack_edges
        edge_features = adj_edge_features + stack_edge_features
        return edges, edge_features

    def featurise(self, graph: Graph, game, methods: List[NodeFeatureMethod]):
        """Return a list of features for this node."""
        for method in methods:
            self.node_features.extend(
                method(self.piece, self.location, self.stack_idx, graph.current_colour, game.grid))

    def __repr__(self):
        return f"Node({self.location}, {self.stack_idx}, {self.piece})"

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id

    def _add_adjacent_edges(self, graph: Graph, game):

        edges, edge_features = [], []

        # if node has no location, no adjacent edges
        if self.location is None:
            return [], []

        # adjacent edges only with nodes at the same stack idx
        for location in positions_around_location(self.location):
            # need to make an edge to every location around - just a case of getting if the height is 0 or 1
            stack = game.grid.get(location, ())

            if self.stack_idx > len(stack):  # if stack idx is 1 but stack is 0, then no edge
                continue

            node = graph.nodes_by_location.get((location, self.stack_idx))
            if node is not None:
                edges.append((self, node))
                edge_features.append([1, 0, 0, 0, 0, 0])  # adjacent edge
        
        return (edges, edge_features)

    def _add_stack_edges(self, graph: Graph, game: Game):

        edges, edge_features = [], []

        # if stack_idx is -1 or 0
        if self.stack_idx <= 0:
            return [], []

        # get the stack at this nodes location
        stack = game.grid.get(self.location, None)
        if stack is None:
            raise ValueError(f"Node {self.node_id} has no stack in the grid.")

        # TODO - need to change this from using the grid - because we can create empty nodes above stacks that are not in the grid.

        # if theres a piece above this piece add an edge to it.  eg stack is len 2 and height is 1
        if len(stack) > self.stack_idx + 1:  # if the total stack size is taller than the current height, then theres a piece above
            above_node = graph.nodes_by_location.get((self.location, self.stack_idx + 1))
            if above_node is None:
                raise ValueError(
                    f"Node {self.node_id} has no above node in the graph (stack height={len(stack)}) - but one was expected.")
            edges.append((self, above_node))
            edge_features.append([0, 1, 0, 0, 0, 0])  # above edge

        # if there is a piece below this piece add an edge to it
        if self.stack_idx >= 1:  # if stack height is 2 then there must be a piece below
            below_node = graph.nodes_by_location.get((self.location, self.stack_idx - 1))
            if below_node is None:
                raise ValueError(f"Node {self.node_id} has no below node in the graph - but one was expected")
            edges.append((self, below_node))
            edge_features.append([0, 0, 1, 0, 0, 0])  # below edge
    
        return (edges, edge_features)