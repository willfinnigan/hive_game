from typing import List, Optional

from hive.game_engine.game_state import BLACK, WHITE, Game, Piece, initial_game
from hive.game_engine.grid_functions import get_placeable_locations, positions_around_location
from hive.game_engine.moves import NoMove, get_possible_moves
from hive.game_engine.player_functions import get_players_possible_moves_or_placements, get_players_moves
from hive.ml.featurise.node_features import NodeFeatureMethod
from hive.ml.featurise.node_features import all_node_feature_methods

class Graph():
    """An intermediate representation on the way to a pytorch geometric graph."""

    def __init__(self, game: Game):

        self.game = game
        self.current_colour = game.current_turn
        
        self.nodes = []
        self.node_dict = {}
        self.nodes_by_location = {}

        self.edges = []
        self.edge_features = []
        self.edge_moves = []  # store the moves for each edge
        
        self._create_nodes_in_play()
        self._create_nodes_unplaced()

        self._create_move_edges()
    
        for node in self.nodes:
            node.featurise(self, all_node_feature_methods)
            node.create_edges(self)
            self.edges += node.edges
            self.edge_features += node.edge_features

        
    def _create_nodes_in_play(self):
        """Create nodes from the grid."""

        # if empty grid, create a single empty node at 0, 0
        if len(self.game.grid) == 0:
            node = Node((0, 0), 0, None, is_current_turn=self.game.current_turn == self.current_colour)
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
                    node = Node(loc, i, piece)
                    self.nodes.append(node)
                    self.node_dict[node.node_id] = node
                    self.nodes_by_location[node.loc_id] = node


        # add an empty node above every stack where there is a piece that can move there
        opponent_colour = BLACK if self.current_colour == WHITE else WHITE
        current_moves = get_players_moves(self.current_colour, self.game)
        opponent_moves = get_players_moves(opponent_colour, self.game)

        move_locations_w_stack_idx = set()
        for move in current_moves+opponent_moves:
            if move.new_stack_idx <= 0:
                continue  # not interested in moves to ground level - already covered

            # add the location and height to the set
            move_locations_w_stack_idx.add((move.new_location, move.new_stack_idx))

        for loc, stack_idx in move_locations_w_stack_idx:
            node = Node(loc, stack_idx, None)
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
                node = Node(None, 0, piece, is_current_turn=self.game.current_turn == self.current_colour)
                self.nodes.append(node)
                self.node_dict[node.node_id] = node
                created.add(piece.name)

    def _create_move_edges(self):
        """Create move edges for all nodes."""
        opponent_colour = BLACK if self.current_colour == WHITE else WHITE
        current_moves = get_players_possible_moves_or_placements(self.current_colour, self.game)
        opponent_moves = get_players_possible_moves_or_placements(self.game.current_turn, self.game)
        
        for move in current_moves:
            # if move is Pass, skip it
            if isinstance(move, NoMove) == True:
                continue

            current_stack_idx = move.current_stack_idx
            if current_stack_idx is None:
                current_stack_idx = 0
            from_node = self.node_dict.get((move.current_location, current_stack_idx, move.piece))
            to_node = self.nodes_by_location.get((move.new_location, move.new_stack_idx))
            if from_node is None:
                raise ValueError(f"Trying to create a move edge from {(move.current_location, current_stack_idx, move.piece)}, but its not in the graph \n nodes by location: {self.nodes_by_location}")
            if to_node is None:
                raise ValueError(f"Trying to create a move edge to {(move.new_location, move.new_stack_idx)}, but its not in the graph \n nodes by location: {self.nodes_by_location}")
            
            # forward
            self.edges.append((from_node, to_node))
            self.edge_features.append([0, 0, 0, 1, 0, 1])
            self.edge_moves.append(move)  # store the move for this edge

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
            from_node = self.node_dict.get((move.current_location, current_stack_idx, move.piece))
            to_node = self.nodes_by_location.get((move.new_location, move.new_stack_idx))
            if from_node is None:
                raise ValueError(f"Trying to create a move edge from {(move.current_location, current_stack_idx)}, but its not in the graph \n nodes by location: {self.nodes_by_location}")
            if to_node is None:
                raise ValueError(f"Trying to create a move edge to {(move.new_location, move.new_stack_idx)}, but its not in the graph \n nodes by location: {self.nodes_by_location}")
            
            # forward
            self.edges.append((from_node, to_node))
            self.edge_features.append([0, 0, 0, 1, 0, 0])

            # retro
            self.edges.append((to_node, from_node))
            self.edge_features.append([0, 0, 0, 0, 1, 0])




class Node():
    def __init__(self, location: tuple, stack_idx: int, piece: Optional[Piece], is_current_turn: bool=False):
        self.location = location
        self.stack_idx = stack_idx
        self.piece = piece
        self.node_id = (location, stack_idx, piece)
        self.loc_id = (location, stack_idx)
        self.is_current_turn = is_current_turn
        self.node_features = []
        self.edges = []
        self.edge_features = []
        
    def create_edges(self, graph: Graph):
        self._add_adjacent_edges(graph)
        self._add_stack_edges(graph)

    def featurise(self, graph: Graph, methods: List[NodeFeatureMethod]) -> List[float|int]:
        """Return a list of features for this node."""

        for method in methods:
            self.node_features.extend(method(self.piece, self.location, self.stack_idx, graph.current_colour, graph.game.grid))

    def __repr__(self):
        return f"Node({self.location}, {self.stack_idx}, {self.piece})"
    
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

        # adjacent edges only with nodes at the same stack idx

        for location in positions_around_location(self.location):
            # need to make an edge to every location around - just a case of getting if the height is 0 or 1
            stack = graph.game.grid.get(location, ())

            if self.stack_idx > len(stack):  # if stack idx is 1 but stack is 0, then no edge
                continue

            node = graph.nodes_by_location.get((location, self.stack_idx))
            if node is not None:
                self.edges.append((self, node))
                self.edge_features.append([1, 0, 0, 0, 0, 0]) # adjacent edge
        
    def _add_stack_edges(self, graph: Graph):

        # if stack_idx is -1 or 0
        if self.stack_idx <= 0:
            return
        
        # get the stack at this nodes location
        stack = graph.game.grid.get(self.location, None)
        if stack is None:
            raise ValueError(f"Node {self.node_id} has no stack in the grid.")
        
        # if theres a piece above this piece add an edge to it.  eg stack is len 2 and height is 1
        if len(stack) > self.stack_idx+1:  # if the total stack size is taller than the current height, then theres a piece above
            above_node = graph.nodes_by_location.get((self.location, self.stack_idx+1))
            if above_node is None:
                raise ValueError(f"Node {self.node_id} has no above node in the graph (stack height={len(stack)}) - but one was expected.")
            self.edges.append((self, above_node))
            self.edge_features.append([0, 1, 0, 0, 0, 0]) # above edge
        
        # if there is a piece below this piece add an edge to it
        if self.stack_idx >= 1:  # if stack height is 2 then there must be a piece below
            below_node = graph.nodes_by_location.get((self.location, self.stack_idx-1))
            if below_node is None:
                raise ValueError(f"Node {self.node_id} has no below node in the graph - but one was expected")
            self.edges.append((self, below_node))
            self.edge_features.append([0, 0, 1, 0, 0, 0]) # below edge

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
    print(graph.edges)