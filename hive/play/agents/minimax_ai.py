import random
import time
from typing import Dict, List, Tuple, Optional, Union, Callable

from hive.game_engine.game_state import Colour, Game
from hive.game_engine.moves import Move, NoMove
from hive.game_engine.player_functions import get_players_possible_moves_or_placements
from hive.play.player import Player
from hive.game_engine.game_functions import opposite_colour
from hive.play.agents.board_score.simple_board_score import score_board_queens
from hive.play.agents.board_score.ai_generated_board_score import score_board_advanced


class TranspositionTable:
    """
    A cache for storing evaluated positions to avoid redundant calculations.
    Uses Zobrist hashing for efficient position identification.
    """
    def __init__(self, max_size: int = 1000000):
        self.max_size = max_size
        self.table: Dict[str, Tuple[int, int, Move]] = {}  # hash -> (score, depth, best_move)
    
    def store(self, game: Game, depth: int, score: int, best_move: Optional[Move] = None):
        """Store a position evaluation in the table"""
        # Simple string representation of the game state as a hash
        # In a real implementation, Zobrist hashing would be more efficient
        game_hash = self._hash_game(game)
        
        self.table[game_hash] = (score, depth, best_move)
        
        # Manage table size if needed
        if len(self.table) > self.max_size:
            # Remove random entries to keep size in check
            # A more sophisticated approach would use LRU or other eviction policies
            keys_to_remove = random.sample(list(self.table.keys()), int(self.max_size * 0.2))
            for key in keys_to_remove:
                del self.table[key]
    
    def lookup(self, game: Game) -> Optional[Tuple[int, int, Move]]:
        """Look up a position in the table"""
        game_hash = self._hash_game(game)
        return self.table.get(game_hash)
    
    def _hash_game(self, game: Game) -> str:
        """
        Create a string hash of the game state.
        This is a simplified approach - a real implementation would use Zobrist hashing.
        """
        # Convert the immutable grid to a string representation
        grid_str = str(sorted([(loc, tuple(stack)) for loc, stack in game.grid.items()]))
        turn_str = str(game.player_turns)
        return f"{grid_str}|{turn_str}"
    
    def clear(self):
        """Clear the transposition table"""
        self.table.clear()


class MinimaxAI(Player):
    """
    AI player that uses the minimax algorithm with alpha-beta pruning to select moves.
    
    Features:
    - Configurable search depth
    - Alpha-beta pruning for efficiency
    - Move ordering to improve pruning
    - Transposition table for caching evaluated positions
    - Iterative deepening for time management
    """
    
    def __init__(self, 
                 colour: Colour, 
                 max_depth: int = 3, 
                 eval_function: Callable[[Game, Colour], int] = score_board_queens,
                 use_iterative_deepening: bool = True,
                 time_limit: float = 5.0):
        """
        Initialize the MinimaxAI.
        
        Args:
            colour: The player's colour (WHITE or BLACK)
            max_depth: Maximum search depth
            eval_function: Function to evaluate board states
            use_iterative_deepening: Whether to use iterative deepening
            time_limit: Time limit for move selection in seconds
        """
        super().__init__(colour)
        self.max_depth = max_depth
        self.eval_function = eval_function
        self.transposition_table = TranspositionTable()
        self.use_iterative_deepening = use_iterative_deepening
        self.time_limit = time_limit
        self.nodes_evaluated = 0
    
    def get_move(self, game: Game) -> Union[Move, NoMove]:
        """
        Select the best move using minimax with alpha-beta pruning.
        
        If iterative deepening is enabled, gradually increase search depth
        until time limit is reached.
        """
        possible_moves = get_players_possible_moves_or_placements(self.colour, game)
        if len(possible_moves) == 0:
            return NoMove(self.colour)
        
        # Reset statistics for this move search
        self.nodes_evaluated = 0
        start_time = time.time()
        
        # If only one move is possible, return it immediately
        if len(possible_moves) == 1:
            return possible_moves[0]
        
        best_move = None
        best_score = float('-inf')
        
        if self.use_iterative_deepening:
            # Start with depth 1 and gradually increase
            for current_depth in range(1, self.max_depth + 1):
                # Clear transposition table between iterations to avoid depth issues
                # In a more sophisticated implementation, we could keep entries but handle depth properly
                self.transposition_table.clear()
                
                temp_best_move, temp_best_score = self._iterative_deepening_search(game, possible_moves, current_depth)
                
                # Update best move if we have a valid result
                if temp_best_move is not None:
                    best_move = temp_best_move
                    best_score = temp_best_score
                
                # Check if we've used up our time budget
                if time.time() - start_time > self.time_limit:
                    break
        else:
            # Regular minimax search at fixed depth
            best_move, best_score = self._find_best_move(game, possible_moves, self.max_depth)
        
        # If we somehow failed to find a move, pick a random one
        if best_move is None:
            best_move = random.choice(possible_moves)
        
        return best_move
    
    def _iterative_deepening_search(self, game: Game, possible_moves: List[Move], depth: int) -> Tuple[Optional[Move], int]:
        """
        Perform minimax search with the specified depth.
        Used as a subroutine for iterative deepening.
        """
        return self._find_best_move(game, possible_moves, depth)
    
    def _find_best_move(self, game: Game, possible_moves: List[Move], depth: int) -> Tuple[Optional[Move], int]:
        """
        Find the best move using minimax with alpha-beta pruning.
        
        Args:
            game: Current game state
            possible_moves: List of possible moves
            depth: Search depth
            
        Returns:
            Tuple of (best_move, best_score)
        """
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Check transposition table first
        tt_entry = self.transposition_table.lookup(game)
        if tt_entry is not None:
            stored_score, stored_depth, stored_move = tt_entry
            # If we have a stored result from an equal or deeper search
            if stored_depth >= depth and stored_move in possible_moves:
                return stored_move, stored_score
        
        # Order moves to improve alpha-beta pruning efficiency
        ordered_moves = self._order_moves(game, possible_moves)
        
        for move in ordered_moves:
            # Apply the move to get the new game state
            new_game = move.play()
            
            # Recursive minimax call for opponent's turn
            score = -self._minimax(new_game, depth - 1, -beta, -alpha, opposite_colour(self.colour))
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
            
            # Alpha-beta pruning
            if alpha >= beta:
                break
        
        # Store result in transposition table
        if best_move is not None:
            self.transposition_table.store(game, depth, best_score, best_move)
        
        return best_move, best_score
    
    def _minimax(self, game: Game, depth: int, alpha: float, beta: float, current_colour: Colour) -> int:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            game: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            current_colour: Colour of the player to move
            
        Returns:
            Evaluation score from the perspective of self.colour
        """
        self.nodes_evaluated += 1
        
        # Check for game over (queen surrounded)
        for colour in [self.colour, opposite_colour(self.colour)]:
            queen_loc = game.queens.get(colour)
            if queen_loc is not None:
                from hive.game_engine.grid_functions import pieces_around_location
                pieces_around_queen = pieces_around_location(game.grid, queen_loc)
                if len(pieces_around_queen) == 6:  # Queen is surrounded
                    if colour == self.colour:
                        return float('-inf')  # We lose
                    else:
                        return float('inf')   # We win
        
        # Check transposition table
        tt_entry = self.transposition_table.lookup(game)
        if tt_entry is not None:
            stored_score, stored_depth, _ = tt_entry
            if stored_depth >= depth:
                return stored_score
        
        # If we've reached the maximum depth or a leaf node, evaluate the position
        if depth == 0:
            score = self._evaluate_position(game)
            self.transposition_table.store(game, depth, score)
            return score
        
        # Create a temporary player to generate moves
        temp_player = Player(current_colour)
        possible_moves = temp_player.possible_moves(game)
        
        # If no moves are available, it's a pass
        if len(possible_moves) == 0:
            # Create a NoMove and apply it
            no_move = NoMove(current_colour, game)
            new_game = no_move.play()
            
            # Continue search with the other player
            return -self._minimax(new_game, depth - 1, -beta, -alpha, opposite_colour(current_colour))
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(game, possible_moves)
        
        # Initialize best score
        if current_colour == self.colour:
            best_score = float('-inf')
            for move in ordered_moves:
                new_game = move.play()
                score = self._minimax(new_game, depth - 1, alpha, beta, opposite_colour(current_colour))
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break  # Beta cutoff
        else:
            best_score = float('inf')
            for move in ordered_moves:
                new_game = move.play()
                score = self._minimax(new_game, depth - 1, alpha, beta, opposite_colour(current_colour))
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if alpha >= beta:
                    break  # Alpha cutoff
        
        # Store result in transposition table
        self.transposition_table.store(game, depth, best_score)
        
        return best_score
    
    def _evaluate_position(self, game: Game) -> int:
        """
        Evaluate the current position from the perspective of self.colour.
        
        Args:
            game: Current game state
            
        Returns:
            Evaluation score (higher is better for self.colour)
        """
        # Use the provided evaluation function
        our_score = self.eval_function(game, self.colour)
        opponent_score = self.eval_function(game, opposite_colour(self.colour))
        
        # Return the difference (positive is good for us)
        return our_score - opponent_score
    
    def _order_moves(self, game: Game, moves: List[Move]) -> List[Move]:
        """
        Order moves to improve alpha-beta pruning efficiency.
        
        Prioritize:
        1. Moves from the transposition table
        2. Captures (moves that surround the opponent's queen)
        3. Queen moves (usually critical)
        4. Beetle moves (can climb)
        5. Other moves
        
        Args:
            game: Current game state
            moves: List of possible moves
            
        Returns:
            Ordered list of moves
        """
        # Check if we have a best move from the transposition table
        tt_entry = self.transposition_table.lookup(game)
        tt_move = None
        if tt_entry is not None:
            _, _, tt_move = tt_entry
        
        # Score each move for ordering
        move_scores = []
        enemy_colour = opposite_colour(self.colour)
        enemy_queen_loc = game.queens.get(enemy_colour)
        
        for move in moves:
            score = 0
            
            # Highest priority: move from transposition table
            if tt_move is not None and move == tt_move:
                score += 10000
            
            # Check if this move surrounds the enemy queen
            if enemy_queen_loc is not None:
                # Apply the move and check if it surrounds the queen
                new_game = move.play()
                from hive.game_engine.grid_functions import pieces_around_location
                pieces_around_enemy_queen = pieces_around_location(new_game.grid, enemy_queen_loc)
                if len(pieces_around_enemy_queen) == 6:  # Queen is surrounded
                    score += 5000
                elif len(pieces_around_enemy_queen) == 5:  # One move away from surrounding
                    score += 1000
            
            # Prioritize queen moves early in the game
            if move.piece.name == "QUEEN":
                # Queen placement is important
                if move.current_location is None:
                    score += 500
                # Moving the queen can be risky but sometimes necessary
                else:
                    score += 200
            
            # Beetles can climb and are versatile
            if move.piece.name == "BEETLE":
                score += 300
            
            # Ants have high mobility
            if move.piece.name == "ANT":
                score += 250
            
            # Add a small random factor to avoid deterministic behavior
            score += random.uniform(0, 10)
            
            move_scores.append((score, move))
        
        # Sort moves by score (descending)
        move_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Return ordered moves
        return [move for _, move in move_scores]