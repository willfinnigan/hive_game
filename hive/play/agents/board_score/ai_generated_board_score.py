from hive.game_engine.game_moves import opposite_colour
from hive.game_engine.game_state import Colour, Game
from hive.game_engine.grid_functions import (
    pieces_around_location, can_remove_piece, get_placeable_locations
)
from hive.game_engine.piece_logic import get_possible_moves




def score_board_queens_improved(game: Game, colour: Colour) -> int:
    """
    An improved version of score_board_queens that maintains simplicity
    while adding targeted enhancements for better performance.
    """
    # Base weights - same as original for consistency
    per_queen_surrounded = -5
    per_enemy_queen_surrounded = 5
    
    # Get enemy color
    enemy_colour = opposite_colour(colour)
    
    # Calculate game progression (0-1 scale)
    total_turns = game.player_turns.get(colour, 0) + game.player_turns.get(enemy_colour, 0)
    late_game_factor = min(1.0, total_turns / 20)  # Reaches 1.0 after 20 turns
    
    # Adjust weights slightly based on game progression
    # Late game: increase importance of queen surroundings
    if late_game_factor > 0.5:
        per_queen_surrounded = int(per_queen_surrounded * (1 + 0.5 * late_game_factor))
        per_enemy_queen_surrounded = int(per_enemy_queen_surrounded * (1 + 0.5 * late_game_factor))
    
    score = 0
    our_queen_location = game.queens.get(colour)
    enemy_queen_location = game.queens.get(enemy_colour)

    # Evaluate our queen's safety
    if our_queen_location is not None:
        pieces_around_queen = pieces_around_location(game.grid, our_queen_location)
        num_around_queen = len(pieces_around_queen)
        
        # Apply base penalty
        score += num_around_queen * per_queen_surrounded
        
        # Add critical danger penalties
        if num_around_queen >= 5:  # One move from losing
            score -= 50
        elif num_around_queen == 4:  # Two moves from losing
            score -= 20
    
    # Evaluate enemy queen's situation
    if enemy_queen_location is not None:
        pieces_around_enemy_queen = pieces_around_location(game.grid, enemy_queen_location)
        num_around_enemy = len(pieces_around_enemy_queen)
        
        # Apply base reward
        score += num_around_enemy * per_enemy_queen_surrounded
        
        # Add critical advantage rewards
        if num_around_enemy >= 5:  # One move from winning
            score += 50
        elif num_around_enemy == 4:  # Two moves from winning
            score += 20
    
    # Small consideration for piece placement ability
    # This helps in early game when queens aren't on board yet
    if total_turns < 10:
        our_placeable = len(get_placeable_locations(game.grid, colour))
        enemy_placeable = len(get_placeable_locations(game.grid, enemy_colour))
        placement_score = (our_placeable - enemy_placeable) * 2
        score += placement_score
    
    return score


def score_board_advanced(game: Game, colour: Colour) -> int:
    """
    Advanced scoring function that considers multiple strategic elements:
    1. Queen surroundings (weighted by game progression)
    2. Piece mobility and freedom
    3. Board control and piece positioning
    4. Piece value and strategic placement
    5. Game stage adaptation
    """
    # Initialize score
    score = 0
    
    # Get player and opponent colors
    enemy_colour = opposite_colour(colour)
    
    # Get queen locations
    our_queen_location = game.queens.get(colour)
    enemy_queen_location = game.queens.get(enemy_colour)
    
    # Calculate game stage (0-1 scale) based on total turns
    total_turns = game.player_turns.get(colour, 0) + game.player_turns.get(enemy_colour, 0)
    early_game = max(0, 1 - (total_turns / 10))  # First ~10 turns
    mid_game = 1 - early_game if total_turns <= 20 else max(0, 1 - ((total_turns - 20) / 10))
    late_game = 1 - early_game - mid_game if mid_game > 0 else max(0, (total_turns - 20) / 10)
    
    # 1. QUEEN SURROUNDINGS EVALUATION
    queen_safety_score = 0
    if our_queen_location is not None:
        # Count pieces around our queen
        pieces_around_our_queen = pieces_around_location(game.grid, our_queen_location)
        num_around_our_queen = len(pieces_around_our_queen)
        
        # Penalize based on game stage - early game: less penalty, late game: severe penalty
        queen_safety_penalty = -5 * (early_game * 0.5 + mid_game * 1.0 + late_game * 2.0)
        queen_safety_score += num_around_our_queen * queen_safety_penalty
        
        # Extra penalty for being close to surrounded (5+ pieces)
        if num_around_our_queen >= 5:
            queen_safety_score -= 50  # Critical danger
        elif num_around_our_queen == 4:
            queen_safety_score -= 20  # High danger
            
    # Enemy queen evaluation
    enemy_queen_score = 0
    if enemy_queen_location is not None:
        # Count pieces around enemy queen
        pieces_around_enemy = pieces_around_location(game.grid, enemy_queen_location)
        num_around_enemy = len(pieces_around_enemy)
        
        # Reward based on game stage - early game: small reward, late game: large reward
        enemy_queen_reward = 5 * (early_game * 0.5 + mid_game * 1.0 + late_game * 2.0)
        enemy_queen_score += num_around_enemy * enemy_queen_reward
        
        # Extra reward for almost surrounding enemy queen
        if num_around_enemy == 5:
            enemy_queen_score += 50  # One move from victory
        elif num_around_enemy == 4:
            enemy_queen_score += 20  # Two moves from victory
    
    # 2. PIECE MOBILITY EVALUATION
    mobility_score = 0
    our_piece_count = 0
    enemy_piece_count = 0
    our_mobility = 0
    enemy_mobility = 0
    
    # Evaluate mobility for all pieces
    for loc, stack in game.grid.items():
        if not stack:
            continue
            
        top_piece = stack[-1]  # Only top piece can move
        
        # Skip if piece can't be removed (would break hive)
        if not can_remove_piece(game.grid, loc):
            continue
            
        # Count pieces by color
        if top_piece.colour == colour:
            our_piece_count += 1
            # Calculate possible moves for our pieces
            possible_moves = get_possible_moves(game.grid, loc)
            our_mobility += len(possible_moves)
        else:
            enemy_piece_count += 1
            # Calculate possible moves for enemy pieces
            possible_moves = get_possible_moves(game.grid, loc)
            enemy_mobility += len(possible_moves)
    
    # Calculate average mobility (avoid division by zero)
    our_avg_mobility = our_mobility / max(1, our_piece_count)
    enemy_avg_mobility = enemy_mobility / max(1, enemy_piece_count)
    
    # Score mobility difference (weighted by game stage)
    mobility_weight = early_game * 3.0 + mid_game * 2.0 + late_game * 1.0
    mobility_score = (our_avg_mobility - enemy_avg_mobility) * mobility_weight * 5
    
    # 3. BOARD CONTROL EVALUATION
    control_score = 0
    
    # Count placeable locations for each player
    our_placeable = len(get_placeable_locations(game.grid, colour))
    enemy_placeable = len(get_placeable_locations(game.grid, enemy_colour))
    
    # Score control difference (weighted by early game importance)
    control_weight = early_game * 2.0 + mid_game * 1.0 + late_game * 0.5
    control_score = (our_placeable - enemy_placeable) * control_weight * 2
    
    # 4. PIECE VALUE AND STRATEGIC PLACEMENT
    piece_value_score = 0
    
    # Define piece values based on their strategic importance
    piece_values = {
        'QUEEN': 100,
        'ANT': 8,    # High mobility
        'BEETLE': 7, # Can climb
        'SPIDER': 5, # Medium mobility
        'GRASSHOPPER': 6  # Jump ability
    }
    
    # Evaluate piece positioning
    for loc, stack in game.grid.items():
        if not stack:
            continue
            
        top_piece = stack[-1]
        piece_value = piece_values.get(top_piece.name, 0)
        
        # Adjust value based on position
        if top_piece.colour == colour:
            # Reward pieces near enemy queen
            if enemy_queen_location is not None:
                # Calculate distance to enemy queen (simplified)
                distance = abs(loc[0] - enemy_queen_location[0]) + abs(loc[1] - enemy_queen_location[1])
                if distance <= 4:  # Close enough to be useful
                    piece_value_score += piece_value * (4 - distance) * 0.5
                    
            # Penalize pieces too close to our queen in early/mid game
            if our_queen_location is not None and (early_game > 0.3 or mid_game > 0.5):
                distance = abs(loc[0] - our_queen_location[0]) + abs(loc[1] - our_queen_location[1])
                if distance <= 2:  # Too close to our queen
                    piece_value_score -= piece_value * (2 - distance) * 0.3
        else:
            # Penalize enemy pieces near our queen
            if our_queen_location is not None:
                distance = abs(loc[0] - our_queen_location[0]) + abs(loc[1] - our_queen_location[1])
                if distance <= 3:  # Close enough to be threatening
                    piece_value_score -= piece_value * (3 - distance) * 0.5
    
    # 5. COMBINE ALL SCORES WITH APPROPRIATE WEIGHTS
    # Adjust weights based on game stage
    queen_weight = early_game * 1.0 + mid_game * 1.5 + late_game * 2.5
    mobility_weight = early_game * 1.5 + mid_game * 1.0 + late_game * 0.5
    control_weight = early_game * 1.5 + mid_game * 1.0 + late_game * 0.5
    piece_value_weight = early_game * 1.0 + mid_game * 1.5 + late_game * 1.0
    
    # Calculate final score
    final_score = (
        queen_safety_score * queen_weight +
        enemy_queen_score * queen_weight +
        mobility_score * mobility_weight +
        control_score * control_weight +
        piece_value_score * piece_value_weight
    )
    
    return int(final_score)  # Convert to integer for consistency with original function

