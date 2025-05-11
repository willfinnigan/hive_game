
from hive.game_engine.game_functions import opposite_colour
from hive.game_engine.game_state import Colour, Game
from hive.game_engine.grid_functions import pieces_around_location



def score_board_queens(game: Game, colour: Colour) -> int:
    """Use hard-coded rules to score the board state for a given colour"""
    per_queen_surrounded = -5
    per_enemy_queen_surrounded = 5

    score = 0
    our_queen_location = game.queens.get(colour)
    enemy_queen_location = game.queens.get(opposite_colour(colour))

    if our_queen_location is not None:
        pieces_around_queen = pieces_around_location(game.grid, our_queen_location)
        score += len(pieces_around_queen) * per_queen_surrounded
    if enemy_queen_location is not None:
        pieces_around_enemy_queen = pieces_around_location(game.grid, enemy_queen_location)
        score += len(pieces_around_enemy_queen) * per_enemy_queen_surrounded
    return score
