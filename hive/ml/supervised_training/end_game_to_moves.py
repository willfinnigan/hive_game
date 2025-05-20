

from hive.game_engine.game_functions import get_winner
from hive.game_engine.game_state import Game


def end_game_to_data(game: Game):
    """
    We start with just the end game state.
    From this we need to extract the data to learn from
    """

    winner = get_winner(game)
