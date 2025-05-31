from hive.game_engine.game_state import Game, Colour
from hive.game_engine.player_functions import get_players_possible_moves_or_placements
from hive.play.agents.board_score.simple_board_score import score_board_queens


def mock_model_callable(game: Game, colour: Colour) -> tuple[list[float], float]:
    possible_moves = get_players_possible_moves_or_placements(colour, game)

    #scored_moves = prioritise_moves(possible_moves, game)

    # assign a random value to every move
    #scores = [np.random.rand() for _ in possible_moves]
    scores = [1] * len(possible_moves)  # for testing, give all moves the same score
    scored_moves = list(zip(scores, possible_moves))

    #scored_moves = score_moves_simple(possible_moves, game)

    scores = [score for score, _ in scored_moves]
    total_score = sum(scores)

    # make scores sum to 1
    if total_score > 0:
        probabilities = [score / total_score for score in scores]
    else:
        # If total score is 0, assign equal probabilities
        probabilities = [0] * len(possible_moves)

    # get value using board score
    value = score_board_queens(game, colour)

    return probabilities, value
