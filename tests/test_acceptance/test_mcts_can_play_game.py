from hive.game_engine.game_state import BLACK, WHITE
from hive.play.agents.mcts import MCTS_AI, mock_model_callable
from hive.play.agents.random_ai import RandomAI
from hive.play.play_game import play


def test_mcts_can_play_game():
    """Test that MCTS can play a game of Hive."""
    from hive.game_engine.game_state import initial_game

    game = initial_game()

    mctsai = MCTS_AI(WHITE, game, model_callable=mock_model_callable)
    randomai = RandomAI(BLACK)

    winner = play(mctsai, randomai, max_turns=500)

