from hive.game_engine.moves import NoMove, Move
from hive.ml.game_to_graph import Graph
from hive.ml.graph_to_pyg import game_to_pytorch
from hive.play.agents.board_score.ai_generated_board_score import score_board_advanced, score_board_queens_improved
from hive.play.agents.board_score.simple_board_score import score_board_queens
from hive.play.agents.minimax_ai import MinimaxAI
from hive.play.agents.random_ai import RandomAI
from hive.game_engine.game_functions import current_turn_colour, get_winner
from hive.play.agents.scored_board_state_ai import ScoreBoardIn1Move_AI
from hive.play.agents.scored_moves_based_ai import ScoreMovesAI
from hive.play.player import Player
from hive.render.to_text import game_to_text
from hive.game_engine.game_state import WHITE, BLACK, Game, initial_game


def _get_next_player(game: Game, player_1, player_2) -> Player:
    colour = current_turn_colour(game)
    if colour == player_1.colour:
        return player_1
    else:
        return player_2

def play(player_1, player_2, game=None, max_turns=None):
    if game is None:
        game = initial_game()

    turn = 0
    while get_winner(game) is None and (max_turns is None or turn < max_turns):
        turn += 1
        player = _get_next_player(game, player_1, player_2)
        move = player.get_move(game)
        print(f"Turn {game.player_turns[player.colour]}: {player.colour} - {move}")
        game = move.play(game)

        #data = game_to_pytorch(game)
        #print(data)

        if not isinstance(move, NoMove):
            print(game_to_text(game, highlight_piece_at=move.new_location))

    winner = get_winner(game)
    print(f"{winner} wins!")
    return winner



if __name__ == '__main__':
    ai_1 = RandomAI(WHITE)
    ai_2 = RandomAI(BLACK)

    #ai_1 = ScoreMovesAI(WHITE)
    #ai_1 = MinimaxAI(WHITE, max_depth=3, eval_function=score_board_queens, use_iterative_deepening = True, time_limit = 4)
    #ai_2 = ScoreBoardIn1Move_AI(BLACK, score_method=score_board_queens)

    max_turns = 1000

    #winner = None
    #while winner is None:
    winner = play(ai_1, ai_2, max_turns=max_turns)








