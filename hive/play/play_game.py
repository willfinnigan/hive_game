from hive.play.agents.random_ai import RandomAI
from hive.game_engine.game_moves import get_winner
from hive.play.move import NoMove
from hive.play.player import Player
from hive.render.to_text import game_to_text
from hive.game_engine.game_state import WHITE, BLACK, Game, initial_game


def _get_next_player(game: Game, player_1, player_2) -> Player:

    # get the colours with the lowest number of turns
    colours = [colour for colour, turns in game.player_turns.items() if turns == min(game.player_turns.values())]

    if colours[0] == player_1.colour:
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

        if not isinstance(move, NoMove):
            print(game_to_text(game, highlight_piece_at=move.new_location))

    winner = get_winner(game)
    print(f"{winner} wins!")
    return winner





if __name__ == '__main__':
    random_ai_1 = RandomAI(WHITE)
    random_ai_2 = RandomAI(BLACK)
    max_turns = 30

    winner = None
    while winner is None:
        winner = play(random_ai_1, random_ai_2, max_turns=max_turns)








