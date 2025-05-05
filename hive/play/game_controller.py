from hive.agents.random_ai import RandomAI
from hive.game import get_winner
from hive.play.move import NoMove
from hive.play.player import Player
from hive.render.to_text import game_to_text
from hive.game_types import WHITE, BLACK, initial_game


class GameController():

    def __init__(self, player_1: Player, player_2: Player):
        self.player_1 = player_1
        self.player_2 = player_2

    def play(self, game=None):
        if game is None:
            game = initial_game()

        while get_winner(game) is None:
            player = self.get_next_player()
            move = player.get_move(game)
            print(f"Turn {self.game.player_turns[player.colour]}: {player.colour} - {move}")
            game = move.play(game)

            if not isinstance(move, NoMove):
                print(game_to_text(game, highlight_piece_at=move.new_location))

        winner = get_winner()
        print(f"{winner} wins!")
        return winner

    def get_next_player(self) -> Player:
        # get the colours with the lowest number of turns
        colours = [colour for colour, turns in self.game.player_turns.items() if turns == min(self.game.player_turns.values())]

        if colours[0] == self.player_1.colour:
            return self.player_1
        else:
            return self.player_2




if __name__ == '__main__':


    for i in range(2):
        random_ai_1 = RandomAI(WHITE)
        random_ai_2 = RandomAI(BLACK)

        game_controller = GameController(random_ai_1, random_ai_2)
        game_controller.play()






