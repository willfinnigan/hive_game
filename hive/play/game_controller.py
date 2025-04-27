from hive.agents.scored_board_state_ai import ScoreBoardStateAI
from hive.agents.scored_moves_based_ai import MoveScores, ScoreMovesAI
from hive.game.game import Game
from hive.game.types_and_errors import WHITE, BLACK
from hive.play.player import Player
from hive.render.text.small_print import game_to_text


class GameController():

    def __init__(self, player_1: Player, player_2: Player, game: Game):
        self.player_1 = player_1
        self.player_2 = player_2
        self.game = game

    def play(self):
        while self.game.get_winner() is None:
            player = self.get_next_player()
            move = player.get_move(self.game)
            print(f"Turn {self.game.player_turns[player.colour]}: {player.colour} - {move}")
            move.play(self.game)
            print(game_to_text(game))

        winner = self.game.get_winner()
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

    from hive.agents.random_ai import RandomAI

    for i in range(2):
        random_ai_1 = RandomAI(WHITE)
        random_ai_2 = RandomAI(BLACK)

        score_queen_only = MoveScores(play_queen=0, play_ant=0, play_beetle=0, play_grasshopper=0, play_spider=0)

        priority_queen_ai_1 = ScoreMovesAI(WHITE)
        priority_queen_ai_2 = ScoreMovesAI(BLACK, scores=score_queen_only)

        board_score_ai_1 = ScoreBoardStateAI(WHITE)

        game = Game()
        game_controller = GameController(board_score_ai_1, priority_queen_ai_2, game)
        game_controller.play()






