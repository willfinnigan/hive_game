import random
from typing import Callable, Union, List

from hive.game_engine.game_state import Colour, Game
from hive.game_engine.move import Move, NoMove, get_players_possible_moves_or_placements
from hive.play.player import Player
from hive.play.agents.board_score.simple_board_score import score_board_queens


ScoreBoardMethod = Callable[[Game, Colour], int]
    

class ScoreBoardIn1Move_AI(Player):
    """ Plays moves according which will give the best board state in 1 moves time"""

    def __init__(self, colour: Colour, score_method: ScoreBoardMethod = score_board_queens):
        super().__init__(colour)
        self.score_method = score_method

    def get_move(self, game) -> Union[Move|NoMove]:
        possible_moves = get_players_possible_moves_or_placements(self.colour, game)
        if len(possible_moves) == 0:
            return NoMove(self.colour, game)

        # score the moves
        scored_moves = self.score_future_board_states(game, possible_moves)
        return scored_moves[0][1]

    def score_future_board_states(self, game: Game, moves: List[Move]):
        """Looks 1 move ahead and scores the board state for each possible move"""

        move_scores = []
        for move in moves:
            mv_game = move.play()
            score = self.score_method(mv_game, self.colour)
            move_scores.append((score, move))

        # shuffle then sort scored_moves
        random.shuffle(move_scores)
        move_scores = sorted(move_scores, key=lambda x: x[0], reverse=True)
        return move_scores

