from __future__ import annotations

import random
from typing import Union

from hive.game_engine.game_state import Game
from hive.game_engine.moves import Move, NoMove
from hive.game_engine.player_functions import get_players_possible_moves_or_placements
from hive.play.player import Player


class RandomAI(Player):
    """ Plays moves completely randomly """
    def get_move(self, game: Game) -> Union[Move|NoMove]:
        possible_moves = get_players_possible_moves_or_placements(self.colour, game)
        if len(possible_moves) == 0:
            return NoMove(self.colour)
        return random.choice(possible_moves)
