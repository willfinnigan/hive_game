from __future__ import annotations

import random
from typing import Union

from v1.hive.play.move import Move, NoMove
from v1.hive.play.player import Player

class RandomAI(Player):
    """ Plays moves completely randomly """
    def get_move(self, game) -> Union[Move|NoMove]:
        possible_moves = self.possible_moves(game)
        if len(possible_moves) == 0:
            return NoMove(self.colour)
        return random.choice(possible_moves)
