from __future__ import annotations
from typing import List

from hive.game_engine.player_functions import Move
from hive.game_engine.game_state import Colour


class Player():
    def __init__(self, colour: Colour):
        self.colour = colour

    def get_move(self, game) -> Move:
        """ AI or Human selection of move """
        pass


