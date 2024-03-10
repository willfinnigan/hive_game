from typing import Tuple

Grid = dict
Location = Tuple[int, int]
Colour = str
WHITE: Colour = 'white'
BLACK: Colour = 'black'


class InvalidPlacementError(Exception):
    pass


class InvalidMoveError(Exception):
    pass


class InvalidLocationError(Exception):
    pass


class PlayerHasLost(Exception):
    pass



class BreaksConnectionError(Exception):
    pass


class NoQueenError(Exception):
    pass

