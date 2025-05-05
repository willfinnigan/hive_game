from typing import Optional, Tuple, Dict
from pyrsistent import PRecord, field, pmap_field, PMap
from typing import NamedTuple


Colour = str
WHITE = 'WHITE'
BLACK = 'BLACK'

class Piece(NamedTuple):
    colour: Colour
    name: str
    number: int

Stack = Tuple[Piece, ...]
Location = Tuple[int, int]
Grid = PMap  # PMap[Location, Stack]

class GridLocation(NamedTuple):
    stack: Stack
    location: Location

# Game state class
class Game(PRecord):
    grid = pmap_field(tuple, tuple)  # Location (tuple of ints) to Stack (tuple of Pieces)
    player_turns = pmap_field(str, int)  # Colour to turn count
    queens = pmap_field(str, tuple)  # Colour to Location (tuple of ints)
    parent = field(initial=None)  # Self-reference to Game, can't use type='Game' directly


def initial_game():
    return Game(
        grid={},
        player_turns={WHITE: 0, BLACK: 0},
        queens={}
    )

if __name__ == "__main__":
    # Example usage
    game = initial_game()
    print(game)
    print(game.grid)
    print(game.player_turns)
    print(game.queens)

