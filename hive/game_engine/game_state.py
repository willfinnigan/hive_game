from typing import Optional, Tuple, Dict
from pyrsistent import PRecord, field, pmap, pmap_field, PMap
from typing import NamedTuple

from hive.game_engine.pieces import QUEEN

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


def initial_game(grid: Optional[Grid] = None) -> Game:
    if grid is None:
        grid = pmap()
    
    # if grid is a dict, convert it to an immutable grid
    if isinstance(grid, dict):
        grid = create_immutable_grid(grid)

    # Find queens if they exist
    queens = {}
    for loc, stack in grid.items():
        for piece in stack:
            if piece.name == QUEEN:
                queens[piece.colour] = loc
    queens = pmap(queens)

    return Game(
        grid=grid,
        player_turns={WHITE: 0, BLACK: 0},
        queens=queens
    )

def create_immutable_grid(grid_dict: dict) -> Grid:
    """ Create a grid from a dictionary of locations and stacks. """
    return pmap(grid_dict)


if __name__ == "__main__":
    # Example usage
    game = initial_game()
    print(game)
    print(game.grid)
    print(game.player_turns)
    print(game.queens)

