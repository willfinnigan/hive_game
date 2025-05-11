from functools import lru_cache
from typing import Optional

from hive.game_engine import pieces
from hive.game_engine.errors import NoQueenError
from hive.game_engine.grid_functions import pieces_around_location, check_is_valid_location, check_is_valid_placement, check_is_valid_move
from hive.game_engine.game_state import BLACK, WHITE, Game, Grid, Piece, Location, Colour

def current_turn_colour(game: Game) -> Colour:
    return [colour for colour, turns in game.player_turns.items() if turns == min(game.player_turns.values())][0]

def check_queen_timely_placement(game: Game, colour: Colour, moves_to_queen=4):
    if game.player_turns[colour] < moves_to_queen:
        return

    if not game.queens.get(colour, False):
        raise NoQueenError(f"No Queen found for {colour}")

@lru_cache(maxsize=None)
def get_queen_location(grid: Grid, colour: Colour) -> Optional[Location]:
    """Get the location of the queen for a given colour."""

    # loop over grid and find queen
    queen_location = None  # if not found, remains None
    for location, stack in grid.items():
        for piece in stack:
            if piece.name == pieces.QUEEN and piece.colour == colour:
                queen_location = location
                break
    return queen_location


def place_piece(game: Game, piece: Piece, location: Location) -> Game:
    check_is_valid_location(location)
    check_is_valid_placement(game.grid, location, piece.colour)

    game_mutable = game.evolver()

    # Add the piece to the stack at the location
    current_stack = game.grid.get(location, ())
    updated_stack = current_stack + (piece,)
    game_mutable = game_mutable.set('grid', game.grid.set(location, updated_stack))

    # Update queen location if needed
    if piece.name == pieces.QUEEN:
        game_mutable = game_mutable.set('queens', game.queens.set(piece.colour, location))

    # Remove the piece from the unplayed pieces
    unplayed_pieces = game.unplayed_pieces.get(piece.colour, ())
    updated_unplayed_pieces = tuple(p for p in unplayed_pieces if p != piece)
    game_mutable = game_mutable.set('unplayed_pieces', game.unplayed_pieces.set(piece.colour, updated_unplayed_pieces))

    # Increment the player's turn count
    current_turn = game.player_turns.get(piece.colour, 0)
    game_mutable = game_mutable.set('player_turns', game.player_turns.set(piece.colour, current_turn + 1))

    check_queen_timely_placement(game_mutable.persistent(), piece.colour)

    new_game = game_mutable.persistent()
    new_game = new_game.set('parent', game)  # Store reference to previous game state
    return new_game

def move_piece(game: Game, current_location: Location, location: Location) -> Game:
    check_is_valid_location(location)
    check_is_valid_move(game.grid, current_location, location)

    game_mutable = game.evolver()

    # Remove piece from current location
    current_stack = game.grid.get(current_location, ())
    if not current_stack:
        raise ValueError("No piece to move at current location")

    piece = current_stack[-1]
    updated_source_stack = current_stack[:-1]

    # Update the grid with the modified source stack or remove the location
    updated_grid = game.grid
    if updated_source_stack:
        updated_grid = updated_grid.set(current_location, updated_source_stack)
    else:
        updated_grid = updated_grid.discard(current_location)
    
    # Add piece to the new location
    destination_stack = game.grid.get(location, ())
    updated_destination_stack = destination_stack + (piece,)
    updated_grid = updated_grid.set(location, updated_destination_stack)
    

    # Update queen position if needed
    if piece.name == pieces.QUEEN:  # Fixed: QUEEN -> pieces.QUEEN
        game_mutable = game_mutable.set('queens', game.queens.set(piece.colour, location))

    # Increment player's turn count
    current_turn = game.player_turns.get(piece.colour, 0)
    game_mutable = game_mutable.set('player_turns', game.player_turns.set(piece.colour, current_turn + 1))

    check_queen_timely_placement(game_mutable.persistent(), piece.colour)

    game_mutable = game_mutable.set('grid', updated_grid)

    new_game = game_mutable.persistent()
    new_game = new_game.set('parent', game)  # Store reference to previous game state
    return new_game

def pass_move(game: Game, colour: str) -> Game:
    game_mutable = game.evolver()

    # Increment player's turn count
    current_turn = game.player_turns.get(colour, 0)
    game_mutable = game_mutable.set('player_turns', game.player_turns.set(colour, current_turn + 1))

    new_game = game_mutable.persistent()
    new_game = new_game.set('parent', game)  # Store reference to previous game state
    return new_game


def has_player_lost(game: Game, colour: Colour) -> bool:
    """Player has lost if their queen is surrounded"""
    queen_location = game.queens.get(colour)
    if queen_location is not None:
        around_queen = pieces_around_location(game.grid, queen_location)
        if len(around_queen) == 6:
            return True
    return False

def get_winner(game) -> Optional[Colour]:
    """If there is only one queen left, that player has won"""
    colours_still_in_game = []
    for colour in game.player_turns.keys():
        if not has_player_lost(game, colour):
            colours_still_in_game.append(colour)

    if len(colours_still_in_game) == 1:
        return colours_still_in_game[0]
    return None

def opposite_colour(colour: str) -> str:
    """Return the opposite colour."""
    if colour == WHITE:
        return BLACK
    elif colour == BLACK:
        return WHITE
    else:
        raise ValueError(f"Invalid colour: {colour}. Expected 'WHITE' or 'BLACK'.")