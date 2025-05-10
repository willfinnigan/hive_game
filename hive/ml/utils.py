
from hive.game_engine.game_state import BLACK, WHITE


def opposite_colour(colour: str) -> str:
    """Return the opposite colour."""
    if colour == WHITE:
        return BLACK
    elif colour == BLACK:
        return WHITE
    else:
        raise ValueError(f"Invalid colour: {colour}. Expected 'WHITE' or 'BLACK'.")
    
