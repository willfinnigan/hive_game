from hive.game.game import Game
from hive.game.pieces.ant import Ant
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.queen import Queen
from hive.game.pieces.spider import Spider
from hive.game.pieces.piece_base_class import Piece
from hive.game.types_and_errors import WHITE, BLACK, Grid, Location, Colour
from typing import Dict, List, Tuple, Optional
import re


def large_game_to_text(game: Game, include_coordinates: bool = False) -> str:
    """
    Render a game state in a large text format.
    
    Args:
        game (Game): The game to render
        include_coordinates (bool): Whether to include coordinate labels
        
    Returns:
        str: A text representation of the game state

    Example output:
  
    HIVE GAME STATE
    Turns: W:0, B:0
    Queens: W:Not placed, B:Not placed

      [ . ][ . ][ . ]
    [ . ][WA0][WA0][ . ]
      [WA0][BA0][WA0]
    [ . ][WA0][WA0][ . ]
      [ . ][ . ][ . ]

    """
    # Find the boundaries of the grid
    if not game.grid:
        return "Empty board"
    
    min_q = min([loc[0] for loc in game.grid.keys()])
    min_r = min([loc[1] for loc in game.grid.keys()])
    max_q = max([loc[0] for loc in game.grid.keys()])
    max_r = max([loc[1] for loc in game.grid.keys()])
    
    # Add a border
    min_q -= 1
    min_r -= 1
    max_q += 1
    max_r += 1
    
    # Create the header with game information
    header = []
    header.append("HIVE GAME STATE")
    
    # Add player turn information
    turns_info = []
    for color, turns in game.player_turns.items():
        turns_info.append(f"{color}:{turns}")
    header.append("Turns: " + ", ".join(turns_info))
    
    # Add queen status information
    queens_info = []
    for color, queen in game.queens.items():
        if queen is None:
            queens_info.append(f"{color}:Not placed")
        else:
            queens_info.append(f"{color}:Placed")
    header.append("Queens: " + ", ".join(queens_info))
    
    # Check if any player has lost
    for color in game.player_turns.keys():
        if game.has_player_lost(color):
            header.append(f"Player {color} has lost!")
    
    # Check if there's a winner
    winner = game.get_winner()
    if winner:
        header.append(f"Winner: {winner}")
    
    # Build the board representation
    rows = []
    
    # Add coordinate header if requested
    if include_coordinates:
        coord_header = "    "
        for q in range(min_q, max_q + 1):
            if q % 2 == 0:  # Only show even q coordinates to avoid crowding
                coord_header += f"{q:^4}"
            else:
                coord_header += "    "
        rows.append(coord_header)
    
    # Generate each row
    for r in range(min_r, max_r + 1):
        # Add row coordinate if requested
        if include_coordinates:
            row = [f"{r:2d} "]
        else:
            row = [""]
        
        # Add offset for even rows to create hexagonal pattern
        if r % 2 == 0:
            row[0] += "  "
        
        # Process each cell in the row
        for q in range(min_q, max_q + 1):
            # Skip cells that don't form valid hex coordinates
            if (q + r) % 2 == 1:
                continue
                
            piece = game.grid.get((q, r))
            if piece is not None:
                # Format: [ColorPieceNumber]
                piece_str = f"[{piece.colour}{piece.piece_letter}{piece.number}]"
                row.append(piece_str)
            else:
                row.append("[   ]")
                
        rows.append("".join(row))
    
    # Combine header and board
    return "\n".join(header + [""] + rows)



if __name__ == "__main__":
    from hive.game.pieces.ant import Ant
    from hive.game.types_and_errors import WHITE, BLACK

    grid = {(6,2): Ant(BLACK),
            (5,1): Ant(WHITE),
            (7,1): Ant(WHITE),
            (8,2): Ant(WHITE),
            (7,3): Ant(WHITE),
            (5,3): Ant(WHITE),
            (4,2): Ant(WHITE),}

    game = Game(grid=grid)
    print(large_game_to_text(game))


