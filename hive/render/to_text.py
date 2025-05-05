from hive.game_engine.game_state import Piece, WHITE, BLACK, Game
from hive.game_engine.game_moves import has_player_lost, get_winner
from hive.game_engine import pieces

red = "\033[31m"
blue = "\033[34m"
white = "\033[37m"
black = "\033[30m"
darkblue = "\033[36m"
yellow = "\033[33m"
reset = "\033[0m"


def piece_as_text(piece, highlight=False, include_number=True):
    # Handle the case where piece is an integer
    if isinstance(piece, int):
        return f"{yellow}{piece}{reset}"
    
    # Get the piece letter
    if hasattr(piece.name, 'name'):  # Handle case where piece.name is an object with a name attribute
        piece_letter = piece.name.name[0]
    else:  # Handle case where piece.name is a string
        piece_letter = piece.name[0]
    
    if highlight:
        return f"{red}{piece_letter}{piece.number if include_number else ''}{reset}"
    elif piece.colour == WHITE:
        return f"{yellow}{piece_letter}{piece.number if include_number else ''}{reset}"
    else:
        return f"{darkblue}{piece_letter}{piece.number if include_number else ''}{reset}"


def game_to_text(game, highlight_piece_at=None, show_moves=None):
    if not game.grid:
        return "Empty board"
        
    min_r = min([loc[1] for loc in game.grid.keys()])
    max_r = max([loc[1] for loc in game.grid.keys()])
    max_q = max([loc[0] for loc in game.grid.keys()])
    min_q = min([loc[0] for loc in game.grid.keys()])


    if min_q % 2 == 1:
        min_q -= 1

    game_text = ""
    for r in range(min_r, max_r+1):
        row = f""
        if r % 2 != 0:
            row += "  "
        for q in range(min_q, max_q+2, 2):
            if (q + r) % 2 == 1:
                q += 1
            stack = game.grid.get((q, r), ())
            if stack:
                piece = stack[-1]
                if (q, r) == highlight_piece_at:
                    row += f"{piece_as_text(piece, highlight=True)}"
                else:
                    row += f"{piece_as_text(piece)}"
            else:
                row += "--"
            row += "  "
        game_text += row + "\n"

    return game_text



def game_to_text_small(game):
    min_r = min([loc[1] for loc in game.grid.keys()])
    max_r = max([loc[1] for loc in game.grid.keys()])
    max_q = max([loc[0] for loc in game.grid.keys()])
    min_q = min([loc[0] for loc in game.grid.keys()])

    if min_q % 2 == 1:
        min_q -= 1

    game_text = ""
    for r in range(min_r, max_r+1):
        row = f""
        if r % 2 != 0:
            row += " "
        for q in range(min_q, max_q+2, 2):
            if (q + r) % 2 == 1:
                q += 1
            stack = game.grid.get((q, r), ())
            if stack:
                piece = stack[-1]  # Get the top piece from the stack
                row += f"{piece_as_text(piece, include_number=False)}"
            else:
                row += "-"
            row += " "
        game_text += row + "\n"

    return game_text


def large_game_to_text(game: Game, include_coordinates: bool = False, use_colors: bool = True) -> str:
    """
    Render a game state in a large text format.
    
    Args:
        game (Game): The game to render
        include_coordinates (bool): Whether to include coordinate labels
        use_colors (bool): Whether to use color coding for pieces
        
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
        color_code = "W" if color == WHITE else "B"
        turns_info.append(f"{color_code}:{turns}")
    header.append("Turns: " + ", ".join(turns_info))
    
    # Add queen status information
    queens_info = []
    for color in game.player_turns.keys():
        color_code = "W" if color == WHITE else "B"
        queen_loc = game.queens.get(color)
        if queen_loc is None:
            queens_info.append(f"{color_code}:Not placed")
        else:
            queens_info.append(f"{color_code}:Placed")
    header.append("Queens: " + ", ".join(queens_info))
    
    # Check if any player has lost
    for color in game.player_turns.keys():
        if has_player_lost(game, color):
            color_code = "W" if color == WHITE else "B"
            header.append(f"Player {color_code} has lost!")
    
    # Check if there's a winner
    winner = get_winner(game)
    if winner:
        winner_code = "W" if winner == WHITE else "B"
        header.append(f"Winner: {winner_code}")
    
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
                
            stack = game.grid.get((q, r), ())
            if stack:
                # Get the top piece from the stack
                piece = stack[-1]
                # Format: [ColorPieceNumber]
                color_code = "W" if piece.colour == WHITE else "B"
                piece_letter = piece.name.name[0]
                
                if use_colors:
                    # Use the same color coding as in piece_as_text
                    if piece.colour == WHITE:
                        piece_str = f"[{yellow}{color_code}{piece_letter}{piece.number}{reset}]"
                    else:
                        piece_str = f"[{darkblue}{color_code}{piece_letter}{piece.number}{reset}]"
                else:
                    piece_str = f"[{color_code}{piece_letter}{piece.number}]"
                    
                row.append(piece_str)
            else:
                row.append("[   ]")
                
        rows.append("".join(row))
    
    # Combine header and board
    return "\n".join(header + [""] + rows)

if __name__ == "__main__":
    grid = {(6,2): (Piece(BLACK, pieces.ANT, 1),),
            (5,1): (Piece(WHITE, pieces.ANT, 2),),
            (7,1): (Piece(BLACK, pieces.ANT, 3),),
            (8,2): (Piece(WHITE, pieces.ANT, 4),),
            (7,3): (Piece(BLACK, pieces.ANT, 5),),
            (5,3): (Piece(WHITE, pieces.ANT, 6),),
            (4,2): (Piece(BLACK, pieces.ANT, 6),)}
    
    game = Game(
        grid=grid,
        player_turns={WHITE: 0, BLACK: 0},
        queens={}
    )
    print(game_to_text(game, highlight_piece_at=(6,2)))

    print("\nSmall representation:")
    print(game_to_text_small(game))

    print("\nLarge representation with colors:")
    print(large_game_to_text(game, use_colors=True))
