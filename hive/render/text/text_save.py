from hive.game.game import Game
from hive.game.pieces.ant import Ant
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.piece_base_class import Piece
from hive.game.pieces.queen import Queen
from hive.game.pieces.spider import Spider


def game_to_string(game: Game) -> str:
    """
    Convert a game to a string representation that can be parsed back.
    
    Format: 
    HIVE:turns=W:n,B:n;pieces=(q,r):CPn,(q,r):CPn,...
    
    Where:
    - C is the color (W/B)
    - P is the piece type (Q=Queen, A=Ant, B=Beetle, G=Grasshopper, S=Spider, P=Generic Piece)
    - n is the piece number
    
    Args:
        game (Game): The game to convert
        
    Returns:
        str: A string representation of the game
    """
    # Build the turns part
    turns_part = []
    for color, turns in game.player_turns.items():
        turns_part.append(f"{color}:{turns}")
    
    # Build the pieces part
    pieces_part = []
    for loc, piece in game.grid.items():
        pieces_part.append(f"({loc[0]},{loc[1]}):{piece.colour}{piece.piece_letter}{piece.number}")
    
    # Combine into the final string
    return f"HIVE:turns={','.join(turns_part)};pieces={','.join(pieces_part)}"




def string_to_game(game_str: str) -> Game:
    """
    Parse a game string representation back into a Game object.
    
    Args:
        game_str (str): The string representation of the game
        
    Returns:
        Game: The reconstructed game
    """
    if not game_str.startswith("HIVE:"):
        raise ValueError("Invalid game string format")
    
    # Split the string into parts
    parts = game_str[5:].split(';')
    if len(parts) != 2:
        raise ValueError("Invalid game string format")
    
    turns_str = parts[0]
    pieces_str = parts[1]
    
    # Parse turns
    if not turns_str.startswith("turns="):
        raise ValueError("Invalid turns format")
    
    turns_data = turns_str[6:].split(',')
    player_colours = []
    player_turns = {}
    
    for turn_data in turns_data:
        color, turns = turn_data.split(':')
        player_colours.append(color)
        player_turns[color] = int(turns)
    
    # Create a new game with the player colors
    game = Game(player_colours=player_colours)
    game.player_turns = player_turns
    
    # Parse pieces
    if not pieces_str.startswith("pieces="):
        raise ValueError("Invalid pieces format")
    
    # Use regex to parse the pieces data correctly
    piece_pattern = r'\((-?\d+),(-?\d+)\):([WB])([QABGSP])(\d+)'
    pieces_matches = re.findall(piece_pattern, pieces_str)
    
    grid = {}
    
    for match in pieces_matches:
        try:
            q = int(match[0])
            r = int(match[1])
            location = (q, r)
            
            color = match[2]
            piece_type = match[3]
            number = int(match[4])
            
            # Create the appropriate piece
            piece = create_piece(piece_type, color, number)
            
            # Add to grid
            grid[location] = piece
            piece.location = location
            
            # Update queen reference if this is a queen
            if piece_type == 'Q':
                game.queens[color] = piece
        except Exception as e:
            print(f"Error parsing piece match: {match}")
            raise ValueError(f"Failed to parse piece data: {match}") from e
    
    game.grid = grid
    return game


def create_piece(piece_type: str, color: str, number: int) -> Piece:
    """
    Create a piece of the specified type.
    
    Args:
        piece_type (str): The type of piece to create (Q, A, B, G, S, P)
        color (str): The color of the piece (W, B)
        number (int): The number of the piece
        
    Returns:
        Piece: The created piece
    """
    if piece_type == 'Q':
        return Queen(color, number)
    elif piece_type == 'A':
        return Ant(color, number)
    elif piece_type == 'B':
        return Beetle(color, number)
    elif piece_type == 'G':
        return GrassHopper(color, number)
    elif piece_type == 'S':
        return Spider(color, number)
    else:  # Default to generic piece
        piece = Piece(color, number)
        piece.piece_letter = piece_type
        return piece
