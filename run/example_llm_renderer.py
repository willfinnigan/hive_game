from hive.game.game import Game
from hive.game.pieces.ant import Ant
from hive.game.pieces.beetle import Beetle
from hive.game.pieces.grasshopper import GrassHopper
from hive.game.pieces.queen import Queen
from hive.game.pieces.spider import Spider
from hive.game.types_and_errors import WHITE, BLACK
from hive.render.text.large_print import large_game_to_text, game_to_string, string_to_game


def create_test_game():
    """Create a test game with various pieces for demonstration"""
    # Create pieces first
    white_queen = Queen(WHITE, 0)
    black_queen = Queen(BLACK, 0)
    white_ant = Ant(WHITE, 1)
    black_ant = Ant(BLACK, 1)
    black_spider = Spider(BLACK, 1)
    white_beetle = Beetle(WHITE, 1)
    white_grasshopper = GrassHopper(WHITE, 1)
    black_grasshopper = GrassHopper(BLACK, 1)
    
    # Set locations
    white_queen.location = (0, 0)
    black_queen.location = (3, -1)
    white_ant.location = (-2, 0)
    black_ant.location = (2, 0)
    black_spider.location = (4, 0)
    white_beetle.location = (1, 1)
    white_grasshopper.location = (-1, -1)
    black_grasshopper.location = (3, 1)
    
    # Create grid
    grid = {
        (0, 0): white_queen,
        (2, 0): black_ant,
        (-2, 0): white_ant,
        (4, 0): black_spider,
        (1, 1): white_beetle,
        (3, -1): black_queen,
        (-1, -1): white_grasshopper,
        (3, 1): black_grasshopper
    }
    
    # Create game
    game = Game(grid=grid)
    game.queens[WHITE] = white_queen
    game.queens[BLACK] = black_queen
    game.player_turns[WHITE] = 4
    game.player_turns[BLACK] = 4
    
    return game


def test_rendering():
    """Test the rendering functionality"""
    game = create_test_game()
    
    print("=" * 50)
    print("LLM-FRIENDLY RENDERER TEST")
    print("=" * 50)
    
    # Test basic rendering
    print("\n1. Basic Rendering:")
    print(large_game_to_text(game))
    
    # Test rendering with coordinates
    print("\n2. Rendering with Coordinates:")
    print(large_game_to_text(game, include_coordinates=True))
    
    # Test game string conversion
    print("\n3. Game String Representation:")
    game_str = game_to_string(game)
    print(game_str)
    
    # Test parsing game string back to game
    print("\n4. Parsed Game from String:")
    parsed_game = string_to_game(game_str)
    print(large_game_to_text(parsed_game))
    
    # Verify the parsed game matches the original
    print("\n5. Verification - Original vs Parsed:")
    original_render = large_game_to_text(game)
    parsed_render = large_game_to_text(parsed_game)
    if original_render == parsed_render:
        print("SUCCESS: Parsed game matches the original!")
    else:
        print("ERROR: Parsed game does not match the original.")
        print("Original:")
        print(original_render)
        print("Parsed:")
        print(parsed_render)


if __name__ == "__main__":
    test_rendering()