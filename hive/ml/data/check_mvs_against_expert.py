

from pathlib import Path

from hive.game_engine.player_functions import get_players_possible_moves_or_placements
from hive.render.to_text import game_to_text
from hive.trajectory.game_dataloader import GameDataLoader


if __name__ == '__main__':
    filepath = f"{Path(__file__).parents[3]}/game_strings/combined.txt"
    batch_size = 100
    loader = GameDataLoader(filepath, batch_size=batch_size)
    total_batches = (len(loader) + batch_size - 1) // batch_size
    
    print(f"Total games: {len(loader)}")
    print("Loading first batch...")

    # Create iterator once
    loader_iter = iter(loader)
    
    for i in range(total_batches):
        games = next(loader_iter)
        for j, game in enumerate(games):
            print(f"Game {i * batch_size + j + 1}/{len(loader)}")
            # want to check that the moves made in the game match the possible moves identified by the engine
            while game.move is not None:
                #print(f"{game.parent.player_turns} - Game turn={game.parent.current_turn} - move={game.move}")
                turn_colour = game.parent.current_turn
                #print(f"Turn: {turn_colour} - Move: {game.move}")
                possible_moves = get_players_possible_moves_or_placements(turn_colour, game.parent)

                #print(game.move)
                #print(game.parent.move)

                matches = [mv for mv in possible_moves if mv == game.move]

                if len(matches) == 0 and possible_moves != []:
                    print(game.player_turns)
                    print(game.move)
                    print(game_to_text(game))
                    print(game_to_text(game.parent))

                    for mv in possible_moves:
                        print(f"{mv} != {game.move}")

                    # Find the game string for debugging - get this from the file
                    game_string_idx = i * batch_size + j
                    print(f"Game string index: {game_string_idx}")
                    filepath = f"{Path(__file__).parents[3]}/game_strings/combined.txt"
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                        if game_string_idx < len(lines):
                            game_string = lines[game_string_idx].strip()
                            print(game_string)
                        else:
                            print(f"Game string index {game_string_idx} out of range")
                    
                    print(game.parent.grid)
                    raise ValueError(f"Move not in possible moves")
                
                

                game = game.parent
    