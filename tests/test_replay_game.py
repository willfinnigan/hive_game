from pathlib import Path

from tqdm import tqdm

from hive.game_engine.game_state import initial_game
from hive.render.to_text import game_to_text
from hive.trajectory.boardspace import MoveString, replay_trajectory, boardspace_to_move
from hive.trajectory.game_string import load_replay_game_strings

data_dir = Path(__file__).parents[0] / "data"
def test_load_game():
    # Test loading a game string
    filepath = f"{data_dir}/BoardGameArena_Base+MLP+NoBots_20240704_110945.txt"
    game_strings = load_replay_game_strings(filepath)

    i = 2
    print(game_strings[i])

    game = replay_trajectory(game_strings[i].moves)

    while game.move is not None:
        print(game.move)
        print(game_to_text(game))
        game = game.parent



def test_load_all_games():
    #filepath = f"../game_strings/BoardGameArena_Base+MLP+NoBots_20240704_110945.txt"
    #game_strings = load_replay_game_strings(filepath)

    # could_not_load = []
    # for i, game_string in enumerate(game_strings):
    #     try:
    #         game = replay_trajectory(game_string.moves)
    #     except:
    #         could_not_load.append((i, game_string))
    #         print(f"Could not load game {i}: {game_string}")
    #         continue

    filepath = f"../game_strings/BoardSpace_Base+MLP+NoBots_20240703_135823.txt"
    game_strings = load_replay_game_strings(filepath)

    could_not_load = []
    game = replay_trajectory(game_strings[37].moves)

    while game.move is not None:
        print(game.move)
        print(game_to_text(game))
        game = game.parent




