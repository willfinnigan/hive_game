from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from hive.game_engine.player_functions import get_players_possible_moves_or_placements
from hive.render.to_text import game_to_text
from hive.trajectory.boardspace import MoveString, replay_trajectory
from hive.game_engine.moves import Move, NoMove, is_pillbug_move


@dataclass
class GameString:
    units: str
    result: str
    turn: str
    moves: List[MoveString]

def load_replay_game_strings(filepath, start_end_idx: Optional[Tuple[int, int]]=None) -> List[GameString]:
    # each line is a game
    game_strings = []
    with open(filepath, "r") as f:

        # only read the lines between start_idx and end_idx
        if start_end_idx is not None:
            lines = f.readlines()[start_end_idx[0]:start_end_idx[1]]
        else:
            lines = f.readlines()

        for line in lines:
            try:
                parts = line.split(";")
                game_strings.append(GameString(units=parts[0],
                                               result=parts[1],
                                               turn=parts[2],
                                               moves=[MoveString(mv) for mv in parts[3:]]))
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(e)
                continue

    return game_strings


def print_game_text_rep(filepath, idx):

    # Find the game string for debugging - get this from the file
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if idx < len(lines):
            game_string = lines[idx].strip()
            print(game_string)
        else:
            print(f"Game string index {idx} out of range")


if __name__ == "__main__":
    # command to clear the terminal
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

    filepath = f"{Path(__file__).parents[2]}/game_strings/combined.txt"
    idx = 4895

    game_strings = load_replay_game_strings(filepath, start_end_idx=(idx, idx + 1))
    game_string = game_strings[0]

    game = replay_trajectory(game_string.moves, game_string.turn)

    game_moves = []
    game_moves.append(game)
    while game.move is not None:
        game = game.parent
        game_moves.append(game)

    game_moves.reverse()

    print(game_to_text(game))
    print()
    for i, game in enumerate(game_moves[0:-1]):
        print(game.current_turn, dict(game.player_turns))
        next_game_state = game_moves[i + 1]
        move = next_game_state.move
        
        print(str(move))
            
        print(game_to_text(next_game_state))
        print()

        turn_colour = move.colour
        possible_moves = get_players_possible_moves_or_placements(turn_colour, game)

        matches = [mv for mv in possible_moves if mv == move]

        if len(matches) == 0 and possible_moves != []:
            print()
            print()

            for mv in possible_moves:
                print(f"{mv} != {move}")


            print()
            print(dict(game.grid))

            print()
            print()
            print_game_text_rep(filepath, idx)
            print()
            raise ValueError(f"Move not in possible moves")

        # game = game.parent
