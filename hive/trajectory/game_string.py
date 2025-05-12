from dataclasses import dataclass
from typing import List

from hive.trajectory.boardspace import MoveString


@dataclass
class GameString:
    units: str
    result: str
    turn: str
    moves: List[MoveString]

def load_replay_game_strings(filepath) -> List[GameString]:

    # each line is a game
    game_strings = []
    with open(filepath, "r") as f:
        for line in f:
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



