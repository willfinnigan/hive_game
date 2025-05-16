from dataclasses import dataclass
from typing import List, Optional, Tuple

from hive.trajectory.boardspace import MoveString


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



