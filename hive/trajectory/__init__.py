"""
Trajectory module for saving and loading Hive game trajectories using the BoardSpace move notation format.

This module provides functionality to:
1. Parse and generate BoardSpace move notation strings
2. Convert between internal move representation and BoardSpace notation
3. Save and load complete game trajectories (sequences of moves)
"""

from hive.trajectory.boardspace import (
    MoveString,
    move_to_boardspace,
    boardspace_to_move,
    save_trajectory,
    load_trajectory,
    replay_trajectory,
    record_game
)