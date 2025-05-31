

"""Create a split of the trajectory dataset for training and validation."""
from pathlib import Path
from typing import List

from tqdm import tqdm

from hive.game_engine.game_functions import get_winner
from hive.game_engine.game_state import Game
from hive.trajectory.game_dataloader import GameDataLoader


filepath = Path(__file__).parents[2] / 'game_strings' / 'combined.txt'
loader = GameDataLoader(filepath)

idxs_with_winner = []
for i in tqdm(range(len(loader))):
    game = loader.get_game(i)
    if get_winner(game) is None:
        continue
    else:
        idxs_with_winner.append(i)

print(f"Total games with winner: {len(idxs_with_winner)}")

# Now create a random split of these indices
import random
random.seed(42)  # For reproducibility
random.shuffle(idxs_with_winner)
split_index = int(0.8 * len(idxs_with_winner))  # 80% for training, 20% for validation
train_idxs = idxs_with_winner[:split_index]
test_idxs = idxs_with_winner[split_index:]

# Using the idxs, create the train and test datasets from combined.txt, writing new files.
train_filepath = filepath.parent / 'train_games.txt'
test_filepath = filepath.parent / 'test_games.txt'

with open(train_filepath, 'w') as train_file, open(test_filepath, 'w') as test_file:
    # copy the relevant lines directly from the original file
    with open(filepath, 'r') as original_file:
        for i, line in enumerate(original_file):
            if i in train_idxs:
                train_file.write(line)
            elif i in test_idxs:
                test_file.write(line)

print(f"Train games written to {train_filepath}")
print(f"Test games written to {test_filepath}")





