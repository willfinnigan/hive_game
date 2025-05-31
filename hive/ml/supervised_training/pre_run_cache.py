from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from hive.ml.data.dataset import HiveLazyGameDataset, collate_fn

# fix for mac
import multiprocessing
multiprocessing.set_start_method('fork', force=True)



folder = Path(__file__).parents[3]
filepath = f"{folder}/game_strings/combined.txt"

train_dataset = HiveLazyGameDataset(filepath, batch_size=64)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    prefetch_factor=3,
    collate_fn=collate_fn)

progress_bar = tqdm(train_loader)

for batch_idx, batch_data in enumerate(progress_bar):
    pass

