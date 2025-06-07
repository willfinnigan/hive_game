from torch.utils.data import DataLoader
from hive.ml.data.dataset import HiveLazyGameDataset, collate_fn
import lightning as L

class HiveDataModule(L.LightningDataModule):
    """
    Lightning DataModule for Hive game data.
    
    This module handles the loading and batching of Hive game data for training.
    Uses a custom collate function to properly batch PyTorch Geometric data.
    """
    
    def __init__(self, data_path: str, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size  # This is the effective batch size after collation
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        """Set up the dataset for the given stage."""
        # The dataset's internal batch_size controls how many games are processed together
        # This is separate from the DataLoader's batch_size
        self.train_dataset = HiveLazyGameDataset(self.data_path, batch_size=100)

    def train_dataloader(self):
        """
        Create the training DataLoader.
        
        The DataLoader will batch multiple dataset items (each containing lists of
        Data objects from one game) and our collate_fn will flatten and batch them
        into a single PyG Batch object.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,  # Batch multiple games together
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=collate_fn
        )

