from torch_geometric.loader import DataLoader
from hive.ml.data.dataset import HiveLazyGameDataset, collate_fn
import lightning as L

class HiveDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        # This is called on every GPU in DDP training
        self.train_dataset = HiveLazyGameDataset(self.data_path, batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=collate_fn
        )

