# In your dataloader.py file

import glob
import json
import os
from io import BytesIO

import torch
import lightning as L
import webdataset as wds # Ensure you have this import
from torch_geometric.data import Batch
# We no longer need the standard DataLoader or IterableDataset here

def pyg_collate(batch):
    """
    This function is still needed for the .batched() stage inside the pipeline.
    """
    return Batch.from_data_list([sample for sample in batch if sample is not None])

class WebdatasetHiveDataModule(L.LightningDataModule):
    # __init__ and decode_sample are the same
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle_buffer_size: int = 10000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.approx_samples_per_epoch = metadata["total_samples"]
            print(f"Successfully loaded metadata. Found {self.approx_samples_per_epoch} samples.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"metadata.json not found in {self.data_dir}. "
                "Please run the data preparation script to generate the dataset and its metadata file."
            )
        self.save_hyperparameters()

    @staticmethod
    def decode_sample(sample):
        with BytesIO(sample["pt"]) as buffer:
            return torch.load(buffer, weights_only=False)


    def setup(self, stage: str = None):
        glob_pattern = os.path.join(self.data_dir, "chunk-*-shard-*.tar")
        self.urls = glob.glob(glob_pattern)
        
        # check to ensure we found some files
        if not self.urls:
            raise FileNotFoundError(f"No .tar files found matching the pattern: {glob_pattern}")
            
        print(f"Found {len(self.urls)} shard files.")

        # Calculate and store the number of batches here!
        if stage == "fit" or stage is None:
            world_size = self.trainer.world_size if self.trainer else 1
            self.num_train_batches = (self.approx_samples_per_epoch // self.batch_size) // world_size


    def train_dataloader(self):

        dataset = wds.DataPipeline(
            wds.WebDataset(self.urls, shardshuffle=200, nodesplitter=wds.split_by_node),
            wds.shuffle(self.shuffle_buffer_size),
            wds.map(self.decode_sample),
            wds.batched(self.batch_size, collation_fn=pyg_collate),
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )
        
        return loader
