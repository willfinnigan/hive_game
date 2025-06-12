import os
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.loggers import WandbLogger

from hive.ml.model.dataloader import WebdatasetHiveDataModule 
from hive.ml.model.lightning_wrapper import HiveLightningModel
from hive.ml.model.models import create_hive_gatv2_gnn


def train_hive_model(model,
                     data_directory: str,
                     experiment_name: str,
                     checkpoint_dir: str,
                     total_epochs: int,
                     batch_size: int,
                     num_workers: int,
                     learning_rate: float,
                     shuffle_buffer_size: int,
                     task_weights: dict,
                     project_name: str = "hive_model_training",):
    

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 1. Set up Data ---
    data_module = WebdatasetHiveDataModule(
        data_dir=data_directory,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_buffer_size=shuffle_buffer_size
    )

    lightning_model = HiveLightningModel(model=model,
                                         learning_rate=learning_rate,
                                         task_weights=task_weights)

    wandb_logger = WandbLogger(log_model="all",
                               project=project_name,
                               name=experiment_name)

    # Checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss_epoch',  # The metric to monitor
        dirpath=f'{checkpoint_dir}/lightning_checkpoints/',
        filename='hive-model-{epoch:02d}-{train_loss_epoch:.2f}',
        mode='min',
        save_top_k=-1,  # Save every epoch
        every_n_epochs=1
    )

    # --- 4. Set up Trainer ---
    data_module.setup()  # self.num_train_batches attribute gets set.
    trainer = Trainer(
        #precision="16-mixed",
        accelerator="auto",  # Automatically uses GPU/MPS if available
        devices="auto",  # Automatically uses all available devices
        max_epochs=total_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        limit_train_batches=data_module.num_train_batches  # for progress bar
    )

    # --- 5. Start Training! ---
    trainer.fit(lightning_model,
                datamodule=data_module)

    print("\nTraining completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")






if __name__ == "__main__":

    folder = Path(__file__).parents[3]

    experiment_name="hive_model_training_lightning"
    filepath = f"{folder}/game_strings/combined.txt"
    data_directory = f"{filepath}.webdataset_1000_games"
    checkpoint_dir = f"{folder}/lightning_checkpoints"

    model = create_hive_gatv2_gnn(hidden_dim=8,
                                  num_layers=2,
                                  heads=1,
                                  dropout=0.05,
                                  residual=False,
                                  batch_norm=False,
                                  task_heads=["value", "mobile_pieces"],
                                  pool_method='mean')
    
    train_hive_model(model=model,
                     data_directory=data_directory,
                     checkpoint_dir=checkpoint_dir,
                     experiment_name=experiment_name,
                     total_epochs=10,
                     batch_size=64,
                     num_workers=4,
                     learning_rate=0.01,
                     shuffle_buffer_size=10000,
                     task_weights={"value": 1, 
                                   "mobile_pieces": 1},)
    

