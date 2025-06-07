import os
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import mlflow
import mlflow.pytorch

from hive.ml.model.components.model import HiveLightningModel
from hive.ml.model.dataloader import HiveDataModule
from hive.ml.model.models import create_hive_gatv2_gnn


if __name__ == "__main__":
    # --- Configuration ---
    TOTAL_EPOCHS = 10
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    LEARNING_RATE = 0.01

    folder = Path(__file__).parents[3]
    filepath = f"{folder}/game_strings/combined.txt"
    mlflow_dir = f"{folder}/mlruns"
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")

    # --- 1. Set up Data ---
    data_module = HiveDataModule(
        data_path=filepath,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # --- 2. Set up Model ---
    model = create_hive_gatv2_gnn(hidden_dim=16,
                                  num_layers=1,
                                  heads=1,
                                  dropout=0.05,
                                  residual=False,
                                  batch_norm=False,
                                  pool_method='add')

    lightning_model = HiveLightningModel(model=model,
                                         learning_rate=LEARNING_RATE)

    # --- 3. Set up Logging & Checkpointing ---
    # MLflow logger automatically logs metrics, hparams, and model artifacts
    mlflow_logger = MLFlowLogger(
        experiment_name="hive_model_training_lightning",
        tracking_uri=f"file://{mlflow_dir}"
    )

    # Checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss_epoch',  # The metric to monitor
        dirpath='lightning_checkpoints/',
        filename='hive-model-{epoch:02d}-{train_loss_epoch:.2f}',
        save_top_k=1,  # Save only the best model
        mode='min',
    )

    # --- 4. Set up Trainer ---
    # The Trainer automates everything
    trainer = Trainer(
        accelerator="auto",  # Automatically uses GPU/MPS if available
        devices="auto",  # Automatically uses all available devices
        max_epochs=TOTAL_EPOCHS,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
    )

    # --- 5. Start Training! ---
    trainer.fit(lightning_model,
                datamodule=data_module)

    print("\nTraining completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"View results in MLflow UI by running: mlflow ui --backend-store-uri {mlflow_dir}")